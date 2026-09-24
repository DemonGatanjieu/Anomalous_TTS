"""GPT-SoVITS inference: turns a Plan (core/planner.py) into audio.

The per-sentence flow follows ``get_tts_wav`` in GPT-SoVITS
``inference_webui.py`` (MIT, RVC-Boss). On top of it:

- sentences that share a voice are decoded together in batches
  (core/t2s_batch.py), each with its own random stream;
- semantic tokens and audio are cached per sentence, so re-running a script
  after editing one line only regenerates that line.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from . import t2s_batch, text_frontend
from .audio import load_mono, resample, silence
from .models import LRU, GPTModel, Resources, SoVITSModel, file_key
from .planner import Gap, Line, Plan, Voice

log = logging.getLogger("Anomalous_TTS")

HZ = 50  # semantic tokens per second
REF_MIN_SEC = 3.0
REF_MAX_SEC = 10.0
AUDIO_CACHE_BYTES = 256 * 1024 * 1024


@dataclass
class SynthesisParams:
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    repetition_penalty: float = 1.35
    speed: float = 1.0
    pause_sec: float = 0.3
    batch_size: int = 8

    def sampling_key(self) -> Tuple:
        return (int(self.top_k), float(self.top_p), float(self.temperature), float(self.repetition_penalty))


@dataclass
class Report:
    lines: int = 0
    generated: int = 0
    from_cache: int = 0
    batches: List[int] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.lines} 句：新生成 {self.generated} 句（{len(self.batches)} 批），"
            f"缓存 {self.from_cache} 句"
        )


@dataclass
class _Reference:
    prompt: Optional[torch.Tensor]  # (P,) semantic tokens, None = no reference text
    phones: List[int]
    bert: Optional[torch.Tensor]  # (1024, len(phones))
    spec: torch.Tensor
    sv_emb: Optional[torch.Tensor]


class Engine:
    """Holds loaded models and caches. One instance per device/dtype."""

    def __init__(self, resources: Resources, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self.resources = resources
        self._lock = threading.Lock()
        self._hubert = None
        self._roberta = None
        self._sv = None
        self._zh_ready = False
        self._en_ready = False
        self._warned: Dict[str, bool] = {}
        self._gpt = LRU(capacity=3)
        self._sovits = LRU(capacity=3)
        self._refs = LRU(capacity=32)
        self._tokens = LRU(capacity=4096)
        self._audio = LRU(max_bytes=AUDIO_CACHE_BYTES, sizeof=lambda a: a[0].nbytes)

    # ================= models =================
    def gpt(self, path: str) -> GPTModel:
        return self._gpt.get_or_create(file_key(path), lambda: GPTModel(path, self.device, self.dtype))

    def sovits(self, path: str) -> SoVITSModel:
        return self._sovits.get_or_create(file_key(path), lambda: SoVITSModel(path, self.device, self.dtype))

    def hubert(self):
        if self._hubert is None:
            from transformers import HubertModel

            model = HubertModel.from_pretrained(self.resources.hubert_dir(), local_files_only=True)
            self._hubert = model.to(device=self.device, dtype=self.dtype).eval()
        return self._hubert

    def roberta(self):
        if self._roberta is None:
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            path = self.resources.roberta_dir()
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            model = AutoModelForMaskedLM.from_pretrained(path, local_files_only=True)
            self._roberta = (tokenizer, model.to(device=self.device, dtype=self.dtype).eval())
        return self._roberta

    def sv(self):
        """ERes2NetV2 speaker encoder for v2Pro / v2ProPlus (GPT-SoVITS ``sv.py``)."""
        if self._sv is None:
            from ..vendor.gpt_sovits.eres2net.ERes2NetV2 import ERes2NetV2

            state = torch.load(self.resources.sv_path(), map_location="cpu", weights_only=True)
            model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
            model.load_state_dict(state)
            self._sv = model.to(device=self.device, dtype=self.dtype).eval()
        return self._sv

    def unload(self) -> None:
        """Free models (called by ComfyUI's "unload models"). Sentence caches are kept: they are small."""
        with self._lock:
            self._hubert = self._roberta = self._sv = None
            self._gpt.clear()
            self._sovits.clear()
            self._refs.clear()

    # ================= text_frontend.Context =================
    def bert(self, norm_text: str, word2ph: List[int]) -> torch.Tensor:
        """GPT-SoVITS ``get_bert_feature``: 3rd-last hidden layer, repeated per phoneme."""
        tokenizer, model = self.roberta()
        inputs = tokenizer(norm_text, return_tensors="pt").to(self.device)
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].float().cpu()[1:-1]
        assert len(word2ph) == len(norm_text) == res.shape[0], (len(word2ph), len(norm_text), res.shape)
        feats = [res[i].repeat(word2ph[i], 1) for i in range(len(word2ph))]
        return torch.cat(feats, dim=0).T

    def prepare(self, lang: str) -> None:
        if lang == "zh":
            self._prepare_chinese()
        elif lang == "en":
            self._prepare_english()

    def can_use(self, lang: str) -> bool:
        """For mixed text: False if the language's packages or data are missing."""
        try:
            self.prepare(lang)
            return True
        except Exception as e:
            if not self._warned.get(lang):
                self._warned[lang] = True
                log.warning("[Anomalous_TTS] %s，文字里的这部分会按主语言处理。", e)
            return False

    def _prepare_english(self) -> None:
        if self._en_ready:
            return
        missing = [m for m in ("g2p_en", "wordsegment", "nltk") if importlib.util.find_spec(m) is None]
        if missing:
            raise RuntimeError(
                f"英语需要安装 {'、'.join(missing)}（ComfyUI 便携版：python_embeded\\python.exe -m pip install "
                f"{' '.join(missing)}）"
            )
        dict_dir, cache_dir, nltk_dir = self.resources.english_dirs()
        import nltk

        if nltk_dir not in nltk.data.path:
            nltk.data.path.insert(0, nltk_dir)
        from ..vendor.gpt_sovits.text import english

        english.configure(dict_dir, cache_dir)
        self._en_ready = True

    def _prepare_chinese(self) -> None:
        """Use g2pW for polyphones when available, like GPT-SoVITS does."""
        if self._zh_ready:
            return
        self._zh_ready = True
        from ..vendor.gpt_sovits.text import chinese2

        if importlib.util.find_spec("opencc") is None:  # g2pW converts to Traditional Chinese first
            log.warning("[Anomalous_TTS] 未安装 opencc，中文多音字改用 pypinyin 判断，准确率会下降。")
            return
        g2pw_dir = self.resources.g2pw_dir()
        if not g2pw_dir:
            log.warning("[Anomalous_TTS] 没有 G2PWModel，中文多音字改用 pypinyin 判断，准确率会下降。")
            return
        try:
            chinese2.enable_g2pw(g2pw_dir, self.resources.roberta_dir())
        except Exception as e:
            log.warning("[Anomalous_TTS] G2PWModel 加载失败（%s），中文多音字改用 pypinyin 判断。", e)

    # ================= reference audio =================
    def _prompt_semantic(self, sovits: SoVITSModel, wav_path: str) -> torch.Tensor:
        wav, sr = load_mono(wav_path)
        wav16k = resample(wav, sr, 16000)
        seconds = len(wav16k) / 16000
        if not REF_MIN_SEC <= seconds <= REF_MAX_SEC:
            raise ValueError(f"参考音频 {os.path.basename(wav_path)} 长 {seconds:.1f} 秒，需要在 3~10 秒之间。")
        # GPT-SoVITS appends 0.3 s of silence measured at the model rate (quirk kept on purpose).
        x = torch.from_numpy(np.concatenate([wav16k, silence(0.3, sovits.sampling_rate)])).to(self.device, self.dtype)
        ssl = self.hubert()(x.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
        return sovits.model.extract_latent(ssl)[0, 0]

    def _spec_and_sv(self, sovits: SoVITSModel, wav_path: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """GPT-SoVITS ``get_spepc``: spectrogram at the model rate, plus the v2Pro speaker embedding."""
        import torchaudio.functional as AF

        from ..vendor.gpt_sovits.module.mel_processing import spectrogram_torch

        wav, sr = load_mono(wav_path)
        audio = torch.from_numpy(wav).unsqueeze(0).float()
        if sr != sovits.sampling_rate:
            audio = AF.resample(audio, sr, sovits.sampling_rate)
        maxx = audio.abs().max()
        if maxx > 1:
            audio /= min(2, maxx)
        spec = spectrogram_torch(
            audio.to(self.device), sovits.filter_length, sovits.sampling_rate,
            sovits.hop_length, sovits.win_length, center=False,
        )
        sv_emb = None
        if sovits.is_v2pro:
            from ..vendor.gpt_sovits.eres2net import kaldi

            wav16 = AF.resample(audio, sovits.sampling_rate, 16000).to(self.device, self.dtype)
            feat = torch.stack(
                [kaldi.fbank(w.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0) for w in wav16]
            )
            sv_emb = self.sv().forward3(feat)
        return spec.to(self.dtype), sv_emb

    def _reference(self, voice: Voice, sovits: SoVITSModel) -> _Reference:
        key = (file_key(voice.sovits_path), file_key(voice.ref_wav), voice.ref_text, voice.ref_lang)

        def build() -> _Reference:
            spec, sv_emb = self._spec_and_sv(sovits, voice.ref_wav)
            if not voice.ref_text:
                return _Reference(None, [], None, spec, sv_emb)
            prompt = self._prompt_semantic(sovits, voice.ref_wav)
            phones, bert = text_frontend.get_phones_and_bert(
                text_frontend.ensure_sentence_end(voice.ref_text, voice.ref_lang), voice.ref_lang, self
            )
            return _Reference(prompt, phones, bert, spec, sv_emb)

        return self._refs.get_or_create(key, build)

    # ================= synthesis =================
    def _line_keys(self, line: Line, params: SynthesisParams) -> Tuple[Tuple, Tuple]:
        v = line.voice
        token_key = (
            file_key(v.gpt_path), file_key(v.sovits_path), file_key(v.ref_wav), v.ref_text, v.ref_lang,
            line.text, line.language, line.seed, params.sampling_key(), str(self.dtype),
        )
        audio_key = token_key + (float(params.speed),)
        return token_key, audio_key

    @torch.inference_mode()
    def _generate_tokens(self, voice: Voice, lines: List[Line], keys: List[Tuple], params: SynthesisParams,
                         out: Dict[Tuple, torch.Tensor], report: Report, tick: Callable[[], None]) -> None:
        gpt = self.gpt(voice.gpt_path)
        sovits = self.sovits(voice.sovits_path)
        ref = self._reference(voice, sovits)
        items = []
        for line, key in zip(lines, keys):
            phones, bert = text_frontend.get_phones_and_bert(
                text_frontend.ensure_sentence_end(line.text, line.language), line.language, self
            )
            if ref.prompt is not None:
                phones, bert = ref.phones + phones, torch.cat([ref.bert, bert], 1)
            items.append((len(phones), phones, bert, line, key))
        items.sort(key=lambda it: it[0])  # similar lengths batch better
        size = max(1, int(params.batch_size))
        for start in range(0, len(items), size):
            chunk = items[start : start + size]
            results = t2s_batch.decode(
                gpt.model,
                [torch.LongTensor(it[1]).to(self.device) for it in chunk],
                [it[2].to(self.device, self.dtype) for it in chunk],
                ref.prompt,
                [it[3].seed for it in chunk],
                top_k=int(params.top_k), top_p=float(params.top_p), temperature=float(params.temperature),
                repetition_penalty=float(params.repetition_penalty), early_stop_num=HZ * gpt.max_sec,
            )
            report.batches.append(len(chunk))
            for it, tokens in zip(chunk, results):
                out[it[4]] = tokens.cpu()
                self._tokens.put(it[4], out[it[4]])
                tick()

    @torch.inference_mode()
    def _decode_audio(self, line: Line, tokens: torch.Tensor, params: SynthesisParams) -> np.ndarray:
        voice = line.voice
        sovits = self.sovits(voice.sovits_path)
        ref = self._reference(voice, sovits)
        phones, _ = text_frontend.get_phones_and_bert(
            text_frontend.ensure_sentence_end(line.text, line.language), line.language, self
        )
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):  # decode adds noise; keep it per-line and leave global RNG alone
            torch.manual_seed(line.seed)
            audio = sovits.model.decode(
                tokens.to(self.device).view(1, 1, -1),
                torch.LongTensor(phones).unsqueeze(0).to(self.device),
                [ref.spec],
                speed=float(params.speed),
                sv_emb=[ref.sv_emb] if sovits.is_v2pro else None,
            )[0][0]
        audio = audio.float().cpu().numpy()
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        return audio / peak if peak > 1 else audio

    def synthesize(
        self, plan: Plan, params: SynthesisParams, progress: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[np.ndarray, int, Report]:
        with self._lock:
            lines = plan.lines
            report = Report(lines=len(lines))
            keys = [self._line_keys(line, params) for line in lines]
            audio: Dict[Tuple, Tuple[np.ndarray, int]] = {}
            tokens: Dict[Tuple, torch.Tensor] = {}
            for token_key, audio_key in keys:
                cached = self._audio.get(audio_key)
                if cached is not None:
                    audio[audio_key] = cached
                elif token_key not in tokens and self._tokens.get(token_key) is not None:
                    tokens[token_key] = self._tokens.get(token_key)
            need_audio = sorted({i for i, (_, ak) in enumerate(keys) if ak not in audio})
            need_tokens = [i for i in need_audio if keys[i][0] not in tokens]
            total = len(need_tokens) + len(need_audio)
            done = [0]

            def tick():
                done[0] += 1
                if progress:
                    progress(done[0], total)

            groups: Dict[Voice, List[int]] = {}
            for i in need_tokens:
                groups.setdefault(lines[i].voice, []).append(i)
            for voice, idxs in groups.items():
                log.info("[Anomalous_TTS] %s：生成 %d 句", voice.label, len(idxs))
                self._generate_tokens(
                    voice, [lines[i] for i in idxs], [keys[i][0] for i in idxs], params, tokens, report, tick
                )
            for i in need_audio:
                token_key, audio_key = keys[i]
                if audio_key not in audio:  # the same line can appear twice with one key only if identical
                    sr = self.sovits(lines[i].voice.sovits_path).sampling_rate
                    audio[audio_key] = (self._decode_audio(lines[i], tokens[token_key], params), sr)
                    self._audio.put(audio_key, audio[audio_key])
                tick()
            report.generated = len(need_audio)
            report.from_cache = len(lines) - len(need_audio)
            return self._assemble(plan, [audio[ak] for _, ak in keys], params) + (report,)

    @staticmethod
    def _assemble(plan: Plan, clips: List[Tuple[np.ndarray, int]], params: SynthesisParams) -> Tuple[np.ndarray, int]:
        """Lines joined by the default pause; a [pause] tag replaces the default pause at that point."""
        sr = clips[0][1]
        pieces: List[np.ndarray] = []
        gap: Optional[float] = None
        n = 0
        for item in plan.items:
            if isinstance(item, Gap):
                gap = item.seconds if gap is None else gap + item.seconds
                continue
            if pieces or gap:
                pieces.append(silence(params.pause_sec if gap is None else gap, sr))
            gap = None
            clip, clip_sr = clips[n]
            n += 1
            pieces.append(resample(clip, clip_sr, sr))
        pieces.append(silence(params.pause_sec if gap is None else gap, sr))  # tail, like GPT-SoVITS
        return np.concatenate(pieces).astype(np.float32), sr
