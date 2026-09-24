"""GPT-SoVITS v1/v2 inference on PyTorch.

The flow follows ``get_tts_wav`` in GPT-SoVITS ``inference_webui.py``
(MIT, RVC-Boss), without Gradio, i18n, v3/v4 or training code.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol, Tuple

import numpy as np
import soundfile as sf
import torch

from . import text_frontend
from .checkpoints import load_gpt_checkpoint, load_sovits_checkpoint

log = logging.getLogger("Anomalous_TTS")

HZ = 50  # semantic tokens per second
REF_MIN_SEC = 3.0
REF_MAX_SEC = 10.0


class Resources(Protocol):
    """Where pretrained files live. Implemented by nodes.py (ComfyUI paths) and tests."""

    def hubert_dir(self) -> str: ...

    def roberta_dir(self) -> str: ...

    def g2pw_dir(self) -> Optional[str]: ...


@dataclass
class Segment:
    """Consecutive sentences that share one reference audio."""

    ref_wav: str
    ref_text: str
    ref_lang: str
    sentences: List[str]
    label: str = "main"


@dataclass
class SynthesisParams:
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    repetition_penalty: float = 1.35
    speed: float = 1.0
    pause_sec: float = 0.3
    seed: int = 0


def _resample(wav: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return wav
    import soxr

    # librosa.load(sr=...) uses soxr "HQ" by default; match it.
    return soxr.resample(wav, sr_from, sr_to, quality="HQ").astype(np.float32)


def load_mono(path: str) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    return wav.mean(axis=1), sr


class _LRU(OrderedDict):
    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity

    def get_or_create(self, key, factory):
        if key in self:
            self.move_to_end(key)
            return self[key]
        value = factory()
        self[key] = value
        while len(self) > self.capacity:
            self.popitem(last=False)
        return value


class GPTModel:
    def __init__(self, path: str, device: torch.device, dtype: torch.dtype):
        from ..vendor.gpt_sovits.AR.models.t2s_model import Text2SemanticDecoder

        data = load_gpt_checkpoint(path)
        config = data["config"]
        self.max_sec = config["data"]["max_sec"]
        model = Text2SemanticDecoder(config=config, top_k=3)
        state = {k[len("model."):]: v for k, v in data["weight"].items() if k.startswith("model.")}
        model.load_state_dict(state)
        self.model = model.to(device=device, dtype=dtype).eval()


class SoVITSModel:
    def __init__(self, path: str, device: torch.device, dtype: torch.dtype):
        from ..vendor.gpt_sovits.module.models import SynthesizerTrn

        data, version = load_sovits_checkpoint(path)
        if version in ("v2Pro", "v2ProPlus"):
            raise NotImplementedError(f"{os.path.basename(path)} 是 {version} 模型，下一阶段支持。")
        hps = data["config"]
        hps["model"]["semantic_frame_rate"] = "25hz"
        hps["model"]["version"] = version
        d = hps["data"]
        self.version = version
        self.sampling_rate = d["sampling_rate"]
        self.filter_length = d["filter_length"]
        self.hop_length = d["hop_length"]
        self.win_length = d["win_length"]
        model = SynthesizerTrn(
            self.filter_length // 2 + 1,
            hps["train"]["segment_size"] // self.hop_length,
            n_speakers=d["n_speakers"],
            **hps["model"],
        )
        if hasattr(model, "enc_q"):
            del model.enc_q  # posterior encoder: training only
        model.load_state_dict(data["weight"], strict=False)
        self.model = model.to(device=device, dtype=dtype).eval()


class Engine:
    """Holds loaded models and caches. One instance per process."""

    def __init__(self, resources: Resources, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self.resources = resources
        self._hubert = None
        self._roberta = None
        self._zh_ready = False
        self._gpt = _LRU(2)
        self._sovits = _LRU(2)
        self._refs = _LRU(16)
        self._lock = threading.Lock()

    # ---------- model loading ----------
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

    def _bert_feature(self, norm_text: str, word2ph: List[int]) -> torch.Tensor:
        """GPT-SoVITS ``get_bert_feature``: 3rd-last hidden layer, repeated per phoneme."""
        tokenizer, model = self.roberta()
        inputs = tokenizer(norm_text, return_tensors="pt").to(self.device)
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].float().cpu()[1:-1]
        assert len(word2ph) == len(norm_text) == res.shape[0], (len(word2ph), len(norm_text), res.shape)
        feats = [res[i].repeat(word2ph[i], 1) for i in range(len(word2ph))]
        return torch.cat(feats, dim=0).T

    def _prepare_language(self, lang: str) -> None:
        """Chinese: use g2pW for polyphones when available, like GPT-SoVITS does."""
        if lang != "zh" or self._zh_ready:
            return
        self._zh_ready = True
        from ..vendor.gpt_sovits.text import chinese2

        import importlib.util

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

    def gpt(self, path: str) -> GPTModel:
        return self._gpt.get_or_create(
            (path, os.path.getmtime(path)), lambda: GPTModel(path, self.device, self.dtype)
        )

    def sovits(self, path: str) -> SoVITSModel:
        return self._sovits.get_or_create(
            (path, os.path.getmtime(path)), lambda: SoVITSModel(path, self.device, self.dtype)
        )

    def unload(self):
        with self._lock:
            self._hubert = None
            self._roberta = None
            self._gpt.clear()
            self._sovits.clear()
            self._refs.clear()

    # ---------- reference audio ----------
    def _prompt_semantic(self, sovits: SoVITSModel, wav_path: str) -> torch.Tensor:
        wav, sr = load_mono(wav_path)
        wav16k = _resample(wav, sr, 16000)
        seconds = len(wav16k) / 16000
        if not REF_MIN_SEC <= seconds <= REF_MAX_SEC:
            raise ValueError(
                f"参考音频 {os.path.basename(wav_path)} 长 {seconds:.1f} 秒，需要在 3~10 秒之间。"
            )
        # GPT-SoVITS appends 0.3 s of silence measured at the model rate (quirk kept on purpose).
        pad = np.zeros(int(sovits.sampling_rate * 0.3), dtype=np.float32)
        x = torch.from_numpy(np.concatenate([wav16k, pad])).to(self.device, self.dtype)
        ssl = self.hubert()(x.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
        codes = sovits.model.extract_latent(ssl)
        return codes[0, 0].unsqueeze(0)

    def _refer_spec(self, sovits: SoVITSModel, wav_path: str) -> torch.Tensor:
        from ..vendor.gpt_sovits.module.mel_processing import spectrogram_torch

        wav, sr = load_mono(wav_path)
        audio = torch.from_numpy(wav).unsqueeze(0).float()
        if sr != sovits.sampling_rate:
            import torchaudio.functional as AF

            audio = AF.resample(audio, sr, sovits.sampling_rate)
        maxx = audio.abs().max()
        if maxx > 1:
            audio /= min(2, maxx)
        spec = spectrogram_torch(
            audio.to(self.device),
            sovits.filter_length,
            sovits.sampling_rate,
            sovits.hop_length,
            sovits.win_length,
            center=False,
        )
        return spec.to(self.dtype)

    def _reference(self, sovits_path: str, sovits: SoVITSModel, wav_path: str, text: str, lang: str):
        key = (sovits_path, wav_path, os.path.getmtime(wav_path), text, lang)

        def build():
            prompt = None
            phones = None
            bert = None
            if text:
                prompt = self._prompt_semantic(sovits, wav_path)
                self._prepare_language(lang)
                phones, bert = text_frontend.get_phones_and_bert(
                    text_frontend.ensure_sentence_end(text, lang), lang, self._bert_feature
                )
            return prompt, phones, bert, self._refer_spec(sovits, wav_path)

        return self._refs.get_or_create(key, build)

    # ---------- synthesis ----------
    @torch.inference_mode()
    def synthesize(
        self,
        gpt_path: str,
        sovits_path: str,
        segments: List[Segment],
        text_lang: str,
        params: SynthesisParams,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[np.ndarray, int]:
        """Synthesize segments (each with its own reference) in order. Returns (float32 mono, sr)."""
        with self._lock:
            gpt = self.gpt(gpt_path)
            sovits = self.sovits(sovits_path)
            self._prepare_language(text_lang)

            gen = torch.Generator(device="cpu").manual_seed(int(params.seed) & 0xFFFFFFFF)
            pause = np.zeros(int(sovits.sampling_rate * params.pause_sec), dtype=np.float32)
            pieces: List[np.ndarray] = []
            total = sum(len(seg.sentences) for seg in segments)
            done = 0
            for seg in segments:
                prompt, phones1, bert1, refer = self._reference(
                    sovits_path, sovits, seg.ref_wav, (seg.ref_text or "").strip(), seg.ref_lang
                )
                for sentence in seg.sentences:
                    done += 1
                    sentence = text_frontend.ensure_sentence_end(sentence, text_lang)
                    if sentence.strip():
                        pieces.append(
                            self._one_sentence(gpt, sovits, prompt, phones1, bert1, refer, sentence, text_lang, params, gen)
                        )
                        pieces.append(pause)
                    if progress:
                        progress(done, total)
            if not pieces:
                raise ValueError("没有可合成的文本。")
            return np.concatenate(pieces).astype(np.float32), sovits.sampling_rate

    def _one_sentence(self, gpt, sovits, prompt, phones1, bert1, refer, sentence, lang, params, gen) -> np.ndarray:
        phones2, bert2 = text_frontend.get_phones_and_bert(sentence, lang, self._bert_feature)
        if prompt is None:  # no reference text
            bert = bert2
            all_ids = phones2
        else:
            bert = torch.cat([bert1, bert2], 1)
            all_ids = phones1 + phones2
        x = torch.LongTensor(all_ids).unsqueeze(0).to(self.device)
        x_len = torch.tensor([x.shape[-1]], device=self.device)
        bert = bert.unsqueeze(0).to(self.device, self.dtype)

        # Sampling uses torch's global RNG; seed it per sentence for reproducibility.
        torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
        pred, idx = gpt.model.infer_panel(
            x,
            x_len,
            prompt,
            bert,
            top_k=int(params.top_k),
            top_p=float(params.top_p),
            temperature=float(params.temperature),
            early_stop_num=HZ * gpt.max_sec,
            repetition_penalty=float(params.repetition_penalty),
        )
        pred = pred[:, -idx:].unsqueeze(0)
        audio = sovits.model.decode(
            pred,
            torch.LongTensor(phones2).unsqueeze(0).to(self.device),
            [refer],
            speed=float(params.speed),
        )[0][0]
        audio = audio.float().cpu().numpy()
        peak = np.abs(audio).max()
        return audio / peak if peak > 1 else audio
