"""ComfyUI nodes for Anomalous_TTS.

Class names (``AnomalousTTS_*``) are part of the public contract: Anomalous
Model Browser and saved workflows refer to them. Do not rename after release.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

import comfy.model_management as mm
import comfy.utils

from .core import characters, paths, text_frontend
from .core.engine import Engine, SynthesisParams
from .vendor.genie.text_splitter import TextSplitter

log = logging.getLogger("Anomalous_TTS")

_engine: Optional[Engine] = None


def get_engine() -> Engine:
    global _engine
    device = mm.get_torch_device()
    dtype = torch.float16 if device.type == "cuda" and mm.should_use_fp16(device) else torch.float32
    if _engine is None or _engine.device != device or _engine.dtype != dtype:
        _engine = Engine(paths.hubert_dir(), device, dtype)
    return _engine


LANG_NAMES = list(text_frontend.LANGUAGES.keys())
IMPLEMENTED_LANGS = ["日语"]  # 中文 / 英语 are next; keep the list honest.


def _none_placeholder(values):
    return values if values else ["（没有找到）"]


class AnomalousTTS_CharacterSpeech:
    """用 GPT-SoVITS 角色模型把文字读出来。"""

    CATEGORY = "Anomalous/TTS"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        chars = list(characters.scan(force=True).keys())
        auto = [characters.AUTO]
        return {
            "required": {
                "character": (_none_placeholder(chars), {"tooltip": "models/gpt_sovits 下的角色文件夹"}),
                "text": ("STRING", {"multiline": True, "default": "", "tooltip": "要读的文字"}),
                "text_language": (IMPLEMENTED_LANGS, {"default": "日语"}),
                "reference_audio": (
                    auto + characters.combo_values("audio"),
                    {"tooltip": "自动：优先用有台词的 3~10 秒参考音频"},
                ),
                "reference_text": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "留空：从同名 .txt 或 .list 标注文件读取；都没有时用无参考文本模式（效果会差一些）",
                    },
                ),
                "reference_language": (auto + IMPLEMENTED_LANGS, {"default": characters.AUTO}),
                "gpt_weights": (auto + characters.combo_values("gpt"), {"tooltip": "自动：轮数最大的 .ckpt"}),
                "sovits_weights": (auto + characters.combo_values("sovits"), {"tooltip": "自动：轮数最大的 .pth"}),
                "top_k": ("INT", {"default": 15, "min": 1, "max": 100}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 1.0, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 2.0, "step": 0.05}),
                "repetition_penalty": ("FLOAT", {"default": 1.35, "min": 1.0, "max": 2.0, "step": 0.05}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
                "pause_seconds": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "句与句之间的停顿"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            }
        }

    def generate(
        self,
        character,
        text,
        text_language,
        reference_audio,
        reference_text,
        reference_language,
        gpt_weights,
        sovits_weights,
        top_k,
        top_p,
        temperature,
        repetition_penalty,
        speed,
        pause_seconds,
        seed,
    ):
        chars = characters.scan(force=True)
        if character not in chars:
            raise ValueError(f"找不到角色：{character}")
        c = chars[character]
        text_lang = text_frontend.LANGUAGES[text_language]

        # Reference audio + text
        list_lang = None
        if reference_audio == characters.AUTO:
            ref_rel, auto_text, list_lang = characters.default_reference(c)
        else:
            owner, ref_rel = characters.split_combo(reference_audio)
            if owner.name != c.name:
                raise ValueError(f"参考音频 {reference_audio} 不属于角色 {c.name}。")
            auto_text, list_lang = characters.reference_text(c, ref_rel)
        ref_text = reference_text.strip() or auto_text
        if reference_language != characters.AUTO:
            ref_lang = text_frontend.LANGUAGES[reference_language]
        else:
            ref_lang = text_frontend.LIST_LANG_CODES.get(list_lang or "", text_lang)
        if ref_text and ref_lang not in ("ja",):
            raise NotImplementedError(f"参考音频语言 {ref_lang} 还没有接入。")
        if not ref_text:
            log.warning("[Anomalous_TTS] %s 没有参考台词，使用无参考文本模式。", ref_rel)

        gpt_path = characters.pick_weight(c, "gpt", gpt_weights)
        sovits_path = characters.pick_weight(c, "sovits", sovits_weights)

        sentences = TextSplitter().split(text)
        if not sentences:
            raise ValueError("请输入要读的文字。")
        if "ja" in (text_lang, ref_lang):
            paths.ensure_ja_userdict()

        engine = get_engine()
        bar = comfy.utils.ProgressBar(len(sentences))

        def progress(done, total):
            bar.update_absolute(done, total)
            mm.throw_exception_if_processing_interrupted()

        wav, sr = engine.synthesize(
            gpt_path,
            sovits_path,
            c.abspath(ref_rel),
            ref_text,
            ref_lang,
            sentences,
            text_lang,
            SynthesisParams(
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                speed=speed,
                pause_sec=pause_seconds,
                seed=seed,
            ),
            progress=progress,
        )
        waveform = torch.from_numpy(wav).reshape(1, 1, -1)
        return ({"waveform": waveform, "sample_rate": sr},)


NODE_CLASS_MAPPINGS = {
    "AnomalousTTS_CharacterSpeech": AnomalousTTS_CharacterSpeech,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnomalousTTS_CharacterSpeech": "角色语音 (GPT-SoVITS)",
}
