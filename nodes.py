"""ComfyUI nodes for Anomalous_TTS. Only widget definitions and glue live here;
the logic is in core/ (planner → engine).

Class names (``AnomalousTTS_*``) and the ``character`` / ``text`` inputs are part
of the public contract (docs/INTERFACE.md §1). Do not rename them.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

import comfy.model_management as mm
import comfy.utils

from .core import characters, paths, planner
from .core.engine import Engine, SynthesisParams

log = logging.getLogger("Anomalous_TTS")

AUTO = characters.AUTO
LANGUAGE_CHOICES = {AUTO: AUTO, "日语": "ja", "中文": "zh", "英语": "en"}

_engine: Optional[Engine] = None


def get_engine() -> Engine:
    global _engine
    device = mm.get_torch_device()
    dtype = torch.float16 if device.type == "cuda" and mm.should_use_fp16(device) else torch.float32
    if _engine is None or _engine.device != device or _engine.dtype != dtype:
        _engine = Engine(paths.ComfyResources(), device, dtype)
    return _engine


def _hook_unload_all_models() -> None:
    """Let ComfyUI's "Unload models" / "Free model and node cache" also free our models.

    Our models are not ComfyUI ModelPatchers, so ComfyUI does not track them. Wrap
    ``unload_all_models`` (called by the /free endpoint) to drop them as well.
    """
    original = mm.unload_all_models
    if getattr(original, "_anomalous_tts", False):
        return

    def unload_all_models(*args, **kwargs):
        if _engine is not None:
            _engine.unload()
        return original(*args, **kwargs)

    unload_all_models._anomalous_tts = True
    mm.unload_all_models = unload_all_models


_hook_unload_all_models()


def _adv(options: dict) -> dict:
    """Hide a widget under the node's "advanced" section (ComfyUI frontend ≥ 1.2x)."""
    return {**options, "advanced": True}


def _relative(value: str, c: characters.Character, chars) -> Optional[str]:
    """Combo value "角色名/相对路径" -> path relative to ``c``; AUTO -> None."""
    if not value or value == AUTO:
        return None
    owner, rel = characters.split_combo(chars, value)
    if owner.name != c.name:
        raise ValueError(f"{value} 不属于角色 {c.name}。请重新选择，或选“{AUTO}”。")
    return rel


class AnomalousTTS_CharacterSpeech:
    """用 GPT-SoVITS 角色模型读剧本。支持 {情绪}、[角色]、[pause:1s]。"""

    CATEGORY = "Anomalous/TTS"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "generate"
    DESCRIPTION = (
        "用 GPT-SoVITS 角色模型读剧本。\n"
        "{开心} 切换情绪，{main} 切回；[角色名] 换人说；[pause:1s] 插入停顿。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        chars = characters.scan(force=True)
        names = list(chars.keys()) or ["（没有找到角色）"]
        return {
            "required": {
                "character": (names, {"tooltip": "gpt_sovits 模型文件夹里的角色"}),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "剧本。{开心} 切换情绪、{main} 切回；[角色名] 换人说；[pause:1s] 插入停顿。",
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, "tooltip": "语速"}),
                "language": (list(LANGUAGE_CHOICES), _adv({"default": AUTO, "tooltip": "自动：按每句文字判断"})),
                "reference_audio": (
                    [AUTO] + characters.combo_values(chars, "audio"),
                    _adv({"tooltip": "主参考音频。自动：角色设置里的，或挑一条有台词的 3~10 秒音频"}),
                ),
                "reference_text": (
                    "STRING",
                    _adv({"multiline": False, "default": "", "tooltip": "主参考的台词。留空：自动查找"}),
                ),
                "gpt_weights": ([AUTO] + characters.combo_values(chars, "gpt"), _adv({"tooltip": "自动：设置里的，或轮数最大的"})),
                "sovits_weights": ([AUTO] + characters.combo_values(chars, "sovits"), _adv({"tooltip": "自动：设置里的，或轮数最大的"})),
                "pause_seconds": ("FLOAT", _adv({"default": 0.3, "min": 0.0, "max": 5.0, "step": 0.05, "tooltip": "句与句之间的停顿"})),
                "top_k": ("INT", _adv({"default": 15, "min": 1, "max": 100})),
                "top_p": ("FLOAT", _adv({"default": 1.0, "min": 0.05, "max": 1.0, "step": 0.05})),
                "temperature": ("FLOAT", _adv({"default": 1.0, "min": 0.05, "max": 2.0, "step": 0.05})),
                "repetition_penalty": ("FLOAT", _adv({"default": 1.35, "min": 1.0, "max": 2.0, "step": 0.05})),
                "batch_size": ("INT", _adv({"default": 8, "min": 1, "max": 64, "tooltip": "一次同时生成几句。显存不够就调小"})),
            }
        }

    @classmethod
    def IS_CHANGED(cls, character, **kwargs):
        """Re-run when the character's files change (settings, audio, weights)."""
        c = characters.scan().get(character)
        if c is None:
            return float("nan")
        stamp = [c.name, len(c.audio), tuple(c.gpt), tuple(c.sovits)]
        path = os.path.join(c.folder, "anomalous_tts.json")
        stamp.append(os.path.getmtime(path) if os.path.exists(path) else 0)
        return repr(stamp)

    def generate(self, character, text, seed, speed, language=AUTO, reference_audio=AUTO, reference_text="",
                 gpt_weights=AUTO, sovits_weights=AUTO, pause_seconds=0.3, top_k=15, top_p=1.0,
                 temperature=1.0, repetition_penalty=1.35, batch_size=8):
        chars = characters.scan(force=True)
        if character not in chars:
            raise ValueError(f"找不到角色：{character}")
        c = chars[character]
        if c.settings_error:
            log.warning("[Anomalous_TTS] %s", c.settings_error)
        opts = planner.NodeOptions(
            character=character,
            language=LANGUAGE_CHOICES.get(language, AUTO),
            reference_audio=_relative(reference_audio, c, chars),
            reference_text=reference_text,
            gpt=_relative(gpt_weights, c, chars),
            sovits=_relative(sovits_weights, c, chars),
            seed=seed,
        )
        plan = planner.build_plan(chars, opts, text)
        for message in plan.warnings:
            log.warning("[Anomalous_TTS] %s", message)
        if "ja" in {line.language for line in plan.lines} | {line.voice.ref_lang for line in plan.lines}:
            paths.ensure_ja_userdict()

        bar = comfy.utils.ProgressBar(1)

        def progress(done, total):
            bar.update_absolute(done, max(total, 1))
            mm.throw_exception_if_processing_interrupted()

        wav, sr, report = get_engine().synthesize(
            plan,
            SynthesisParams(
                top_k=top_k, top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
                speed=speed, pause_sec=pause_seconds, batch_size=batch_size,
            ),
            progress=progress,
        )
        log.info("[Anomalous_TTS] %s", report.summary())
        info = "\n".join([report.summary()] + plan.warnings)
        return ({"waveform": torch.from_numpy(wav).reshape(1, 1, -1), "sample_rate": sr}, info)


NODE_CLASS_MAPPINGS = {
    "AnomalousTTS_CharacterSpeech": AnomalousTTS_CharacterSpeech,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnomalousTTS_CharacterSpeech": "角色语音 (GPT-SoVITS)",
}
