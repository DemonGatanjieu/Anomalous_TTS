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
# Other spellings API callers use for ``language``; matched case-insensitively.
LANGUAGE_ALIASES = {
    AUTO: ["auto", ""],
    "ja": ["日文", "日本語", "ja", "jp", "japanese"],
    "zh": ["汉语", "普通话", "zh", "cn", "chinese", "mandarin"],
    "en": ["英文", "en", "english"],
}
CROSS_LINGUAL_CHOICES = {"自动调整": True, "不调整": False}
VOLUME_CHOICES = {"统一音量": -20.0, "不调整": None}


def resolve_language(value) -> Optional[str]:
    """Widget value or alias -> AUTO / ja / zh / en; None when unknown."""
    text = str(value if value is not None else "").strip()
    if text in LANGUAGE_CHOICES:
        return LANGUAGE_CHOICES[text]
    for code, names in LANGUAGE_ALIASES.items():
        if text.lower() in names:
            return code
    return None

_engine: Optional[Engine] = None


def get_engine() -> Engine:
    global _engine
    device = mm.get_torch_device()
    dtype = torch.float16 if device.type == "cuda" and mm.should_use_fp16(device) else torch.float32
    if _engine is None or _engine.device != device or _engine.dtype != dtype:
        _engine = Engine(paths.ComfyResources(), device, dtype)
    return _engine


def _hook_free_memory() -> None:
    """Give our VRAM back when ComfyUI needs it.

    Our models are not ComfyUI ModelPatchers, so ComfyUI does not track them. Wrap
    ``free_memory``: when ComfyUI asks for more than is free on our device (loading an
    image model), drop our models and the g2pW session first. "Unload models" asks for
    everything: then the sentence caches go too, whatever device it names. With enough
    free memory everything stays, so the next speech needs no reload.
    """
    original = mm.free_memory
    if getattr(original, "_anomalous_tts", False):
        return

    def free_memory(memory_required, device, *args, **kwargs):
        everything = memory_required >= 1e29  # comfy.model_management.unload_all_models
        if _engine is not None and everything and _engine.holds_anything():
            log.info("[Anomalous_TTS] 卸载模型：释放 GPT-SoVITS 模型、中文多音字模型和句子缓存")
            _engine.unload(everything=True)
            mm.soft_empty_cache()
        elif _engine is not None and _engine.holds_models() and _engine.on(device) \
                and mm.get_free_memory(device) < memory_required:
            log.info("[Anomalous_TTS] 显存不够，先让出 GPT-SoVITS 模型（下次生成语音时重新加载）")
            _engine.unload()
            mm.soft_empty_cache()
        return original(memory_required, device, *args, **kwargs)

    free_memory._anomalous_tts = True
    mm.free_memory = free_memory


_hook_free_memory()


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


def _reference_path(value: str, c: characters.Character) -> Optional[str]:
    value = (value or "").strip().replace("\\", "/")
    if not value or value == AUTO:
        return None
    if value not in c.audio:
        raise ValueError(f"角色 {c.name} 里没有参考音频 {value}（填角色文件夹里的相对路径，或留空自动选择）。")
    return value


class AnomalousTTS_CharacterSpeech:
    """用 GPT-SoVITS 角色模型读剧本。支持 {情绪}、[角色]、[pause:1s]。"""

    CATEGORY = "Anomalous/TTS"
    RETURN_TYPES = ("AUDIO", "STRING")  # info also goes to the UI output, so /history carries it
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "generate"
    DESCRIPTION = (
        "用 GPT-SoVITS 角色模型读剧本。\n"
        "{开心} 切换情绪，{main} 切回；[角色名] 换人说；[pause:1s] 插入停顿。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        chars = characters.scan()
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
            },
            # Everything below has a default, so an API prompt only needs character + text. The order
            # is unchanged from when these were required: ComfyUI saves widget values by position.
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, "tooltip": "语速"}),
                "language": (list(LANGUAGE_CHOICES), _adv({"default": AUTO, "tooltip": "自动：按每句文字判断"})),
                "reference_audio": (
                    "STRING",
                    _adv({
                        "multiline": False,
                        "default": "",
                        "tooltip": "主参考音频：角色文件夹里的相对路径，例如 参考音频/xxx.wav。留空：用角色设置里的，或自动挑一条有台词的 3~10 秒音频",
                    }),
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
                "cross_lingual": (list(CROSS_LINGUAL_CHOICES), _adv({
                    "default": "自动调整",
                    "tooltip": "句子和参考音频不是同一种语言时（比如日语角色说中文），这些句子的 top_k 最多 10、temperature 最多 0.8，更稳；你设得更低时按你的",
                })),
                "volume": (list(VOLUME_CHOICES), _adv({
                    "default": "统一音量",
                    "tooltip": "统一音量：每句的人声都调到约 -20 dBFS（峰值不超过 -1 dBFS），换种子、换句子音量不再忽大忽小。不调整：保持模型输出的音量",
                })),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, language=AUTO):
        """``language`` also takes aliases (日文, ja, japanese, zh, cn, en ...), so ComfyUI's own
        list check is replaced by this one."""
        if resolve_language(language) is None:
            return f"language 只能是 {' / '.join(LANGUAGE_CHOICES)}，或 ja / zh / en 等写法：{language!r}"
        return True

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

    def generate(self, character, text, seed=0, speed=1.0, language=AUTO, reference_audio="", reference_text="",
                 gpt_weights=AUTO, sovits_weights=AUTO, pause_seconds=0.3, top_k=15, top_p=1.0,
                 temperature=1.0, repetition_penalty=1.35, batch_size=8, cross_lingual="自动调整",
                 volume="统一音量"):
        chars = characters.scan(max_age=2.0)
        if character not in chars:
            raise ValueError(f"找不到角色：{character}")
        c = chars[character]
        if c.settings_error:
            log.warning("[Anomalous_TTS] %s", c.settings_error)
        opts = planner.NodeOptions(
            character=character,
            language=resolve_language(language) or AUTO,
            reference_audio=_reference_path(reference_audio, c),
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
                cross_lingual=CROSS_LINGUAL_CHOICES.get(cross_lingual, True),
                loudness_db=VOLUME_CHOICES.get(volume, VOLUME_CHOICES["统一音量"]),
            ),
            progress=progress,
        )
        log.info("[Anomalous_TTS] %s", report.summary())
        info = "\n".join([report.summary()] + plan.warnings)
        audio = {"waveform": torch.from_numpy(wav).reshape(1, 1, -1), "sample_rate": sr}
        return {"ui": {"text": [info]}, "result": (audio, info)}


NODE_CLASS_MAPPINGS = {
    "AnomalousTTS_CharacterSpeech": AnomalousTTS_CharacterSpeech,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnomalousTTS_CharacterSpeech": "角色语音 (GPT-SoVITS)",
}
