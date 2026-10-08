"""ComfyUI nodes for Anomalous_TTS. Only widget definitions and glue live here;
the logic is in core/ (planner → engine).

Class names (``AnomalousTTS_*``) and the ``character`` / ``text`` inputs are part
of the public contract (docs/INTERFACE.md §1). Do not rename them.

The node's text is English; other languages come from locales/<lang>/nodeDefs.json, which
ComfyUI's frontend reads.
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
LANGUAGE_CHOICES = {AUTO: AUTO, "Japanese": "ja", "Chinese": "zh", "English": "en"}
# Other spellings API callers use for ``language``, and the options' names before they were
# English (saved workflows); matched case-insensitively.
LANGUAGE_ALIASES = {
    AUTO: ["", "自动"],
    "ja": ["日语", "日文", "日本語", "ja", "jp", "japanese"],
    "zh": ["中文", "汉语", "普通话", "zh", "cn", "chinese", "mandarin"],
    "en": ["英语", "英文", "en", "english"],
}
CROSS_LINGUAL_CHOICES = {"adjust": True, "off": False}
VOLUME_CHOICES = {"normalize": -20.0, "off": None}
# The options' names before they were English, still taken from saved workflows.
OLD_AUTO = "自动"
OLD_CHOICES = {
    "cross_lingual": {"自动调整": "adjust", "不调整": "off"},
    "volume": {"统一音量": "normalize", "不调整": "off"},
}


def current_choice(name: str, value) -> str:
    """A cross_lingual / volume value as today's option (an old name becomes the new one)."""
    return OLD_CHOICES[name].get(value, value)


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
    """Combo value "character/relative path" -> path relative to ``c``; AUTO -> None."""
    if not value or value in (AUTO, OLD_AUTO):
        return None
    owner, rel = characters.split_combo(chars, value)
    if owner.name != c.name:
        raise ValueError(f"{value} does not belong to the character {c.name}. Pick another one, or {AUTO!r}.")
    return rel


def _reference_path(value: str, c: characters.Character) -> Optional[str]:
    value = (value or "").strip().replace("\\", "/")
    if not value or value in (AUTO, OLD_AUTO):
        return None
    if value not in c.audio:
        raise ValueError(f"The character {c.name} has no reference clip {value} "
                         "(give a path inside the character's folder, or leave it empty to pick one).")
    return value


class AnomalousTTS_CharacterSpeech:
    """Reads a script with GPT-SoVITS character models: {emotion}, [character], [pause:1s]."""

    CATEGORY = "Anomalous/TTS"
    RETURN_TYPES = ("AUDIO", "STRING")  # info also goes to the UI output, so /history carries it
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "generate"
    DESCRIPTION = (
        "Reads a script with GPT-SoVITS character models.\n"
        "{happy} switches the emotion, {main} switches back; [name] switches the speaker; [pause:1s] inserts a pause."
    )

    @classmethod
    def INPUT_TYPES(cls):
        chars = characters.scan()
        names = list(chars.keys()) or ["(no characters found)"]
        return {
            "required": {
                "character": (names, {"tooltip": "A character in the gpt_sovits models folder"}),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "The script. {happy} switches the emotion, {main} switches back; "
                                   "[name] switches the speaker; [pause:1s] inserts a pause.",
                    },
                ),
            },
            # Everything below has a default, so an API prompt only needs character + text. The order
            # is unchanged from when these were required: ComfyUI saves widget values by position.
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, "tooltip": "Speaking speed"}),
                "language": (list(LANGUAGE_CHOICES), _adv({"default": AUTO, "tooltip": "auto: decided for each sentence from its text"})),
                "reference_audio": (
                    "STRING",
                    _adv({
                        "multiline": False,
                        "default": "",
                        "tooltip": "Main reference clip: a path inside the character's folder, e.g. refs/xxx.wav. "
                                   "Empty: the one in the character's settings, or a 3–10 s clip with a transcript, picked automatically",
                    }),
                ),
                "reference_text": (
                    "STRING",
                    _adv({"multiline": False, "default": "", "tooltip": "What the main reference clip says. Empty: looked up automatically"}),
                ),
                "gpt_weights": ([AUTO] + characters.combo_values(chars, "gpt"), _adv({"tooltip": "auto: the one in the character's settings, else the most trained"})),
                "sovits_weights": ([AUTO] + characters.combo_values(chars, "sovits"), _adv({"tooltip": "auto: the one in the character's settings, else the most trained"})),
                "pause_seconds": ("FLOAT", _adv({"default": 0.3, "min": 0.0, "max": 5.0, "step": 0.05, "tooltip": "Pause between sentences"})),
                "top_k": ("INT", _adv({"default": 15, "min": 1, "max": 100})),
                "top_p": ("FLOAT", _adv({"default": 1.0, "min": 0.05, "max": 1.0, "step": 0.05})),
                "temperature": ("FLOAT", _adv({"default": 1.0, "min": 0.05, "max": 2.0, "step": 0.05})),
                "repetition_penalty": ("FLOAT", _adv({"default": 1.35, "min": 1.0, "max": 2.0, "step": 0.05})),
                "batch_size": ("INT", _adv({"default": 8, "min": 1, "max": 64, "tooltip": "Sentences generated at once. Lower it when VRAM runs out"})),
                "cross_lingual": (list(CROSS_LINGUAL_CHOICES), _adv({
                    "default": "adjust",
                    "tooltip": "adjust: a sentence in another language than the reference clip (a Japanese character speaking "
                               "Chinese, say) uses top_k at most 10 and temperature at most 0.8, which is steadier; lower values you set stay",
                })),
                "volume": (list(VOLUME_CHOICES), _adv({
                    "default": "normalize",
                    "tooltip": "normalize: each sentence's voice is brought to about -20 dBFS (peaks under -1 dBFS), so a new seed "
                               "or sentence does not jump in loudness. off: the loudness the model gives",
                })),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, language=AUTO, gpt_weights=AUTO, sovits_weights=AUTO, cross_lingual="adjust", volume="normalize"):
        """These inputs also take other spellings (ja, japanese, zh, cn …) and the options' names from
        before they were English, so ComfyUI's own list check is replaced by this one."""
        if resolve_language(language) is None:
            return f"language must be one of {' / '.join(LANGUAGE_CHOICES)}, or a code such as ja / zh / en: {language!r}"
        for name, value in (("cross_lingual", cross_lingual), ("volume", volume)):
            choices = CROSS_LINGUAL_CHOICES if name == "cross_lingual" else VOLUME_CHOICES
            if current_choice(name, value) not in choices:
                return f"{name} must be one of {' / '.join(choices)}: {value!r}"
        chars = characters.scan()
        for name, kind, value in (("gpt_weights", "gpt", gpt_weights), ("sovits_weights", "sovits", sovits_weights)):
            if value not in (AUTO, OLD_AUTO) and value not in characters.combo_values(chars, kind):
                return f"{name}: no such weights file: {value!r}"
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
                 temperature=1.0, repetition_penalty=1.35, batch_size=8, cross_lingual="adjust",
                 volume="normalize"):
        chars = characters.scan(max_age=2.0)
        if character not in chars:
            raise ValueError(f"Character not found: {character}")
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
                cross_lingual=CROSS_LINGUAL_CHOICES.get(current_choice("cross_lingual", cross_lingual), True),
                loudness_db=VOLUME_CHOICES.get(current_choice("volume", volume), VOLUME_CHOICES["normalize"]),
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
    "AnomalousTTS_CharacterSpeech": "Character Speech (GPT-SoVITS)",
}
