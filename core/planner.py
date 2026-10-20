"""Turn a script into a list of lines to synthesize (pure logic, no torch).

Input: characters, the node's settings and the script text.
Output: ``Plan`` = ordered ``Line`` / ``Gap`` items plus warnings.
Every ``Line`` carries everything the engine needs, so the engine never has to
look at characters or settings itself.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from . import langdetect, script
from .characters import AUTO, MAIN, Character, Reference, resolve_name
from .tuning import Tuner, Tuning

MIN_SPEED, MAX_SPEED = 0.5, 2.0  # the node's speed input
# Models that average the timbre of several references (GPT-SoVITS inp_refs); v3 / v4 / v5 use one.
AUX_VERSIONS = {"v1", "v2", "v2Pro", "v2ProPlus"}
LANGUAGES = {"ja": "ja", "jp": "ja", "zh": "zh", "cn": "zh", "en": "en", "日语": "ja", "中文": "zh", "英语": "en",
             AUTO: AUTO, "自动": AUTO}  # auto: detected for this line even when the node's language is set


@dataclass(frozen=True)
class Voice:
    """Model + reference. Lines with the same Voice can be batched together."""

    gpt_path: str
    sovits_path: str
    ref_wav: str
    ref_text: str
    ref_lang: str
    label: str  # for logs, e.g. "阿罗娜/日配{开心}"
    ref_aux: Tuple[str, ...] = ()  # auxiliary references (absolute): timbre averaged in (v1 / v2 / v2Pro)


@dataclass(frozen=True)
class Line:
    voice: Voice
    text: str
    language: str
    seed: int
    tuning: Tuning = Tuning()  # sampling and speed (core/tuning.py)


@dataclass(frozen=True)
class Gap:
    seconds: float


@dataclass
class Plan:
    items: List[Union[Line, Gap]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def lines(self) -> List[Line]:
        return [i for i in self.items if isinstance(i, Line)]


@dataclass
class NodeOptions:
    """What the node's widgets say about its own (first) character."""

    character: str
    language: str = AUTO  # AUTO or ja / zh / en
    reference_audio: Optional[str] = None  # path relative to the character folder
    reference_text: str = ""
    gpt: Optional[str] = None  # relative paths
    sovits: Optional[str] = None
    seed: int = 0


def line_seed(seed: int, speaker: str, emotion: str, text: str, occurrence: int, take: int = 1) -> int:
    """Depends only on the node seed and the line itself, so editing one line keeps the others.
    ``take`` > 1 ([take:N]) gives the line another voice without touching the other lines;
    take 1 keeps the seeds scripts had before takes existed."""
    key = f"{seed}|{speaker}|{emotion}|{text}|{occurrence}" + (f"|take{take}" if take > 1 else "")
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def split_sentences(text: str) -> List[str]:
    from ..vendor.genie.text_splitter import TextSplitter

    return TextSplitter().split(text)


class _Builder:
    def __init__(self, chars: Dict[str, Character], opts: NodeOptions, tuner: Tuner):
        if opts.character not in chars:
            raise ValueError(f"找不到角色：{opts.character}")
        self.chars = chars
        self.opts = opts
        self.tuner = tuner
        self.plan = Plan()
        self.voices: Dict[Tuple[str, str], Voice] = {}
        self.counts: Counter = Counter()
        self.speaker = chars[opts.character]
        self.emotion = MAIN
        self.pending_speaker: Optional[str] = None
        self.last_lang: Optional[str] = None
        self.take = 1  # [take:N] until the end of its line or the next tag
        self.line: Dict[str, object] = {}  # [speed:…] [preset:…] [lang:…], the same span as a take

    def warn(self, message: str) -> None:
        if message not in self.plan.warnings:
            self.plan.warnings.append(message)

    # ----- voices -----
    def _reference(self, c: Character, emotion: str) -> Reference:
        is_node_char = c.name == self.opts.character
        if emotion != MAIN:
            emotions = c.emotions()
            if emotion in emotions:
                return emotions[emotion]
            self.warn(
                f"角色 {c.name} 没有情绪 {{{emotion}}}，改用主参考。"
                f"已有：{'、'.join(emotions) or '无'}（文件名写成 原名.{emotion}.wav，或在 Anomalous 里标注）"
            )
        if is_node_char and self.opts.reference_audio:
            ref = c.make_reference(self.opts.reference_audio, "node")
        else:
            ref = c.default_reference()
            if ref is None:
                raise ValueError(f"角色 {c.name} 里没有 3~10 秒的参考音频。")
        if is_node_char and self.opts.reference_text.strip():
            ref = Reference(ref.audio, self.opts.reference_text.strip(), ref.language, "node", ref.aux)
        return ref

    def voice(self, c: Character, emotion: str, sentence_lang: str) -> Voice:
        if emotion != MAIN and emotion not in c.emotions():
            self._reference(c, emotion)  # records the warning
            emotion = MAIN
        key = (c.name, emotion)
        if key not in self.voices:
            ref = self._reference(c, emotion)
            is_node_char = c.name == self.opts.character
            sovits_path = c.weight("sovits", self.opts.sovits if is_node_char else None)
            label = c.name if emotion == MAIN else f"{c.name}{{{emotion}}}"
            aux = ref.aux
            kind = c.sovits_kind(sovits_path) if aux else None
            if kind is not None and kind.version not in AUX_VERSIONS:
                self.warn(f"{label} 的副参考音频只在 v1 / v2 / v2Pro 模型里有用，这个模型是 {kind.version}，已略过。")
                aux = ()
            self.voices[key] = Voice(
                gpt_path=c.weight("gpt", self.opts.gpt if is_node_char else None),
                sovits_path=sovits_path,
                ref_wav=c.abspath(ref.audio),
                ref_text=ref.text,
                ref_lang=ref.language or c.language or sentence_lang,
                label=label,
                ref_aux=tuple(c.abspath(rel) for rel in aux),
            )
            if not ref.text:
                self.warn(f"{self.voices[key].label} 的参考音频 {ref.audio} 没有台词，使用无参考文本模式（效果会差一些）。")
        return self.voices[key]

    # ----- tokens -----
    def language_of(self, text: str) -> str:
        own = self.line.get("language")
        if own and own != AUTO:
            return own
        if self.opts.language != AUTO and own != AUTO:
            return self.opts.language
        model_lang = self.speaker.language
        lang = langdetect.detect(text, model_lang) or self.last_lang or model_lang or "zh"
        return lang

    def resolve_pending_speaker(self, next_text: Optional[str]) -> None:
        if self.pending_speaker is None:
            return
        tag, self.pending_speaker = self.pending_speaker, None
        own = self.line.get("language")
        if own and own != AUTO:
            lang = own
        elif self.opts.language != AUTO and own != AUTO:
            lang = self.opts.language
        else:
            lang = langdetect.detect(next_text) if next_text else None
        try:
            self.speaker = resolve_name(self.chars, tag, lang)
        except KeyError:
            self.warn(f"找不到角色 [{tag}]，继续由 {self.speaker.name} 说。")
            return
        self.emotion = MAIN

    def setting(self, token: script.LineSetting) -> None:
        """A line's own speed, preset or language; a value it cannot use is left out with a warning."""
        value = token.value.strip()
        if token.name == "speed":
            try:
                self.line["speed"] = min(MAX_SPEED, max(MIN_SPEED, float(value)))
            except ValueError:
                self.warn(f"[speed:{value}] 不是数字，这句按原来的语速。")
        elif token.name == "preset":
            if value.lower() == AUTO:
                self.line["preset"] = ""  # the presets decide for this line, not the node's preset input
            elif value in self.tuner.names:
                self.line["preset"] = value
            else:
                self.warn(f"没有叫 {value} 的预设，这句按平时的语气走。")
        else:
            code = LANGUAGES.get(value.lower())
            if code:
                self.line["language"] = code
            else:
                self.warn(f"[lang:{value}] 只能是 ja / zh / en / auto，这句自动判断语言。")

    def text(self, text: str) -> None:
        if self.take > 1 or self.line:
            text = text.lstrip("\n")
            head, _, rest = text.partition("\n")
            self.sentences(head)
            self.take = 1
            self.line = {}
            if rest.strip():
                self.sentences(rest)
            return
        self.sentences(text)

    def sentences(self, text: str) -> None:
        self.resolve_pending_speaker(text)
        for sentence in split_sentences(self.speaker.respell(text)):
            lang = self.language_of(sentence)
            self.last_lang = lang
            voice = self.voice(self.speaker, self.emotion, lang)
            key = (self.speaker.name, self.emotion, sentence)
            occurrence = self.counts[key]
            self.counts[key] += 1
            seed = line_seed(self.opts.seed, self.speaker.name, self.emotion, sentence, occurrence, self.take)
            own = {k: v for k, v in self.line.items() if k != "language"}
            tuning = self.tuner.for_line(self.speaker.settings.get("defaults"), voice.ref_lang, lang, own)
            self.plan.items.append(Line(voice, sentence, lang, seed, tuning))

    def build(self, text: str) -> Plan:
        for token in script.tokenize(text):
            if not isinstance(token, (script.Text, script.Take, script.LineSetting)):
                self.take = 1  # a take and a line's settings cover only the text right after them
                self.line = {}
            if isinstance(token, script.Text):
                self.text(token.text)
            elif isinstance(token, script.Take):
                self.take = token.number
            elif isinstance(token, script.LineSetting):
                self.setting(token)
            elif isinstance(token, script.Emotion):
                self.resolve_pending_speaker(None)
                self.emotion = token.name
            elif isinstance(token, script.Speaker):
                self.resolve_pending_speaker(None)
                self.pending_speaker = token.name
            elif isinstance(token, script.Pause):
                self.plan.items.append(Gap(token.seconds))
        self.resolve_pending_speaker(None)
        if not self.plan.lines:
            raise ValueError("请输入要读的文字。")
        return self.plan


def build_plan(chars: Dict[str, Character], opts: NodeOptions, text: str, tuner: Optional[Tuner] = None) -> Plan:
    """``tuner``: the presets file and the node's changed inputs; None = built-in presets only."""
    return _Builder(chars, opts, tuner or Tuner({})).build(text)
