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


@dataclass(frozen=True)
class Voice:
    """Model + reference. Lines with the same Voice can be batched together."""

    gpt_path: str
    sovits_path: str
    ref_wav: str
    ref_text: str
    ref_lang: str
    label: str  # for logs, e.g. "阿罗娜/日配{开心}"


@dataclass(frozen=True)
class Line:
    voice: Voice
    text: str
    language: str
    seed: int


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


def line_seed(seed: int, speaker: str, emotion: str, text: str, occurrence: int) -> int:
    """Depends only on the node seed and the line itself, so editing one line keeps the others."""
    digest = hashlib.sha256(f"{seed}|{speaker}|{emotion}|{text}|{occurrence}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def split_sentences(text: str) -> List[str]:
    from ..vendor.genie.text_splitter import TextSplitter

    return TextSplitter().split(text)


class _Builder:
    def __init__(self, chars: Dict[str, Character], opts: NodeOptions):
        if opts.character not in chars:
            raise ValueError(f"找不到角色：{opts.character}")
        self.chars = chars
        self.opts = opts
        self.plan = Plan()
        self.voices: Dict[Tuple[str, str], Voice] = {}
        self.counts: Counter = Counter()
        self.speaker = chars[opts.character]
        self.emotion = MAIN
        self.pending_speaker: Optional[str] = None
        self.last_lang: Optional[str] = None

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
            ref = Reference(ref.audio, self.opts.reference_text.strip(), ref.language, "node")
        return ref

    def voice(self, c: Character, emotion: str, sentence_lang: str) -> Voice:
        if emotion != MAIN and emotion not in c.emotions():
            self._reference(c, emotion)  # records the warning
            emotion = MAIN
        key = (c.name, emotion)
        if key not in self.voices:
            ref = self._reference(c, emotion)
            is_node_char = c.name == self.opts.character
            self.voices[key] = Voice(
                gpt_path=c.weight("gpt", self.opts.gpt if is_node_char else None),
                sovits_path=c.weight("sovits", self.opts.sovits if is_node_char else None),
                ref_wav=c.abspath(ref.audio),
                ref_text=ref.text,
                ref_lang=ref.language or c.language or sentence_lang,
                label=c.name if emotion == MAIN else f"{c.name}{{{emotion}}}",
            )
            if not ref.text:
                self.warn(f"{self.voices[key].label} 的参考音频 {ref.audio} 没有台词，使用无参考文本模式（效果会差一些）。")
        return self.voices[key]

    # ----- tokens -----
    def language_of(self, text: str) -> str:
        if self.opts.language != AUTO:
            return self.opts.language
        model_lang = self.speaker.language
        lang = langdetect.detect(text, model_lang) or self.last_lang or model_lang or "zh"
        return lang

    def resolve_pending_speaker(self, next_text: Optional[str]) -> None:
        if self.pending_speaker is None:
            return
        tag, self.pending_speaker = self.pending_speaker, None
        if self.opts.language != AUTO:
            lang = self.opts.language
        else:
            lang = langdetect.detect(next_text) if next_text else None
        try:
            self.speaker = resolve_name(self.chars, tag, lang)
        except KeyError:
            self.warn(f"找不到角色 [{tag}]，继续由 {self.speaker.name} 说。")
            return
        self.emotion = MAIN

    def text(self, text: str) -> None:
        self.resolve_pending_speaker(text)
        for sentence in split_sentences(self.speaker.respell(text)):
            lang = self.language_of(sentence)
            self.last_lang = lang
            voice = self.voice(self.speaker, self.emotion, lang)
            key = (self.speaker.name, self.emotion, sentence)
            occurrence = self.counts[key]
            self.counts[key] += 1
            seed = line_seed(self.opts.seed, self.speaker.name, self.emotion, sentence, occurrence)
            self.plan.items.append(Line(voice, sentence, lang, seed))

    def build(self, text: str) -> Plan:
        for token in script.tokenize(text):
            if isinstance(token, script.Text):
                self.text(token.text)
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


def build_plan(chars: Dict[str, Character], opts: NodeOptions, text: str) -> Plan:
    return _Builder(chars, opts).build(text)
