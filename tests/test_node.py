"""Node inputs for API callers: only character + text are required; language takes aliases."""

from Anomalous_TTS import nodes

Node = nodes.AnomalousTTS_CharacterSpeech


def test_only_character_and_text_are_required():
    spec = Node.INPUT_TYPES()
    assert list(spec["required"]) == ["character", "text"]
    # ComfyUI saves widget values by position: the order must stay as it was, new inputs go last
    assert list(spec["optional"])[:3] == ["seed", "speed", "language"]
    assert list(spec["optional"])[-1] == "volume"


def test_language_aliases():
    for value, code in [("日语", "ja"), ("日文", "ja"), ("JA", "ja"), ("japanese", "ja"), ("中文", "zh"),
                        ("cn", "zh"), ("Chinese", "zh"), ("en", "en"), ("英文", "en"), ("自动", "自动"), ("auto", "自动")]:
        assert nodes.resolve_language(value) == code, value
    assert Node.VALIDATE_INPUTS(language="日文") is True
    assert Node.VALIDATE_INPUTS() is True
    assert "language" in Node.VALIDATE_INPUTS(language="klingon")
