"""Node inputs for API callers: only character + text are required; language takes aliases; the
options' names from before they were English (saved workflows) still work."""

from Anomalous_TTS import nodes

Node = nodes.AnomalousTTS_CharacterSpeech


def test_only_character_and_text_are_required():
    spec = Node.INPUT_TYPES()
    assert list(spec["required"]) == ["character", "text"]
    # ComfyUI saves widget values by position: the order must stay as it was, new inputs go last
    assert list(spec["optional"])[:3] == ["seed", "speed", "language"]
    assert list(spec["optional"])[-1] == "volume"


def test_language_aliases():
    for value, code in [("Japanese", "ja"), ("日语", "ja"), ("日文", "ja"), ("JA", "ja"), ("japanese", "ja"),
                        ("Chinese", "zh"), ("中文", "zh"), ("cn", "zh"), ("English", "en"), ("en", "en"), ("英文", "en"),
                        ("auto", "auto"), ("自动", "auto"), ("", "auto")]:
        assert nodes.resolve_language(value) == code, value
    assert Node.VALIDATE_INPUTS(language="日文") is True
    assert Node.VALIDATE_INPUTS() is True
    assert "language" in Node.VALIDATE_INPUTS(language="klingon")


def test_options_are_english():
    spec = Node.INPUT_TYPES()["optional"]
    assert spec["language"][0] == ["auto", "Japanese", "Chinese", "English"]
    assert spec["cross_lingual"][0] == ["adjust", "off"] and spec["cross_lingual"][1]["default"] == "adjust"
    assert spec["volume"][0] == ["normalize", "off"] and spec["volume"][1]["default"] == "normalize"
    assert spec["gpt_weights"][0][0] == "auto"
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS["AnomalousTTS_CharacterSpeech"] == "Character Speech (GPT-SoVITS)"


def test_old_option_names_still_work():
    """A workflow saved before the options were English keeps its choices."""
    assert Node.VALIDATE_INPUTS(language="自动", gpt_weights="自动", sovits_weights="自动",
                                cross_lingual="不调整", volume="统一音量") is True
    assert nodes.current_choice("cross_lingual", "自动调整") == "adjust"
    assert nodes.current_choice("cross_lingual", "不调整") == "off"
    assert nodes.current_choice("volume", "统一音量") == "normalize"
    assert nodes.current_choice("volume", "不调整") == "off"
    assert nodes.current_choice("volume", "off") == "off"
    assert "volume" in Node.VALIDATE_INPUTS(volume="loud")
    assert "gpt_weights" in Node.VALIDATE_INPUTS(gpt_weights="nobody/missing.ckpt")
