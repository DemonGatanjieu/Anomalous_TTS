from Anomalous_TTS.core.script import Emotion, Pause, Speaker, Text, tokenize


def test_plain_text():
    assert tokenize("先生、おはよう！") == [Text("先生、おはよう！")]


def test_tags_in_order():
    tokens = tokenize("A{开心}B[pause:1.5][普拉娜]C{ main }D")
    assert tokens == [Text("A"), Emotion("开心"), Text("B"), Pause(1.5), Speaker("普拉娜"), Text("C"), Emotion("main"), Text("D")]


def test_pause_units_and_aliases():
    assert tokenize("[pause:500ms]") == [Pause(0.5)]
    assert tokenize("[停顿：2秒]") == [Pause(2.0)]
    assert tokenize("[wait: 1s]") == [Pause(1.0)]
    assert tokenize("[pause:999]") == [Pause(30.0)]  # capped


def test_empty_parts_dropped_like_anomalous():
    # Anomalous parseTaggedSpeech drops empty segments; a trailing tag alone produces nothing.
    assert tokenize("{开心}") == [Emotion("开心")]
    assert tokenize("  \n") == []


def test_braces_without_content_are_text():
    assert tokenize("{}") == [Text("{}")]


def test_take_tags():
    from Anomalous_TTS.core.script import Take
    assert tokenize("{开心}[take:3]好的。") == [Emotion("开心"), Take(3), Text("好的。")]
    assert tokenize("[版本：2]好的。") == [Take(2), Text("好的。")]
    assert tokenize("[take:0]好的。")[0] == Take(1)


def test_line_setting_tags():
    from Anomalous_TTS.core.script import LineSetting, Take
    assert tokenize("{开心}[lang:ja][speed:1.2][preset:lively][take:2]好的。") == [
        Emotion("开心"), LineSetting("language", "ja"), LineSetting("speed", "1.2"), LineSetting("preset", "lively"),
        Take(2), Text("好的。")]
    assert tokenize("[语速：0.9][预设:custom_a1]好的。")[:2] == [LineSetting("speed", "0.9"), LineSetting("preset", "custom_a1")]
    assert tokenize("[speed fast]") == [Speaker("speed fast")]
