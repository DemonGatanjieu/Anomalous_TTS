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
