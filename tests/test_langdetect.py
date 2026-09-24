from Anomalous_TTS.core.langdetect import detect


def test_detect():
    assert detect("先生、おはようございます") == "ja"
    assert detect("老师，早上好") == "zh"
    assert detect("通常授業！", model_language="ja") == "ja"
    assert detect("Hello, Sensei!") == "en"
    assert detect("……123！") is None
