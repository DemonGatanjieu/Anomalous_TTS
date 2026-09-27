from Anomalous_TTS.core.langdetect import detect


def test_detect():
    assert detect("先生、おはようございます") == "ja"
    assert detect("老师，早上好") == "zh"
    assert detect("通常授業！", model_language="ja") == "ja"
    assert detect("了解！", model_language="ja") == "ja"
    # A Japanese voice speaking Chinese: characters Japanese does not write decide.
    assert detect("你好，今天天气很好", model_language="ja") == "zh"
    assert detect("谢谢老师", model_language="ja") == "zh"
    assert detect("这是什么", model_language="ja") == "zh"
    assert detect("老师，早上好。", model_language="ja") == "zh"
    for line in ("本当？", "歓迎！", "出発！", "大丈夫", "移動開始！", "感謝！", "先生！"):
        assert detect(line, model_language="ja") == "ja", line
    assert detect("Hello, Sensei!") == "en"
    assert detect("……123！") is None
