# 测试

```
python -m pytest tests
```

不需要 ComfyUI（`conftest.py` 会替换掉 `folder_paths` 和 `comfy`）。

## 需要模型的测试

`test_frontend.py`、`test_engine.py` 需要模型文件，没有时自动跳过。准备一个文件夹，用环境变量 `ANOMALOUS_TTS_ASSETS` 指过去：

```
assets/
  chinese-hubert-base/
  chinese-roberta-wwm-ext-large/
  G2PWModel/
  sv/pretrained_eres2netv2w24s4ep4.ckpt
  en_dict/            cmudict.rep、cmudict-fast.rep、engdict-hot.rep、namedict_cache.pickle
  nltk_data/          taggers/averaged_perceptron_tagger_eng、taggers/averaged_perceptron_tagger、corpora/cmudict
  character/          任意一个日语 v2 角色：GPT_weights_v2/*.ckpt、SoVITS_weights_v2/*.pth、ref.wav（3~10 秒）、ref.txt（台词）
```

（这些文件节点第一次运行时都会下载到 `models/gpt_sovits/pretrained/`，可以直接指向那里再加一个 `character/`。）

| 测试 | 检查什么 |
|---|---|
| `test_script.py` | 剧本标签解析 |
| `test_langdetect.py` | 自动判断语言 |
| `test_settings.py` | 角色设置文件的校验和读写 |
| `test_planner.py` | 角色识别、情绪、换人、停顿、每句种子 |
| `test_frontend.py` | 日 / 中 / 英音素与基准一致（基准已和官方 GPT-SoVITS 对过，见 UPSTREAM.md） |
| `test_engine.py` | 批量解码与官方逐句解码结果完全相同；批量分组不影响结果；缓存和可复现性 |

改了文字处理并确认和官方一致后，用 `tools/make_frontend_fixtures.py` 重新生成基准。
