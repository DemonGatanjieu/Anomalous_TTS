# 上游代码来源

本节点包不依赖 GPT-SoVITS 整合包或 `genie_tts` 包。推理需要的代码直接复制在 `vendor/` 里，两者都是 MIT 许可证，原许可证文件放在各自目录中。

## GPT-SoVITS（`vendor/gpt_sovits/`）

- 仓库：https://github.com/RVC-Boss/GPT-SoVITS
- 固定版本：`48b1a0169a28582a8984402f82cf438d3bfa6aca`（2026-08-18）
- 许可证：`vendor/gpt_sovits/LICENSE`
- 同步：`python tools/sync_upstream.py <GPT-SoVITS 源码目录>`，脚本会复制下列文件并重新打补丁。任何一个补丁对不上都会停下，需要人工检查。

| 文件 | 用途 |
|---|---|
| `AR/models/t2s_model.py`、`AR/models/utils.py` | GPT：由音素生成语义 token（含 top_k / top_p / temperature / repetition_penalty 采样） |
| `AR/modules/*.py`（5 个） | GPT 的 Transformer 结构 |
| `module/models.py` 及 `commons` `modules` `attentions` `mrte_model` `quantize` `core_vq` `transforms` `mel_processing` | SoVITS：由语义 token 生成波形 |
| `text/symbols.py`、`text/symbols2.py` | 音素表 |

没有复制：训练代码、WebUI、`TTS_infer_pack`（官方推理主流程，由 `core/engine.py` 替代）、v3/v4 的 BigVGAN 和 CFM、UVR5、ASR、说话人识别（v2Pro 用，下一阶段加入）。

### 补丁（见 `tools/sync_upstream.py`）

1. 所有 `from AR...`、`from module...`、`from text...` 改成包内相对 import。上游靠修改 `sys.path` 找模块，放进 ComfyUI 会和其他节点冲突。
2. `module/models.py`：删掉 `from f5_tts.model import DiT`（只有 v3/v4 用）。
3. `AR/models/t2s_model.py`：删掉训练用的 `torchmetrics` 精度统计。
4. `module/core_vq.py`：分布式训练的三个工具函数换成空实现，推理用不到。
5. `module/mel_processing.py`：librosa 改为用到时才导入（推理只用 `spectrogram_torch`，不需要 librosa）。

`core/engine.py` 的推理流程照 `inference_webui.py` 的 `get_tts_wav` 重写，保留了上游的行为细节：参考音频限 3~10 秒、参考音频后补 0.3 秒静音、短句前补 "."、句末补标点、每句单独解码后按峰值归一化。

`core/checkpoints.py` 的版本判断照 `process_ckpt.py`。不同之处：用 `torch.load(weights_only=True)` 加载，把权重里的 `utils.HParams` 映射到本地的空类，避免执行网上下载的模型里的任意代码。

## Genie-TTS（`vendor/genie/`）

- 仓库：https://github.com/High-Logic/Genie-TTS
- 版本：2.0.2（PyPI `genie-tts`）
- 许可证：`vendor/genie/LICENSE`

| 文件 | 来源 | 改动 |
|---|---|---|
| `text_splitter.py` | `genie_tts/Utils/TextSplitter.py` | 无 |
| `japanese_g2p.py` | `genie_tts/G2P/Japanese/JapaneseG2P.py` | 连续标点合并规则、未知音素记为 `UNK`，这两处改回和 GPT-SoVITS 一致；音素表用 GPT-SoVITS 的 `symbols2.py` |

只借鉴 Genie 的做法，不兼容它的 ONNX 角色格式。

### 日语前端对拍结果（2026-09-24）

用阿罗娜数据集 `all.txt` 的 138 句加 5 句自造句子，对比本包和 GPT-SoVITS `text/japanese.py` 输出的音素：

- 138 句数据集台词完全一致（1 句是标注文件本身的坏行，不计）。
- 不同的只有两类：
  - 日文里夹英文单词（如 "NiCe"、"Windows"）：GPT-SoVITS 用自带的用户词典读成片假名。本包首次使用日语时会从上游（固定版本）下载这份词典（`userdict.csv`，17MB）并编译到 `models/gpt_sovits/pretrained/ja_userdic/`，下载失败才退回按字母读。
  - 半角 `%`：本包读作「パーセント」，GPT-SoVITS 只转换全角 `％`。
