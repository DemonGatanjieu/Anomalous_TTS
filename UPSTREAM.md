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
| `text/chinese2.py`、`tone_sandhi.py`、`opencpop-strict.txt`、`zh_normalization/`、`g2pw/`（含多音字词表 `polyphonic.rep`、`polyphonic-fix.rep`；上游的 `polyphonic.pickle` 不复制） | 中文前端：文本规整、分词、变调、儿化、g2pW 多音字 |
| `text/english.py`、`text/en_normalization/expend.py` | 英语前端（CMU 词典首次使用时从同一版本的 GitHub 下载，不放进仓库） |
| `text/namedict.rep`（脚本生成） | 英语人名读音：上游只有 `namedict_cache.pickle`，同步脚本从固定版本里把它转成文本 |
| `eres2net/`（4 个文件） | v2Pro / v2ProPlus 的说话人识别模型结构 |

没有复制：训练代码、WebUI、`TTS_infer_pack`（官方推理主流程，由 `core/engine.py` 替代）、v3/v4 的 BigVGAN 和 CFM、UVR5、ASR、`sv.py`（由 `core/engine.py` 的 `sv()` 替代）、`LangSegmenter`（中英混合改用正则切分）。

### 补丁（见 `tools/sync_upstream.py`）

1. 所有 `from AR...`、`from module...`、`from text...` 改成包内相对 import。上游靠修改 `sys.path` 找模块，放进 ComfyUI 会和其他节点冲突。
2. `module/models.py`：删掉 `from f5_tts.model import DiT`（只有 v3/v4 用）。
3. `AR/models/t2s_model.py`：删掉训练用的 `torchmetrics` 精度统计。
4. `module/core_vq.py`：分布式训练的三个工具函数换成空实现，推理用不到。
5. `module/mel_processing.py`：librosa 改为用到时才导入（推理只用 `spectrogram_torch`，不需要 librosa）。
6. `text/chinese2.py`：
   - 删掉没用到的 `cn2an`。
   - g2pW 不在导入时加载（上游写死了相对路径并会自动下载），改由 `enable_g2pw(模型目录, 分词器目录)` 按需加载；加载前走上游自带的 pypinyin 分支。`disable_g2pw()` 再把它放掉（约 600 MB 内存），引擎卸载模型时调用。
   - 删掉 pypinyin 分支里每次都打印的调试输出。
   - `jieba_fast` 装不上时退回纯 Python 的 `jieba`（70 句测试结果一致）。
7. `text/tone_sandhi.py`：
   - 同上，`jieba_fast` → `jieba` 兜底。
   - **有意和官方不同**：叠字变轻声只用在真正的叠字上。上游把一个词里任何两个挨着的相同字的第二个读成轻声，jieba 把“银行行长”分成一个词，于是读成 yin2 hang2 hang5 zhang3。现在 3 个字以内的词照旧；4 个字以上的词只有开头两个字相同（好好学习）或 AABB（高高兴兴）才变，“银行行长”“人民民主”这种跨了两个词的不变。
8. `text/g2pw/onnx_api.py`：`requests`、`opencc` 改为用到时才导入；g2pW 只在 CPU 上跑，去掉 `preload_dlls()`（onnxruntime-gpu 要的 CUDA 版本和 torch 自带的不一样时，上游每次都会报一段红字再退回 CPU）。
9. `text/english.py`：相对 import；CMU 词典的位置由 `configure(词典目录)` 指定；`en_G2p()` 改为首次使用时才创建（上游在导入时就加载词典）；词典按 UTF-8 读（上游用系统默认编码，中文 Windows 上是 GBK）。
10. `eres2net/ERes2NetV2.py`：相对 import。
11. **数据文件只当数据读，不执行、不反序列化**（ComfyUI-Manager 审核时提出：G2PWModel 是下载来的，界面里添加的文件夹又能决定读哪一份）：
    - `text/g2pw/utils.py`：g2pW 的 `config.py` 不再执行，只读顶层的 `名字 = 字面量`；
    - `text/g2pw/onnx_api.py`：`download_and_decompress` 不再自己下载（上游从 ModelScope 的 `master` 下载、不校验）；下载由 `core/paths.py` 做，固定到一个版本，解压前核对 SHA-256；
    - `text/g2pw/g2pw.py`：多音字词表每次从 `.rep` 文本读，不再用 pickle 缓存，也不再 `eval()`（值是带引号的拼音列表，按文本解析）；
    - `text/english.py`：CMU 词典每次从 `.rep` 文本建（约 0.25 秒），不再读写 `engdict_cache.pickle`；人名读音读 `namedict.rep`。
    结果和原来完全一样：多音字 45047 条、人名 19929 条、CMU 词典 143205 条逐条相同，g2pW 的设置项相同，三种语言的音素基准测试不变。旧版本留在 `en_dict` 里的 `engdict_cache.pickle`、`namedict_cache.pickle` 不再读。

`core/t2s_batch.py` 是批量 GPT 解码，改写自 `t2s_model.py` 的 `infer_panel_batch_infer` 和 `infer_panel_naive`：每句用自己的随机数生成器（结果只取决于这一句的种子，和同批的其他句子无关），支持无参考文本，停止规则同 `infer_panel_naive`。测试确认：单句时与官方 `infer_panel_naive` 生成的 token 完全相同；多句批量与逐句结果完全相同（CPU、fp32）。唯一不同：到 54 秒上限被强制截断时，官方会丢掉第一个 token，这里保留。

`core/engine.py` 的推理流程照 `inference_webui.py` 的 `get_tts_wav` 重写，保留了上游的行为细节：参考音频限 3~10 秒、参考音频后补 0.3 秒静音、短句前补 "."、句末补标点、每句单独解码后按峰值归一化。

`core/checkpoints.py` 的版本判断照 `process_ckpt.py`。不同之处：用 `torch.load(weights_only=True)` 加载，把权重里的 `utils.HParams` 映射到本地的空类，避免执行网上下载的模型里的任意代码。

### 中文前端对拍结果（2026-09-24）

70 句自编测试句（多音字、数字、日期、电话、儿化、语气词），本包与 GPT-SoVITS 官方 `chinese2.py`（g2pW 模式）音素完全一致，BERT 特征按官方 `get_bert_feature` 计算。上面第 7 条的叠字修改后重新对拍（2026-09-27），基准里的句子仍然全部一致（基准里没有被修改影响的词）。

为什么中文没用 Genie 的前端：Genie 用 g2pM 判断多音字，其中前 50 句里有 16 句和官方不同，而且多是明显读错（"都市"读 dou、"还你"读 hai、"便宜"读 bian、"很长"读 zhang、"一只猫"读 zhi3）。同样 50 句，不用模型的 pypinyin 也只有 11 句不同。模型是用官方 g2pW 前端训练的，所以中文照搬官方。
g2pW 需要 `opencc`（先把简体转成繁体再判断）。试过用 `zh_normalization/char_convert.py` 的逐字转换代替，会把"了"转成"瞭"读成 liao，不可用。

### 英语对拍结果（2026-09-24）

16 句测试句（多音词、数字、日期、金额、缩写、专有名词），本包与 GPT-SoVITS 官方 `english.py` 音素完全一致。

### v2Pro / v2ProPlus

用官方底模 `s2Gv2Pro.pth`、`s2Gv2ProPlus.pth` 测试：权重全部加载（除训练用的 `enc_q`），说话人向量按 `sv.py` 的 `compute_embedding3` 计算，能正常生成。

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
