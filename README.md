# Anomalous TTS (GPT-SoVITS)

在 ComfyUI 里用 GPT-SoVITS 角色模型读剧本。**不需要另外安装 GPT-SoVITS 整合包**：推理代码已经精简后放在节点包里，直接用 ComfyUI 的 PyTorch 和显卡。

姊妹项目：[Anomalous Model Browser](https://github.com/DemonGatanjieu/Anomalous_Model_Browser)（管理角色、标注情绪、写剧本）。两边的约定见 [docs/INTERFACE.md](docs/INTERFACE.md)。

> 开发中，尚未发布。

## 怎么用

1. 把角色模型放进 `ComfyUI/models/gpt_sovits/`（或用 `extra_model_paths.yaml` 指到现有文件夹，见下文）。装了 Anomalous Model Browser 的话，也可以在它的音频页里直接导入：把 GPT、SoVITS 权重和参考音频拖进去，自动建好角色文件夹。
2. 添加节点 **角色语音 (GPT-SoVITS)**（分类 `Anomalous/TTS`），选角色，写剧本，接 Save Audio / Preview Audio。

节点上平时只有：角色、剧本、「插入标签」按钮、种子、语速。其他（语言、参考音频、权重、采样参数、批量大小）在「高级参数」里，一般不用动。

## 剧本写法

```
先生、おはようございます！{开心}今日もがんばりましょう！[pause:0.8]
[普拉娜]……おはようございます、先生。
```

| 写法 | 作用 |
|---|---|
| `{开心}` | 之后改用「开心」的参考音频；`{main}` 切回主参考 |
| `[普拉娜]` | 之后换这个角色说（角色名或别名） |
| `[pause:1]`、`[pause:500ms]`、`[停顿:1s]` | 插入停顿 |

「插入标签」按钮会列出当前角色有的情绪、其他角色、常用停顿，点一下就插到光标位置。
写了不存在的情绪或角色不会报错：控制台提示，并继续用主参考 / 当前角色。

语言默认自动判断（有假名 → 日语；只有汉字 → 中文，日语模型则按日语，但句子里有日语不用的字（吗、你、们、这、说、气这类）时仍按中文；只有字母 → 英语）。中文里夹的英文单词按英语读；日文里的英文单词按片假名读（同官方）。

## 角色文件夹

```
models/gpt_sovits/
  阿罗娜/
    日配/                 ← 一个角色下有多套模型时，分别列为「阿罗娜/日配」「阿罗娜/中配」
      GPT_weights_v2/*.ckpt
      SoVITS_weights_v2/*.pth
      参考音频/*.wav
      all.txt              ← GPT-SoVITS 训练标注文件，用来查参考台词
      anomalous_tts.json   ← 可选：角色设置（通常由 Anomalous 生成）
  pretrained/              ← 底模，首次使用自动下载
```

- 权重默认用轮数最大的；设置文件或高级参数里可以指定。
- 参考台词依次从：设置文件 → 同名 `.txt` 或 `.lab` → 标注文件（`.list`，或同格式 `.txt`：`音频路径|说话人|语言|台词`）里找；都没有就用无参考文本模式。
- 情绪参考：设置文件里的 `emotions`，或把文件改名为 `原名.情绪.wav`（情绪名是第一个点后面的部分，和 F5-TTS、Anomalous 相同）。
- 高级参数里的「reference_audio」填角色文件夹里的相对路径（例如 `参考音频/xxx.wav`），留空就自动选。
- 参考音频的语气会带到角色说的每一句话里：拿问句当主参考，每句结尾都会上扬。没指定主参考时，会自动挑有台词、语气平稳的陈述句（不选问句，其次避开感叹句），4～8 秒优先。已经指定了的不会变，想换就在 Anomalous 里换一条语气平稳的。
- 设置文件格式见 [docs/INTERFACE.md](docs/INTERFACE.md) 第 3 节。

已有的模型不用复制：装了 [Anomalous Model Browser](https://github.com/DemonGatanjieu/Anomalous_Model_Browser) 的话，在它的音频页里导入角色、选存放位置、指定整合包就行，不用重启。也可以在 `ComfyUI/extra_model_paths.yaml` 里加目录：

```yaml
anomalous_tts:
    base_path: D:/voices
    gpt_sovits: 模型
```

电脑上有 GPT-SoVITS 整合包的话，把它的底模也加进来，省掉下载：

```yaml
gpt_sovits_pretrained:
    base_path: D:/GPT-SoVITS/GPT_SoVITS
    gpt_sovits: |
        pretrained_models
        text
```

## 读音替换

有些词角色总是读错时，在角色的设置文件 `anomalous_tts.json` 里加 `replace`：左边是剧本里写的，右边是让角色实际读的。剧本和字幕照常写，只有送进模型的文字被替换，只对这个角色生效。

```json
{
  "replace": { "C站": "西站", "LoRA": "萝拉", "粘进": "沾进" }
}
```

长的先匹配，只替换一遍（替换出来的字不会再被替换）。

想让 AI 助手替你调某个角色（试音、找出读错的词、换参考音频、写替换表），把 [docs/AGENT_GUIDE.md](docs/AGENT_GUIDE.md) 给它看：里面是一步步的做法和规矩，只用 ComfyUI 已有的接口。

日语模型说中文时特别有用：

- 中文里夹的英文字母和单词按英语发音读，只学过日语的模型读不好；写成读音相近的汉字。
- 只学过日语的模型发不好 ü（选 → 显、全 → 敲）、后鼻音（模型 → 墨西）和声调。这是模型本身的限制，文字处理给它的读音是对的。在难读的词前面加个字（插件 → 这个插件）、加逗号断开，或者换个说法，往往就好了。
- 多音字偶尔判断错，例如“粘进”会读成 nián；换成同音的字。

## 用 API 调用

```json
{
  "1": { "class_type": "AnomalousTTS_CharacterSpeech", "inputs": { "character": "阿罗娜", "text": "老师好。" } },
  "2": { "class_type": "SaveAudio", "inputs": { "audio": ["1", 0], "filename_prefix": "tts/line" } }
}
```

- 除了 `character`、`text` 都可以不传。`language` 写 `中文`、`zh`、`cn`、`日文`、`ja`、`japanese` 等都可以。
- 提示（角色没有这个情绪、参考没有台词等）在节点的 `info` 输出里，也在 `/history` 里这个节点的 `outputs.text`。
- 单独跑一个只有本节点的 ComfyUI 时，别的程序（比如 Whisper）要用显存，本节点不会知道；用完调一次 ComfyUI 的 `POST /free`（内容 `{"unload_models": true}`）就会释放模型。

## 支持范围

- 模型：v1、v2、v2Pro、v2ProPlus。不支持 v3 / v4。
- 语言：日语、中文、英语。
- 速度：同一角色、同一参考的句子一起批量生成（`batch_size`，默认 8；显存不够就调小）。
- 跨语言（比如日语角色说中文）：这些句子默认用更稳的采样（`top_k` 最多 10、`temperature` 最多 0.8，你设得更低时按你的），`cross_lingual` 选“不调整”可以关掉。只用日语训练的模型说中文会带口音，这是模型本身的，调参数去不掉；有中文模型的角色用中文模型更地道。
- 缓存：每句结果按「种子 + 句子内容 + 参数」缓存。改了剧本里的一句，再运行只重新生成这一句。同样的输入得到同样的声音；在 CPU 上已验证批量分组不影响结果（显卡半精度下可能有极细微的差别）。
- 音量：默认每句都调到同样的响度（人声约 -20 dBFS，峰值不超过 -1 dBFS），换种子、换句子不会忽大忽小；`volume` 选“不调整”保持模型原样。
- 显存：ComfyUI 要加载别的模型而显存不够时，或者点「卸载模型 / 释放缓存」时，会一起释放本节点的模型。

## 底模（首次使用时自动下载到 `models/gpt_sovits/pretrained/`）

| 用途 | 文件 | 大小 | 来源 |
|---|---|---|---|
| 所有 | `chinese-hubert-base` | 190MB | HuggingFace `lj1995/GPT-SoVITS` |
| 中文 | `chinese-roberta-wwm-ext-large` | 650MB | HuggingFace |
| 中文多音字 | `G2PWModel` | 600MB | ModelScope |
| v2Pro | `sv/pretrained_eres2netv2w24s4ep4.ckpt` | 100MB | HuggingFace |
| 日语里的英文单词 | `ja_userdic/userdict.csv` | 17MB | GitHub（GPT-SoVITS 固定版本） |
| 英语 | 英语词典 + nltk 词性数据 | 约 30MB | GitHub |

网络不通时按报错提示手动下载。G2PWModel、`opencc` 或 `onnxruntime` 缺失时，中文多音字改用 pypinyin 判断（能用，但准确率下降）。

## 依赖

见 `requirements.txt`。只用到的语言才加载对应的包：日语 `pyopenjtalk-plus`；中文 `pypinyin`、`jieba`、`opencc`；英语 `g2p_en`、`wordsegment`、`nltk`。

## 开发

- 代码结构和维护规矩：[ARCHITECTURE.md](ARCHITECTURE.md)。
- 更新记录：[CHANGELOG.md](CHANGELOG.md)。
- 测试：见 [tests/README.md](tests/README.md)。
- 上游代码来源与改动：[UPSTREAM.md](UPSTREAM.md)。

## 许可证

MIT。内含 GPT-SoVITS 与 Genie-TTS 的部分代码（均为 MIT），出处见 [UPSTREAM.md](UPSTREAM.md)。本包不附带任何角色模型或声音。
