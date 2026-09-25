# Anomalous TTS (GPT-SoVITS)

在 ComfyUI 里用 GPT-SoVITS 角色模型读剧本。**不需要另外安装 GPT-SoVITS 整合包**：推理代码已经精简后放在节点包里，直接用 ComfyUI 的 PyTorch 和显卡。

姊妹项目：[Anomalous Model Browser](https://github.com/DemonGatanjieu/Anomalous_Model_Browser)（管理角色、标注情绪、写剧本）。两边的约定见 [docs/INTERFACE.md](docs/INTERFACE.md)。

> 开发中，尚未发布。

## 怎么用

1. 把角色模型放进 `ComfyUI/models/gpt_sovits/`（或用 `extra_model_paths.yaml` 指到现有文件夹，见下文）。
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

语言默认自动判断（有假名 → 日语；只有汉字 → 中文，日语模型则按日语；只有字母 → 英语）。中文里夹的英文单词按英语读；日文里的英文单词按片假名读（同官方）。

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
- 参考台词依次从：设置文件 → 同名 `.txt` → 标注文件（`.list`，或同格式 `.txt`：`音频路径|说话人|语言|台词`）里找；都没有就用无参考文本模式。
- 情绪参考：设置文件里的 `emotions`，或把文件改名为 `原名.情绪.wav`（情绪名是第一个点后面的部分，和 F5-TTS、Anomalous 相同）。
- 高级参数里的「reference_audio」填角色文件夹里的相对路径（例如 `参考音频/xxx.wav`），留空就自动选。
- 设置文件格式见 [docs/INTERFACE.md](docs/INTERFACE.md) 第 3 节。

已有的模型不用复制：装了 [Anomalous Model Browser](https://github.com/DemonGatanjieu/Anomalous_Model_Browser) 的话，在它的音频页里添加角色库、指定整合包就行，不用重启。也可以在 `ComfyUI/extra_model_paths.yaml` 里加目录：

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

## 支持范围

- 模型：v1、v2、v2Pro、v2ProPlus。不支持 v3 / v4。
- 语言：日语、中文、英语。
- 速度：同一角色、同一参考的句子一起批量生成（`batch_size`，默认 8；显存不够就调小）。
- 缓存：每句结果按「种子 + 句子内容 + 参数」缓存。改了剧本里的一句，再运行只重新生成这一句。同样的输入得到同样的声音；在 CPU 上已验证批量分组不影响结果（显卡半精度下可能有极细微的差别）。
- ComfyUI 的「卸载模型 / 释放缓存」会一起释放本节点的模型。

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
