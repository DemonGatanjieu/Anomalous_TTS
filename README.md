# Anomalous TTS (GPT-SoVITS)

在 ComfyUI 里用训练好的 GPT-SoVITS 角色模型读台词：多句一起生成，可以切换情绪、多人对话、插入停顿。**不需要另外安装 GPT-SoVITS 整合包**：推理代码已经精简后放在节点包里，直接用 ComfyUI 的 PyTorch 和显卡。

它是 [Anomalous Model Browser](https://github.com/DemonGatanjieu/Anomalous_Model_Browser) 的配套节点包。两个一起装，可以在 Anomalous 里拖入文件导入角色、标注情绪、改读音，然后在剧本台直接生成；只装这一个也能用，角色文件夹需要自己放好。

> **English**: A ComfyUI node that speaks scripts with your trained GPT-SoVITS character models (v1, v2, v2Pro, v2ProPlus; Japanese, Chinese, English), with emotion tags, multiple speakers and pauses. No separate GPT-SoVITS install is needed. It is the companion of [Anomalous Model Browser](https://github.com/DemonGatanjieu/Anomalous_Model_Browser), which imports characters, edits emotions and pronunciations, and generates scripts from its Script Director. The node's buttons follow ComfyUI's language; the rest of this page is in Chinese.

## 安装

- **ComfyUI Manager**：搜索 `Anomalous TTS`，安装后重启 ComfyUI。
- **手动**：在 `ComfyUI/custom_nodes` 里执行

  ```bash
  git clone https://github.com/DemonGatanjieu/Anomalous_TTS.git
  ```

  然后用 ComfyUI 自己的 Python 安装依赖（便携版是 `python_embeded\python.exe -m pip install -r requirements.txt`），重启 ComfyUI。

第一次生成时会自动下载需要的底模（按语言，最多约 1.6 GB）到 `models/gpt_sovits/pretrained/`。电脑上已经有 GPT-SoVITS 整合包的话，可以直接用它的底模，见下面“已有的模型和底模”。

## 开始使用

1. **准备角色**
   - 装了 Anomalous Model Browser：打开它的“角色语音”页，把 GPT、SoVITS 权重和参考音频拖进去，自动建好角色。
   - 只用本节点：按下面“角色文件夹”的样子把模型放进 `ComfyUI/models/gpt_sovits/`。
2. **添加节点** `角色语音 (GPT-SoVITS)`（英文界面叫 `Character Speech (GPT-SoVITS)`，分类 `Anomalous/TTS`），选角色，写台词，接 Save Audio 或 Preview Audio。

节点上平时只有：角色、台词、种子、语速，以及几个按钮：

| 按钮 | 作用 |
|---|---|
| 插入标签 | 列出当前角色的情绪、其他角色、常用停顿，点一下插到光标处 |
| 角色：试听 / 刷新 / 导入与编辑 | 试听主声音和各个情绪的参考音频；刷新角色列表（新放进去的角色不用重启）；在 Anomalous 里导入或编辑这个角色（没装 Anomalous 时会告诉你去哪装） |
| 高级参数 | 语言、参考音频、权重、采样参数、批量大小，一般不用动 |

角色有问题，或者缺这个语言要用的 Python 包时，节点上会多出一行 ⚠ 提示，点它可以看原因或复制安装命令。

## 台词写法

```
先生、おはようございます！{开心}今日もがんばりましょう！[pause:0.8]
[普拉娜]……おはようございます、先生。
```

| 写法 | 作用 |
|---|---|
| `{开心}` | 之后改用「开心」的参考音频；`{main}` 切回主参考 |
| `[普拉娜]` | 之后换这个角色说（角色名或别名） |
| `[pause:1]`、`[pause:500ms]`、`[停顿:1s]` | 插入停顿 |
| `{开心}[take:2]好的。` | 只把这一段换一版重新生成（到换行或下一个标签为止），其他句子不变 |

写了不存在的情绪或角色不会报错：控制台提示，并继续用主参考 / 当前角色。语言默认自动判断，中文里夹的英文单词按英语读。

## 支持范围

- 模型：v1、v2、v2Pro、v2ProPlus。不支持 v3 / v4。
- 语言：日语、中文、英语。跨语言（比如日语角色说中文）也可以，会带一点口音，这是模型本身的。
- 改了台词里的一句，再运行只重新生成这一句；同样的输入得到同样的声音。
- 每句自动调到同样的音量，换种子、换句子不会忽大忽小。
- 显存：ComfyUI 要加载别的模型而显存不够时，或者点“卸载模型”时，会一起释放本节点的模型。

## 使用声音的提醒

本包不附带任何角色模型或声音。请只使用你有权使用的声音模型，生成的语音不要用来冒充真人或误导他人；发布作品时遵守声音来源的授权要求和当地法规。

---

以下是参考资料，用到时再看。

<details>
<summary><b>角色文件夹</b></summary>

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
- 高级参数里的 `reference_audio` 填角色文件夹里的相对路径（例如 `参考音频/xxx.wav`），留空就自动选。
- 参考音频的语气会带到每一句话里：拿问句当主参考，每句结尾都会上扬。没指定主参考时，会自动挑有台词、语气平稳的陈述句，4～8 秒优先。
- 设置文件格式见 [docs/INTERFACE.md](docs/INTERFACE.md) 第 3 节。

</details>

<details>
<summary><b>已有的模型和底模</b></summary>

已有的模型不用复制：装了 Anomalous Model Browser 的话，在它的音频页里选存放位置、指定整合包就行，不用重启。也可以在 `ComfyUI/extra_model_paths.yaml` 里加目录：

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

底模（首次使用时自动下载到 `models/gpt_sovits/pretrained/`）：

| 用途 | 文件 | 大小 | 来源 |
|---|---|---|---|
| 所有 | `chinese-hubert-base` | 190MB | HuggingFace `lj1995/GPT-SoVITS` |
| 中文 | `chinese-roberta-wwm-ext-large` | 650MB | HuggingFace |
| 中文多音字 | `G2PWModel` | 600MB | ModelScope（固定版本，下载后核对 SHA-256） |
| v2Pro | `sv/pretrained_eres2netv2w24s4ep4.ckpt` | 100MB | HuggingFace |
| 日语里的英文单词 | `ja_userdic/userdict.csv` | 17MB | GitHub（GPT-SoVITS 固定版本） |
| 英语 | 英语词典 + nltk 词性数据 | 约 30MB | GitHub |

网络不通时按报错提示手动下载。G2PWModel、`opencc` 或 `onnxruntime` 缺失时，中文多音字改用 pypinyin 判断（能用，但准确率下降）。

依赖见 `requirements.txt`，只用到的语言才加载对应的包：日语 `pyopenjtalk-plus`；中文 `pypinyin`、`jieba`、`opencc`；英语 `g2p_en`、`wordsegment`、`nltk`。

</details>

<details>
<summary><b>读音替换</b></summary>

有些词角色总是读错时，在角色的设置文件 `anomalous_tts.json` 里加 `replace`（Anomalous 的“读音”按钮就是改这里）：左边是台词里写的，右边是让角色实际读的。台词和字幕照常写，只有送进模型的文字被替换，只对这个角色生效。

```json
{
  "replace": { "C站": "西站", "LoRA": "萝拉", "粘进": "沾进" }
}
```

长的先匹配，只替换一遍（替换出来的字不会再被替换）。日语模型说中文时特别有用：

- 中文里夹的英文字母和单词按英语发音读，只学过日语的模型读不好；写成读音相近的汉字。
- 只学过日语的模型发不好 ü（选 → 显、全 → 敲）、后鼻音（模型 → 墨西）和声调。这是模型本身的限制。在难读的词前面加个字（插件 → 这个插件）、加逗号断开，或者换个说法，往往就好了。
- 多音字偶尔判断错，例如“粘进”会读成 nián；换成同音的字。

想让 AI 助手替你调某个角色（试音、找出读错的词、换参考音频、写替换表），把 [docs/AGENT_GUIDE.md](docs/AGENT_GUIDE.md) 给它看。

</details>

<details>
<summary><b>用 API 调用</b></summary>

```json
{
  "1": { "class_type": "AnomalousTTS_CharacterSpeech", "inputs": { "character": "阿罗娜", "text": "老师好。" } },
  "2": { "class_type": "SaveAudio", "inputs": { "audio": ["1", 0], "filename_prefix": "tts/line" } }
}
```

- 除了 `character`、`text` 都可以不传。`language` 写 `中文`、`zh`、`cn`、`日文`、`ja`、`japanese` 等都可以。
- 提示（角色没有这个情绪、参考没有台词等）在节点的 `info` 输出里，也在 `/history` 里这个节点的 `outputs.text`。
- 同一角色、同一参考的句子一起批量生成（`batch_size`，默认 8；显存不够就调小）。
- 单独跑一个只有本节点的 ComfyUI 时，别的程序要用显存，本节点不会知道；用完调一次 ComfyUI 的 `POST /free`（内容 `{"unload_models": true}`）就会释放模型。

</details>

<details>
<summary><b>开发</b></summary>

- 代码结构和维护规矩：[ARCHITECTURE.md](ARCHITECTURE.md)。
- 和 Anomalous 的约定：[docs/INTERFACE.md](docs/INTERFACE.md)。
- 更新记录：[CHANGELOG.md](CHANGELOG.md)。
- 测试：见 [tests/README.md](tests/README.md)。
- 上游代码来源与改动：[UPSTREAM.md](UPSTREAM.md)。

</details>

## 许可证

MIT。内含 GPT-SoVITS 与 Genie-TTS 的部分代码（均为 MIT），出处见 [UPSTREAM.md](UPSTREAM.md)。
