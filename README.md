# Anomalous TTS (GPT-SoVITS)

在 ComfyUI 里用自己训练的 GPT-SoVITS 角色模型读文字。**不需要另外安装 GPT-SoVITS 整合包**，推理代码已经精简后放在节点包里，直接用 ComfyUI 的 PyTorch 和显卡。

姊妹项目：[Anomalous Model Browser](https://github.com/DemonGatanjieu/Anomalous_Model_Browser)。

> 开发中，尚未发布。

## 目前能做什么

- 节点 **角色语音 (GPT-SoVITS)**（`AnomalousTTS_CharacterSpeech`），输出 ComfyUI 的 `AUDIO`，可接 Save Audio / Preview Audio。
- 模型版本：v1、v2。v2Pro / v2ProPlus 下一步加入；v3 / v4 不支持。
- 语言：日语、中文。英语下一步加入。中文里的英文字母目前会被忽略，日文里的英文单词按片假名读。
- 可调：top_k、top_p、temperature、repetition_penalty、语速、句间停顿、随机种子（同一种子可复现同样的结果）。
- 参考台词自动读取：同名 `.txt`，或 GPT-SoVITS 训练用的标注文件（`.list`，或同格式的 `.txt`，每行 `音频路径|角色|JA|台词`）。
- 没有台词时自动改用无参考文本模式。

## 放模型

默认目录是 `ComfyUI/models/gpt_sovits/`。每个角色一个文件夹，里面怎么分子文件夹都可以：

```
models/gpt_sovits/
  阿罗娜/
    日配/            ← 一个角色下有两套模型时，分别列为「阿罗娜/日配」「阿罗娜/中配」
      GPT_weights_v2/ALuoNa-e15.ckpt
      SoVITS_weights_v2/ALuoNa_e16_s224.pth
      参考音频/*.wav
      all.txt        ← 标注文件，用来查参考台词
    中配/
      ...
  pretrained/        ← 底模，首次使用自动下载
```

GPT 权重（`.ckpt`）和 SoVITS 权重（`.pth`）默认选轮数最大的，也可以在节点里指定。

已有的模型不用复制，在 `ComfyUI/extra_model_paths.yaml` 里加一个目录即可：

```yaml
anomalous_tts:
    base_path: D:/voices
    gpt_sovits: 模型
```

如果电脑上有 GPT-SoVITS 整合包，也可以把它的底模目录加进来，省掉下载（`text` 里有 G2PWModel）：

```yaml
gpt_sovits_pretrained:
    base_path: D:/GPT-SoVITS/GPT_SoVITS
    gpt_sovits: |
        pretrained_models
        text
```

## 底模

首次生成时自动下载到 `models/gpt_sovits/pretrained/`：

- `chinese-hubert-base`（约 190MB，HuggingFace `lj1995/GPT-SoVITS`）。网络不通时按报错提示手动下载。
- 日语用户词典 `ja_userdic/userdict.csv`（17MB，GitHub）。下载失败不影响使用，只是日文里的英文单词会按字母读。
- 中文：`chinese-roberta-wwm-ext-large`（约 650MB，HuggingFace）和多音字模型 `G2PWModel`（约 600MB，ModelScope）。G2PWModel 下载失败、或没装 `opencc` / `onnxruntime` 时，多音字改用 pypinyin 判断，能用但准确率明显下降（测试的 70 句里有 11 句读音和官方不同）。

## 依赖

`requirements.txt` 里的包大多是 ComfyUI 自带或常见的。日语需要 `pyopenjtalk-plus`。

## 许可证

MIT。内含 GPT-SoVITS 与 Genie-TTS 的部分代码（均为 MIT），出处和改动见 [UPSTREAM.md](UPSTREAM.md)。本包不附带任何角色模型或声音。
