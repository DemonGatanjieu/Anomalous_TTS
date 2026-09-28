# 给 AI 助手：调一个角色

这份说明写给替用户调角色的 AI 助手（Claude、Codex、Gemini 等），人也可以照着做。只用 ComfyUI 已有的 HTTP 接口，不需要装别的东西。接口的完整定义在 [INTERFACE.md](INTERFACE.md)，这里只讲调一个角色要用到的部分。

## 能调什么

每个角色的设置在它文件夹里的 `anomalous_tts.json`，字段见 INTERFACE.md 第 3 节。调角色主要改这几项：

| 问题 | 改什么 |
|---|---|
| 某些词总是读错 | `replace`：剧本里写的 → 让角色读的。只影响送进模型的文字，剧本和字幕照常写 |
| 每句结尾都上扬、语气怪 | `reference`：换一条语气平稳的陈述句当主参考（4~8 秒最好，不要用问句） |
| 某种情绪不像 | `emotions`：给这个情绪换一条参考音频 |
| 用错了模型版本 | `gpt` / `sovits`：指定角色文件夹里的某个权重 |

语速、采样参数（`top_k`、`temperature` 等）、音量是节点上的输入，不存在角色设置里，试音时在 `/prompt` 里传。

## 规矩

- 只改用户要你调的那个角色，不动别的角色。
- 通过 `POST /anomalous_tts/settings` 写设置，不要直接改文件：服务器会检查引用的文件都在角色文件夹里，并原子写入。这个接口只接受本机请求。
- 不删、不移动、不复制角色里的音频和权重。要新增参考音频，让用户在 Anomalous 的音频页里导入。
- 设置是**整体替换**：先读出原来的，只改你要改的字段，其余字段（包括你不认识的）原样写回。
- 改之前告诉用户你要改什么；改完用同一句台词、同样的种子试音，对比前后效果。
- 调完调一次 `POST /free`，把显存还给用户的其他程序。

下面的例子里，`http://127.0.0.1:8188` 换成用户 ComfyUI 的地址，`<角色名>` 换成角色名（如 `阿罗娜/日配`，放进网址时要 URL 编码）。

## 1. 看角色现在的样子

```
GET /anomalous_tts/characters?name=<角色名>&refresh=1
```

返回的 `character` 里：

- `settings`：设置文件原文（没有设置文件时是 `{}`）；`settings_error` 不为空说明文件本身有问题。
- `reference`、`emotions`：实际生效的参考音频和台词，`source` 说明来自哪里（`settings` 设置文件、`filename` 文件名、`auto` 自动挑选）。
- `audio`：角色文件夹里所有音频的相对路径，换参考时从这里挑。
- 某条音频的台词：用 `GET /anomalous_tts/audio?character=<角色名>&path=<相对路径>` 可以取到音频本身；台词在角色的 `.list` / 同名 `.txt` / `.lab` 里，自动挑选时节点已经帮你找好了（看 `reference.text`）。

## 2. 试音

```
POST /prompt
{
  "prompt": {
    "1": { "class_type": "AnomalousTTS_CharacterSpeech",
           "inputs": { "character": "<角色名>", "text": "老师，这是今天的文件。", "language": "zh", "seed": 1 } },
    "2": { "class_type": "SaveAudio", "inputs": { "audio": ["1", 0], "filename_prefix": "tuning/before" } }
  }
}
```

- 除了 `character`、`text` 都可以不传。`language` 写 `zh` / `ja` / `en` 或 `自动`。
- 同一句换几个 `seed` 各生成一次，才看得出是角色的问题还是某一次运气不好。
- 返回的 `prompt_id` 用 `GET /history/<prompt_id>` 查结果：`outputs["2"].audio` 是生成的文件（在 ComfyUI 的 output 文件夹里），`outputs["1"].text` 是节点的提示（比如“角色没有情绪 {开心}，改用主参考”）。
- 能听就让用户听；有语音识别工具时，把识别出来的字和台词对比，找出读错的词。

## 3. 改设置

先把第 1 步读到的 `settings` 拿出来，只改要改的字段，再整体写回：

```
POST /anomalous_tts/settings
{
  "character": "<角色名>",
  "settings": {
    "format": 1,
    "language": "ja",
    "reference": { "audio": "参考音频/xxx.wav" },
    "replace": { "C站": "西站", "LoRA": "萝拉" }
  }
}
```

- 成功返回 `{"ok": true, "character": {…}}`，里面就是改完后生效的样子，核对一下 `reference` 和 `settings`。
- 返回 400 时正文是原因（例如引用的音频不在这个角色里、`replace` 的原文是空的），按原因改了再写。
- `reference` 只写 `audio` 时，台词和语言按原来的规则自动找；台词找不到或不对时，把 `text`、`language` 一起写上。

## 4. 对比，收尾

用和第 2 步完全相同的台词、种子再生成一次（`filename_prefix` 换成 `tuning/after`），和改之前对比。效果变差就把第 1 步读到的原设置写回去。最后：

```
POST /free
{ "unload_models": true }
```

## 调中文读音的经验

只学过日语的模型说中文时：

- 文字处理给模型的读音基本是对的，读错多半是模型发不好 ü（选 → 显）、后鼻音（模型 → 墨西）和声调。用 `replace` 改写成好读的字，或在难读的词前面加个字（插件 → 这个插件）、加逗号断开。
- 中文里夹的英文字母和单词按英语发音读，日语模型读不好；写成读音相近的汉字（C站 → 西站、LoRA → 萝拉）。
- 多音字偶尔判断错（“粘进”读成 nián），换成同音字（沾进）。
- 短句、句首的词最容易错；先试试在句首多加一两个字。
