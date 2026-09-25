# Anomalous_TTS ↔ Anomalous Model Browser 接口约定

版本：3（2026-09-25）

两个项目在不同的对话里开发。这份文档是双方唯一的约定：**Anomalous 只依赖这里写的东西**，其余都是 Anomalous_TTS 的内部实现，可以随时改。

- 修改约定的一方负责同时更新两处：Anomalous_TTS 仓库的 `docs/INTERFACE.md`，以及 Claude 项目里的 `claude/anomalous-tts-interface.md`（两处内容相同）。
- 只增加字段不算破坏；删除或改名字段、改变含义要把版本号加 1，并在文末“变更记录”写清楚。
- Anomalous 读取时要容忍不认识的字段；写回角色设置时要原样保留不认识的字段。

---

## 1. 节点

| 项 | 值 |
|---|---|
| 节点类名 | `AnomalousTTS_CharacterSpeech`（发布后不改） |
| 显示名 | 角色语音 (GPT-SoVITS) |
| 输出 | `AUDIO`（`{"waveform": [1, 1, T], "sample_rate": int}`） |

Anomalous 推送剧本时只需要设置两个输入：

| 输入 | 类型 | 说明 |
|---|---|---|
| `character` | 下拉 | 值 = 角色名（见第 2 节 `name`） |
| `text` | 多行文本 | 剧本，语法见第 4 节 |

其他输入都有默认值，Anomalous 不需要碰。需要时可以设置：`language`（`自动` / `日语` / `中文` / `英语`）、`seed`（整数）、`speed`（0.5–2.0）。

## 2. 角色

一个角色 = 一个文件夹，放在某个**角色库**里。角色库就是 ComfyUI 模型分类 `gpt_sovits` 的根目录：默认 `ComfyUI/models/gpt_sovits/`，`extra_model_paths.yaml` 里写的，以及在界面里添加的（`POST /anomalous_tts/libraries`，不用重启）。

- 根目录下的每个子文件夹是一个角色；文件夹里（含子文件夹）至少要有一个 `.ckpt`（GPT 权重）和一个 `.pth`（SoVITS 权重）。
- 如果这个子文件夹下面有 2 个及以上各自带权重的子文件夹（比如日配、中配），每个子文件夹单独算一个角色，名字是 `角色/子文件夹`。
- 名字里的 `pretrained` 文件夹会被跳过。

角色名（`name`）就是上面的相对名字，用 `/` 分隔，例如 `阿罗娜/日配数据集制`。

## 3. 角色设置文件 `anomalous_tts.json`

放在角色文件夹里，可选。**Anomalous 负责写，节点负责读。**所有字段都可选；路径都相对角色文件夹、用 `/` 分隔。

```json
{
  "format": 1,
  "aliases": ["阿罗娜"],
  "language": "ja",
  "gpt": "成品模型/GPT_weights_v2/ALuoNa-e15.ckpt",
  "sovits": "成品模型/SoVITS_weights_v2/ALuoNa_e16_s224.pth",
  "reference": {
    "audio": "参考音频/Arona_Academy_Talk_3.wav",
    "text": "通常授業！課外授業！自由時間！どれを選びますか？",
    "language": "ja"
  },
  "emotions": {
    "开心": { "audio": "参考音频/Arona_AttendanceEvent03_Enter_1.wav" },
    "生气": { "audio": "参考音频/Arona_Work_Talk_3.wav", "text": "…", "language": "ja" }
  }
}
```

| 字段 | 含义 |
|---|---|
| `format` | 固定为 `1` |
| `aliases` | 剧本里 `[名字]` 可以用的别名 |
| `language` | 这个模型说的语言：`ja` / `zh` / `en`。用于同名角色有多个版本时挑选，以及参考台词语言的默认值 |
| `gpt` / `sovits` | 默认权重；不写就用轮数最大的 |
| `reference` | 主参考（剧本里没标情绪、或 `{main}` 的部分） |
| `emotions` | 情绪名 → 参考。情绪名就是剧本里 `{情绪}` 里写的字 |
| `text` | 参考台词；不写就依次找：同名 `.txt` → 文件夹里的 GPT-SoVITS 标注文件（`.list` 或同格式 `.txt`，`路径|说话人|语言|台词`，先按文件名找，再按去掉情绪后缀的文件名找）→ 都没有则用无参考文本模式 |
| `language`（参考里） | 参考台词语言；不写就用标注文件里的，再没有就按台词文字自动判断 |

**优先级**（高 → 低）：节点上手动填的 → 设置文件 → 文件名约定（`原名.情绪.wav`，情绪名是文件名第一个点之后的部分；只是音频扩展名的部分不算情绪，例如 `X.ogg.wav`、`X.ogg (1).ogg`）→ 自动挑选。

## 4. 剧本语法

```
先生、おはようございます！{开心}今日もがんばりましょう！[pause:0.8]
[普拉娜]……おはようございます、先生。{main}今日の予定です。
```

| 写法 | 作用 |
|---|---|
| `{情绪}` | 之后改用这个情绪的参考；`{main}` 切回主参考。找不到的情绪：控制台警告，用主参考 |
| `[角色]` | 之后改由这个角色说（名字或别名）。情绪回到 `main`。找不到的角色：控制台警告，继续用当前角色 |
| `[pause:1.5]`、`[pause:500ms]`、`[停顿:1s]` | 插入停顿 |

- 第一个 `[角色]` 之前的文字由节点上选的 `character` 说。
- `[名字]` 同时匹配到多个角色版本时（比如 `阿罗娜` 对应日配、中配两个），选 `language` 和这句话语言一致的那个；还分不出来就报错，并列出候选。
- 节点把文字按句切开生成；每句的随机数只由 种子 + 句子内容 决定，所以改了某一句，其他句子的结果不变，并且可以直接用缓存。

## 5. HTTP 接口

都挂在 ComfyUI 服务器上（默认 `http://127.0.0.1:8188`）。所有 JSON 响应都带 `"format": 3`。

标了 🔒 的接口会读写服务器上的文件，只接受本机的请求（`127.0.0.1` / `::1`），其他电脑访问返回 403。`status.local` 告诉界面当前是不是本机。

旧版节点没有 `/anomalous_tts/status`（404），界面据此隐藏第 5.2 节的功能。

### 5.1 角色

#### `GET /anomalous_tts/characters`

所有角色的摘要。服务器会缓存扫描结果 30 秒；加 `?refresh=1` 强制重新扫描（给“刷新”按钮用）。

```json
{
  "format": 2,
  "characters": [
    {
      "name": "阿罗娜/日配数据集制",
      "aliases": ["阿罗娜"],
      "language": "ja",
      "has_settings": true,
      "settings": { "...": "anomalous_tts.json 原文，没有则为 {}" },
      "settings_error": null,
      "counts": { "gpt": 3, "sovits": 4, "audio": 135 },
      "reference": { "audio": "参考音频/Arona_Academy_Talk_3.wav", "text": "…", "language": "ja", "source": "auto" },
      "emotions": {
        "开心": { "audio": "…", "text": "…", "language": "ja", "source": "settings" }
      }
    }
  ]
}
```

- `reference` / `emotions` 是解析后的最终结果（已按第 3 节的优先级合并），`source` 为 `settings`、`filename` 或 `auto`。`reference` 可能为 `null`（没有 3~10 秒的音频）。
- 摘要里**没有**文件列表（一个角色可能有上千个音频）。要文件列表时取单个角色的详情。
- 读取失败的角色只有 `name` 和 `error`。

#### `GET /anomalous_tts/characters?name=<角色名>`

单个角色的详情：`{"format": 2, "character": {…摘要字段…, "gpt": [...], "sovits": [...], "audio": [...]}}`。`gpt` / `sovits` / `audio` 是角色文件夹里的全部相对路径。找不到返回 404。

#### `GET /anomalous_tts/audio?character=<name>&path=<相对路径>`

返回角色文件夹里的一个音频文件（用于试听）。只允许 `audio` 列表里的文件。

#### 🔒 `POST /anomalous_tts/settings`

写入角色设置文件。

```json
{ "character": "阿罗娜/日配数据集制", "settings": { "format": 1, "emotions": { "开心": { "audio": "参考音频/xxx.wav" } } } }
```

- `settings` 整体替换 `anomalous_tts.json`（先 GET、改、再 POST；保留不认识的字段）。
- 服务器会校验：`gpt` / `sovits` / 各 `audio` 必须是这个角色文件夹里存在的文件；不合格返回 400 和原因。
- 成功返回 `{"ok": true, "character": {…同单个角色详情…}}`。

### 5.2 准备状态（角色库、底模、依赖）

在界面里添加的角色库和底模来源记在 `ComfyUI/user/anomalous_tts.json`（节点自己管理，Anomalous 不直接读写）。

#### `GET /anomalous_tts/status`

```json
{
  "format": 3,
  "local": true,
  "libraries": [
    { "path": "…/ComfyUI/models/gpt_sovits", "source": "default", "exists": true, "writable": true, "characters": 0 },
    { "path": "D:/voices/模型", "source": "app", "exists": true, "writable": true, "characters": 29 }
  ],
  "pretrained": [
    { "id": "hubert", "label": "chinese-hubert-base", "needed_for": "all", "size": 190000000, "required": true, "state": "ok", "path": "…" },
    { "id": "g2pw", "label": "G2PWModel", "needed_for": "zh", "size": 610000000, "required": false, "state": "downloading", "done": 123000000 }
  ],
  "pretrained_sources": ["D:/GPT-SoVITS/GPT_SoVITS"],
  "dependencies": {
    "ja": { "ok": true, "missing": [] },
    "zh": { "ok": false, "missing": ["opencc"], "command": "\"…/python.exe\" -m pip install opencc" }
  }
}
```

- 路径一律用 `/` 分隔。
- `libraries[].source`：`default`（`models/gpt_sovits`）、`yaml`（`extra_model_paths.yaml`）、`app`（在界面里添加的，只有这种能移除）。`characters` 是这个角色库里的角色数。
- `pretrained[].id`：`hubert`、`roberta`、`g2pw`、`sv`、`english`、`ja_userdic`。`needed_for`：`all` / `zh` / `en` / `ja` / `v2pro`。`required: false` 表示缺了也能用，只是效果差一点（g2pw 缺失时多音字改用 pypinyin；日语用户词典缺失时英文单词按字母读）。
- `pretrained[].state`：`ok`（带 `path`）、`missing`、`queued`、`downloading`（带 `done`，已写入的字节数；`size` 是大约的总大小）、`error`（带 `error`）。
- `dependencies` 只报告缺哪些包，`command` 是给运行 ComfyUI 的那个 Python 安装它们的命令。节点不替用户安装。
- 每次实时计算；下载中可以每秒查一次。

#### 🔒 `POST /anomalous_tts/libraries`

`{ "path": "D:/voices/模型" }` 添加，`{ "path": "…", "remove": true }` 移除（只能移除 `app` 来源的）。文件夹必须存在，不能重复添加。**不复制、不移动任何文件。**成功返回新的 `status`（角色列表同时重新扫描）；不合格返回 400 和原因。

#### 🔒 `POST /anomalous_tts/pretrained/source`

`{ "path": "D:/GPT-SoVITS/GPT_SoVITS" }`：把一个 GPT-SoVITS 整合包当作底模来源（会在它自己、`pretrained_models`、`text` 里找）。一个底模都找不到返回 400。`"remove": true` 移除。成功返回新的 `status`。

#### 🔒 `POST /anomalous_tts/pretrained/download`

`{ "ids": ["roberta", "g2pw"] }`，不写 `ids` = 下载所有缺的。立即返回 `{"ok": true}`，在后台逐个下载，进度看 `status`。已有的、正在下载的会跳过。只有用户点了按钮才调用。

#### 🔒 `GET /anomalous_tts/browse?path=<文件夹>`

服务器端的文件夹浏览（浏览器拿不到本机路径），给“选择文件夹 / 文件”用。

```json
{ "path": "D:/GPT-SoVITS", "parent": "D:/", "dirs": ["GPT_weights_v2", "SoVITS_weights_v2"],
  "files": [{ "name": "xxx-e15.ckpt", "kind": "gpt", "size": 155000000 }], "truncated": false }
```

- 不写 `path`：`dirs` 是所有磁盘（Windows）或 `/`，`parent` 为 `null`。`parent` 为 `""` 表示上一级就是磁盘列表。
- `files` 只列导入能用的：`gpt`（.ckpt）、`sovits`（.pth）、`audio`（.wav/.flac/.ogg/.mp3）、`text`（.txt/.list）。隐藏文件夹不列，不进入符号链接。
- 文件夹不存在或没有权限返回 400。

---

## 变更记录

- 1（2026-09-24）：初版。
- 2（2026-09-25）：`/anomalous_tts/characters` 列表只返回摘要（去掉 `gpt` / `sovits` / `audio`，加 `counts`），文件列表改由 `?name=` 取单个角色；加 `?refresh=1`；`format` 改为 2。文件名里只是扩展名的部分（`X.ogg.wav`）不再当作情绪。节点的 `reference_audio` 改为文本框（相对路径）。
- 3（2026-09-25）：加第 5.2 节（`status`、`libraries`、`pretrained/source`、`pretrained/download`、`browse`）；角色库可以在界面里添加；`POST /settings` 只接受本机请求；`format` 改为 3。
