# Anomalous_TTS ↔ Anomalous Model Browser 接口约定

版本：13（2026-10-08）

两个项目在不同的对话里开发。这份文档是双方唯一的约定：**Anomalous 只依赖这里写的东西**，其余都是 Anomalous_TTS 的内部实现，可以随时改。

- 约定只在这里维护：修改约定的一方负责更新 Anomalous_TTS 仓库的 `docs/INTERFACE.md`。
- 只增加字段不算破坏；删除或改名字段、改变含义要把版本号加 1，并在文末“变更记录”写清楚。
- Anomalous 读取时要容忍不认识的字段；写回角色设置时要原样保留不认识的字段。

---

## 1. 节点

| 项 | 值 |
|---|---|
| 节点类名 | `AnomalousTTS_CharacterSpeech`（发布后不改） |
| 显示名 | Character Speech (GPT-SoVITS)（中文界面里显示“角色语音 (GPT-SoVITS)”，来自 `locales/zh/nodeDefs.json`） |
| 输出 | `AUDIO`（`{"waveform": [1, 1, T], "sample_rate": int}`）；`info`（文字：生成了几句、用了几句缓存，以及提示，例如角色没有某个情绪、参考没有台词）。`info` 同时作为界面输出 `text` 出现在 `/history` 里 |

Anomalous 推送剧本时只需要设置两个输入：

| 输入 | 类型 | 说明 |
|---|---|---|
| `character` | 下拉 | 值 = 角色名（见第 2 节 `name`） |
| `text` | 多行文本 | 剧本，语法见第 4 节 |

其他输入都是可选的，都有默认值（通过 `/prompt` 只传这两个也能运行），Anomalous 不需要碰。需要时可以设置：`language`（`auto` / `Japanese` / `Chinese` / `English`，也接受 `ja` / `zh` / `en`、`日文`、`japanese`、`cn`、`english` 等写法）、`seed`（整数）、`speed`（0.5–2.0）、`volume`（`normalize`：每句人声调到约 -20 dBFS，默认；`off`）。下拉选项以前是中文（`自动`、`日语`、`统一音量`、`自动调整`、`不调整` 等），旧名字照样接受。

## 2. 角色

一个角色 = 一个文件夹，放在某个**角色库**里。角色库就是 ComfyUI 模型分类 `gpt_sovits` 的根目录：默认 `ComfyUI/models/gpt_sovits/`、`extra_model_paths.yaml` 里写的，以及节点设置文件 `ComfyUI/user/anomalous_tts.json` 里写的**存放位置**（导入的角色放在这里，见 5.2）和其他角色文件夹。

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
  },
  "replace": { "C站": "西站", "LoRA": "萝拉" },
  "defaults": { "language": "zh", "speed": 1.1 }
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
| `defaults` | 这个角色常用的生成设置：`language`（`auto` / `ja` / `zh` / `en`）、`speed`（0.5~2.0），以及采样参数 `top_k`（1~100 整数）、`top_p`（0.05~1）、`temperature`（0.05~2）、`repetition_penalty`（1~2）。都可以不写（用节点的默认值）。**节点不读**（节点上的输入照旧生效）；Anomalous 在界面里直接生成时用它填节点输入 |
| `replace` | 读音替换：剧本里写的 → 这个角色实际读的。合成前替换（长的先匹配，只替换一遍），剧本和字幕照常写。原文不能为空，读法可以为空（不读） |
| `text` | 参考台词；不写就依次找：同名 `.txt` 或 `.lab` → 文件夹里的 GPT-SoVITS 标注文件（`.list` 或同格式 `.txt`，`路径|说话人|语言|台词`，先按文件名找，再按去掉情绪后缀的文件名找）→ 都没有则用无参考文本模式 |
| `language`（参考里） | 参考台词语言；不写就用标注文件里的，再没有就按台词文字自动判断 |

**优先级**（高 → 低）：节点上手动填的 → 设置文件 → 文件名约定（`原名.情绪.wav`，情绪名是文件名第一个点之后的部分；只是音频扩展名的部分不算情绪，例如 `X.ogg.wav`、`X.ogg (1).ogg`）→ 自动挑选。

**自动挑主参考**（3~10 秒的音频里）：有台词的优先 → 和角色同名、不带情绪后缀的文件优先 → 陈述句优先，其次感叹句（台词以 `！` 结尾），问句（以 `？` 结尾）最后 → 4~8 秒优先。参考音频的语气会带到角色说的每一句话里，所以问句只在没有别的可选时才用。

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
| `[take:2]`、`[版本:2]` | 紧跟着的文字（到换行或下一个标签为止）换第 2 版：只有这段的随机数变，其他句子不变、可以直接用缓存。写在情绪标签之后，例如 `{开心}[take:2]好的。`；`[take:1]` 和不写一样 |

- 第一个 `[角色]` 之前的文字由节点上选的 `character` 说。
- `[名字]` 同时匹配到多个角色版本时（比如 `阿罗娜` 对应日配、中配两个），选 `language` 和这句话语言一致的那个；还分不出来就报错，并列出候选。
- 节点把文字按句切开生成；每句的随机数只由 种子 + 句子内容（+ `[take]` 的版本号）决定，所以改了某一句，其他句子的结果不变，并且可以直接用缓存。

## 5. HTTP 接口

都挂在 ComfyUI 服务器上（默认 `http://127.0.0.1:8188`）。所有 JSON 响应都带 `"format": 13`。

标了 🔒 的接口会读写服务器上的文件，只接受本机的请求（`127.0.0.1` / `::1`），其他电脑访问返回 403。`status.local` 告诉界面当前是不是本机。

旧版节点没有 `/anomalous_tts/status`（404），界面据此隐藏第 5.2、5.3 节的功能。

出错时：400 = 请求不对，正文是原因（文字）；409 = 已经存在 / 偏移不对，正文是 JSON `{"error": 原因, …}`；403 = 不是本机。

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

### 5.2 准备状态（存放位置、底模、依赖）

节点设置文件 `ComfyUI/user/anomalous_tts.json`（路径见 `status.settings_file`）：

```json
{
  "format": 1,
  "storage": "D:/voices",
  "libraries": ["E:/old voices"],
  "pretrained": ["D:/GPT-SoVITS/GPT_SoVITS"]
}
```

- `storage`（存放位置）、`libraries`（其他角色文件夹）、`pretrained`（底模来源：GPT-SoVITS 整合包，用里面现成的底模，不用下载）**只由用户编辑这个文件来设置**。节点只读这个文件，没有接口能改它，所以请求决定不了在哪里新建文件夹、读哪些文件夹。
- 都要写绝对路径；空的、相对路径、不是字符串的项会被忽略。改了以后不用重启：存放位置和底模来源马上生效，新写的角色文件夹马上读取（从文件里删掉的，重启 ComfyUI 后才不再读取）。
- Anomalous 不直接读写这个文件；要换存放位置、加底模来源时，告诉用户改这个文件（显示 `status.settings_file`）。
- 没有接口接受本机的文件或文件夹路径：导入的文件一律由浏览器上传（5.3）。

#### `GET /anomalous_tts/status`

```json
{
  "format": 13,
  "local": true,
  "settings_file": "…/ComfyUI/user/anomalous_tts.json",
  "storage": "D:/voices",
  "libraries": [
    { "path": "…/ComfyUI/models/gpt_sovits", "source": "default", "storage": false, "exists": true, "writable": true, "characters": 26 },
    { "path": "D:/voices", "source": "storage", "storage": true, "exists": true, "writable": true, "characters": 3 }
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
- `storage`：存放位置，导入的角色放在这里（没设置过就是 `models/gpt_sovits`）。`libraries[].storage` 标出它是哪一项。
- `libraries[].source`：`default`（`models/gpt_sovits`）、`storage`（设置文件里的存放位置）、`app`（设置文件里的 `libraries`）、`yaml`（`extra_model_paths.yaml`）。`characters` 是这里的角色数。
- `pretrained_sources`：设置文件里的 `pretrained`。
- `pretrained[].id`：`hubert`、`roberta`、`g2pw`、`sv`、`english`、`ja_userdic`。`needed_for`：`all` / `zh` / `en` / `ja` / `v2pro`。`required: false` 表示缺了也能用，只是效果差一点（g2pw 缺失时多音字改用 pypinyin；日语用户词典缺失时英文单词按字母读）。
- `pretrained[].state`：`ok`（带 `path`）、`missing`、`queued`、`downloading`（带 `done`，已写入的字节数；`size` 是大约的总大小）、`error`（带 `error`）。
- `dependencies` 只报告缺哪些包，`command` 是给运行 ComfyUI 的那个 Python 安装它们的命令。节点不替用户安装。
- 每次实时计算；下载中可以每秒查一次。

#### 🔒 `POST /anomalous_tts/pretrained/download`

`{ "ids": ["roberta", "g2pw"] }`，`ids` 必须写（要全部下载就列出 `status` 里所有缺的），不写返回 400。立即返回 `{"ok": true}`，在后台逐个下载，进度看 `status`。已有的、正在下载的会跳过。只有用户点了按钮才调用。

### 5.3 导入角色（一律复制）

把 GPT、SoVITS 权重、参考音频、台词文件做成一个角色，或加到已有角色里。分三步：**上传 → 检查 → 创建**。文件一律由浏览器上传（拖进来或在浏览器里选），节点不按路径读本机文件；用户的原文件永远不动。

每个文件在请求里写成 `{"upload": "<上传 id>"}`（写成别的，例如 `{"path": …}`，返回 400）。只收 `.ckpt`、`.pth`、音频（.wav/.flac/.ogg/.mp3）、`.txt`、`.lab`、`.list`。

可以另带 `"name"`：这个文件放进角色里用的名字（比如两个情绪文件夹里都有 `01.wav`，一个改成 `难过_01.wav`）。扩展名必须和原文件一样，名字要合法。`inspect` 返回的 `name` 是新名字；台词还是按原文件名在标注文件里找，同名 `.txt` / `.lab` 按新名字配对（一起改名即可）。

#### 🔒 `POST /anomalous_tts/import/upload`

ComfyUI 默认每个请求最大 100MB，权重常常更大，所以分块上传。

- 开始：JSON `{ "name": "ALuoNa-e15.ckpt", "size": 155312957, "library": "D:/voices/模型" }`（`library` 可省略，默认存放位置）→ `{ "upload": "<id>" }`。
- 传块：`POST /anomalous_tts/import/upload?upload=<id>&offset=<字节>`，请求体是这一块的原始字节（建议 8MB 一块）→ `{ "received": <已收字节> }`。`offset` 不等于已收字节时返回 409 `{"error": …, "received": n}`，从 `n` 继续即可。超过声明的 `size` 返回 400。
- 上传的文件暂存在角色库里的 `.anomalous_tts_staging/`（扫描角色时跳过）。24 小时没动的暂存文件，下次开始上传时清理；节点重启后没用完的上传也会失效。

#### 🔒 `POST /anomalous_tts/import/discard`

`{ "uploads": ["<id>", …] }`：用户取消导入时删掉暂存的上传。→ `{"ok": true}`

#### 🔒 `POST /anomalous_tts/import/inspect`

`{ "files": [ … ] }`（往已有角色里加时再带 `"target": "<角色名>"`）→ 每个文件的判断，加上界面预填用的建议：

```json
{
  "files": [
    { "ref": 0, "name": "ALuoNa-e15.ckpt", "kind": "gpt", "size": 155312957 },
    { "ref": 1, "name": "ALuoNa_e16_s224.pth", "kind": "sovits", "size": 85007879, "version": "v2", "supported": true },
    { "ref": 2, "name": "Arona_Talk_3.wav", "kind": "audio", "size": 460834, "seconds": 5.22,
      "text": "通常授業！…", "text_source": "list", "language": "ja" },
    { "ref": 3, "name": "all.txt", "kind": "text", "size": 17278 }
  ],
  "suggested": { "name": "ALuoNa", "language": "ja", "reference": 2 },
  "problems": []
}
```

- `ref` 是这个文件在 `files` 里的序号，创建时用它指文件。
- SoVITS 报告 `version`；v3 / v4 标 `supported: false`。
- 音频报告 `seconds`；台词按第 3 节的规则从一起给的文件里找（同名 `.txt` / `.lab` → `.list` / 同格式 `.txt` 标注文件）；都没有、而文件名读起来像一句话时（至少 4 个汉字或假名，或带句读符号；开头的 `【开心】` 这类标签去掉），用文件名作建议。`text_source` 是 `txt`、`lab`、`list`、`filename` 或 `null`（没找到，让用户粘贴）。`filename` 只是导入时的建议，生成语音时节点从不按文件名猜台词。`language` 来自标注文件，没有就按台词判断。
- `suggested.name` 取自 GPT 文件名（去掉 `-e<轮数>`）；`suggested.reference` 按第 3 节“自动挑主参考”的顺序（不看文件名）从 3~10 秒的音频里挑：有台词、陈述句、4~8 秒优先。
- 带 `target` 时：每个文件多一个 `existing`，说明角色里同一位置有没有这个文件：`null`（没有，会复制）、`same`（完全一样，会跳过）、`merge`（都是标注文件，新的行会追加进去）、`different`（同名但内容不同，创建时会被拒绝）。台词也会在角色已有的标注文件和同名 `.txt` / `.lab` 里找（一条一条加音频时不用再带标注文件）。
- `problems` 是给用户看的提醒（不支持的版本、同名但内容不同的文件）。缺不缺权重、音频长度合不合适（`seconds` 不在 3~10 秒）由界面自己判断，这里不提醒。有提醒也可以继续创建。

#### 🔒 `POST /anomalous_tts/import/commit`

新建角色：

```json
{
  "library": "D:/voices/模型",
  "character": "阿罗娜",
  "files": [ { "upload": "<id>" }, { "upload": "<id>" }, { "upload": "<id>" } ],
  "settings": {
    "aliases": ["阿罗娜"],
    "language": "ja",
    "reference": { "file": 2, "text": "通常授業！…", "language": "ja" },
    "emotions": { "开心": { "file": 3, "text": "…" } }
  }
}
```

加到已有角色：把 `library` + `character` 换成 `"target": "<角色名>"`。`settings` 合并进已有设置：写到的键覆盖，`emotions` 按情绪名合并，没写到的都保留。

- `settings` 就是第 3 节的格式，只是 `reference` / `emotions` 里用 `"file": <序号>` 指文件（`gpt` / `sovits` 也可以写序号），节点换成放好后的相对路径。可以省略。
- 文件放到：

  ```text
  <角色库>/<角色名>/
    GPT_weights/<原文件名>.ckpt
    SoVITS_weights/<原文件名>.pth
    参考音频/<原文件名>（音频和 .txt / .list）
    anomalous_tts.json
  ```

- 新建角色至少要有一个 `.ckpt` 和一个 `.pth`。`character` 必须是合法的文件夹名，`library` 必须是 `status.libraries` 里 `writable` 的一项，通常就是 `status.storage`。
- 同名文件夹已存在：409，什么都不改。往已有角色里加文件时，完全一样的文件跳过，标注文件（`.list` / 同格式 `.txt`）把没有的行追加进去，同名但内容不同的其他文件：409，什么都不改。设置不合格、序号不对：400，什么都不改。
- 全有或全无：先在暂存区拼好，最后一步才放进角色库；中途失败时角色库保持原样，上传的文件回到暂存区，可以直接重试。
- 成功 → `{"ok": true, "character": {…单个角色详情…}, "skipped": ["参考音频/a.wav"], "merged": ["参考音频/all.list"]}`（`skipped` / `merged` 是相对路径，新建角色时都是空列表），角色列表同时刷新。

## 6. 节点调用 Anomalous

Anomalous Model Browser 加载后在页面上提供：

```js
window.anomalous_open_voice(character)  // → true：已打开浏览器的“角色语音”页并定位到这个角色
```

- `character` 是节点 `character` 控件的值（角色名，多套模型时是 `角色/子文件夹`）；空字符串打开整个页面。
- 节点“角色”菜单里的“导入或编辑角色”调用它。函数不存在（没装 Anomalous 或版本太旧）时，节点改为提示安装并打开 Anomalous 的项目主页。
- 只负责打开界面，不读写任何文件。

---

## 变更记录

- 1（2026-09-24）：初版。
- 2（2026-09-25）：`/anomalous_tts/characters` 列表只返回摘要（去掉 `gpt` / `sovits` / `audio`，加 `counts`），文件列表改由 `?name=` 取单个角色；加 `?refresh=1`；`format` 改为 2。文件名里只是扩展名的部分（`X.ogg.wav`）不再当作情绪。节点的 `reference_audio` 改为文本框（相对路径）。
- 3（2026-09-25）：加第 5.2 节（`status`、`libraries`、`pretrained/source`、`pretrained/download`、`browse`）和第 5.3 节（`import/upload`、`import/discard`、`import/inspect`、`import/commit`）；角色库可以在界面里添加；`POST /settings` 只接受本机请求；`format` 改为 3。
- 4（2026-09-25）：只有一个存放位置：加 `POST /anomalous_tts/storage`（可以把角色移过去），`status` 加 `storage`、`move`，`libraries[]` 加 `storage` 字段和 `source: storage`；`POST /anomalous_tts/libraries` 只能移除以前的存放位置，**不能再添加**；上传默认放在存放位置；`format` 改为 4。台词文件也认 `.lab`，导入检查的 `text_source` 加 `lab`、`filename`。
- 5（2026-09-26）：往已有角色里加文件更宽容：完全一样的文件跳过，标注文件合并新行，只有同名但内容不同的才 409。`import/inspect` 可以带 `target`，文件多 `existing` 字段，并会用角色已有的台词文件找台词；`import/commit` 的结果加 `skipped`、`merged`；`format` 改为 5。
- 6（2026-09-26）：`browse` 加 `recursive=1`，一次列出文件夹里所有能用的文件（批量导入）；`format` 改为 6。
- 7（2026-09-26）：`import/inspect` 的 `problems` 不再包含“还缺 GPT / SoVITS 权重”（界面的待办清单自己显示）；`format` 改为 7。
- 8（2026-09-26）：`import/inspect` 的 `problems` 不再逐条提醒不在 3~10 秒的音频（界面按 `seconds` 自己处理）；`browse?recursive=1` 不进入 GPT-SoVITS 程序和训练用的文件夹；`format` 改为 8。
- 11（2026-09-29）：剧本语法加 `[take:N]`（只重做一段）；设置文件加 `defaults`（Anomalous 直接生成时用的语言、语速）；`pretrained/download` 必须写 `ids`，空请求体不再下载全部；文件名、角色名不能含不可见字符和改变文字方向的字符（如 U+202E）；`format` 改为 11。
- 13（2026-10-08）：没有接口再接受本机路径。存放位置、其他角色文件夹、底模来源只由用户编辑 `ComfyUI/user/anomalous_tts.json` 设置，节点只读：去掉 `POST /anomalous_tts/storage`（以及移动角色）、`POST /anomalous_tts/libraries`、`POST /anomalous_tts/pretrained/source`；`status` 去掉 `move`，加 `settings_file`；`libraries[].source: app` 改为指设置文件里的 `libraries`。去掉 `GET /anomalous_tts/browse`（含 `recursive=1`）和 `GET /anomalous_tts/import/preview`；导入的文件只能上传，`{"path": …}` 返回 400。`format` 改为 13。
- 12（2026-10-08）：节点的下拉选项改为英文：`language` 是 `auto` / `Japanese` / `Chinese` / `English`，`gpt_weights`、`sovits_weights` 的自动是 `auto`，`cross_lingual` 是 `adjust` / `off`，`volume` 是 `normalize` / `off`；以前的中文选项名照样接受。显示名改为 `Character Speech (GPT-SoVITS)`，中文界面的名字和提示来自 `locales/zh/`。`format` 改为 12。
- 10（2026-09-29）：节点除 `character`、`text` 外的输入都改为可选；`language` 接受别名；加 `volume` 输入（默认统一音量）；`info` 也作为界面输出 `text` 进 `/history`。设置文件加 `replace`（读音替换）。自动挑主参考（以及 `import/inspect` 的 `suggested.reference`）改为优先陈述句和 4~8 秒，避开问句；`format` 改为 10。
- 9（2026-09-26）：`browse?recursive=1` 的 `output`、`temp`、`tools`、`logs`、`GPT_SoVITS` 只在整合包里跳过，别处照常扫描；`runtime`、`pretrained_models` 一律跳过；最多 6 层；结果加 `skipped`、`too_deep`。导入的文件可以带 `name` 改名。加 `GET /anomalous_tts/import/preview`（试听本机音频）。`format` 改为 9。
