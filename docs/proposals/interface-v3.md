# 草案：接口版本 3（在界面里完成配置和导入）

> **状态：草案，还没实现。** 确认后分步实现；实现完的部分并入 [../INTERFACE.md](../INTERFACE.md)，这份草案随之删除。

## 目标

新用户不改 `extra_model_paths.yaml`、不重启、不手动整理文件夹，就能在 Anomalous 的音频页里：

1. 看到还缺什么（底模、依赖、角色）；
2. 下载底模，或者指定一个已有的 GPT-SoVITS 整合包；
3. 把 GPT、SoVITS、参考音频、参考台词拖进来，建好一个能直接用的角色；
4. 选一个已经整理好的文件夹，作为角色库。

Anomalous 仍然只做界面，所有文件操作都由节点完成。版本 3 只**增加**接口，版本 2 的接口不变；为了让 Anomalous 能判断节点支不支持，所有响应里的 `format` 改为 3。旧版节点没有 `/anomalous_tts/status`（返回 404），Anomalous 就隐藏这些功能。

## 只允许本机

下面标了 🔒 的接口会读写服务器上的文件。只有从本机访问（请求来自 `127.0.0.1` / `::1`）时才会执行，否则返回 403。用 `--listen` 让局域网里的其他电脑打开 ComfyUI 时，它们只能查看状态，不能导入和改设置。`status` 里的 `local` 字段告诉界面要不要把按钮变灰。

## 节点自己的配置

`ComfyUI/user/anomalous_tts.json`（在 ComfyUI 的用户目录里，更新插件不会丢）：

```json
{ "format": 1, "libraries": ["D:/voices/模型"], "pretrained": ["D:/GPT-SoVITS/GPT_SoVITS"] }
```

- `libraries`：额外的角色库。启动时和添加时用 `folder_paths.add_model_folder_path` 注册，和 yaml 里写的效果一样，不用重启。
- `pretrained`：额外的底模来源（整合包的 `GPT_SoVITS` 文件夹，里面有 `pretrained_models`、`text`）。
- `extra_model_paths.yaml` 仍然有效，两者一起用。

---

## `GET /anomalous_tts/status`

```json
{
  "format": 3,
  "version": "0.1.0",
  "local": true,
  "libraries": [
    { "path": "…/ComfyUI/models/gpt_sovits", "source": "default", "characters": 0, "writable": true },
    { "path": "D:/voices/模型", "source": "app", "characters": 29, "writable": true }
  ],
  "pretrained": [
    { "id": "hubert", "label": "chinese-hubert-base", "needed_for": "all", "size": 190000000, "state": "ok" },
    { "id": "roberta", "label": "chinese-roberta-wwm-ext-large", "needed_for": "zh", "size": 650000000, "state": "missing" },
    { "id": "g2pw", "label": "G2PWModel", "needed_for": "zh", "size": 600000000, "state": "downloading", "done": 123000000 }
  ],
  "dependencies": {
    "ja": { "ok": true, "missing": [] },
    "zh": { "ok": false, "missing": ["opencc"] },
    "en": { "ok": true, "missing": [] }
  }
}
```

- `source`：`default`（`models/gpt_sovits`）、`yaml`（`extra_model_paths.yaml`）、`app`（在界面里添加的）。只有 `app` 的能在界面里移除。
- `state`：`ok` / `missing` / `downloading` / `error`（带 `error` 文字）。
- `dependencies` 只报告缺哪些包；界面给出可复制的安装命令，不替用户安装。
- 状态每次实时计算，不需要缓存；界面在下载中每秒查询一次。

## 🔒 `POST /anomalous_tts/pretrained/download`

`{ "ids": ["roberta", "g2pw"] }`（不写 `ids` = 下载所有缺的）。立即返回 `{"ok": true}`，在后台下载；进度看 `status`。同一个文件不会同时下载两次。只有用户点了按钮才下载。

## 🔒 `POST /anomalous_tts/pretrained/source`

`{ "path": "D:/GPT-SoVITS/GPT_SoVITS" }` → 检查里面有没有可用的底模，有就记进配置，返回新的 `status`；一个都没有就返回 400 并说明找过哪些文件夹。`{ "path": "…", "remove": true }` 移除。

## 🔒 `POST /anomalous_tts/libraries`

`{ "path": "D:/voices/模型" }` 添加，`{ "path": "…", "remove": true }` 移除（只能移除 `app` 来源的）。路径必须是已存在的文件夹。返回新的 `status`，角色列表会重新扫描。**不复制、不移动任何文件。**

## 🔒 `GET /anomalous_tts/browse?path=<文件夹>`

给“手动选择路径”用的服务器端文件夹浏览（浏览器拿不到本机路径）。

```json
{
  "path": "D:/GPT-SoVITS",
  "parent": "D:/",
  "dirs": ["GPT_weights_v2", "SoVITS_weights_v2", "output"],
  "files": [{ "name": "xxx-e15.ckpt", "kind": "gpt", "size": 155000000 }]
}
```

- 不写 `path` 时列出磁盘（Windows）或 `/`（其他系统）。
- `files` 只列相关文件：`gpt`（.ckpt）、`sovits`（.pth）、`audio`（.wav/.ogg/.mp3/.flac）、`text`（.txt/.list）。
- 只读，不跟随符号链接出去。

---

## 导入角色（复制）

分三步：**上传或指定文件 → 检查 → 创建**。浏览器拖进来的文件走上传；手动选的文件直接给路径。两种方式在“检查”和“创建”这两步完全一样。

ComfyUI 默认每个请求最大 100MB，而 GPT、SoVITS 权重常常更大，所以上传分块进行。

### 🔒 `POST /anomalous_tts/import/upload`

- 开始：`{ "name": "ALuoNa-e15.ckpt", "size": 155000000 }` → `{ "upload": "<id>" }`
- 传块：`POST /anomalous_tts/import/upload?upload=<id>&offset=<字节>`，请求体是这一块的原始字节（每块 8MB）。返回 `{ "received": <已收字节> }`。偏移不对返回 409 和正确的偏移，便于续传。
- 暂存在目标角色库下的 `.anomalous_tts_staging/`（和最终位置在同一个盘，创建时直接改名，不再复制一次）。超过 24 小时没用的暂存文件在下次上传时清理。

### 🔒 `POST /anomalous_tts/import/inspect`

```json
{ "files": [ { "upload": "<id>" }, { "path": "D:/GPT-SoVITS/SoVITS_weights_v2/ALuoNa_e16_s224.pth" }, { "upload": "<id>" } ] }
```

返回每个文件的判断，以及界面预填用的建议：

```json
{
  "files": [
    { "ref": 0, "name": "ALuoNa-e15.ckpt", "kind": "gpt", "size": 155000000 },
    { "ref": 1, "name": "ALuoNa_e16_s224.pth", "kind": "sovits", "version": "v2", "supported": true },
    { "ref": 2, "name": "Arona_Talk_3.wav", "kind": "audio", "seconds": 4.2, "text": "通常授業！…", "text_source": "list", "language": "ja" }
  ],
  "suggested": { "name": "ALuoNa", "language": "ja", "reference": 2 },
  "problems": []
}
```

- `kind` 按扩展名，GPT / SoVITS 还会读文件头确认；SoVITS 报告版本，v3 / v4 标 `supported: false`。
- 音频报告时长；3~10 秒以外在 `problems` 里提醒（GPT-SoVITS 的参考音频要求）。
- 台词按现有规则自动找：同名 `.txt` → 一起传来的 `.list` / `.txt` 标注文件（按音频文件名查）。找不到就留空，用户手动粘贴。
- 语言按台词文字判断。
- 名字取 GPT 文件名去掉 `-e<轮数>` 之后的部分。

### 🔒 `POST /anomalous_tts/import/commit`

新建角色：

```json
{
  "library": "D:/voices/模型",
  "character": "阿罗娜",
  "files": [ { "upload": "<id>" }, { "path": "D:/GPT-SoVITS/SoVITS_weights_v2/ALuoNa_e16_s224.pth" }, { "upload": "<id>" } ],
  "settings": {
    "format": 1,
    "aliases": ["阿罗娜"],
    "language": "ja",
    "reference": { "file": 2, "text": "通常授業！…", "language": "ja" },
    "emotions": { "开心": { "file": 3, "text": "…" } }
  }
}
```

往已有角色里加文件：用 `"target": "<角色名>"` 代替 `library` + `character`；`settings` 里的内容合并进已有的设置（只改这次提到的情绪和主参考，其他字段保留）。

节点按下面的结构放文件，然后写 `anomalous_tts.json`：

```text
<角色库>/<角色名>/
  GPT_weights/<原文件名>.ckpt
  SoVITS_weights/<原文件名>.pth
  参考音频/<原文件名>
  anomalous_tts.json
```

- `settings` 里的 `"file": <序号>` 指向 `files` 里的文件，节点换成放好之后的相对路径。
- **一律复制**，原文件不动。上传来的文件从暂存区改名过去，不再复制第二次。
- 先把全部文件准备在临时文件夹里，全部成功后再一次改名到位；任何一步失败，角色库里不留半个角色，已有的角色也不受影响。
- 新建时同名文件夹已存在返回 409；往已有角色加文件时，同名文件已存在也返回 409，不覆盖。
- `character` 必须是合法的文件夹名；`library` 必须是 `status.libraries` 里 `writable` 的一项。
- 成功返回 `{"ok": true, "character": {…单个角色详情…}}`，角色列表缓存同时清空。

---

## 实现顺序

1. `status`、`libraries`、`browse`、`pretrained/*`（新用户最先卡住的地方）。
2. `import/*`（上传、检查、创建、追加）。
3. Anomalous 界面：准备状态卡 → 拖放导入和确认面板 → 手动选择路径。

## 待定

- 以后如果要“试听一句话”（`POST /anomalous_tts/preview`），另起一版。
