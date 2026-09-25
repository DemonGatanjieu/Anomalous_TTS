# 草案：接口版本 3（在界面里完成配置和导入）

> **状态：草案。** 第 1 步已实现并并入 [../INTERFACE.md](../INTERFACE.md)；导入（第 2 步）还没实现，实现后并入并删除这份草案。

## 目标

新用户不改 `extra_model_paths.yaml`、不重启、不手动整理文件夹，就能在 Anomalous 的音频页里：

1. 看到还缺什么（底模、依赖、角色）；
2. 下载底模，或者指定一个已有的 GPT-SoVITS 整合包；
3. 把 GPT、SoVITS、参考音频、参考台词拖进来，建好一个能直接用的角色；
4. 选一个已经整理好的文件夹，作为角色库。

Anomalous 仍然只做界面，所有文件操作都由节点完成。导入接口只**增加**，不改已有接口。

## 已实现的部分

准备状态、角色库、底模来源和下载、文件夹浏览已经实现，写在 [../INTERFACE.md](../INTERFACE.md) 第 5.2 节（接口版本 3）。下面只剩导入。🔒 的含义也见那里。

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

1. ~~`status`、`libraries`、`browse`、`pretrained/*`~~（已完成）。
2. `import/*`（上传、检查、创建、追加）。
3. Anomalous 界面：准备状态卡 → 拖放导入和确认面板 → 手动选择路径。

## 待定

- 以后如果要“试听一句话”（`POST /anomalous_tts/preview`），另起一版。
