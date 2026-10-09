# Anomalous_TTS 架构

给维护者和 AI 看的入口：模块归谁管、数据怎么走、哪些规矩不能破。“模块职责”一节同时是文件目录：每个源码文件都有一行，找代码先看这里。改代码前先读 [AGENTS.md](AGENTS.md)（开发规范，第 0 节每次提交都要遵守）。用户说明看 [README.md](README.md)，版本记录看 [CHANGELOG.md](CHANGELOG.md)，这两份都不是架构的依据。

## 阅读地图

| 要改…… | 先读 |
|---|---|
| 节点输入、HTTP 接口、角色文件夹规则、设置文件、剧本语法 | [docs/INTERFACE.md](docs/INTERFACE.md)（和 Anomalous 的唯一约定） |
| `vendor/` 里的上游代码、对拍记录 | [UPSTREAM.md](UPSTREAM.md) |
| 测试、需要模型的测试怎么准备 | [tests/README.md](tests/README.md) |
| 替用户调一个角色（读音、参考音频），给 AI 助手的步骤 | [docs/AGENT_GUIDE.md](docs/AGENT_GUIDE.md) |

## 整体结构

```text
ComfyUI 前端
  web/anomalous_tts.js         节点上的辅助按钮、下拉过滤、标签菜单、角色菜单、警告行
  locales/zh/                  中文界面：nodeDefs.json（节点名、提示）、main.json（按钮文字），ComfyUI 读
  Anomalous Model Browser      只通过 HTTP 接口和节点输入交流（INTERFACE.md）

ComfyUI Python
  __init__.py                  注册模型分类、节点、HTTP 路由
  nodes.py                     节点定义：把输入交给 core，把结果变成 AUDIO
  server.py                    /anomalous_tts/* 路由：只做请求解析和响应拼装

  core/script.py               剧本 → 标记（文字 / {情绪} / [角色] / [停顿]）
  core/planner.py              标记 + 角色 + 节点选项 → Plan（纯逻辑，不碰 torch）
  core/engine.py               Plan → 音频：分批、缓存、调用模型
  vendor/                      GPT-SoVITS、Genie-TTS 原代码（只经 tools/sync_upstream.py 打补丁）

用户数据
  角色库（gpt_sovits 模型分类的根目录）  角色文件夹、anomalous_tts.json
  models/gpt_sovits/pretrained/          底模，首次使用或在界面里点下载时下载
  ComfyUI/user/anomalous_tts.json        存放位置、其他角色文件夹、底模来源（用户手写，节点只读）
```

一次生成：`nodes.py` 扫描角色 → `planner.build_plan` 得到每一句的声音、语言、种子 → `engine` 按声音分批解码，每句的结果按内容缓存 → 拼成一条音频。

## 模块职责

- `core/paths.py`：角色库（ComfyUI `folder_paths` 的 `gpt_sovits` 分类，加上设置文件里的存放位置和其他角色文件夹）、底模来源、找底模和下载底模。每个底模的查找规则只在这里写一次（`PRETRAINED` 表 + `locate` / `fetch`）。别的模块不自己拼模型路径。设好的存放位置暂时不存在时照样报告它，不会换成另一个文件夹。
- `core/app_config.py`：读节点自己的设置文件 `ComfyUI/user/anomalous_tts.json`（存放位置、其他角色文件夹、底模来源，用户手写）。节点从不写它；不是绝对路径的项忽略。
- `core/downloads.py`：界面发起的底模下载，一个后台线程逐个下载，报告进度和错误；真正的下载仍是 `paths.fetch`。
- `core/dependencies.py`：每种语言需要的 Python 包和安装命令；引擎报错和准备状态都用它。
- `core/importer.py`：导入角色：分块上传的暂存（只收上传的文件，不按路径读本机文件）、按扩展名判断文件种类（`kind_of`）、检查（版本、时长、台词）、创建或追加。台词查找复用 `characters.find_text`，设置校验复用 `settings.validate`，不另写一套规则。往已有角色里加时，同样的文件跳过、标注文件追加新行（失败时截回原长度），只有同名不同内容才拒绝。
- `core/characters.py`：角色发现、默认权重、参考音频和情绪的解析、读音替换（`respell`），是 INTERFACE.md 第 2、3 节规则的**唯一实现**。自动挑主参考的顺序（`reference_rank`）导入时也用它。扫描结果缓存 30 秒，`invalidate()` 清空。
- `core/settings.py`：`anomalous_tts.json` 的读、校验、原子写入；不认识的字段原样保留。
- `core/script.py`、`core/langdetect.py`：剧本语法、按句判断语言。
- `core/planner.py`：把剧本变成 `Plan`。每个 `Line` 带齐引擎需要的全部信息，引擎不再回头看角色或设置。
- `core/engine.py`：推理流程和每句缓存。`core/t2s_batch.py` 是批量 GPT 解码（每句独立随机数），`core/models.py` 是模型加载和 LRU，`core/checkpoints.py` 安全加载权重（`weights_only=True`）并识别版本，`core/text_frontend.py` 是文字 → 音素和 BERT 特征，`core/audio.py` 是读取、重采样、静音和每句的音量统一（拼接时做，缓存里存的是模型原样的声音）。
- `nodes.py`：只放控件定义和胶水代码。我们的模型不是 ComfyUI 的 ModelPatcher，所以这里包了一层 `comfy.model_management.free_memory`：ComfyUI 要的显存比空余的多时（加载图像模型、点“卸载模型”），先卸掉我们的模型；显存够时留着。
- `server.py`：路由和响应拼装。耗时的扫描和磁盘操作放进 `run_in_executor`；读写文件的接口先过 `_require_local`。
- `web/anomalous_tts.js`：节点界面。文字在代码里是英文，其他语言从 ComfyUI 的 `/i18n` 读（`locales/<语言>/main.json` 的 `anomalousTTS`），跟随 ComfyUI 的语言；节点名和输入提示的翻译在 `locales/<语言>/nodeDefs.json`，由 ComfyUI 前端自己读。旧工作流里的中文选项值打开时换成英文的。辅助按钮必须加在所有真实输入之后（ComfyUI 按位置保存控件值）。角色菜单试听参考音频、刷新角色列表，并通过 Anomalous 提供的 `window.anomalous_open_voice` 打开角色（INTERFACE.md 第 6 节）；警告行只在角色出错或缺这个语言的 Python 包时出现，数据来自 `characters` 和 `status`。

## 不能破的规矩

1. **Anomalous 只依赖 INTERFACE.md。** 节点类名、`character` / `text` 输入、HTTP 接口、设置文件格式都在那里。只加字段不算破坏；删字段、改名、改含义要升版本号并写变更记录。
2. **`vendor/` 不手改。** 改动写成 `tools/sync_upstream.py` 里的补丁，每个补丁必须恰好匹配一次；换上游版本后在 UPSTREAM.md 记下提交号。
3. **结果可复现。** 每句的随机数只由种子和这句话本身决定，与批量分组、其他句子无关。改动 planner 或 engine 时用 `test_engine.py` 确认。
4. **文字处理和官方一致。** 改前端后先和官方 GPT-SoVITS 对拍，再用 `tools/make_frontend_fixtures.py` 更新基准，不能为了让测试通过而改基准。
5. **不执行模型里的代码。** 权重一律 `weights_only=True` 加载。
6. **不阻塞事件循环。** HTTP 路由里的扫描、读写文件都放到工作线程。
7. **一个坏角色不影响其他角色。** 读取失败的角色只返回 `name` 和 `error`。
8. **可选依赖缺失时降级并说明。** 比如没有 g2pW 就用 pypinyin，控制台给一条短提示；不静默吞掉。
9. **不附带任何角色模型或声音；代码和文档里不写本机路径**，示例用 `D:/voices` 这种明显是示例的路径。
10. **联网只在需要时发生**：只下载用户要用的底模（第一次用到，或用户点了下载），不做更新检查、统计之类的请求。
11. **会读写文件的接口只接受本机请求。** 用 `--listen` 开放到局域网时，其他电脑只能查看，不能导入、改设置或浏览文件夹。
12. **没有接口接受本机路径。** 存放位置、角色文件夹、底模来源只来自用户手写的设置文件，节点只读它；导入的文件只能上传；接口只能在角色库里新建文件夹。
13. **导入一律复制，全有或全无。** 用户的原文件不动；先在暂存区拼好再一步放进角色库，失败时角色库保持原样。已存在的文件不覆盖（409）。

## 改动流程

每次提交要做的事（该更新哪份文档、测试、提交和推送）写在 [AGENTS.md](AGENTS.md) 第 0 节，这里不重复。
