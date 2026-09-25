# Anomalous_TTS 架构

给维护者和 AI 看的入口：模块归谁管、数据怎么走、哪些规矩不能破。用户说明看 [README.md](README.md)，版本记录看 [CHANGELOG.md](CHANGELOG.md)，这两份都不是架构的依据。

## 阅读地图

| 要改…… | 先读 |
|---|---|
| 节点输入、HTTP 接口、角色文件夹规则、设置文件、剧本语法 | [docs/INTERFACE.md](docs/INTERFACE.md)（和 Anomalous 的唯一约定） |
| `vendor/` 里的上游代码、对拍记录 | [UPSTREAM.md](UPSTREAM.md) |
| 测试、需要模型的测试怎么准备 | [tests/README.md](tests/README.md) |
| 还没做的接口设计 | `docs/proposals/`（草案，不代表现状） |

## 整体结构

```text
ComfyUI 前端
  web/anomalous_tts.js         节点上的辅助按钮、下拉过滤、标签菜单
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
  ComfyUI/user/anomalous_tts.json        在界面里添加的角色库和底模来源
```

一次生成：`nodes.py` 扫描角色 → `planner.build_plan` 得到每一句的声音、语言、种子 → `engine` 按声音分批解码，每句的结果按内容缓存 → 拼成一条音频。

## 模块职责

- `core/paths.py`：角色库（ComfyUI `folder_paths` 的 `gpt_sovits` 分类，加上在界面里添加的）、底模来源、找底模和下载底模。每个底模的查找规则只在这里写一次（`PRETRAINED` 表 + `locate` / `fetch`）。别的模块不自己拼模型路径。
- `core/app_config.py`：节点自己的设置文件 `ComfyUI/user/anomalous_tts.json`（角色库、底模来源），原子写入。
- `core/downloads.py`：界面发起的底模下载，一个后台线程逐个下载，报告进度和错误；真正的下载仍是 `paths.fetch`。
- `core/dependencies.py`：每种语言需要的 Python 包和安装命令；引擎报错和准备状态都用它。
- `core/browse.py`：服务器端的文件夹浏览（浏览器拿不到本机路径）。
- `core/characters.py`：角色发现、默认权重、参考音频和情绪的解析，是 INTERFACE.md 第 2、3 节规则的**唯一实现**。扫描结果缓存 30 秒，`invalidate()` 清空。
- `core/settings.py`：`anomalous_tts.json` 的读、校验、原子写入；不认识的字段原样保留。
- `core/script.py`、`core/langdetect.py`：剧本语法、按句判断语言。
- `core/planner.py`：把剧本变成 `Plan`。每个 `Line` 带齐引擎需要的全部信息，引擎不再回头看角色或设置。
- `core/engine.py`：推理流程和每句缓存。`core/t2s_batch.py` 是批量 GPT 解码（每句独立随机数），`core/models.py` 是模型加载和 LRU，`core/checkpoints.py` 安全加载权重（`weights_only=True`）并识别版本，`core/text_frontend.py` 是文字 → 音素和 BERT 特征，`core/audio.py` 是读取、重采样、静音。
- `nodes.py`：只放控件定义和胶水代码。
- `server.py`：路由和响应拼装。耗时的扫描和磁盘操作放进 `run_in_executor`；读写文件的接口先过 `_require_local`。
- `web/anomalous_tts.js`：节点界面。辅助按钮必须加在所有真实输入之后（ComfyUI 按位置保存控件值）。

## 不能破的规矩

1. **Anomalous 只依赖 INTERFACE.md。** 节点类名、`character` / `text` 输入、HTTP 接口、设置文件格式都在那里。只加字段不算破坏；删字段、改名、改含义要升版本号并写变更记录，同时更新 Claude 项目里的 `claude/anomalous-tts-interface.md`。
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

## 改动流程

1. 改之前 `git status`，不覆盖已有的改动。
2. 跑 `python -m pytest tests`；动了前端或引擎，带上 `ANOMALOUS_TTS_ASSETS` 跑需要模型的测试。
3. 改了模块职责、数据流、接口或存储格式才更新这份文档，更新最小的相关部分；普通修复不写进来。
4. 用户能感觉到的变化写进 CHANGELOG.md 的“未发布”。
5. 验证后本地提交，只包含本次的文件；**没有明确同意不推送**。
