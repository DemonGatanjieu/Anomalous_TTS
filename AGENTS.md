# 开发与维护规范

适用于本节点的所有修改，人工维护和任何 AI（Claude、Codex、Gemini……）都一样。这里是规范的唯一正文；模块职责、数据流和不能破的规矩在 [ARCHITECTURE.md](ARCHITECTURE.md)，和 Anomalous 的约定在 [docs/INTERFACE.md](docs/INTERFACE.md)。

用户明确的当前要求优先；确需偏离时，采用最小例外，并在交付说明里写明原因和影响。

## 0. 每次提交的硬性要求

以下几条每次提交都要做到，没有“这次改动小”的例外。

1. **改之前** `git status`，不覆盖已有的改动。先读 ARCHITECTURE.md，按阅读地图只打开相关的文档。
2. **结构和文档同步。** 按下表更新文档，与代码放在同一次提交里。新增、删除、改名、拆分源码文件一律算改了模块职责：“模块职责”一节同时是文件目录，每个源码文件都要有一行。
3. **跑测试。** `python -m pytest tests`，失败不提交。其中 `tests/test_structure.py` 检查每个源码文件都登记在 ARCHITECTURE.md 里、每个模块都能从 `__init__.py` 导入到，文件超过 600 行时给出警告：先看能否按职责拆分，继续扩展要在交付说明里写原因。动了文字处理或引擎，带上 `ANOMALOUS_TTS_ASSETS` 跑需要模型的测试（见 [tests/README.md](tests/README.md)）。
4. **不留旧实现。** 被取代的函数、字段、路由和文档说明同一次提交删掉。
5. **只提交本次的文件**，没有明确同意不推送。
6. **交付说明**写清：改了什么和为什么、验证结果和没验证到的地方、遗留问题、提交号。

| 改了…… | 更新 |
| --- | --- |
| 新增 / 删除 / 改名 / 拆分源码文件，模块职责、数据流 | ARCHITECTURE.md 的“模块职责”或“整体结构”，改最小的相关部分 |
| 节点输入、HTTP 接口、角色文件夹规则、设置文件、剧本语法 | docs/INTERFACE.md。删字段、改名、改含义要升版本号 |
| `vendor/` 的上游代码 | 只通过 `tools/sync_upstream.py` 打补丁，并在 UPSTREAM.md 记下提交号 |
| 测试的新增或用途 | tests/README.md 的表 |
| 用户能感觉到的变化 | CHANGELOG.md 的“未发布”；用法变了同步 README.md |
| 开发规范本身 | 只改本文件；CLAUDE.md 只做阅读入口 |

普通修复不写进 ARCHITECTURE.md；没有变化的地方不写“本次未改变”。
