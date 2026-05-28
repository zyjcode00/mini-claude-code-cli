# 新长期记忆系统说明

本文档总结当前 `mini-claude-code-cli` 项目在阶段 1 到阶段 5 重构后形成的新长期记忆系统。

保存路径：`docs/new_long_term_memory_system.md`

---

## 1. 总体目标

新的长期记忆系统目标不是简单保存聊天记录，而是为 code-agent 提供一个可持续积累项目经验的记忆闭环：

```text
任务执行 → 观察记录 → 结构化长期记忆 → 混合检索 → Prompt 预算控制注入 → 辅助后续决策
```

它现在覆盖：

- 用户偏好保存
- 项目事实保存
- 架构决策保存
- 文件修改历史保存
- 历史错误与修复经验保存
- 任务执行经验保存
- 长期记忆检索
- Prompt 自动注入
- 文件编辑前自动召回
- 测试失败后自动召回
- 记忆统计查看

---

## 2. 核心模块

当前记忆系统主要由以下模块组成：

| 文件 | 职责 |
|---|---|
| `core/memory_manager.py` | 统一记忆编排入口，负责保存、召回、统计、Prompt 上下文构建 |
| `core/memory_items.py` | 新长期记忆数据模型，例如 `MemoryItem`、`RawObservation` |
| `core/memory_layers.py` | 工作记忆、情景记忆、长期记忆三层结构 |
| `core/memory_retrieval.py` | 统一检索层与 Hybrid Recall |
| `core/memory_context_builder.py` | 将召回结果转换为可注入 Prompt 的上下文，并控制 token budget |
| `core/engine.py` | Agent 主循环，接入主动记忆保存、任务开始召回、工具前后自动召回 |
| `tools/memory_tool.py` | 暴露 `memory_save`、`memory_recall`、`memory_file_history`、`memory_error_history`、`memory_stats` 等工具 |

---

## 3. 记忆保存机制

### 3.1 自动保存

Agent 在执行任务过程中会自动记录关键观察，例如：

- 用户提出的任务
- 工具调用结果
- 文件修改行为
- 测试失败信息
- 错误修复过程
- 用户明确表达的偏好
- 项目架构或工作流决策

大致链路：

```text
AgentEngine
  ↓
记录 RawObservation
  ↓
MemoryManager 统一处理
  ↓
生成或更新 MemoryItem
  ↓
落盘到本地 memory/long_term
```

### 3.2 手动保存

系统提供手动保存工具：

```text
memory_save
```

可以用于显式保存一条长期记忆，例如用户偏好、架构决策、历史问题等。

示例概念：

```text
memory_save:
  content: "用户偏好：修改 core/ 或 tools/ 逻辑后必须补测试并运行 pytest"
  memory_type: "preference"
  importance: 0.9
```

---

## 4. 记忆数据模型

新的主线长期记忆格式是 `MemoryItem`。

一条记忆通常包含：

```text
id
content
memory_type
importance
confidence
created_at
updated_at
tags
source
file_paths
error_type
task_type
metadata
```

常见 `memory_type` 包括：

| 类型 | 含义 |
|---|---|
| `preference` | 用户偏好 |
| `bug` | 历史错误和修复经验 |
| `architecture` | 架构设计信息 |
| `workflow` | 工作流程经验 |
| `decision` | 技术决策 |
| `fact` | 项目事实 |
| `procedural` | 操作步骤型知识 |
| `task` | 历史任务摘要 |

旧版 `SessionSummary` 仍然兼容召回，因此当前系统可以同时利用新旧两类长期记忆。

---

## 5. 存储位置

当前记忆使用本地文件式存储。

主要目录：

```text
memory/
  long_term/
    index.json
    memory_item_*.json
```

其中：

- `memory_item_*.json` 保存结构化长期记忆
- `index.json` 保存长期记忆索引信息
- 旧会话摘要仍可从 sessions 或历史存储中兼容读取

可以通过 `memory_stats` 查看当前存储统计。

---

## 6. 检索机制：Hybrid Recall

长期记忆检索统一走 `MemoryManager` 与 `HybridRecall`。

检索时不只是简单字符串匹配，而是综合考虑：

- 查询关键词
- 记忆类型
- 文件路径
- 错误类型
- 任务类型
- 重要度 `importance`
- 置信度 `confidence`
- 新旧记忆来源

常见检索入口：

```text
memory_recall
memory_file_history
memory_error_history
MemoryManager.hybrid_recall(...)
```

### 6.1 普通召回

工具：

```text
memory_recall
```

用途：根据普通 query 查找相关长期记忆。

### 6.2 文件历史召回

工具：

```text
memory_file_history
```

用途：根据文件路径查找历史修改、历史 bug、相关决策与注意事项。

例如：

```text
memory_file_history:
  file_path: "core/engine.py"
```

### 6.3 错误历史召回

工具：

```text
memory_error_history
```

用途：根据错误类型或错误文本召回历史类似问题和修复经验。

例如：

```text
memory_error_history:
  error_type: "AssertionError"
  query: "pytest failed"
```

---

## 7. Prompt 注入机制

阶段 5 引入了：

```text
core/memory_context_builder.py
```

核心类：

```text
MemoryContextBuilder
```

它负责把召回结果转换成可注入 Prompt 的上下文片段。

任务开始时，`AgentEngine` 会调用：

```text
MemoryManager.build_prompt_memory_context(...)
```

大致流程：

```text
用户输入
  ↓
AgentEngine 推断任务类型
  ↓
MemoryManager 执行 Hybrid Recall
  ↓
MemoryContextBuilder 排序、过滤、去重、截断
  ↓
生成 [相关长期记忆] Prompt 片段
  ↓
注入模型上下文
```

注入片段概念示例：

```text
[相关长期记忆]
### 相关长期记忆
任务类型: code_edit

1. [bug] 修改 core/engine.py 后需要运行相关 engine 测试
   来源: core/engine.py; long_term_items
   相关度: 3.00
   原因: 文件路径匹配、任务类型匹配
   内容: 之前在 AgentEngine 中接入工具调用时出现过异步处理遗漏，修改后必须运行 pytest tests/test_engine.py
```

---

## 8. Token Budget 与上下文污染控制

为了避免长期记忆挤占上下文或污染当前任务，系统加入了预算控制。

当前关键配置概念：

```text
memory_token_budget = 1500
memory_recall_top_k = 8
```

`MemoryContextBuilder` 会执行：

- 限制最大召回条数
- 限制总注入长度
- 截断单条过长记忆
- 过滤低置信度记忆
- 去除重复或高度相似记忆
- 按任务类型重排记忆优先级

这保证 Prompt 中只注入与当前任务最相关、最有价值的一小部分长期记忆。

---

## 9. 任务类型感知召回

系统会根据用户输入推断任务类型。

当前常见任务类型：

```text
code_edit
architecture
test_failure
documentation
git
general
```

不同任务类型优先召回不同记忆：

| 任务类型 | 优先记忆 |
|---|---|
| `code_edit` | `bug`、`workflow`、`decision`、`architecture` |
| `architecture` | `architecture`、`decision`、`fact`、`workflow` |
| `test_failure` | `bug`、`procedural`、`workflow`、`fact` |
| `documentation` | `preference`、`architecture`、`decision`、`task` |
| `git` | `workflow`、`preference`、`bug`、`procedural` |
| `general` | 综合召回 |

---

## 10. 文件编辑前自动召回

在调用文件编辑类工具前，系统会自动进行文件历史召回。

触发工具：

```text
edit_file
write_full_file
```

触发条件：工具参数中存在：

```text
path
file_path
```

系统会生成类似上下文：

```text
[自动文件历史召回: core/engine.py]
...
```

用途：

- 在修改文件前了解历史 bug
- 避免重复踩坑
- 识别该文件相关架构约束
- 提醒应运行哪些测试

---

## 11. 测试或工具失败后自动召回

当工具失败时，系统会尝试从失败输出中提取错误类型和关键信息，然后召回历史错误经验。

触发工具：

```text
run_pytest
execute_bash
edit_file
write_full_file
```

系统会生成类似上下文：

```text
[自动错误历史召回: AssertionError]
...
```

用途：

- 利用过去类似错误的修复经验
- 提升 Agent 自愈能力
- 减少重复排查成本

---

## 12. 记忆工具列表

当前默认工具集中包含以下记忆工具：

| 工具名 | 用途 |
|---|---|
| `memory_save` | 手动保存长期记忆 |
| `memory_recall` | 根据 query 召回相关长期记忆 |
| `memory_file_history` | 根据文件路径召回文件历史 |
| `memory_error_history` | 根据错误类型或错误文本召回历史错误经验 |
| `memory_stats` | 查看记忆系统统计信息 |

`memory_stats` 可查看：

- 工作记忆数量
- 情景记忆数量
- 长期记忆数量
- `MemoryItem` 数量
- `SessionSummary` 数量
- 记忆存储目录
- 使用率信息

---

## 13. 当前测试状态

记忆系统重构阶段已经配套了测试。

相关测试文件：

```text
tests/test_memory_items.py
tests/test_memory_layers.py
tests/test_memory_manager.py
tests/test_memory_manager_phase2.py
tests/test_memory_models.py
tests/test_memory_persistence.py
tests/test_memory_phase3.py
tests/test_memory_phase4.py
tests/test_memory_phase5.py
tests/test_memory_retrieval.py
```

阶段 5 完成时的完整测试结果：

```text
121 passed
```

说明当前主线长期记忆系统已经通过测试验证。

---

## 14. 当前系统闭环总结

新的长期记忆系统已经形成以下闭环：

```text
用户任务 / 工具执行 / 文件修改 / 错误输出
        ↓
AgentEngine 自动记录观察
        ↓
MemoryManager 统一编排
        ↓
RawObservation 原始事件
        ↓
MemoryItem 结构化长期记忆
        ↓
本地 memory/long_term 存储
        ↓
Hybrid Recall 混合检索
        ↓
MemoryContextBuilder 控制预算、排序、去重、截断
        ↓
[相关长期记忆] 注入 Prompt
        ↓
辅助后续 Agent 决策
```

同时还有两个自动增强点：

```text
文件编辑前 → 自动召回文件历史
测试或工具失败后 → 自动召回错误历史
```

因此，当前系统已经从“简单会话摘要”升级为：

```text
结构化长期记忆 + 混合检索 + Prompt 预算控制注入 + 文件历史召回 + 错误历史召回 + 统计工具
```

---

## 15. 后续可选优化方向

当前重构主线已经完成，但如果继续产品化，可以考虑：

1. 记忆质量治理
   - 合并重复记忆
   - 降低过期记忆权重
   - 清理低价值记忆

2. 更智能的经验压缩
   - 将连续错误修复过程压缩成可复用 bug 模板
   - 将多轮任务总结为高质量项目经验

3. 记忆可观测性
   - 展示每次 Prompt 注入了哪些记忆
   - 展示召回命中原因
   - 提供 memory dashboard

4. 记忆安全机制
   - 支持用户删除或禁用某条记忆
   - 对高影响记忆加入确认机制
   - 防止错误记忆长期污染后续任务

5. 更强的语义检索
   - 后续可接入 embedding/vector store
   - 在 Hybrid Recall 中融合语义相似度

---

## 16. 一句话结论

当前 `mini-claude-code-cli` 的新长期记忆系统已经完成主线重构，具备可用闭环：

```text
自动保存经验，结构化落盘，混合检索召回，受控注入 Prompt，并在文件编辑和错误修复场景中自动利用历史记忆。
```
