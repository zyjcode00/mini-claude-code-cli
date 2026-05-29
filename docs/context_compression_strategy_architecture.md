# Mini Claude Code CLI 上下文压缩策略优化架构规划

> 项目路径：`D:\LLM\mini-claude-code-cli`
> 文档位置：`docs/context_compression_strategy_architecture.md`
> 关联文档：
> - `docs/architecture_summary.md`
> - `docs/memory_system_refactor_roadmap.md`
> - `docs/new_long_term_memory_system.md`
>
> 核心结论：下一步压缩策略不应继续停留在“按消息裁剪 + 出错后清理”的层面，而应升级为 **按 turn 原子化建模、分层压缩、结构化状态迁移、长期记忆联动、发送前协议校验** 的上下文管理系统。

---

## 1. 背景与当前问题

当前项目已经具备：

- `CompressionEngine` 多策略压缩：`LLM_SUMMARY` / `KEYFRAME` / `SLIDING_WINDOW` / `IMPORTANCE_FILTER`；
- `ContextManager` 负责消息管理、压缩触发、摘要写入；
- `MemoryManager` / `MemoryItem` / `MemoryContextBuilder` 负责长期记忆、召回和 Prompt 预算注入；
- OpenAI-compatible tool pair sanitizer，用于避免 `assistant.tool_calls` 缺失对应 `tool` 响应导致 400。

但是当前压缩系统仍有几个结构性缺陷：

1. **压缩单位仍偏 message 级别**
   很多策略按消息列表裁剪，容易破坏一次用户请求、assistant 工具调用、tool 返回、assistant 总结之间的完整语义链。

2. **tool call pair 只是在最后被清理，而不是从建模阶段就原子化**
   当前 sanitizer 能止血，但理想系统应该在压缩前就把 `assistant(tool_calls) + tool responses` 视为不可拆分块。

3. **压缩结果偏自然语言 summary，缺少结构化状态迁移**
   Coding agent 真正需要保留的是：当前目标、已改文件、失败错误、测试状态、关键决策、未完成事项，而不是完整聊天。

4. **长期记忆和上下文压缩边界还可以更清晰**
   压缩应负责“当前会话继续执行所需状态”；长期记忆应负责“跨会话可复用经验”。两者应协同，但不能互相替代。

5. **缺少压缩质量可观测性**
   目前很难知道一次压缩删除了什么、摘要保留了什么、是否丢失关键状态、是否发生 sanitizer 兜底。

---

## 2. 优化目标

下一阶段上下文压缩策略的目标不是简单降低 token，而是：

```text
在不破坏模型消息协议的前提下，把长对话迁移成可继续执行任务的 compact state。
```

具体目标：

1. **协议安全**
   - OpenAI-compatible：永远不发送孤立 `assistant.tool_calls` 或孤立 `tool` 消息；
   - Anthropic-compatible：保持 `tool_use` / `tool_result` 成对语义；
   - 记忆注入不得插入 assistant/tool pair 中间。

2. **任务连续性**
   - 保留当前用户目标；
   - 保留当前 plan；
   - 保留未完成步骤；
   - 保留最近失败 traceback；
   - 保留最近文件修改与测试结果。

3. **信息保真**
   - 大工具输出不长期保留全文，但必须摘要关键结果；
   - 失败尝试不能完全丢弃，应转成“已尝试但失败”的经验；
   - 文件全文应尽量不进 summary，需要时重新 `read_file`。

4. **分层预算控制**
   - System prompt；
   - 当前 plan；
   - 相关长期记忆；
   - 压缩历史摘要；
   - 最近完整 turns；
   - 当前用户请求。

5. **可测试、可观测、可回退**
   - 每个压缩阶段都有测试；
   - 每次压缩输出统计信息；
   - LLM summary 超时或失败时有确定性 fallback；
   - 发送前永远做最终 validator。

---

## 3. 目标架构总览

建议引入一个明确的压缩流水线：

```text
Raw messages
  ↓
MessageNormalizer
  ↓
TurnBuilder
  ↓
CompressionPlanner
  ↓
TurnCompressor
  ↓
StateExtractor
  ↓
MemoryPromoter
  ↓
ContextAssembler
  ↓
ProviderMessageValidator
  ↓
LLM API request
```

对应职责：

| 模块 | 职责 |
|---|---|
| `MessageNormalizer` | 清理空消息、兼容字段、reasoning_content、provider 差异 |
| `TurnBuilder` | 把 message list 切成用户轮次、assistant 轮次、tool 轮次，保证 tool pair 原子化 |
| `CompressionPlanner` | 根据 token 预算决定哪些 turn 原样保留、摘要、丢弃或晋升长期记忆 |
| `TurnCompressor` | 对旧 turn 做结构化摘要，避免保留大 stdout / 文件全文 |
| `StateExtractor` | 从旧上下文提取任务状态、文件变化、错误、测试、决策 |
| `MemoryPromoter` | 把跨会话有价值的信息保存为 `MemoryItem` |
| `ContextAssembler` | 按预算组装 system + memory + summary + recent turns |
| `ProviderMessageValidator` | 发送前做 OpenAI/Anthropic 协议校验和兜底修复 |

---

## 4. 核心设计：从 message 级压缩升级为 turn 级压缩

### 4.1 Turn 的定义

建议定义内部结构 `ConversationTurn`：

```python
@dataclass
class ConversationTurn:
    id: str
    start_index: int
    end_index: int
    user_message: dict | None
    assistant_messages: list[dict]
    tool_messages: list[dict]
    messages: list[dict]
    has_tool_calls: bool
    is_tool_pair_complete: bool
    token_estimate: int
    importance: float
    categories: list[str]  # code_edit / test / error / planning / search / file_read
    files_touched: list[str]
    errors: list[str]
    tests: list[str]
```

Turn 是压缩的最小语义单位。压缩策略只处理 turn，不直接乱切 messages。

### 4.2 Tool pair 原子化规则

OpenAI-compatible 场景下：

```text
assistant(tool_calls=[call_a, call_b])
tool(tool_call_id=call_a)
tool(tool_call_id=call_b)
```

必须视为一个原子块。

规则：

1. `assistant.tool_calls` 后必须紧邻对应 `tool` 响应块；
2. 如果缺任意一个 tool response，则该 assistant/tool group 不能原样发送；
3. 对不完整 group 的处理优先级：
   - 如果可以从历史中找到完整 pair：补齐；
   - 如果无法补齐：把整个坏 group 转为纯文本摘要；
   - 如果摘要也不可行：删除整个 group；
4. 禁止只删除 tool 而保留 assistant.tool_calls；
5. 禁止在 assistant.tool_calls 和 tool responses 中间插入 memory/user/system 消息。

### 4.3 Turn 压缩决策

每个 turn 可进入四种状态：

| 状态 | 含义 | 适用场景 |
|---|---|---|
| `KEEP_RAW` | 原样保留 | 最近交互、当前任务关键步骤、最近失败 traceback |
| `COMPRESS` | 压缩成结构化摘要 | 较旧但有价值的文件编辑、测试、错误修复过程 |
| `PROMOTE_MEMORY` | 写入长期记忆，只在当前上下文保留引用或短摘要 | 跨会话可复用经验、用户偏好、架构决策 |
| `DROP` | 丢弃 | 重复日志、无关 stdout、旧文件全文、成功安装日志 |

---

## 5. 推荐上下文布局

最终发送给 LLM 的上下文建议按以下顺序构建：

```text
1. System Prompt
2. Runtime Safety Instructions
3. Current Plan / Task State
4. Relevant Long-term Memory Recall
5. Compressed Session State
6. Recent Complete Turns
7. Current User Message
```

其中各层预算建议：

| 区域 | 默认预算建议 | 说明 |
|---|---:|---|
| System Prompt | 固定 | 核心行为规则，不参与普通压缩 |
| Current Plan | 500 - 1200 tokens | 必须保留 |
| Long-term Memory Recall | 800 - 1800 tokens | 由 `MemoryContextBuilder` 控制 |
| Compressed Session State | 1200 - 3000 tokens | 当前会话摘要、keyframe |
| Recent Raw Turns | 总预算的 40% - 60% | 最近完整 turns 原样保留 |
| Emergency Buffer | 10% - 15% | 给模型输出和工具参数预留 |

---

## 6. 压缩后的结构化摘要格式

建议不要只生成一段自然语言 summary，而是生成可解析的 `CompressedSessionState`。

```python
@dataclass
class CompressedSessionState:
    task_goal: str
    current_status: str
    completed_steps: list[str]
    pending_steps: list[str]
    files_changed: list[dict]
    commands_run: list[dict]
    tests_run: list[dict]
    errors_encountered: list[dict]
    key_decisions: list[str]
    user_preferences: list[str]
    tool_safety_notes: list[str]
    dropped_content_summary: str
    source_turn_ids: list[str]
    created_at: str
```

渲染到 prompt 时可以是：

```text
### 压缩后的会话状态
当前目标：...
当前状态：...
已完成：
- ...
未完成：
- ...
文件修改：
- core/engine.py: 增加发送前 OpenAI tool pair validator
测试：
- tests/test_openai_tool_pairing.py passed
关键错误：
- 400 assistant.tool_calls missing tool response，原因是压缩保留半截 tool pair
注意事项：
- 不要在 assistant.tool_calls 和 tool response 中间插入记忆内容
```

优势：

- 后续可以测试字段是否保留；
- 可以和 `SessionSummary` / `MemoryItem` 互相转换；
- 可以做 diff 和质量评估；
- 可以把高价值字段晋升为长期记忆。

---

## 7. 压缩策略分层

### 7.1 第一层：协议清理与规范化

位置：压缩前 + 发送前。

职责：

- 去除非法 role；
- 规范 OpenAI message 字段；
- 检查 tool pair 完整性；
- 清理孤立 tool；
- 清理孤立 assistant.tool_calls；
- 记录 sanitizer 统计。

注意：发送前 validator 永远保留，即使前面所有策略都正确，也不能删除最后防线。

### 7.2 第二层：最近完整 turn 保留

规则：

- 最近 N 个完整 turn 原样保留；
- N 不固定，应由 token budget 动态决定；
- 最近一次用户请求和之后的工具调用必须保留；
- 最近一次失败 traceback 优先保留。

### 7.3 第三层：旧 turn 结构化压缩

对较旧 turns：

- 文件读取：只保留路径、目的、关键发现，不保留全文；
- 文件修改：保留文件路径、修改意图、修改结果；
- 测试运行：保留命令、passed/failed、关键失败；
- 搜索结果：保留搜索目标和命中结论；
- 长 stdout：截断或摘要；
- 重复日志：合并。

### 7.4 第四层：关键帧 Keyframe

每次任务完成、测试通过、plan 完成、session 保存前，可生成 keyframe：

```text
Project State Keyframe:
- branch
- task
- plan status
- changed files
- tests status
- known caveats
- next recommended action
```

Keyframe 是断点恢复的核心，不应该依赖完整聊天。

### 7.5 第五层：长期记忆晋升

以下内容应该保存到长期记忆，而不是只放会话摘要：

| 内容 | Memory 类型 |
|---|---|
| 用户长期偏好 | `preference` |
| 架构决策 | `architecture` / `decision` |
| 可复用 bug 修复经验 | `bug` |
| 项目约定 | `workflow` / `procedural` |
| 文件历史经验 | `file_history` |
| 工具调用注意事项 | `tooling` |

---

## 8. CompressionPlanner 决策算法建议

伪代码：

```python
def plan_compression(turns, token_budget, provider):
    normalized = normalize_messages(turns)
    safe_turns = build_complete_turns(normalized, provider)

    scored = score_turn_importance(safe_turns)

    recent = select_recent_complete_turns(scored, min_recent=4)
    critical = select_critical_turns(scored, categories=["error", "test_failure", "plan", "current_edit"])
    promotable = select_promotable_memory(scored)
    compressible = select_old_useful_turns(scored)
    droppable = select_low_value_turns(scored)

    while estimated_tokens(recent + critical + compressed(compressible)) > token_budget:
        demote_lowest_value_turn()

    return CompressionPlan(
        keep_raw=recent + critical,
        compress=compressible,
        promote_memory=promotable,
        drop=droppable,
    )
```

重要性评分建议：

```text
importance =
  recency_score
+ user_intent_score
+ file_change_score
+ error_score
+ test_score
+ plan_score
+ memory_reusability_score
- bulk_output_penalty
- duplicate_penalty
```

---

## 9. ProviderMessageValidator 设计

建议将当前 OpenAI sanitizer 升级为显式 validator：

```python
class ProviderMessageValidator:
    def validate(self, messages, provider) -> ValidationResult:
        ...

    def repair(self, messages, provider) -> RepairResult:
        ...
```

`ValidationResult`：

```python
@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]
    orphan_tool_call_ids: list[str]
    orphan_tool_messages: list[str]
    repaired: bool = False
```

必须覆盖：

1. OpenAI assistant.tool_calls 必须紧邻 tool responses；
2. tool_call_id 必须完全匹配；
3. 不允许 tool message 没有前置 assistant.tool_calls；
4. memory 注入不能出现在 tool pair 中间；
5. 压缩 summary 不能携带 `tool_calls` 字段；
6. 修复后再次 validate，不能修完仍非法。

---

## 10. 和长期记忆系统的协作边界

### 10.1 压缩系统负责

- 当前会话状态压缩；
- turn 级别裁剪；
- 保留当前任务连续性；
- 生成 `CompressedSessionState`；
- provider message 安全校验。

### 10.2 长期记忆系统负责

- 保存跨会话可复用经验；
- 按当前任务召回相关记忆；
- 通过 `MemoryContextBuilder` 控制注入预算；
- 提供 file history / error history / workflow preference。

### 10.3 两者之间的接口

建议：

```python
class CompressionEngine:
    async def compress(
        self,
        messages: list[dict],
        memory_manager: MemoryManager | None,
        token_budget: int,
        provider: str,
    ) -> CompressionResult:
        ...
```

`CompressionResult` 应包含：

```python
@dataclass
class CompressionResult:
    success: bool
    messages: list[dict]
    compressed_state: CompressedSessionState | None
    promoted_memory_ids: list[str]
    validation_result: ValidationResult
    stats: CompressionStats
    fallback_used: bool
```

---

## 11. 可观测性与日志

每次压缩建议输出结构化日志：

```text
[Compression]
- provider: openai
- original_messages: 86
- original_tokens_est: 52000
- turns_built: 18
- keep_raw_turns: 5
- compressed_turns: 9
- promoted_memories: 3
- dropped_turns: 4
- final_messages: 23
- final_tokens_est: 14200
- sanitizer_repairs: 1
- fallback_used: false
```

同时可保存调试快照：

```text
memory/compression_logs/{session_id}/{timestamp}.json
```

调试快照建议只保存 metadata 和摘要，不默认保存完整敏感内容。

---

## 12. 测试计划

任何 `core/` 或 `tools/` 逻辑修改都必须配套测试。建议新增或扩展以下测试文件：

### 12.1 `tests/test_turn_builder.py`

覆盖：

- 普通 user/assistant turn 构建；
- assistant.tool_calls + tool responses 构建为同一 turn；
- 多 tool_calls 连续响应；
- 缺失 tool response 时标记 incomplete；
- user 插入 tool pair 中间时标记非法。

### 12.2 `tests/test_provider_message_validator.py`

覆盖：

- OpenAI 完整 tool pair validate 通过；
- 孤立 assistant.tool_calls 被 repair 或删除；
- 孤立 tool message 被 repair 或删除；
- memory 注入打断 tool pair 会被检测；
- repair 后再次 validate 必须通过。

### 12.3 `tests/test_compression_planner.py`

覆盖：

- 最近 turns 优先保留；
- 错误和测试失败优先保留；
- 大 stdout 优先压缩或丢弃；
- 文件修改 turn 进入摘要；
- 长期可复用经验进入 promote list。

### 12.4 `tests/test_compressed_session_state.py`

覆盖：

- 从消息中提取 task_goal；
- 提取 files_changed；
- 提取 tests_run；
- 提取 errors_encountered；
- 渲染后的 prompt 不包含 tool_calls 字段。

### 12.5 `tests/test_context_assembly_budget.py`

覆盖：

- memory budget 不挤占 recent turns；
- summary 超长会截断；
- recent turn 不能截断半个 tool pair；
- 当前用户请求永远保留。

### 12.6 保留现有回归测试

必须继续保留并加强：

- `tests/test_openai_tool_pairing.py`
- `tests/test_compression_engine.py`
- `tests/test_memory_context_builder.py` 如果已有则扩展，否则新增

---

## 13. 分阶段实施路线

### Phase 0：巩固现有安全兜底（已基本完成）

目标：确保不会再因为 tool pair 半截导致 OpenAI 400。

任务：

1. 保留发送前 sanitizer；
2. 压缩失败后也清理上下文；
3. 将压缩 LLM 超时配置化；
4. 增加 tool pair 回归测试。

验收：

```bash
pytest tests/test_openai_tool_pairing.py -v
pytest -q
```

### Phase 1：引入 TurnBuilder

目标：压缩策略从 message 级别迁移到 turn 级别。

任务：

1. 新增 `ConversationTurn` 数据结构；
2. 新增 `TurnBuilder`；
3. 对 OpenAI tool pair 做 turn 原子化；
4. 修改压缩策略先 build turns 再裁剪；
5. 补 `tests/test_turn_builder.py`。

验收：

- 不完整 tool pair 不会进入 `KEEP_RAW`；
- 最近 turn 保留时一定完整；
- 所有现有测试通过。

### Phase 2：引入 CompressionPlanner

目标：让压缩决策从固定策略变成预算驱动、重要性驱动。

任务：

1. 新增 `CompressionPlan`；
2. 实现 turn importance scoring；
3. 实现 `KEEP_RAW` / `COMPRESS` / `PROMOTE_MEMORY` / `DROP`；
4. 增加 token budget 配置；
5. 补 `tests/test_compression_planner.py`。

验收：

- 最近交互优先保留；
- 错误/测试/文件修改不会被无声丢弃；
- 大 stdout 被压缩或丢弃。

### Phase 3：结构化 CompressedSessionState

目标：压缩结果从普通文本摘要升级为结构化任务状态。

任务：

1. 新增 `CompressedSessionState`；
2. 从 turns 中提取目标、状态、文件、错误、测试、决策；
3. LLM summary 只作为增强，不作为唯一来源；
4. fallback 使用规则摘要生成；
5. 补 `tests/test_compressed_session_state.py`。

验收：

- LLM summary 超时仍能生成可用状态；
- 渲染结果能明确表达“已完成/未完成”；
- 不包含非法 tool_calls 字段。

### Phase 4：和 MemoryManager 联动

目标：把可复用经验从压缩摘要晋升为长期记忆。

任务：

1. CompressionEngine 接收可选 `MemoryManager`；
2. 从压缩结果中生成 MemoryItem 候选；
3. 对 bug / decision / preference / workflow 做自动晋升；
4. 记录 source_turn_ids；
5. 补 memory promotion 测试。

验收：

- 修复过的 bug 可通过 `memory_error_history` 搜到；
- 文件修改经验可通过 `memory_file_history` 搜到；
- 长期记忆有来源引用。

### Phase 5：ContextAssembler 统一上下文预算

目标：统一 system、plan、memory、summary、recent turns 的预算和顺序。

当前状态：**已具备长期记忆 Prompt 预算注入的 MVP，尚未形成完整的全量 ContextAssembler。**

已经落地：

1. 新增 `core/memory_context_builder.py` 中的 `MemoryContextBuilder`；
2. `MemoryManager.build_prompt_memory_context(...)` 统一执行 Hybrid Recall + budget formatting；
3. `AgentEngine._build_relevant_memory_context(...)` 在任务开始前通过 `MemoryManager` 注入相关长期记忆；
4. 工具执行前后通过文件历史 / 错误历史召回，且延后写入，避免插入 `assistant.tool_calls` 与 `tool` 响应之间；
5. `tests/test_memory_phase5.py` 已覆盖 budget、任务类型排序、统计工具、memory_recall 委托、文件/错误历史召回和 Plan branch 回归。

Phase 5 的完整目标边界：

| 层级 | 目标 | 当前状态 | 下一步 |
|---|---|---|---|
| System Prompt | 固定规则、CLAUDE.md、运行约束 | 由 `prompts.py` / engine 既有逻辑处理 | 统一纳入 ContextAssembler 输入 |
| Current Plan | 当前任务目标、步骤、完成状态 | 由系统提示和 PlanManager 注入 | 设置硬预算与不可丢弃规则 |
| Long-term Memory | 相关 MemoryItem / 历史摘要 | `MemoryContextBuilder` 已实现预算控制 | 和 summary/recent turns 统一竞争预算 |
| Compressed Session State | 当前会话结构化状态 | Phase 3 已实现 `CompressedSessionState` | 明确注入位置和最大长度 |
| Recent Complete Turns | 最近完整 turn，tool pair 原子化 | Phase 1/2 已建立 TurnBuilder 基础 | ContextAssembler 按 turn 装配，禁止半截截断 |
| Provider Validation | 发送前协议安全 | 已有 sanitizer / tool pair 测试 | Phase 5 最终输出必须过 validator |

建议补齐任务：

1. 新增 `core/context_assembler.py`，定义 `ContextAssemblyInput` / `ContextAssemblyResult`；
2. 明确默认预算，例如：
   - plan：不可丢弃，约 500-1200 tokens；
   - memory：默认 800-1500 tokens；
   - compressed state：默认 1200-2500 tokens；
   - recent turns：保留总预算的 40%-60%，且最后一条用户请求必须存在；
3. ContextAssembler 只在安全区域插入 memory/summary：
   - system prompt 附加段；或
   - 新一轮 user 前的独立 user context；
   - 禁止插入 assistant.tool_calls 与 tool response 中间；
4. 对旧消息先走 TurnBuilder，按完整 turn 裁剪；
5. 补 `tests/test_context_assembly_budget.py`，覆盖预算、顺序、最近 turn、当前请求和 provider validator。

验收：

- 当前用户请求永远存在；
- 当前 Plan 永远存在或明确标记为空；
- 最近完整 turns 不被半截截断；
- memory 注入有长度上限，不会挤掉 recent turns；
- `CompressedSessionState` 能以稳定结构注入；
- 注入内容只能进入 system/user 安全区域，不能插入 tool pair；
- provider validator 通过；
- 相关测试至少包括 `tests/test_memory_phase5.py` 与新增 `tests/test_context_assembly_budget.py`。

推荐执行顺序：

```text
Phase 5A：保持现有 MemoryContextBuilder，补文档和回归测试
Phase 5B：新增 ContextAssembler 数据结构，只做 deterministic assembly，不接 LLM
Phase 5C：AgentEngine / ContextManager 改用 ContextAssembler 组装发送前上下文
Phase 5D：接入 ProviderMessageValidator，形成发送前最终安全门
```

### Phase 6：可观测性与压缩质量评估

目标：让压缩行为可解释、可调试。

任务：

1. 增加 `CompressionStats`；
2. 输出压缩日志；
3. 可选保存 debug snapshot；
4. 增加 `compression_stats` 或 session debug 命令；
5. 增加压缩质量测试和快照测试。

验收：

- 用户能看到压缩了多少、丢了多少、是否 fallback；
- 出错时能快速定位是哪一层破坏消息结构；
- debug snapshot 不泄漏敏感内容。

---

## 14. 推荐优先级

如果只做最有价值的下一步，建议顺序是：

```text
P0: TurnBuilder + ProviderMessageValidator 显式化
P1: CompressionPlanner 预算驱动
P1: CompressedSessionState 结构化摘要
P2: MemoryManager 自动晋升长期记忆
P2: ContextAssembler 统一预算
P3: 可观测性 dashboard / compression logs
```

短期 MVP：

```text
1. 新增 TurnBuilder
2. 把 OpenAI tool pair 当作不可拆分 turn
3. CompressionEngine 改为基于 turns 保留最近窗口
4. 发送前 ProviderMessageValidator 再校验
5. 增加 tests/test_turn_builder.py + tests/test_provider_message_validator.py
```

这个 MVP 做完后，压缩系统的稳定性会明显高于单纯 sanitizer。

---

## 15. 成功标准

压缩策略优化完成后，应满足：

1. 任意压缩策略输出都能通过 provider validator；
2. OpenAI-compatible 不再出现 `assistant.tool_calls missing tool response`；
3. 最近任务状态可在压缩后继续执行，不需要用户重复说明；
4. 重要错误和修复经验能进入长期记忆并被召回；
5. 压缩失败时有 deterministic fallback；
6. 测试覆盖 turn 构建、validator、planner、state extraction、context budget；
7. 压缩日志能解释“保留了什么、压缩了什么、丢弃了什么、晋升了什么”。

---

## 16. 总结

下一步压缩策略的核心不是继续堆更多摘要 prompt，而是改变抽象层级：

```text
message list 裁剪
  ↓
conversation turn 原子化
  ↓
结构化任务状态迁移
  ↓
长期记忆晋升
  ↓
预算化上下文组装
  ↓
provider 协议校验
```

当前 sanitizer 是必要的安全地基；真正成熟的 code-agent 压缩系统，需要在它之上建立 turn-level compression、structured state 和 memory-aware context assembly。
