# Mini Claude Code CLI 记忆系统重构路线与架构设计

> 项目路径：`D:\LLM\mini-claude-code-cli`  
> 文档位置：`docs/memory_system_refactor_roadmap.md`  
> 参考项目：`D:\LLM\memory\agentmemory`  
> 结论：**建议重构，但不建议推倒重写。应保留现有三层记忆雏形，按 agentmemory 的“事件捕获 + 结构化记忆 + 混合检索 + 预算注入”思路渐进增强。**

---

## 1. 结论：现在的记忆系统需要重构吗？

需要，但不是因为现在完全没用，而是因为它已经有了雏形，却缺少形成长期价值的闭环。

当前 `mini-claude-code-cli` 已经具备：

- `WorkingMemory`：保存最近消息；
- `EpisodicMemory`：保存结构化 `SessionSummary`；
- `LongTermMemory`：基于 JSON 文件和倒排索引做长期记忆；
- `CompressionEngine`：多策略压缩；
- `get_system_prompt()`：把历史摘要、情景记忆、长期记忆注入 System Prompt；
- `sessions/*.json`：保存会话状态、Plan 状态、消息和记忆；
- `tests/test_memory_*.py`：已经有基础测试覆盖。

所以不应该重写成另一个项目，而应该进行“架构级重构”：

```text
保留现有三层记忆和测试
  -> 抽出统一 MemoryManager / MemoryStore / MemoryRetriever
  -> 增加 Observation 事件捕获
  -> 增加显式 Memory 类型
  -> 改进检索和注入策略
  -> 最后考虑接入 agentmemory 或 MCP
```

如果直接推倒重写，风险是：

1. 现有 Plan 状态同步、Session 恢复、压缩测试都会受影响；
2. 当前项目是 Python CLI Agent，agentmemory 是 TypeScript 独立服务，直接移植成本高；
3. 你的项目已有可运行链路，更适合渐进式演进。

---

## 2. 当前记忆系统现状

### 2.1 现有相关模块

```text
core/
├── context.py              # 上下文与记忆编排，目前承担过多职责
├── memory_layers.py        # Working / Episodic / LongTerm 三层记忆
├── memory_models.py        # SessionSummary、FileChange、ErrorRecord、ToolUsage
├── memory_retrieval.py     # 三层记忆检索器
├── compression_engine.py   # 压缩策略与摘要生成
├── prompts.py              # 记忆注入 System Prompt
├── engine.py               # LLM 调用、工具循环、session 保存恢复
├── keyword_indexer.py      # 关键词索引
└── bm25_retriever.py       # BM25 检索

tools/
└── retrieval_tool.py       # 独立的关键词/BM25 检索工具

memory/
└── long_term/              # 长期记忆 JSON 文件和 index.json

sessions/
└── *.json                  # 会话状态、消息、Plan、memories
```

### 2.2 当前数据流

当前大致数据流是：

```text
用户输入 / 工具结果
  -> AgentEngine 往 ContextManager.messages 添加消息
  -> ContextManager 同步写入 WorkingMemory
  -> 超过阈值时调用 CompressionEngine.compress()
  -> 生成 SessionSummary
  -> 写入 EpisodicMemory 和 session_summaries
  -> 如果 EpisodicMemory 满了，才淘汰旧摘要到 LongTermMemory
  -> save_session() 把 messages / plan / memories 存到 sessions/{session_id}.json
  -> 下一次 _call_llm() 时，从 LongTermMemory 检索，再拼入 System Prompt
```

---

## 3. 为什么现在会感觉“鸡肋”？

### 3.1 没有原始 Observation 层

agentmemory 的核心是先记录 `RawObservation`：

- 用户提交了什么 prompt；
- 调用了什么工具；
- 工具输入是什么；
- 工具输出是什么；
- 工具失败的 traceback 是什么；
- 哪些文件被读取或修改；
- 当前项目路径、session id、时间戳是什么。

而当前项目主要依赖 `messages` 和压缩后的 `SessionSummary`。问题是：

- 工具调用的原始上下文没有被单独结构化保存；
- 很多有价值的信息只散落在对话消息里；
- 压缩失败或摘要质量差时，长期记忆质量就会下降；
- 后续无法追溯一条记忆来自哪次工具调用。

### 3.2 长期记忆写入太被动

当前 `LongTermMemory` 主要在 `EpisodicMemory` 超过上限时才写入：

```text
EpisodicMemory max_size = 50
超过 50 条后，重要性最低的摘要才被 store 到 LongTermMemory
```

这会导致：

- 普通项目长期记忆很久都不会真正落入 `memory/long_term`；
- 跨会话检索可能经常检索不到有价值内容；
- 明明完成了重要任务，却没有立刻沉淀成长期知识。

### 3.3 只有 SessionSummary，没有显式 Memory 类型

当前核心长期单元是 `SessionSummary`，更像“这次会话发生了什么”。

但 coding agent 更需要区分：

| 类型 | 示例 |
|---|---|
| `fact` | 项目使用 OpenAI 兼容接口调用 DeepSeek |
| `architecture` | AgentEngine 负责异步 ReAct 循环，ContextManager 负责编排上下文 |
| `preference` | 用户希望文档使用中文，保存到 docs 目录 |
| `bug` | OpenAI SDK 回传 reasoning_content 必须原样保留，否则 DeepSeek 报错 |
| `workflow` | 修改 core/tools 逻辑后必须写测试并 run_pytest |
| `decision` | 不推倒重写记忆系统，采用渐进式重构 |
| `procedural` | edit_file 连续失败后切换 write_full_file |

如果只有 `SessionSummary`，Agent 很难精确回答：

- “这个项目的架构约定是什么？”
- “用户偏好是什么？”
- “之前这个 bug 怎么修的？”
- “修改这个文件前有什么历史坑？”

### 3.4 检索能力割裂

当前有三套检索相关代码：

1. `core.memory_retrieval.MemoryRetriever`；
2. `core.keyword_indexer.KeywordIndexer`；
3. `core.bm25_retriever.BM25Retriever`；
4. `tools.retrieval_tool.SearchMemoryTool` 又自己持有独立索引器。

问题是：

- `SearchMemoryTool` 并没有直接查询 `ContextManager` 的三层记忆；
- `AgentEngine` 初始化了 `KeywordIndexer` 和 `BM25Retriever`，但和 `LongTermMemory` 不是同一套索引；
- README 中说“BM25 + 语义向量混合检索”，但实际 `memory_retrieval.py` 主要是简化版 BM25 和关键词匹配；
- 检索结果缺少来源、引用、时间、文件、任务状态等丰富上下文。

### 3.5 注入策略不够稳定

当前 `get_system_prompt()` 中会注入：

- `history_summary`；
- `long_term_memories`；
- `episodic_memories`；
- `session_summaries`；
- completed_goals；
- plan；
- CLAUDE.md 相关段落。

已有去重和条件注入，这是优点。

但仍然存在：

- 没有统一 token budget；
- 没有按任务类型决定注入哪类记忆；
- `episodic_memory.get_all()` 先取全量，再由 prompt 里过滤；
- 记忆内容没有标准格式，容易噪声过多；
- 长期记忆召回质量依赖简单关键词。

### 3.6 缺少隐私过滤

agentmemory 在写入前有 privacy filter，会过滤：

- API key；
- secret；
- token；
- private 标签；
- 其他敏感片段。

当前项目会把会话和工具输出保存到 `sessions/*.json`，如果工具输出里包含敏感内容，就可能被写入本地会话文件。

### 3.7 一些实现细节有潜在问题

当前代码中有几个值得后续修复的点：

1. `LongTermMemory._extract_keywords()` 从文件变更里读取 `file_path`：

```python
file_path = fc.get("file_path", "")
```

但 `FileChange.to_dict()` 输出字段是：

```python
"path": self.path
```

所以长期记忆索引可能没有正确索引文件路径。

2. `README.md` 里写 `memory_models.py` 是 Pydantic，但实际是 `dataclass`。

3. `tools/retrieval_tool.py` 中的 `SearchMemoryTool` 并不是直接检索现有三层记忆，而是独立索引器，容易让 Agent 调用后搜不到真正的会话记忆。

4. `SessionSummary.task_status` 测试里有时使用 `done`，主逻辑里使用 `completed/in_progress/failed/cancelled`，状态值不统一。

---

## 4. 参考 agentmemory 后，适合本项目的目标架构

建议目标架构如下：

```text
┌─────────────────────────────────────────────────────────────┐
│                      AgentEngine                            │
│  - 用户输入                                                  │
│  - LLM 调用                                                  │
│  - 工具执行                                                  │
│  - Plan 状态                                                 │
└───────────────┬─────────────────────────────────────────────┘
                │ 事件回调 / hook
                ▼
┌─────────────────────────────────────────────────────────────┐
│                    MemoryManager                            │
│  统一入口：observe / remember / recall / build_context       │
│                                                             │
│  - observe_prompt()                                         │
│  - observe_tool_use()                                       │
│  - observe_tool_failure()                                   │
│  - compress_session()                                       │
│  - promote_to_long_term()                                   │
│  - save_memory()                                            │
│  - search()                                                 │
│  - build_memory_context()                                   │
└───────────────┬─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│                    MemoryStore                              │
│                                                             │
│  Working Store       当前对话消息 / 最近观察                 │
│  Observation Store   原始事件：prompt、tool、error           │
│  Episodic Store      会话摘要 / 任务片段                     │
│  Semantic Store      架构事实 / 用户偏好 / 项目约定          │
│  Procedural Store    工作流 / 修复步骤 / 工具使用经验        │
│  Index Store         BM25 / Keyword / optional Vector        │
└───────────────┬─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│                    MemoryRetriever                          │
│  - BM25 keyword search                                      │
│  - file history search                                      │
│  - error history search                                     │
│  - memory type filtering                                    │
│  - recency / importance / source fusion                     │
│  - optional vector retrieval                                │
└───────────────┬─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│                  Prompt Memory Injector                      │
│  - token budget                                              │
│  - 去重                                                      │
│  - 按任务类型注入                                            │
│  - 引用来源                                                  │
│  - 控制噪声                                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 推荐新增 / 调整的数据模型

### 5.1 RawObservation

用于保存原始事件，不直接替代 `SessionSummary`。

```python
@dataclass
class RawObservation:
    id: str
    session_id: str
    project: str
    cwd: str
    timestamp: str
    event_type: str  # prompt_submit / pre_tool_use / post_tool_use / tool_failure / session_start / session_end
    tool_name: str | None = None
    tool_input: dict | None = None
    tool_output: str | None = None
    user_prompt: str | None = None
    error: str | None = None
    files: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
```

### 5.2 MemoryItem

用于保存真正的长期知识。

```python
@dataclass
class MemoryItem:
    id: str
    type: str  # fact / architecture / preference / bug / workflow / decision / procedural
    title: str
    content: str
    project: str
    created_at: str
    updated_at: str
    concepts: list[str]
    files: list[str]
    source_observation_ids: list[str]
    source_session_ids: list[str]
    importance: float = 0.5
    confidence: float = 0.8
    status: str = "active"  # active / superseded / archived
    version: int = 1
```

### 5.3 SessionSummary 保留但升级

当前 `SessionSummary` 可以保留，但建议增加：

```python
project: str
source_observation_ids: list[str]
concepts: list[str]
memory_types: list[str]
created_by: str  # synthetic / llm / manual
```

---

## 6. 分阶段工作路线

### 阶段 0：先修正现有小问题，不改架构

目标：让当前记忆系统更可靠，避免“明明写了记忆却搜不到”。

#### 任务 0.1：修复 LongTermMemory 文件路径索引字段

**做什么：**

把 `LongTermMemory._extract_keywords()` 中：

```python
file_path = fc.get("file_path", "")
```

改为兼容：

```python
file_path = fc.get("path") or fc.get("file_path") or ""
```

**怎么做：**

1. 修改 `core/memory_layers.py`；
2. 新增测试 `tests/test_long_term_memory_file_path_index.py`；
3. 构造一个包含 `FileChange(path="core/context.py")` 的 summary；
4. store 后搜索 `context.py` 或 `core`，验证能召回。

#### 任务 0.2：统一 task_status 枚举

**做什么：**

统一使用：

```text
completed / in_progress / failed / cancelled
```

兼容旧数据中的 `done`：读入时映射为 `completed`。

**怎么做：**

1. 在 `SessionSummary.from_dict()` 中增加状态归一化；
2. 测试旧数据 `task_status="done"` 能恢复为 `completed`；
3. 更新测试中不一致的写法。

#### 任务 0.3：让 SearchMemoryTool 真正检索当前记忆

**做什么：**

当前 `SearchMemoryTool` 是独立 BM25/关键词索引器，和 `ContextManager` 没有直接关系。建议先不暴露它，或者重写为调用 `MemoryRetriever`。

**怎么做：**

1. 设计 `SearchMemoryTool(context_manager)`；
2. `run(query)` 调用 `MemoryRetriever.retrieve()`；
3. 返回来源层：working / episodic / long_term；
4. 新增测试，验证插入情景记忆后工具能搜到。

---

### 阶段 1：抽出 MemoryManager，集中管理记忆入口

目标：把分散在 `ContextManager`、`AgentEngine`、`prompts.py`、`retrieval_tool.py` 的记忆逻辑集中到一个服务类中。

#### 任务 1.1：新增 `core/memory_manager.py`

**做什么：**

新增统一门面类：

```python
class MemoryManager:
    def __init__(self, session_id, project, cwd, storage_dir="memory"):
        ...

    def observe_prompt(self, prompt: str):
        ...

    def observe_tool_use(self, tool_name: str, tool_input: dict, tool_output: str):
        ...

    def observe_tool_failure(self, tool_name: str, tool_input: dict, error: str):
        ...

    async def compress_if_needed(self, llm_summarizer_func):
        ...

    def remember(self, type: str, title: str, content: str, files=None, concepts=None):
        ...

    def recall(self, query: str, top_k=5):
        ...

    def build_context(self, query: str, token_budget=1500):
        ...
```

**怎么做：**

1. 第一版内部可以复用现有 `ContextManager`；
2. 不急着迁移全部逻辑；
3. 先把新接口打通，并写测试；
4. `AgentEngine` 后续只和 `MemoryManager` 交互。

#### 任务 1.2：AgentEngine 接入 MemoryManager

**做什么：**

在 `AgentEngine.__init__` 中初始化：

```python
self.memory = MemoryManager(
    session_id=session_id,
    project="mini-claude-code-cli",
    cwd=os.getcwd(),
    context=self.context,
)
```

**怎么做：**

1. 不马上删除 `self.context`；
2. 先让 `self.memory` 包装它；
3. 在用户输入处调用 `observe_prompt()`；
4. 在工具执行成功处调用 `observe_tool_use()`；
5. 在工具失败处调用 `observe_tool_failure()`。

---

### 阶段 2：增加 Observation 事件捕获

目标：参考 agentmemory 的 hooks 思路，为自己的 CLI Agent 增加“自动观察”能力。

#### 任务 2.1：新增 `core/memory_observation.py`

**做什么：**

定义：

```python
RawObservation
ObservationType
ObservationStore
```

保存目录建议：

```text
memory/
├── observations/
│   ├── {session_id}.jsonl
│   └── index.json
├── long_term/
└── semantic/
```

**怎么做：**

1. 每个 observation 一行 JSONL；
2. 避免每次重写大 JSON；
3. 写入前做输出截断；
4. 写入前做隐私过滤。

#### 任务 2.2：捕获用户 prompt

**做什么：**

每次用户输入后记录：

```json
{
  "event_type": "prompt_submit",
  "user_prompt": "...",
  "session_id": "...",
  "cwd": "...",
  "timestamp": "..."
}
```

**怎么做：**

在 `main.py` 或 `AgentEngine.execute_query()` 用户输入进入前后调用。

#### 任务 2.3：捕获工具调用成功

**做什么：**

每次工具执行成功后记录：

```json
{
  "event_type": "post_tool_use",
  "tool_name": "read_file",
  "tool_input": {"path": "core/engine.py"},
  "tool_output": "截断后的输出",
  "files": ["core/engine.py"]
}
```

**怎么做：**

1. 在工具执行统一分发处加 hook；
2. 输出最多保留 4000～8000 字符；
3. 文件工具、bash 工具、pytest 工具分别抽取不同 metadata；
4. 失败时走 `observe_tool_failure()`。

#### 任务 2.4：捕获工具失败和测试失败

**做什么：**

对失败单独记录，方便后续检索“这个错误以前怎么修”。

```json
{
  "event_type": "tool_failure",
  "tool_name": "run_pytest",
  "error": "Traceback...",
  "files": ["tests/test_xxx.py", "core/xxx.py"]
}
```

**怎么做：**

在工具异常捕获和 `run_pytest` 返回失败时都写 observation。

---

### 阶段 3：增加显式长期记忆 MemoryItem

目标：从“会话摘要记忆”升级到“知识型长期记忆”。

#### 任务 3.1：新增 `core/memory_items.py`

**做什么：**

定义 `MemoryItem` 和 `MemoryItemStore`。

存储目录：

```text
memory/
├── semantic/
│   ├── memories.jsonl
│   └── index.json
├── procedural/
│   ├── workflows.jsonl
│   └── index.json
```

#### 任务 3.2：新增 remember 接口

**做什么：**

让 Agent 可以显式保存：

```python
memory.remember(
    type="bug",
    title="DeepSeek reasoning_content 必须回传",
    content="OpenAI 兼容接口返回 reasoning_content 时，下一轮 messages 必须原样保留，否则会报 400。",
    files=["core/engine.py"],
    concepts=["deepseek", "reasoning_content", "openai-compatible"]
)
```

**怎么做：**

1. 先做本地 Python 函数；
2. 再做工具 `save_memory`；
3. 再在系统提示词中要求 Agent 在重要任务完成后主动保存。

#### 任务 3.3：自动从 SessionSummary 提炼 MemoryItem

**做什么：**

任务完成时，从摘要中提取：

- 架构决策；
- bug 修复经验；
- 用户偏好；
- 文件历史；
- workflow。

第一版不一定要 LLM，可以用规则：

```text
如果 summary.errors_encountered 非空 -> 生成 bug memory 候选
如果 summary.key_decisions 非空 -> 生成 decision memory 候选
如果 files_changed 包含 CLAUDE.md / README.md / docs -> 生成 documentation/project convention 候选
如果用户 prompt 包含“以后都...” -> 生成 preference memory 候选
```

后续再接 LLM 提炼。

---

### 阶段 4：统一检索层，做 Hybrid Recall

目标：把现在分散的检索器合并成一个稳定的 `MemoryRetriever`。

#### 任务 4.1：统一索引入口

**做什么：**

所有可检索内容都转成统一文档：

```python
@dataclass
class MemoryDocument:
    id: str
    source_type: str  # observation / session_summary / memory_item
    content: str
    project: str
    session_id: str | None
    files: list[str]
    concepts: list[str]
    timestamp: str
    importance: float
```

#### 任务 4.2：BM25 + 关键词 + 元数据加权

**做什么：**

先不要急着做向量，先做好：

```text
score = bm25_score
      + recency_weight
      + importance_weight
      + file_match_weight
      + type_weight
```

#### 任务 4.3：增加 file_history / error_history

**做什么：**

参考 agentmemory 的 `memory_file_history`：

- 修改某文件前，先查这个文件历史；
- 测试失败时，查相似错误历史；
- 运行 git 操作前，查用户偏好和历史风险。

工具建议：

```text
memory_recall(query, top_k=5)
memory_save(type, title, content, files, concepts)
memory_file_history(path, top_k=5)
memory_error_history(error, top_k=5)
memory_stats()
```

---

### 阶段 5：Prompt 注入重构，引入 token budget

目标：不再简单拼接记忆，而是按预算构造可控上下文。

#### 任务 5.1：新增 MemoryContextBuilder

**做什么：**

```python
class MemoryContextBuilder:
    def build(self, query, memories, token_budget=1500):
        ...
```

输出结构：

```text
### 相关长期记忆
1. [bug] DeepSeek reasoning_content 必须回传
   来源: core/engine.py, session_xxx
   内容: ...

### 相关文件历史
- core/memory_layers.py: 曾修复 path/file_path 索引字段不一致

### 用户偏好
- 文档优先中文，保存到 docs 目录
```

#### 任务 5.2：按任务类型选择记忆

**做什么：**

| 当前任务 | 优先注入 |
|---|---|
| 代码修改 | 文件历史、bug、workflow |
| 架构设计 | architecture、decision、semantic |
| 测试失败 | error_history、bug、procedural |
| 文档写作 | preference、architecture、recent docs |
| git 操作 | workflow、用户安全偏好、历史事故 |

#### 任务 5.3：控制噪声

**做什么：**

1. 最多注入 5～8 条；
2. 每条 80～200 字；
3. 保留来源；
4. 去重相似内容；
5. 不注入低置信度过期记忆。

---

### 阶段 6：可选接入 agentmemory

目标：决定是否把 agentmemory 作为外部长期记忆服务。

#### 方案 A：内置增强版记忆系统

适合当前项目短期路线。

优点：

- 纯 Python，依赖少；
- 和当前 AgentEngine 深度融合；
- 测试和调试简单；
- 不需要启动额外服务。

缺点：

- 没有现成 viewer；
- 没有成熟 MCP；
- 向量/图谱/隐私过滤都要自己做。

#### 方案 B：接 agentmemory REST API

适合中期增强。

怎么做：

1. 新增 `core/agentmemory_client.py`；
2. 在 session start / prompt submit / post tool use / failure 时调用 REST API；
3. 在 `_call_llm()` 前调用 smart-search 或 context；
4. 把返回结果并入 `MemoryContextBuilder`。

优点：

- 直接获得成熟的 observation pipeline；
- 支持 BM25 + vector + graph；
- 有 viewer；
- 后续可 MCP 化。

缺点：

- 需要 Node 服务常驻；
- 本项目部署复杂度上升；
- 要处理服务不可用降级。

#### 方案 C：接 agentmemory MCP

适合后期工具生态化。

前提：

- mini-claude-code 支持 MCP client；
- 或者先写一个 MCP adapter，把 MCP 工具转成本项目 BaseTool。

推荐顺序：

```text
先内置重构
  -> 再 REST API 对接 agentmemory
    -> 最后 MCP 工具化
```

---

## 7. 目标目录结构建议

重构后建议结构：

```text
core/
├── memory/
│   ├── __init__.py
│   ├── manager.py              # MemoryManager 统一入口
│   ├── models.py               # RawObservation / MemoryItem / MemoryDocument
│   ├── store.py                # JSONL / index / persistence
│   ├── retriever.py            # hybrid recall
│   ├── context_builder.py      # token budget 注入
│   ├── privacy.py              # secret 过滤
│   ├── promotion.py            # observation -> summary -> memory item
│   └── agentmemory_client.py   # 可选外部 agentmemory REST 客户端
│
├── context.py                  # 保留对话上下文，但逐步移出记忆职责
├── engine.py
└── prompts.py

tools/
├── memory_tool.py              # memory_recall / memory_save / file_history / error_history
└── retrieval_tool.py           # 可逐步废弃或改造成通用搜索

memory/
├── observations/
├── episodic/
├── semantic/
├── procedural/
├── long_term/
└── indexes/
```

短期为了避免大迁移，可以先不新建 `core/memory/` 包，只新增：

```text
core/memory_manager.py
core/memory_observation.py
core/memory_items.py
core/memory_context_builder.py
tools/memory_tool.py
```

等稳定后再归并到包目录。

---

## 8. 每一步具体实施顺序

### 第 1 批：一两天内可以完成的小修

1. 修复 `LongTermMemory._extract_keywords()` 的 `path/file_path` 字段兼容；
2. 统一 `task_status`；
3. 让 `SearchMemoryTool` 能使用真实三层记忆；
4. 为以上每项补测试；
5. 跑：

```bash
pytest tests/test_memory_layers.py tests/test_memory_retrieval.py tests/test_memory_persistence.py -v
```

### 第 2 批：建立 MemoryManager

1. 新增 `core/memory_manager.py`；
2. 包装 `ContextManager`；
3. 提供 `observe_prompt / observe_tool_use / recall / build_context`；
4. 写 `tests/test_memory_manager.py`；
5. `AgentEngine` 先初始化但不强依赖，灰度接入。

### 第 3 批：Observation Store

1. 新增 `RawObservation`；
2. JSONL 写入；
3. 隐私过滤；
4. 输出截断；
5. 在 AgentEngine 工具调用处接入；
6. 写测试验证 prompt/tool/failure observation 都能保存。

### 第 4 批：MemoryItem 长期知识

1. 新增 `MemoryItem`；
2. 新增 `memory_save` 工具；
3. Agent 完成任务后主动保存关键知识；
4. 从 `SessionSummary` 自动提炼候选；
5. 检索时优先召回 MemoryItem。

### 第 5 批：Prompt 注入改造

1. 新增 `MemoryContextBuilder`；
2. 设置默认 `token_budget=1500`；
3. 按任务类型过滤；
4. 保留来源引用；
5. 替换 `prompts.py` 中零散拼接记忆的逻辑。

### 第 6 批：外部 agentmemory 对接实验

1. 新增 `AgentMemoryClient`；
2. 启动 agentmemory；
3. 做 health check；
4. mirror observation 到 agentmemory；
5. recall 时同时查本地和 agentmemory；
6. 失败时自动降级到本地。

---

## 9. MVP 版本建议

如果你想尽快让记忆“明显有用”，最小 MVP 是：

```text
1. 修复长期记忆索引 bug
2. 增加 memory_save 工具
3. 任务完成后保存 MemoryItem
4. 新任务开始前 recall MemoryItem
5. 修改文件前自动查 file_history
```

MVP 后，你应该能做到：

- 用户问“之前怎么处理 DeepSeek 报错的？”可以召回；
- 修改 `core/engine.py` 前能看到这个文件的历史坑；
- 完成一次重构后能保存“架构决策”；
- 用户偏好不再只靠 session 摘要碰运气；
- Agent 不会把所有历史摘要都塞进 prompt，而是按任务相关性注入。

---

## 10. 和 agentmemory 的能力对照

| agentmemory 能力 | 当前项目现状 | 建议 |
|---|---|---|
| RawObservation | 缺失 | 阶段 2 实现 |
| Privacy filter | 缺失 | 阶段 2 实现 |
| Session summary | 已有 | 保留并升级 |
| Semantic memory | 缺失 | 阶段 3 实现 MemoryItem |
| Procedural memory | 缺失 | 阶段 3 实现 workflow/procedural |
| BM25 | 有多个实现但割裂 | 阶段 4 统一 |
| Vector search | 暂无 | 后期可选 |
| Graph search | 暂无 | 暂不建议优先做 |
| MCP tools | 暂无 MCP client | 后期可选 |
| Viewer | 暂无 | 不优先，可先 CLI stats |
| Context budget | 不完善 | 阶段 5 实现 |
| Citation provenance | 缺失 | MemoryItem 加 source ids |
| Auto-forgetting | 简单重要性淘汰 | 后期增强 |

---

## 11. 推荐优先级

| 优先级 | 功能 | 理由 |
|---|---|---|
| P0 | 修复长期记忆索引字段 | 低成本，高收益 |
| P0 | 统一 task_status | 避免状态混乱 |
| P0 | memory_save / memory_recall 工具 | 让 Agent 主动使用记忆 |
| P0 | Observation Store | 让记忆有原始来源 |
| P1 | MemoryManager | 降低耦合，便于后续演进 |
| P1 | MemoryItem | 从“摘要”升级为“知识” |
| P1 | file_history / error_history | 对代码 Agent 最实用 |
| P1 | token budget 注入 | 控制上下文噪声 |
| P2 | agentmemory REST 对接 | 获取外部成熟能力 |
| P2 | vector search | 提升语义召回 |
| P3 | graph / viewer / MCP | 后期增强，不急 |

---

## 12. 最终建议

你的项目当前记忆系统的方向是对的，但实现还停留在“压缩会话摘要 + 简单检索”阶段，因此使用体验会显得鸡肋。

最应该参考 agentmemory 的不是代码语言或服务形态，而是这几个设计原则：

1. **先捕获原始事件，再压缩总结**；
2. **长期记忆不应该只等情景记忆溢出才生成**；
3. **区分事实、架构、偏好、bug、workflow 等记忆类型**；
4. **检索必须统一入口，不能多个索引各搜各的**；
5. **注入上下文要有 token budget 和任务相关性**；
6. **每条重要记忆都应该能追溯来源**；
7. **隐私过滤必须在写入前完成**。

推荐执行路线：

```text
先修 bug 和统一状态
  -> 抽 MemoryManager
    -> 加 Observation Store
      -> 加 MemoryItem / memory_save
        -> 统一检索和注入
          -> 再考虑 agentmemory REST / MCP
```

这样改完后，`mini-claude-code-cli` 的记忆系统就会从“会话摘要展示”升级成真正可用的“长期工程经验系统”。
