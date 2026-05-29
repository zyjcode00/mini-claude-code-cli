# Mini Claude Code CLI 长期记忆系统不足分析与下一代改进方案

> 项目路径：`D:\LLM\mini-claude-code-cli`  
> 文档位置：`docs/memory_system_next_architecture.md`  
> 参考项目：`D:\LLM\memory\agentmemory`  
> 目标：明确当前长期记忆系统的真实不足，判断 agentmemory 的 `BM25 + Vector + Graph + RRF + optional rerank` 是否适合复现，并给出后续可对照实施的系统架构与分阶段改进路线。

---

## 1. 总结结论

当前 `mini-claude-code-cli` 的记忆系统不是“架构完全不行”，而是已经完成了一个可用的长期记忆 MVP，但还没有达到 agentmemory 那种成熟的工程记忆服务水平。

更准确地说：

```text
当前问题不是没有长期记忆，
而是：
1. 记忆写入质量和生命周期治理还不够；
2. 检索召回仍偏启发式，不是真正多路召回融合；
3. Observation 没有独立持久化，来源追溯链不完整；
4. MemoryItem 的去重、版本化、过期、合并、纠错机制不足；
5. 向量检索、图谱检索、RRF、rerank 尚未实现；
6. Prompt 注入已有预算控制，但缺少召回质量评测闭环。
```

因此建议：

```text
不要推倒重写；
不要一开始就全量复刻 agentmemory；
应在现有 MemoryManager / MemoryItem / Hybrid Recall / ContextAssembler 基础上渐进增强。
```

最推荐路线是：

```text
先把 BM25 做真、把索引持久化做好、把记忆质量治理做好
  → 再加本地轻量 VectorIndex
    → 再用 RRF 融合 BM25 + Vector + metadata
      → 最后视规模和收益再做 Graph / rerank / agentmemory 对接。
```

---

## 2. 当前项目已经具备什么

经过现有代码和文档梳理，当前项目已经具备以下能力。

### 2.1 三层记忆架构

相关文件：

```text
core/memory_layers.py
core/memory_manager.py
core/context.py
```

已有三层：

```text
WorkingMemory
  - 保存最近消息
  - FIFO 淘汰

EpisodicMemory
  - 保存 SessionSummary
  - 有 max_size 和重要性淘汰

LongTermMemory
  - 基于 memory/long_term/*.json 持久化
  - index.json 保存索引
  - 支持 SessionSummary 和 MemoryItem
```

这说明项目已经不是简单聊天历史，而是有分层记忆雏形。

---

### 2.2 MemoryManager 统一入口

相关文件：

```text
core/memory_manager.py
```

当前 `MemoryManager` 已经统一持有：

```text
WorkingMemory
EpisodicMemory
LongTermMemory
MemoryRetriever
CompressionEngine
MemoryContextBuilder
```

并提供：

```text
add_message
reset_working_memory
save_summary
save_memory_item
save_observation
recall
hybrid_recall
retrieve_file_history
retrieve_error_history
build_prompt_memory_context
get_statistics
export/import
```

这是很重要的地基。后续升级不应该绕开它，而应继续让它作为统一门面。

---

### 2.3 MemoryItem 模型已经存在

相关文件：

```text
core/memory_items.py
```

当前 `MemoryItem` 已有字段：

```text
id
kind
title
content
project
created_at / updated_at
concepts
files
source_observation_ids
source_session_ids
importance
confidence
status
version
metadata
```

支持类型：

```text
fact
architecture
preference
bug
workflow
decision
procedural
task
summary
other
```

这已经接近 agentmemory 的 Memory 设计，只是缺少更完整的生命周期关系字段，例如：

```text
parent_id
supersedes
related_ids
is_latest
forget_after
last_accessed_at
access_count
quality_score
```

---

### 2.4 RawObservation 模型已经存在，但没有独立 Observation Store

当前已有：

```text
RawObservation
ObservationType
```

并且 `AgentEngine` 中会记录：

```text
prompt submit
tool result
tool failure
task completion
```

但关键问题是：

```text
RawObservation 目前主要作为 MemoryItem.metadata["raw_observation"] 被嵌入保存，
没有独立落盘为 observation 事件流。
```

这导致：

```text
1. 不能重新从原始事件重建记忆；
2. 不能对 observation 单独建索引；
3. 不能做“原始事件 → 压缩观察 → 长期知识”的完整 pipeline；
4. 来源追溯依赖 MemoryItem 内嵌元数据，不够规范。
```

---

### 2.5 长期记忆工具已经接入

相关文件：

```text
tools/memory_tool.py
tools/__init__.py
```

已有工具：

```text
memory_save
memory_recall
memory_file_history
memory_error_history
memory_stats
```

并且默认工具集中已经注册，工具间共享同一个 `MemoryManager`，避免重复加载长期记忆索引。

这是可用闭环的一部分。

---

### 2.6 Prompt 预算注入已经具备 MVP

相关文件：

```text
core/memory_context_builder.py
core/context_assembler.py
core/engine.py
```

当前已有：

```text
MemoryContextBuilder
  - token budget
  - max_items
  - per-item 截断
  - 低置信度过滤
  - 近似去重
  - 按任务类型优先不同 kind

ContextAssembler
  - system / memory / compressed_state / recent_turns 分区预算
  - 避免把 memory 注入 assistant.tool_calls 和 tool response 中间
```

这是一个很好的设计，后续应保留。

---

### 2.7 当前检索已经叫 Hybrid Recall，但不是 agentmemory 意义上的 Hybrid Search

当前 `MemoryRetriever.hybrid_recall()` 逻辑大致是：

```text
1. 读取所有 MemoryItem；
2. 读取 EpisodicMemory 中的 SessionSummary；
3. 读取 LongTermMemory 中的 SessionSummary；
4. 转成 MemoryDocument；
5. 对 query 做 tokenize；
6. 对每个 doc 计算：
   score = 简化文本匹配分
         + importance_weight
         + recency_weight
         + file_match
         + error_match
         + type_weight
7. 排序返回 top_k。
```

这属于：

```text
关键词/伪 BM25 + 元数据加权召回
```

不是 agentmemory 的：

```text
BM25 index search + VectorIndex search + GraphRetrieval search + RRF fusion + optional rerank
```

---

## 3. 当前主要不足在哪里

### 3.1 检索层最大问题：没有真正的多路召回

现在的 `hybrid_recall` 是单流程评分，不是多路检索器融合。

当前：

```text
MemoryDocument 全量构建
  → 每条 doc 计算一个综合分
  → 排序
```

agentmemory：

```text
BM25 search topN
Vector search topN
Graph search topN
  → RRF rank fusion
  → session diversity
  → enrich full observation
  → optional rerank
```

差别非常大。

当前项目的问题是：

```text
1. 每次召回会调用 get_all_items() 读取所有 MemoryItem 文件；
2. 文本相关分不是完整 BM25，没有 IDF，没有文档长度归一化；
3. 没有语义向量召回；
4. 没有图谱/关系扩展；
5. 没有 RRF，所有分数硬加，尺度不可控；
6. 没有 reranker；
7. 没有召回多样性控制；
8. 没有 query expansion / synonyms；
9. 中文分词仍是简化 2/3 字片段。
```

所以，如果用户感觉“长期记忆鸡肋”，主要原因大概率是：

```text
召回检索质量和记忆质量治理不足，
不是 MemoryManager 这个总体架构方向错了。
```

---

### 3.2 当前 BM25 不够真

项目里有：

```text
core/bm25_retriever.py
core/keyword_indexer.py
```

但长期记忆召回主线没有真正使用 `BM25Retriever` 的完整倒排索引，而是在 `memory_retrieval.py` 中用：

```python
score += (tf * 2.5) / (tf + 1.5)
```

这个评分缺少：

```text
IDF
平均文档长度
文档长度归一化
全局倒排索引
字段权重
持久化 BM25 快照
```

因此它更像“词频相关分”，不是标准 BM25。

---

### 3.3 长期记忆存储仍是文件级 JSON，索引能力有限

当前存储：

```text
memory/long_term/index.json
memory/long_term/memory_item_*.json
memory/long_term/summary_*.json
```

优点：

```text
简单
可读
易调试
不依赖外部数据库
```

缺点：

```text
1. 召回时读取 item 文件多，规模上来后变慢；
2. index.json 只存 keyword -> ids，不存 BM25 term freq/doc length；
3. 没有 vector 文件；
4. 没有 graph nodes/edges；
5. 没有 access log；
6. 没有 memory quality metadata；
7. Windows 下 index.json 原子替换已做保护，但频繁写仍可能有压力。
```

---

### 3.4 MemoryItem 生命周期不够完整

当前有：

```text
status: active / superseded / archived
version
```

但缺少主动使用这些字段的机制：

```text
1. 保存新记忆时不会自动判断是否 supersede 旧记忆；
2. 没有 parent_id / supersedes / related_ids；
3. 没有 is_latest 标记；
4. 没有 TTL / forget_after；
5. 没有 last_accessed_at / access_count；
6. 没有质量分数 quality_score；
7. 没有错误记忆纠正机制；
8. 没有合并相似记忆的 maintenance job。
```

这会导致长期运行后出现：

```text
重复记忆越来越多；
旧决策和新决策冲突；
低质量工具日志污染召回；
同一个 bug 被保存很多次；
过期项目状态仍被注入 Prompt。
```

---

### 3.5 Observation pipeline 还没真正成型

agentmemory 的核心是：

```text
RawObservation
  → CompressedObservation
    → Memory
      → Index
```

当前项目更像：

```text
RawObservation
  → 直接嵌入 MemoryItem.metadata
```

或者：

```text
messages
  → CompressionEngine
    → SessionSummary
      → MemoryItem candidates
```

缺少中间的：

```text
CompressedObservation / ObservationStore
```

因此不容易做到：

```text
1. 重建索引；
2. 对每次工具调用单独召回；
3. 按 observation 类型过滤；
4. 从原始事件重新提炼更好的长期记忆；
5. 生成知识图谱关系边。
```

---

### 3.6 没有真正的向量检索

当前没有：

```text
EmbeddingProvider
VectorIndex
embedding cache
vector dimension validation
query embedding
cosine similarity search
```

因此对于语义相似但关键词不同的问题召回较弱，例如：

```text
用户问：启动很慢为什么？
历史记忆写的是：main.py 初始化时重复加载 long_term index 导致延迟。
```

如果 query 没命中“初始化/加载/index/延迟”等关键词，就可能召回失败。

---

### 3.7 没有知识图谱召回

当前 MemoryItem 有：

```text
concepts
files
source_session_ids
source_observation_ids
```

这已经具备构图素材，但没有显式 graph：

```text
nodes: file / concept / error / memory / session / symbol
edges: mentions / modifies / fixes / caused_by / supersedes / related_to
```

所以不能做类似：

```text
query 命中 core/engine.py
  → 找到相关 bug memory
    → 扩展到 OpenAI tool pair ordering
      → 扩展到 tests/test_openai_tool_pairing.py
```

---

### 3.8 没有 RRF 融合，分数尺度会混乱

当前综合分是直接相加：

```text
bm25_score + importance + recency + file_match + error_match + type_weight
```

问题：

```text
1. 各分量尺度不一致；
2. BM25/Vector/Graph 如果后续加入，不能直接加原始分；
3. 一个强 file_match 可能压过真正语义相关内容；
4. 重要性过高的旧记忆可能长期霸榜。
```

agentmemory 用 RRF：

```text
score = w_bm25   * 1/(k + rank_bm25)
      + w_vector * 1/(k + rank_vector)
      + w_graph  * 1/(k + rank_graph)
```

优点是：

```text
不同检索器的原始分数不用统一尺度；
只按排名融合；
鲁棒、简单、适合本地系统。
```

---

## 4. agentmemory 的 BM25 + Vector + Graph + RRF + rerank 是否可以复现？

结论：可以复现，但不建议一次性完整复现。

### 4.1 可以复现的部分

| 能力 | 是否适合本项目复现 | 原因 |
|---|---:|---|
| BM25 倒排索引 | 强烈建议 | 对代码符号、文件路径、错误信息最有效 |
| VectorIndex | 建议第二阶段做 | 提升语义召回，可用本地轻量实现 |
| RRF | 强烈建议 | 简单稳健，适合融合 BM25/Vector/Graph |
| metadata filter | 强烈建议 | file/error/kind/project 过滤对 code-agent 很关键 |
| rerank | 可选 | 本地 cross-encoder 成本高，可先用轻量 LLM/规则 rerank |
| Graph | 后置 | 有价值但复杂度高，必须先有高质量实体和关系 |
| agentmemory REST/MCP | 可选实验 | 可作为外部服务，但不应替代内置系统主线 |

---

### 4.2 不建议完全复刻的部分

agentmemory 是 TypeScript 独立服务，具备 REST API、MCP、Viewer、iii-sdk state 等能力。

你的项目是 Python CLI code-agent，当前更适合：

```text
内置轻量长期记忆系统
  + 可选 agentmemory 适配层
```

不建议现在直接把 agentmemory 作为唯一记忆后端，原因：

```text
1. 引入独立服务会增加启动、部署、调试复杂度；
2. 当前项目已有 MemoryManager/工具/测试，不应废弃；
3. 本地 JSON 可读可控，适合快速迭代；
4. agentmemory 的 graph/viewer/MCP 后期再接入收益更高。
```

---

## 5. 推荐的下一代架构

### 5.1 总体架构

推荐目标：

```text
MemoryManager 继续作为统一入口，下面拆成 Store / Index / Retrieval / Promotion / Context 五个子系统。
```

目标结构：

```text
core/
  memory_manager.py                 # 统一门面，兼容现有接口
  memory_items.py                   # RawObservation / MemoryItem / MemoryRecallResult
  memory_layers.py                  # 逐步瘦身为 Store facade
  memory_retrieval.py               # 新 HybridMemoryRetriever
  memory_context_builder.py         # Prompt 预算注入，保留并增强

  memory_store.py                   # 新增：ObservationStore / MemoryItemStore / SummaryStore
  memory_index.py                   # 新增：BM25Index / VectorIndex / GraphIndex / IndexPersistence
  memory_embedding.py               # 新增：EmbeddingProvider / HashEmbeddingProvider / OpenAI-compatible provider
  memory_graph.py                   # 新增：实体抽取、关系边、图扩展
  memory_reranker.py                # 新增：可选 rerank
  memory_maintenance.py             # 新增：去重、supersede、archive、TTL、质量治理
  memory_privacy.py                 # 新增：secret 脱敏
```

数据目录建议：

```text
memory/
  long_term/
    items/
      memory_item_*.json
    summaries/
      summary_*.json
    observations/
      observation_*.jsonl
    indexes/
      bm25.json
      vector.json
      graph_nodes.json
      graph_edges.json
      meta.json
    index.json                      # 兼容旧版，逐步迁移
```

---

### 5.2 写入 pipeline

目标写入链路：

```text
AgentEngine 事件
  ↓
MemoryManager.observe(raw_event)
  ↓
PrivacyFilter.strip_private_data()
  ↓
ObservationStore.append(RawObservation)
  ↓
ObservationCompressor / PromotionPolicy
  ↓
MemoryItem 候选生成
  ↓
MemoryMaintenance.deduplicate_or_supersede()
  ↓
MemoryItemStore.save(item)
  ↓
IndexManager.add(item/observation/summary)
  ↓
BM25Index + optional VectorIndex + optional GraphIndex
```

其中：

```text
RawObservation：所有原始事件。
CompressedObservation：结构化事件摘要，可选新增。
MemoryItem：跨会话可复用知识。
SessionSummary：会话级兼容摘要。
```

---

### 5.3 检索 pipeline

目标召回链路：

```text
query
  ↓
QueryAnalyzer
  - tokenize
  - detect file path
  - detect error type
  - detect task type
  - extract entities
  ↓
Candidate generation
  ├── BM25Index.search(query, top_n)
  ├── VectorIndex.search(query_embedding, top_n)          # optional
  ├── GraphIndex.expand(entities / top candidates, top_n) # optional
  └── MetadataRetriever(file/error/kind/project)
  ↓
RRF Fusion
  ↓
Filters
  - project/cwd
  - status active/is_latest
  - confidence threshold
  - not expired
  ↓
Diversity
  - max items per session/source/file/kind
  ↓
Optional rerank
  - rule-based rerank first
  - local/LLM rerank later
  ↓
MemoryContextBuilder
  - token budget
  - task type priority
  - dedupe
  - formatting
```

---

## 6. 分阶段改进路线

### Phase 0：清理和基线评测

目标：知道现在召回到底差在哪里。

任务：

```text
1. 新增 memory recall eval fixtures：
   - 给定 query，期望命中 memory_id；
   - 包含 file history、error history、architecture、workflow、preference。

2. 新增 tests/test_memory_recall_quality.py：
   - 测 top1 / top3 / top5 命中；
   - 测中文 query；
   - 测文件路径 query；
   - 测错误 traceback query。

3. 清理旧的独立 SearchMemoryTool 或标记为 legacy：
   - 主线只允许 MemoryManager.hybrid_recall。
```

验收：

```text
pytest tests/test_memory_phase*.py tests/test_memory_recall_quality.py -q
```

---

### Phase 1：实现真正 BM25Index

目标：把当前伪 BM25 升级为标准 BM25。

新增/改造：

```text
core/memory_index.py
  class BM25MemoryIndex
```

索引字段：

```text
title      权重 2.0
content    权重 1.0
concepts   权重 1.8
files      权重 2.2
kind       权重 0.8
error      权重 2.0
project    权重 0.5
```

需要保存：

```text
doc_store: doc_id -> metadata
term_freqs: doc_id -> term -> count
doc_lengths: doc_id -> length
inverted_index: term -> doc_ids
avg_doc_length
```

检索：

```text
idf = log((N - df + 0.5) / (df + 0.5) + 1)
score = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
```

中文分词：

```text
优先 jieba；
没有 jieba 时 fallback 到当前 2/3 字片段；
代码路径和符号必须保留 / . _ -。
```

验收：

```text
1. BM25 能优先命中稀有错误词，如 WinError 5、ModuleNotFoundError；
2. 文件路径 query 能命中文件相关记忆；
3. 中文 query 不再大量无意义碎片命中；
4. 索引可持久化，启动不必全量读取所有文件计算分数。
```

---

### Phase 2：IndexPersistence 和增量索引

目标：避免每次 recall 都从磁盘全量构建 documents。

任务：

```text
1. MemoryItemStore.save 后调用 IndexManager.add_or_update(doc)
2. MemoryItem archived/superseded 后调用 IndexManager.remove(doc_id)
3. 启动时加载 indexes/bm25.json
4. 如果索引损坏或版本不一致，后台 rebuild
5. 保存使用 debounce + atomic write
```

索引版本：

```json
{
  "schema_version": 1,
  "embedding_provider": null,
  "created_at": "...",
  "updated_at": "..."
}
```

验收：

```text
1. 保存记忆后无需重启即可召回；
2. 重启后不全量读取所有 memory_item_*.json 也可检索；
3. index 损坏时自动 rebuild；
4. Windows 下不会遗留大量 index.*.tmp。
```

---

### Phase 3：MemoryItem 生命周期治理

目标：防止长期记忆污染。

新增机制：

```text
1. 保存前相似度检查：
   - exact hash
   - normalized content hash
   - Jaccard similarity
   - title/content overlap

2. supersede：
   - 新记忆明显更新旧记忆时，旧记忆 status=superseded
   - 新记忆 version=old.version+1
   - 新记忆 parent_id=old.id
   - 新记忆 supersedes=[old.id]

3. archive：
   - 低 confidence
   - 长期未命中
   - 过期 project state

4. access tracking：
   - last_accessed_at
   - access_count
   - last_injected_at

5. quality_score：
   - 来源可靠度
   - 测试是否通过
   - 是否来自用户显式 memory_save
   - 是否被多次召回有效
```

MemoryItem 建议新增字段：

```python
parent_id: str | None
supersedes: list[str]
related_ids: list[str]
is_latest: bool
forget_after: str | None
last_accessed_at: str | None
access_count: int
quality_score: float
```

验收：

```text
1. 重复保存相同 memory 不产生重复污染；
2. 新决策能 supersede 旧决策；
3. recall 默认只返回 active + is_latest；
4. memory_stats 能显示 active/superseded/archived 数量。
```

---

### Phase 4：轻量 VectorIndex

目标：提升语义召回。

建议先做轻量内置，不上外部向量数据库。

新增：

```text
core/memory_embedding.py
  EmbeddingProvider
  HashEmbeddingProvider        # 默认、无外部依赖、用于测试和降级
  OpenAICompatibleEmbeddingProvider # 可选

core/memory_index.py
  VectorMemoryIndex
```

存储：

```text
memory/long_term/indexes/vector.json
```

向量字段：

```text
embedding
provider
model
dimensions
doc_id
updated_at
```

检索：

```text
query_embedding = provider.embed(query)
cosine_similarity(query, doc.embedding)
return top_n
```

维度保护：

```text
如果 persisted dimension != active provider dimension：
  - 默认禁用 vector index 并提示 rebuild；
  - 可配置 drop stale vector index。
```

验收：

```text
1. 无 API key 时 HashEmbeddingProvider 仍能让测试通过；
2. 有 embedding 配置时可生成真实向量；
3. 换 provider 维度不一致不会静默污染召回；
4. Vector 召回结果进入 RRF 融合。
```

---

### Phase 5：RRF 融合

目标：把 BM25、Vector、Metadata、Graph 的结果用统一机制融合。

新增：

```python
class RetrievalHit:
    doc_id: str
    rank: int
    score: float
    source: str  # bm25/vector/graph/metadata
```

RRF：

```python
RRF_K = 60
final_score = sum(weight[source] * 1 / (RRF_K + rank[source]))
```

推荐默认权重：

```text
BM25:    0.45
Vector:  0.30
Metadata:0.20
Graph:   0.05 initially disabled
```

对于特殊场景动态调权：

```text
file_history:
  Metadata/File: 0.45
  BM25: 0.35
  Vector: 0.15
  Graph: 0.05

error_history:
  BM25: 0.45
  Metadata/Error: 0.35
  Vector: 0.15
  Graph: 0.05

architecture/general:
  BM25: 0.40
  Vector: 0.35
  Metadata: 0.20
  Graph: 0.05
```

验收：

```text
1. 不同召回器分数不直接相加；
2. 可以在 reason 中展示命中来源：BM25 rank2 + Vector rank4 + file exact；
3. top_k 结果比单 BM25 更稳；
4. 文件/错误专用召回仍能强命中。
```

---

### Phase 6：GraphIndex，后置实现

目标：做实体关系扩展，而不是一开始追求复杂知识图谱。

建议先做轻量图：

```text
Node:
  memory:<id>
  file:<path>
  concept:<name>
  error:<type>
  session:<id>
  tool:<name>

Edge:
  memory -> file       mentions_file
  memory -> concept    has_concept
  memory -> error      fixes_error
  memory -> session    from_session
  memory -> memory     supersedes / related
```

图召回方式：

```text
1. query 提取 file/error/concept；
2. 找到对应 node；
3. BFS 1~2 hops；
4. 返回相关 memory ids；
5. 作为 Graph hits 进入 RRF。
```

不建议一开始做复杂 GraphRAG，因为当前记忆规模和实体质量还不够。

验收：

```text
1. 查询 core/engine.py 能扩展到相关 bug/workflow；
2. 查询 ModuleNotFoundError 能扩展到相关测试文件和修复经验；
3. Graph 结果不会单独霸榜，只进入 RRF。
```

---

### Phase 7：Optional rerank

目标：对 RRF top20 做精排。

优先级建议：

```text
先做 rule-based rerank，后做模型 rerank。
```

规则 rerank：

```text
+ exact file path
+ exact error type
+ same project
+ active/is_latest
+ high confidence
+ recently accessed and useful
- archived/superseded
- low confidence
- expired
- too long/noisy content
```

模型 rerank 可选：

```text
1. 本地 sentence-transformers / cross-encoder；
2. LLM rerank top10；
3. 只在 high-value query 启用，避免慢。
```

验收：

```text
1. rerank 可配置开关；
2. 默认不拖慢普通 recall；
3. rerank reason 可解释；
4. 测试中可用 fake reranker 保证确定性。
```

---

## 7. 是否需要直接接 agentmemory？

建议作为 Phase 8 实验，而不是主线依赖。

### 方案 A：继续内置增强，推荐主线

优点：

```text
1. 不增加外部服务；
2. Python 内部测试容易；
3. 和现有 MemoryManager/工具/ContextAssembler 无缝；
4. 数据可控、可读、可迁移。
```

缺点：

```text
需要自己实现 vector/graph/rerank/viewer 等能力。
```

### 方案 B：agentmemory 作为外部镜像服务

方式：

```text
MemoryManager.observe()
  → 本地保存
  → async mirror 到 agentmemory REST

MemoryManager.hybrid_recall()
  → 本地 recall
  → 可选 agentmemory smart-search
  → RRF 合并
```

优点：

```text
1. 可使用 agentmemory 的 viewer、graph、MCP；
2. 本地系统仍能独立工作；
3. 外部服务挂了可降级。
```

缺点：

```text
1. 架构复杂；
2. 要处理同步失败；
3. 要处理两边 memory id/source 对齐；
4. 要处理隐私和重复写入。
```

推荐：

```text
先完成 Phase 1~5，再做 agentmemory mirror 实验。
```

---

## 8. 当前与 agentmemory 能力对照

| 能力 | mini 当前状态 | agentmemory 状态 | 建议 |
|---|---|---|---|
| Working memory | 已有 | 有 session/observation | 保留 |
| Episodic summary | 已有 | 有 compressed observation/summary | 保留并拆出 ObservationStore |
| Long-term MemoryItem | 已有 | 有 Memory + version/supersede | 增强生命周期 |
| RawObservation | 有模型，未独立存储 | 完整 hook observation | 新增 ObservationStore |
| BM25 | 有独立类，但主线未真用 | 完整 BM25 index | 优先升级 |
| Vector | 无 | 可选 VectorIndex | 第二阶段实现 |
| Graph | 无 | GraphRetrieval | 后置实现 |
| RRF | 无 | 有 RRF | 强烈建议实现 |
| rerank | 无 | optional local rerank | 可选后置 |
| index persistence | 有 index.json，但弱 | BM25/vector snapshot | 增强 |
| memory versioning | 有 version/status 字段 | parent/supersedes/isLatest | 增强 |
| TTL/forget | 无 | 有 forgetAfter | 增强 |
| privacy filter | 无独立模块 | 有 stripPrivateData | 必须补 |
| context budget | 已有 MVP | 有 context endpoint/token budget | 继续增强 |
| viewer/dashboard | 无 | 有 viewer | 可选 |

---

## 9. 最小可行优先级

如果只做最关键的 5 件事，推荐顺序：

```text
P0. 新增召回质量测试集，量化 recall 是否真的变好
P1. 把长期记忆主线检索升级为真正 BM25Index
P2. 做 MemoryItem 去重 + supersede + is_latest
P3. 做 RRF 框架，即使先只有 BM25 + Metadata 两路
P4. 加轻量 VectorIndex，再接入 RRF
```

不建议最先做：

```text
Graph
Viewer
MCP
复杂 reranker
直接替换为 agentmemory 服务
```

原因：这些能力在低质量基础索引和重复记忆未治理前，收益不稳定，反而会增加复杂度。

---

## 10. 后续实施检查清单

### 10.1 代码模块清单

建议新增：

```text
core/memory_index.py
core/memory_store.py
core/memory_embedding.py
core/memory_maintenance.py
core/memory_privacy.py
core/memory_graph.py        # 后置
core/memory_reranker.py     # 后置
```

建议增强：

```text
core/memory_manager.py
core/memory_retrieval.py
core/memory_layers.py
core/memory_items.py
core/memory_context_builder.py
tools/memory_tool.py
```

建议逐步废弃或改造：

```text
core/keyword_indexer.py     # 可作为 legacy/general search，不作为长期记忆主线
core/bm25_retriever.py      # 可迁移能力到 memory_index.py
tools/retrieval_tool.py     # 不应再维护独立记忆索引
```

---

### 10.2 测试清单

必须新增或扩展：

```text
tests/test_memory_bm25_index.py
tests/test_memory_index_persistence.py
tests/test_memory_item_lifecycle.py
tests/test_memory_rrf_fusion.py
tests/test_memory_vector_index.py
tests/test_memory_recall_quality.py
tests/test_memory_privacy.py
tests/test_memory_graph_index.py      # 后置
tests/test_memory_reranker.py         # 后置
```

每阶段验收都必须跑对应 pytest。

---

## 11. 最终目标效果

完成后，长期记忆系统应该达到以下效果：

```text
1. 用户问“之前这个错误怎么修的？”
   → 能命中历史 bug memory，带文件、错误、修复步骤、测试结果。

2. 编辑 core/engine.py 前
   → 自动召回这个文件历史坑点和相关 workflow。

3. 新架构任务开始前
   → 自动召回历史架构决策、用户偏好、相关文档。

4. 重复记忆不会无限堆积
   → 新记忆 supersede 旧记忆，默认只召回 latest active。

5. query 关键词不完全一致也能召回
   → VectorIndex 补足语义相似。

6. 文件/错误/概念能互相扩展
   → GraphIndex 提供关系召回。

7. 多路召回结果稳定融合
   → RRF 避免 BM25/vector/graph 分数尺度混乱。

8. Prompt 不被记忆污染
   → MemoryContextBuilder 和 ContextAssembler 继续按预算、任务类型、置信度控制注入。
```

---

## 12. 最终建议

这个项目当前长期记忆系统的方向是对的，问题主要在“检索和治理还不够成熟”，不是整体架构需要推倒。

推荐路线：

```text
保留现有 MemoryManager 统一入口
  → 新增真正 BM25Index + IndexPersistence
    → 增强 MemoryItem 生命周期治理
      → 引入 RRF 融合框架
        → 加轻量 VectorIndex
          → 再考虑 Graph / rerank / agentmemory mirror
```

一句话：

```text
先把本地长期记忆做成稳定、可评测、可治理、可解释的 BM25+Metadata+RRF 系统，
再逐步增强为 BM25+Vector+Graph+RRF+optional rerank。
```
