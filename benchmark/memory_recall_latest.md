# Memory Recall Benchmark Report

## Summary

- Cases: 12
- Hit@1: 91.67%
- Hit@3: 100.00%
- Hit@5: 100.00%
- MRR: 0.958
- Forbidden violation rate: 0.00%

## By Category

| Category | Cases | Hit@1 | Hit@3 | Hit@5 | MRR | Forbidden |
|---|---:|---:|---:|---:|---:|---:|
| architecture | 2 | 100.00% | 100.00% | 100.00% | 1.000 | 0.00% |
| chinese_query | 1 | 0.00% | 100.00% | 100.00% | 0.500 | 0.00% |
| error_history | 3 | 100.00% | 100.00% | 100.00% | 1.000 | 0.00% |
| file_history | 1 | 100.00% | 100.00% | 100.00% | 1.000 | 0.00% |
| lifecycle | 1 | 100.00% | 100.00% | 100.00% | 1.000 | 0.00% |
| phase_task | 2 | 100.00% | 100.00% | 100.00% | 1.000 | 0.00% |
| preference | 1 | 100.00% | 100.00% | 100.00% | 1.000 | 0.00% |
| semantic_rewrite | 1 | 100.00% | 100.00% | 100.00% | 1.000 | 0.00% |

## Failures

No failed cases or forbidden hits.

## Case Details

### phase5_rrf_done

- Category: phase_task
- Query: Phase 5 是不是已经做了 RRF，多路召回现在怎么融合排序？
- Expected any: bench_phase5_rrf_fusion
- Hit rank: 1
- Ranked ids: bench_phase5_rrf_fusion, bench_arch_retrieval_not_rewrite, bench_phase3_lifecycle_governance, bench_phase4_vector_index, bench_phase2_index_persistence

### phase4_vector_index

- Category: phase_task
- Query: Phase 4 加的是不是向量检索和 HashEmbeddingProvider？
- Expected any: bench_phase4_vector_index
- Hit rank: 1
- Ranked ids: bench_phase4_vector_index, bench_phase5_rrf_fusion, bench_phase3_lifecycle_governance, bench_arch_retrieval_not_rewrite, bench_phase2_index_persistence

### index_persistence_incremental

- Category: file_history
- Query: core/memory_index.py 里 IndexPersistence 和增量 BM25 索引是什么时候做的？
- Expected any: bench_phase2_index_persistence
- Hit rank: 1
- Ranked ids: bench_phase2_index_persistence, bench_phase5_rrf_fusion, bench_phase4_vector_index, bench_bug_winerror5_index_json, bench_phase3_lifecycle_governance

### lifecycle_filters_old_memory

- Category: lifecycle
- Query: 长期记忆现在怎么过滤 archived superseded expired 旧记忆？
- Expected any: bench_phase3_lifecycle_governance
- Hit rank: 1
- Ranked ids: bench_phase3_lifecycle_governance, bench_arch_retrieval_not_rewrite, bench_bug_openai_tool_pairing, bench_phase4_vector_index, bench_bug_module_not_found_pytest

### architecture_not_rewrite

- Category: architecture
- Query: 现在记忆检索还要推倒重写吗，下一步应该先做 Graph 还是质量测评？
- Expected any: bench_arch_retrieval_not_rewrite
- Hit rank: 1
- Ranked ids: bench_arch_retrieval_not_rewrite, bench_phase3_lifecycle_governance, bench_phase2_index_persistence, bench_context_compression_strategy_doc, bench_bug_winerror5_index_json

### winerror5_index_failure

- Category: error_history
- Query: PS D:\LLM\mini-claude-code-cli> python .\main.py 保存索引失败 WinError 5 拒绝访问 index.json
- Expected any: bench_bug_winerror5_index_json
- Hit rank: 1
- Ranked ids: bench_bug_winerror5_index_json, bench_bug_module_not_found_pytest, bench_phase2_index_persistence, bench_phase5_rrf_fusion, bench_phase4_vector_index

### openai_tool_pairing_bad_request

- Category: error_history
- Query: BadRequestError assistant tool_calls 后没有紧邻 tool response 是怎么修的？
- Expected any: bench_bug_openai_tool_pairing
- Hit rank: 1
- Ranked ids: bench_bug_openai_tool_pairing, bench_bug_winerror5_index_json, bench_bug_module_not_found_pytest, bench_phase3_lifecycle_governance, bench_phase5_rrf_fusion

### module_not_found_core_pytest

- Category: error_history
- Query: Traceback ModuleNotFoundError No module named core pytest 应该怎么办？
- Expected any: bench_bug_module_not_found_pytest
- Hit rank: 1
- Ranked ids: bench_bug_module_not_found_pytest, bench_bug_winerror5_index_json, bench_preference_tdd_pytest_required, bench_phase3_lifecycle_governance, bench_phase5_rrf_fusion

### user_tdd_preference

- Category: preference
- Query: 用户对实现功能和测试有什么要求，是不是必须 pytest 通过才能说完成？
- Expected any: bench_preference_tdd_pytest_required
- Hit rank: 1
- Ranked ids: bench_preference_tdd_pytest_required, bench_phase5_rrf_fusion, bench_bug_module_not_found_pytest, bench_phase3_lifecycle_governance, bench_phase2_index_persistence

### context_compression_doc

- Category: architecture
- Query: 上下文压缩系统重构文档里说 token budget 和结构化摘要怎么做？
- Expected any: bench_context_compression_strategy_doc
- Hit rank: 1
- Ranked ids: bench_context_compression_strategy_doc, bench_phase5_rrf_fusion, bench_phase2_index_persistence, bench_phase3_lifecycle_governance, bench_arch_retrieval_not_rewrite

### semantic_rewrite_rrf

- Category: semantic_rewrite
- Query: BM25 和向量结果冲突时现在还是简单加权吗，还是倒数排名融合？
- Expected any: bench_phase5_rrf_fusion
- Hit rank: 1
- Ranked ids: bench_phase5_rrf_fusion, bench_phase2_index_persistence, bench_arch_retrieval_not_rewrite, bench_phase4_vector_index, bench_bug_openai_tool_pairing

### chinese_quality_question

- Category: chinese_query
- Query: 现在的长期记忆检索是不是已经不是以前那种简单关键词鸡肋检索了？
- Expected any: bench_arch_retrieval_not_rewrite, bench_phase5_rrf_fusion
- Hit rank: 2
- Ranked ids: bench_phase3_lifecycle_governance, bench_arch_retrieval_not_rewrite, bench_context_compression_strategy_doc, bench_bug_module_not_found_pytest, bench_preference_tdd_pytest_required
