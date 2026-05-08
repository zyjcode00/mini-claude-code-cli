# Architecture Summary

## Project Overview

Mini Claude Code CLI is a lightweight AI engineering agent system that replicates the core capabilities of Claude Code:

- ✅ Tool calling (file operations, code search, Git management, test execution)
- ✅ Task planning system (structured Plan management)
- ✅ Three-layer memory architecture (Working / Episodic / Long-term)
- ✅ Multi-strategy context compression (LLM_SUMMARY / KEYFRAME / SLIDING_WINDOW / IMPORTANCE_FILTER)
- ✅ Session persistence and recovery
- ✅ Git automation safety net (shadow branches, auto snapshots, failure rollback)

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Interaction (main.py)                 │
│  REPL loop · Command parsing · Rich console display             │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      AgentEngine (core/engine.py)               │
│  execute_query() · _call_llm() · Parallel tool execution        │
│  Git safety net · Session save/restore                          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
          ┌─────────────┬───────┴───────┬─────────────┐
          ▼             ▼               ▼             ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ ContextManager│ │  PlanManager  │ │  Tool System  │ │ BM25Retriever │
│  (context.py) │ │  (plan.py)    │ │ (tools/*.py)  │ │ (bm25.py)     │
│ Messages      │ │ Tasks & Goal  │ │ Bash · File   │ │ Keyword index │
│ 3-layer mem   │ │ Validation    │ │ Search · Test │ │ BM25 search   │
│ Compression   │ │ Auto-fix      │ │ Plan · Git    │ │ History inject│
└───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘
        │                   │
        ▼                   ▼
┌───────────────┐ ┌───────────────┐
│ Compression   │ │SessionManager │
│   Engine      │ │(session_mgr)  │
│ LLM_SUMMARY   │ │ Session list  │
│ KEYFRAME      │ │ Incomplete    │
│ SLIDING_WINDOW│ │ detection     │
└───────────────┘ └───────────────┘
```

## Three-Layer Memory Architecture

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Working Memory (20 messages, FIFO eviction)         │
│  Real-time access · Latest context                   │
└───────────────────────────┬─────────────────────────┘
                            │ Eviction
                            ▼
┌─────────────────────────────────────────────────────┐
│  Episodic Memory (50 summaries, importance eviction) │
│  Structured summaries · Keyword search               │
└───────────────────────────┬─────────────────────────┘
                            │ Archive
                            ▼
┌─────────────────────────────────────────────────────┐
│  Long-term Memory (unlimited, disk JSON storage)     │
│  Cross-session retrieval · Inverted index + IDF      │
│  Storage path: memory/long_term/*.json                │
└─────────────────────────────────────────────────────┘
```

## Core Modules

### engine.py — Main Loop

| Method | Description |
|--------|-------------|
| `execute_query(input)` | Main reasoning loop (max 80 steps) |
| `_call_llm()` | Async LLM call + 3-layer memory injection |
| `compress_messages()` | Async context compression |
| `save_session()` / `load_session()` | Session state persistence |

### context.py — Context & Memory Manager

- Message history management
- Three-layer memory orchestration
- Compression engine integration
- `asyncio.Lock` for concurrent safety (`get_messages_snapshot()`)

### plan.py — Task Planning

- Structured task list with `plan_id` (MD5 hash)
- `validate_state()` + `auto_fix()` for consistency
- `is_goal_completed()` with similarity detection (prevents duplicate execution)
- Completed goals archive (last 10 entries)

### compression_engine.py — Multi-Strategy Compression

| Strategy | Trigger Condition | Description |
|----------|------------------|-------------|
| `LLM_SUMMARY` | Error ratio > 30% | LLM-generated structured summary (with cache + rate limit) |
| `KEYFRAME` | Tool ratio > 40% | Extract key frames (user + error + tool calls) |
| `SLIDING_WINDOW` | Default | Keep most recent messages |
| `IMPORTANCE_FILTER` | Avg importance > 0.6 | Filter by importance score (preserving order) |

Additional features: LLM call cache, hourly rate limiting (10 calls/hr), tool-call pairing validation.

### Tools (tools/*.py)

| Category | Tools |
|----------|-------|
| File ops | ReadTool, FileEditTool, WriteFullFileTool, FileTreeTool |
| Code analysis | SearchTool, ListSymbolsTool, FindSymbolTool |
| Execution | BashTool, PytestTool |
| Planning | UpdatePlanTool, MarkDoneTool |
| Git | GitStatusTool, GitCommitTool, GitRollbackTool |
| Session | SessionCleanerTool |

## Key Design Decisions

1. **Async architecture** — `async/await` + `asyncio.gather` for parallel tool execution
2. **Three-layer memory** — Working → Episodic → Long-term, with BM25 retrieval
3. **Adaptive compression** — 4 strategies with intelligent selection + fallback
4. **State synchronization** — Plan completion triggers summary state updates
5. **Git safety net** — Shadow branches (`agent/plan-{id}`) with auto squash merge on completion
6. **Session persistence** — Full state save (messages + plan + memories) with auto recovery

## Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.11+ |
| Async | asyncio |
| LLM API | OpenAI API (Anthropic-compatible) |
| Data validation | Pydantic |
| CLI display | Rich |
| Testing | pytest |
| Storage | JSON |
| Git | subprocess |