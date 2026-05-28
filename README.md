<div align="center">

# 🤖 Mini Claude Code CLI

**一个具备异步并发推理、显式规划与确定性自愈能力的 AI 工程代理**

基于工程化 Agent 架构 · 闭环测试驱动 · 三层记忆系统 · 影子分支保护

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![Architecture](https://img.shields.io/badge/架构-Agentic--Loop-orange.svg)]()
[![VCS](https://img.shields.io/badge/VCS-Shadow--Branching-blueviolet.svg)]()
[![Symbol](https://img.shields.io/badge/智能-AST--Symbol--Map-red.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

</div>

---

## ✨ 为什么选择 Mini Claude Code CLI？

Mini Claude Code CLI 不仅仅是一个代码生成器——它是一个**闭环的代码进化系统**。

| 传统代码助手 | Mini Claude Code CLI |
|:---|:---|
| 串行调用工具，效率低下 | ⚡ 异步并发推理，3-5x 速度提升 |
| 长对话后遗忘上下文 | 🧠 三层记忆架构，跨会话知识沉淀 |
| 改错了就束手无策 | 🛡️ 影子分支 + Git 物理回滚，零风险重构 |
| 盲目 grep 搜索代码 | 🗺️ AST 符号地图，精准定位跨文件定义 |
| 写完代码就算完成 | 🧪 TDD 闭环：测试失败 → 自动分析 → 修复 → 验证 |
| 随意开工没有计划 | ⏳ 显式规划管理，实时看板追踪任务进度 |

---

## 🧭 核心架构概览

```
┌──────────────────────────────────────────────────────────────┐
│                        用户输入 (REPL)                        │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                  Agent Engine (异步决策中枢)                   │
│  ┌────────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │  PlanManager   │  │   Context    │  │  SessionManager  │ │
│  │  任务规划看板   │  │  上下文管家   │  │   会话持久化     │ │
│  └────────────────┘  └──────────────┘  └──────────────────┘ │
└──────────────────────────┬───────────────────────────────────┘
                           │ asyncio.gather 并发分发
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                       工具层 (Tools)                          │
│  Bash │ File │ Git │ Search │ Symbol │ Pytest │ Plan │ ...  │
└──────────────────────────┬───────────────────────────────────┘
                           │ 物理执行反馈
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    三层记忆系统 (Memory)                       │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │  工作记忆    │ → │  情景记忆    │ → │  长期记忆    │      │
│  │ 最近N条消息 │   │ 结构化摘要   │   │ BM25检索库  │       │
│  └─────────────┘   └─────────────┘   └─────────────┘       │
└──────────────────────────────────────────────────────────────┘
```

---

## 📂 项目结构

```
mini-claude-code-cli/
├── main.py                    # 系统入口：异步 REPL 与配置加载
├── CLAUDE.md                  # Agent 行为准则与工程规范
├── .env.example               # 环境变量示例
├── requirements.txt           # Python 依赖清单
├── LICENSE                    # MIT 开源许可证
│
├── core/                      # 🧠 核心引擎层
│   ├── engine.py              # 异步决策中枢：并发 ReAct 循环，接入主动记忆召回/保存
│   ├── context.py             # 上下文管家：压缩与持久化
│   ├── plan.py                # PlanManager：显式任务规划
│   ├── prompts.py             # 提示词工厂：动态注入环境/记忆/计划
│   ├── session_manager.py     # 会话管理器：跨会话状态持久化
│   ├── compression_engine.py  # 智能压缩引擎：4 种自适应策略
│   ├── memory_layers.py       # 三层记忆架构管理
│   ├── memory_models.py       # 会话摘要、文件变更、错误记录等记忆数据模型
│   ├── memory_items.py        # 长期记忆实体：MemoryItem / RawObservation
│   ├── memory_manager.py      # 统一记忆编排：观察、保存、召回、统计、Prompt 构建
│   ├── memory_context_builder.py # 记忆 Prompt 注入与 Token Budget 控制
│   ├── memory_retrieval.py    # 统一检索层：Hybrid Recall / 文件历史 / 错误历史
│   ├── keyword_indexer.py     # 关键词提取：jieba + TF-IDF
│   └── bm25_retriever.py      # BM25 全文检索引擎
│
├── tools/                     # 🛠️ 工具层
│   ├── base.py                # 工具基类与 Schema 规范
│   ├── bash_tool.py           # Shell 执行 (Windows 编码自愈)
│   ├── file_tool.py           # 文件读写 (CRLF/LF 归一化)
│   ├── git_tool.py            # Git 操作 (影子分支/快照/回滚)
│   ├── search_tool.py         # 正则代码搜索
│   ├── symbol_tool.py         # AST 符号导航
│   ├── pytest_tool.py         # 测试执行与 Traceback 捕获
│   ├── plan_tool.py           # 任务计划管理接口
│   ├── session_tool.py        # 会话清理
│   ├── retrieval_tool.py      # 记忆检索工具
│   └── memory_tool.py         # 长期记忆工具：save/recall/file_history/error_history/stats
│
├── tests/                     # 🧪 测试套件 (40+ 测试文件)
├── docs/                      # 📚 设计文档与架构记录
│   ├── architecture_summary.md
│   ├── shadow_branch_system.md
│   ├── memory_system_refactor_roadmap.md
│   └── new_long_term_memory_system.md
└── memory/                    # 持久化长期记忆存储 (运行时生成)
```

---

## 🔥 核心能力详解

### 1. ⚡ 异步并发推理 (Parallel Tool Use)

基于 AsyncIO 链路的并行架构，当 LLM 一次性提出多个工具请求时，引擎使用 `asyncio.gather` 并行执行，交互轮次减少约 60%。

### 2. ⏳ 显式规划与任务看板 (Explicit Planning)

执行前强制调用 `manage_plan` 创建结构化任务清单，Plan 状态被动态注入 System Prompt，确保 Agent 在长对话中始终"不忘初心"。任务看板实时展示 ⏳/✅ 进度。

### 3. 🧪 确定性自愈与 TDD 闭环

- **物理验证**：通过 `run_pytest` 将代码修改与测试绑定
- **报错驱动修复**：Agent 准解析 Traceback，自主进入 **Read → Edit → Test** 闭环
- **只有测试通过，任务才算真正完成**

### 4. 🛡️ 影子分支与 Git 物理回滚

- **零风险重构**：执行 Plan 前自动创建影子分支，实验与 main 物理隔离
- **确定性回退**：连续 edit_file 失败时自动触发 Git 回滚
- **优雅提交**：任务完成后 Squash Merge，保持 main 分支历史纯净

### 5. 🗺️ AST 符号地图与智能搜索

- **符号导航**：基于 Python AST 建立全局类/函数索引，按名称精准定位
- **代码搜索**：正则表达式 + 上下文行展示 + glob 文件过滤

### 6. 🧬 三层记忆架构与智能压缩

```
工作记忆 → 情景记忆 → 长期记忆
(最近消息)   (结构化摘要)  (BM25检索库)
     ↑ 溢出压缩     ↑ 持久化存储
```

**压缩策略自适应切换**：

| 策略 | 适用场景 | 特点 |
|:---|:---|:---|
| keyframe | 代码重构/Bug修复 | 保留所有文件变更和错误记录 |
| semantic | 分析讨论/知识问答 | 提取核心观点和决策 |
| hybrid | 混合型任务 | 关键帧 + 语义摘要 |
| timeline | 时序任务 | 保留时序关键节点 |

### 7. 🧠 新长期记忆闭环 (MemoryManager + Hybrid Recall)

在原有三层记忆基础上，项目进一步补充了面向 Code Agent 的长期记忆闭环：

```text
任务执行 → RawObservation 事件记录 → MemoryItem 结构化长期记忆
       → Hybrid Recall 混合检索 → MemoryContextBuilder 预算控制注入
       → 后续任务决策 / 文件编辑 / 错误修复
```

新增能力包括：

- **统一入口**：`core/memory_manager.py` 负责观察、保存、召回、统计和 Prompt 记忆上下文构建
- **结构化长期记忆**：`core/memory_items.py` 提供 `MemoryItem` 与 `RawObservation`，不再只依赖会话摘要溢出
- **主动记忆保存**：任务完成、工具调用、文件修改、测试失败、错误修复等关键事件可沉淀为长期经验
- **Hybrid Recall**：统一检索 `MemoryItem`、会话摘要和历史观察，并融合关键词、BM25、重要性、时间新鲜度和来源等信号
- **文件/错误历史召回**：编辑文件前可查询 `memory_file_history`，测试或工具失败后可查询 `memory_error_history`
- **Prompt 预算控制**：`MemoryContextBuilder` 对召回结果排序、去重、截断，避免长期记忆污染当前上下文
- **工具化接口**：提供 `memory_save`、`memory_recall`、`memory_file_history`、`memory_error_history`、`memory_stats`

> 详细说明见：[`docs/new_long_term_memory_system.md`](docs/new_long_term_memory_system.md)
> 重构路线见：[`docs/memory_system_refactor_roadmap.md`](docs/memory_system_refactor_roadmap.md)

### 8. 🪟 工业级环境兼容性

- **编码自愈**：自动识别 Windows CMD 乱码，内置 `chcp 65001` 与 `chardet` 解码
- **换行符归一化**：`edit_file` 自动处理 CRLF/LF 差异
- **原子化防错**：规避 Shell echo 嵌套引号陷阱

---

## 🛠️ 工具生态

| 工具 | 功能 | 核心特性 |
|:---|:---|:---|
| `execute_bash` | 执行 Shell 命令 | Windows 编码自愈、错误流捕获 |
| `read_file` | 读取文件内容 | 行范围选择、raw_mode 无装饰输出 |
| `edit_file` | 精准修改文件 | 局部替换、CRLF/LF 自动归一化 |
| `write_full_file` | 全量写入文件 | 创建新文件或完全覆盖 |
| `search_code` | 代码搜索 | 正则表达式、上下文行、glob 过滤 |
| `list_all_symbols` | 项目符号大纲 | AST 解析、类/函数索引 |
| `find_symbol_definition` | 符号定义定位 | 精准行号、代码片段展示 |
| `get_git_status` | Git 状态检查 | 分支/暂存区/工作区状态 |
| `commit_snapshot` | Git 快照提交 | 自动暂存与提交 |
| `git_rollback` | Git 安全回滚 | 回退到指定提交 |
| `run_pytest` | 运行 pytest 测试 | Traceback 捕获与结果解析 |
| `manage_plan` | 创建/更新任务计划 | 强制规划、状态管理 |
| `mark_task_done` | 标记任务完成 | 实时进度更新 |
| `clean_old_sessions` | 清理旧会话 | 释放磁盘空间 |
| `retrieve_memory` | 检索历史记忆 | BM25 + 语义向量混合检索 |
| `memory_save` | 保存长期记忆 | 手动沉淀用户偏好、架构决策、Bug 经验、工作流 |
| `memory_recall` | 召回长期记忆 | 基于 Hybrid Recall 查询相关历史经验 |
| `memory_file_history` | 查询文件历史 | 修改文件前召回相关变更记录和注意事项 |
| `memory_error_history` | 查询错误历史 | 测试/工具失败后召回相似错误与修复经验 |
| `memory_stats` | 记忆统计 | 查看工作记忆、情景记忆、长期记忆和 MemoryItem 数量 |

---

## 🚀 快速上手

### 1. 克隆项目并安装依赖

```bash
git clone https://github.com/zyjcode00/mini-claude-code-cli.git
cd mini-claude-code-cli
pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 API Key 和端点 URL：

```bash
cp .env.example .env
```

`.env` 内容示例：

```ini
# 必填：API Key（支持 OpenAI 兼容端点）
MINI_CLAUDE_API_KEY=sk-your-api-key-here

# 可选：自定义 API 端点 URL（默认 https://api.openai.com/v1）
MINI_CLAUDE_BASE_URL=https://api.openai.com/v1
```

> 也支持通过环境变量直接配置：`export MINI_CLAUDE_API_KEY=sk-xxx`

### 3. 启动系统

```bash
# 默认启动
python main.py

# 指定会话和模型
python main.py --session my_project --model your-model-name
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|:---|:---|:---|
| `--session` | 会话 ID（用于持久化与恢复） | `default` |
| `--model` | 使用的模型名称 | 项目默认模型 |

### REPL 交互命令

| 命令 | 说明 |
|:---|:---|
| `exit` / `quit` | 退出系统 |
| `/clear` | 清空上下文、摘要记忆和任务计划 |

---

## 🧪 测试验证

```bash
# 运行完整测试套件
pytest tests/ -v

# 运行指定模块测试
pytest tests/test_compression_engine.py -v
pytest tests/test_memory_layers.py -v
pytest tests/test_memory_manager.py -v
pytest tests/test_memory_phase3.py -v
pytest tests/test_memory_phase4.py -v
pytest tests/test_memory_phase5.py -v
```

**测试覆盖范围**：

- ✅ Git 工具功能与影子分支系统
- ✅ 文件读写与编码兼容性
- ✅ 代码搜索与 AST 符号导航
- ✅ 会话恢复与隔离
- ✅ 压缩引擎 (4 种策略)
- ✅ 三层记忆流转与持久化
- ✅ MemoryManager 统一编排、MemoryItem 长期记忆与主动保存
- ✅ Hybrid Recall、文件历史召回、错误历史召回与 Prompt 预算控制
- ✅ BM25 检索器与记忆召回
- ✅ Plan 状态管理与中断恢复

---

## 📅 开发里程碑

### ✅ 已完成

- [x] **Phase 1-4**：基础 ReAct 循环与工具调用体系
- [x] **PlanManager**：显式任务规划与实时看板
- [x] **AsyncIO 引擎**：并发工具调用重构
- [x] **Pytest 闭环**：确定性自愈与 TDD 流程
- [x] **上下文压缩**：滑动窗口 + 递归式摘要
- [x] **Git 影子分支**：原子快照与物理回滚
- [x] **AST 符号地图**：全局类/函数索引与精准导航
- [x] **智能压缩引擎**：4 策略自适应切换
- [x] **三层记忆架构**：工作记忆 → 情景记忆 → 长期记忆
- [x] **长期记忆重构 Phase 1-5**：MemoryManager、MemoryItem、主动保存、Hybrid Recall、Prompt 预算注入
- [x] **记忆工具化接口**：memory_save / memory_recall / memory_file_history / memory_error_history / memory_stats
- [x] **检索增强**：BM25 + 关键词 + 重要性/新鲜度融合的混合召回
- [x] **工业级兼容**：Windows 编码自愈与 CRLF/LF 归一化

### 🚧 计划中

- [ ] 多文件依赖图分析
- [ ] SWE-bench 自动评测流水线
- [ ] 多模型切换与负载均衡
- [ ] 可视化任务看板与执行轨迹
- [ ] Docker 环境沙箱隔离
- [ ] 向量检索深度集成与可选 agentmemory REST/MCP 对接

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发规范

1. **TDD 强制约束**：任何对 `core/` 或 `tools/` 的修改必须伴随测试脚本
2. **符号查询优先**：修改核心逻辑前，先用 `find_symbol_definition` 评估影响面
3. **全局搜索评估**：跨文件重构前，必须使用 `search_code` 确认影响范围
4. **Git 保护原则**：禁止删除或破坏 `.git` 目录

### 提交规范

```
<type>(<scope>): <subject>

类型：feat / fix / docs / test / refactor / chore
范围：core / tools / tests / docs
```

---

## 📖 设计理念

> **"AI 的推理能力必须与物理环境的反馈及版本控制深度解耦又高度协作，才能解决长程软件工程中的幻觉问题。"**

五大架构原则：

1. **推理与执行分离** — LLM 负责战略规划，工具链负责物理执行
2. **确定性自愈闭环** — 测试失败 → 错误分析 → 自动修复 → 重新验证
3. **状态显式化** — 任务进度、记忆摘要、Git 状态全程可追踪
4. **容错优先** — 影子分支、原子快照、自动回滚三重保险
5. **记忆分层化与可召回** — 工作记忆 → 情景记忆 → 长期记忆，并通过 Hybrid Recall 与 Token Budget 控制服务后续任务

---

## 📄 License

[MIT License](LICENSE)

---

## 🙏 致谢

感谢 Claude (Anthropic) 提供的灵感与技术参考，以及开源社区的优秀工具库。