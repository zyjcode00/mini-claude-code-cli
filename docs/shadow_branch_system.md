# 影子分支系统设计文档

## 概述

影子分支系统是一种 Plan 级别的 Git 分支隔离机制，旨在实现：
- **主分支历史清晰**：每个 Plan 只留下一条干净的提交记录
- **开发过程隔离**：Plan 中的所有 Task 尝试和修改都在独立的影子分支上进行
- **自动归档**：Plan 完成后自动 squash merge 并删除影子分支

## 架构设计

### 核心组件

#### 1. `start_plan_branch(plan_id)`
**位置**：`tools/git_tool.py`

**功能**：
- 创建并切换到影子分支 `agent/plan-{id}`
- 如果分支已存在，直接切换
- 记录原分支名用于后续归档

**调用时机**：
- `engine.py` 在检测到 Plan 且当前不在影子分支时创建
- 使用 Plan ID 作为分支标识符

**代码示例**：
```python
success, msg = start_plan_branch("abc12345")
# success: True/False
# msg: "✅ 已切换到影子分支: agent/plan-abc12345\n📍 原分支: main"
```

#### 2. `finalize_plan(plan_id, description)`
**位置**：`tools/git_tool.py`

**功能**：
- 暂存影子分支上的所有未提交修改
- 切换回主分支（自动检测 `main` 或 `master`）
- 执行 squash merge，将影子分支合并到主分支
- 生成干净的提交信息：`🎯 [Plan-{id}] {description}`
- 删除影子分支

**调用时机**：
- `engine.py` 检测到 Plan 全部完成时
- 所有 Task 都标记为完成后才执行

**代码示例**：
```python
success, msg = finalize_plan("abc12345", "实现用户管理功能")
# success: True/False
# msg: "✅ Plan 已归档: abc123de\n📝 🎯 [Plan-abc12345] 实现用户管理功能\n🗑️ 已删除影子分支: agent/plan-abc12345"
```

### 引擎集成

**文件**：`core/engine.py`

**状态追踪**：
```python
self.current_plan_branch = None  # 当前影子分支对应的 Plan ID
```

**Plan 循环逻辑**：
```python
while step < max_steps:
    # 1. 检测 Plan 并创建影子分支
    plan_id = self.plan_manager.get_plan_id()
    if plan_id and not self.current_plan_branch:
        success, msg = start_plan_branch(plan_id)
        if success:
            self.current_plan_branch = plan_id

    # ... 工具执行逻辑 ...

    # 2. 检测 Plan 完成并归档
    if t_name == "mark_task_done" and "✅" in str(res):
        if self.plan_manager.is_plan_complete():
            # 获取 Plan 描述
            plan_desc = self.plan_manager.current_goal
            # 归档影子分支
            success, msg = finalize_plan(self.current_plan_branch, plan_desc)
            if success:
                self.current_plan_branch = None
```

## 工作流程示例

### 单 Plan 场景

```
时间线：
main (Initial commit)
  ↓
  [创建影子分支]
agent/plan-abc12345 (Task-1 快照)
agent/plan-abc12345 (Task-2 快照)
agent/plan-abc12345 (Task-3 快照)
  ↓
  [Plan 完成，归档]
main (🎯 [Plan-abc12345] 实现功能 X)  ← 只有这一条记录
```

### 多 Plan 场景

```
时间线：
main (Initial commit)
  ↓
  [Plan 1]
agent/plan-11111111 (Task 提交...)
  ↓
  [归档 Plan 1]
main (🎯 [Plan-11111111] 实现功能 A)
  ↓
  [Plan 2]
agent/plan-22222222 (Task 提交...)
  ↓
  [归档 Plan 2]
main (🎯 [Plan-22222222] 实现功能 B)
```

## 测试验证

### 测试脚本
**位置**：`tests/test_plan_shadow_branch.py`

### 测试用例

#### Plan 级别影子分支测试
- 创建 Plan 影子分支
- 执行 3 个 Task（每个创建快照）
- 归档 Plan
- **验证**：主分支只有 2 条提交（Initial + Plan）

### 运行测试
```bash
python tests/test_plan_shadow_branch.py
```

**预期输出**：
```
🎉 所有测试通过！
```

## 优势与限制

### 优势
✅ **历史清晰**：主分支每个 Plan 只有一条记录
✅ **隔离性好**：Plan 中的所有 Task 尝试不影响主分支
✅ **自动管理**：无需手动创建/删除分支
✅ **兼容性强**：自动适配 `main` 和 `master` 分支名
✅ **减少碎片化**：避免每个 Task 都产生一条提交

### 限制
⚠️ **并发限制**：同一时间只能在一个影子分支上工作
⚠️ **合并冲突**：如果多个 Plan 修改同一文件，squash merge 可能失败
⚠️ **不可逆**：影子分支归档后会被删除（可通过 Git reflog 恢复）

## 最佳实践

### 适用场景
- ✅ 适合：需要多个 Task 协同完成的复杂功能
- ✅ 适合：希望保持主分支历史清晰的项目
- ✅ 适合：Plan 中的 Task 需要多次尝试和修改

### 不适用场景
- ❌ 不适合：多个 Plan 需要并行修改同一文件
- ❌ 不适合：需要保留详细的开发历史供回溯
- ❌ 不适合：多人协作频繁的场景（可能产生大量合并冲突）

## 未来改进方向

1. **保留影子分支**：归档后可选择保留影子分支作为备份
2. **冲突自动解决**：检测冲突并提示用户手动处理
3. **分支命名策略**：支持自定义影子分支命名规则（如 `feature/{plan-name}`）
4. **部分归档**：支持归档 Plan 中已完成的 Task

## 相关文件

- **Plan 管理**：`core/plan.py`
- **核心实现**：`tools/git_tool.py`
- **引擎集成**：`core/engine.py`
- **测试脚本**：`tests/test_plan_shadow_branch.py`
- **本文档**：`docs/shadow_branch_system.md`

---

**创建日期**：2026-04-06
**版本**：v2.0（Plan 级别）
**维护者**：Claude Code Team
