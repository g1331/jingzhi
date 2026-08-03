# 参与开发

境织使用 GitHub Issues 记录需求和缺陷。开始修改前，应先确认对应 Issue 已具备完整验收条件，且所有前置事项已经关闭。

## 本地环境

```powershell
uv sync --extra dev
uv run jingzhi
```

提交前必须完成：

```powershell
uv lock --check
uv run ruff format --check src tests
uv run ruff check src tests
uv run pytest -q
uv build
```

需要自动整理 Python 代码时运行：

```powershell
uv run ruff format src tests
```

## 分支

每个 Issue 使用一个独立分支，不直接在 `main` 上开发。

- 代理分支：`codex/issue-<编号>-<简短主题>`
- 人工分支：`issue-<编号>-<简短主题>`

主题使用小写英文和连字符，例如 `codex/issue-21-repository-engineering`。

## 提交

提交标题采用：

```text
<类型>(<范围>): <动宾短语> (#<Issue>)
```

允许的类型：`功能`、`修复`、`测试`、`重构`、`文档`、`构建`、`杂务`。

示例：

```text
功能(问答): 持久化回答的确切证据 (#4)
修复(字幕): 保留用户编辑版本的最高优先级 (#3)
构建(仓库): 建立公开仓库工程化基线 (#21)
```

正文只在需要解释原因、迁移方式或风险时添加。多行正文必须使用真实换行，不得写入字面量 `\n`。

Dependabot 自动生成的依赖更新使用 `构建(依赖)` 前缀，不要求关联人工 Issue；其他提交不得使用这一例外。

## Pull Request

- PR 标题与提交标题使用相同格式，并关联一个主要 Issue；经过 GitHub 身份校验的 Dependabot PR 除外。
- PR 正文必须包含 `Closes #<编号>`、修改内容、验证结果和剩余风险。
- 一个 PR 只交付一个主要 Issue；前置 Issue 应先单独完成。
- 合并前必须通过 `CI` 和 `PR title` 检查。
- 仓库使用 squash merge；进入 `main` 的最终提交采用该 PR 的单一提交标题和正文，多提交 PR 则采用 PR 标题。
- 合并后删除功能分支。

## 领域与架构

修改业务代码前先阅读 `CONTEXT.md` 和相关 `docs/adr/`。如果实现与现有 ADR 冲突，应先记录新的决策，不得静默覆盖原有约束。
