---
name: worktree-pr-workflow
description: "Use to go from a GitHub Issue to a worktree/branch, implementation, commit, and PR following repo Git rules. Trigger: worktree, create branch, open PR, ブランチ作成, PR作成, ワークツリー."
---

# Worktree / PR Workflow

GitHub Issue から worktree（またはブランチ）→ 実装 → コミット → PR までを、リポジトリの Git ルールに従って進める。

## Hard Rules

- **main への直接コミット禁止**。必ず feature ブランチで作業する
- worktree / ブランチは **必ず `origin/main` から作成**（ローカル `main` からは作らない）
- `git commit --no-verify` / `git push --no-verify` は **禁止**（hooks は必ず通す）
- ブランチ名 prefix: `feat/` `fix/` `refactor/` `test/` `docs/` `perf/`

## Workflow

```bash
# 1. 最新の origin/main を取得
git fetch origin

# 2a. worktree を使う場合（worktrees/ 配下に issue 番号入りで作成）
git worktree add -b feat/issue-NNN-short-name worktrees/feat-issue-NNN-short-name origin/main

# 2b. 通常ブランチの場合
git checkout -b feat/issue-NNN-short-name origin/main

# 3. 実装 → quality-gate スキルの手順でチェック

# 4. 対象ファイルのみ明示的に add してコミット（Conventional Commits）
git add <files>
git commit -m "feat: short description

Longer explanation with physical basis if relevant.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# 5. push（pre-push hooks: mypy + pytest + C++ build/ctest が走る）
git push -u origin feat/issue-NNN-short-name

# 6. PR 作成
gh pr create --title "feat: Title" --body "..."
```

## Constraints

- コミットメッセージは Conventional Commits + `Co-Authored-By` trailer
- PR は "Squash and merge"。マージ後は `git worktree remove` / `git branch -d` で掃除
- pre-push hook が失敗したら原因を修正する（skip しない）。他人由来のテスト失敗も無視せず修正または報告
- worktree 内にも `.claude/skills` 等の symlink が必要な場合は `ln -s ../.agent/skills .claude/skills` を張り直す

## References

- `AGENTS.md` — ブランチ/worktree ルールの原典
- `CLAUDE.md` — Development Workflow セクション
