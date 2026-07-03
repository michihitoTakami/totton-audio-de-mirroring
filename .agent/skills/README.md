# Agent Skills

AI エージェント（Claude Code / Codex CLI など）向けスキルの単一ソース。

## 構成

```
.agent/skills/<skill-name>/
├── SKILL.md          # 必須。YAML frontmatter (name, description) + 本文
├── references/       # 任意。必要なときだけ読む詳細資料
└── agents/           # 任意。ツール固有メタデータ (例: openai.yaml)
```

- `name` はディレクトリ名と一致させる（kebab-case）
- `description` にはスキルの用途と発動条件（trigger words、日英）を書く。エージェントはこの1行で使用可否を判断する
- 本文は日本語。コマンドは実際に動く形（検証済み引数）で書く

## 各ツールからの参照

- **Claude Code**: `.claude/skills` → `../.agent/skills` の symlink 経由で自動認識（追加後は Claude Code 再起動）
- **Codex CLI**: `.codex/skills` → `../.agent/skills` の symlink。`scripts/development/install_codex_skills.sh` が `~/.codex/skills/` へ実体コピーする
- **その他のツール**: このディレクトリを直接参照するか、同様の symlink を張る

ルール類（coding-style / security / testing）も同様に `.agent/rules/` が実体で、
`.claude/rules` は symlink。

## スキルの追加・変更

1. `.agent/skills/<name>/SKILL.md` を作成・編集する（symlink 先の `.claude/skills/` を直接編集しても実体は同じ）
2. 記載するコマンドは `--help` かソースで引数を検証してから書く
3. 一時的な調査メモはスキルに書かず Issue / PR に残す
