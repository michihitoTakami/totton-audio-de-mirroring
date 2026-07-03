#!/bin/bash
# Install repo-local agent skills (and recommended public skills) into Codex.
#
# Repo-local skills live in .agent/skills (both .claude/skills and .codex/skills
# are symlinks to it), so copies below resolve symlinks with -L.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
CODEX_HOME_DIR="${CODEX_HOME:-$HOME/.codex}"
SKILLS_DIR="$CODEX_HOME_DIR/skills"
INSTALLER="$SKILLS_DIR/.system/skill-installer/scripts/install-skill-from-github.py"
LOCAL_SKILLS_SRC="$REPO_ROOT/.agent/skills"

RECOMMENDED_SKILLS=(
    doc
    gh-address-comments
    security-best-practices
)

INSTALLED_ANY=false

if [ -f "$INSTALLER" ]; then
    PENDING_PATHS=()
    for skill in "${RECOMMENDED_SKILLS[@]}"; do
        if [ -d "$SKILLS_DIR/$skill" ]; then
            echo "Skipping $skill (already installed)"
            continue
        fi
        PENDING_PATHS+=("skills/.curated/$skill")
    done
    if [ "${#PENDING_PATHS[@]}" -gt 0 ]; then
        python3 "$INSTALLER" --repo openai/skills --path "${PENDING_PATHS[@]}"
        INSTALLED_ANY=true
    fi
else
    echo "Codex skill installer not found; skipping public skills: $INSTALLER" >&2
fi

if [ ! -d "$LOCAL_SKILLS_SRC" ]; then
    echo "Repository skills directory not found: $LOCAL_SKILLS_SRC" >&2
    exit 1
fi

mkdir -p "$SKILLS_DIR"
for skill_src in "$LOCAL_SKILLS_SRC"/*/; do
    skill_name="$(basename "$skill_src")"
    skill_dest="$SKILLS_DIR/$skill_name"
    if [ -d "$skill_dest" ]; then
        echo "Skipping $skill_name (already installed)"
        continue
    fi
    cp -RL "$skill_src" "$skill_dest"
    echo "Installed $skill_name from repository"
    INSTALLED_ANY=true
done

if ! $INSTALLED_ANY; then
    echo "All Codex skills are already installed."
    exit 0
fi

echo "Restart Codex to pick up new skills."
