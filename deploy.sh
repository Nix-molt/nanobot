#!/bin/bash
# Deploy nanobot from our fork's deploy branch
# Usage: ./deploy.sh [--sync]  (--sync pulls upstream first)
set -e

FORK_DIR="$(cd "$(dirname "$0")" && pwd)"
SITE="/root/.local/share/pipx/venvs/nanobot-ai/lib/python3.12/site-packages/nanobot"

if [ "$1" = "--sync" ]; then
    echo "Syncing with upstream..."
    cd "$FORK_DIR"
    git fetch origin
    git checkout deploy
    git rebase origin/main
    git push fork deploy --force
    echo "Synced."
fi

# Our patched files
FILES=(
    "agent/loop.py"
    "agent/tools/shell.py"
    "agent/tools/filesystem.py"
    "cli/commands.py"
    "config/schema.py"
    "providers/claude_code_auth.py"
    "providers/claude_code_provider.py"
)

echo "Copying patched files..."
for f in "${FILES[@]}"; do
    cp "$FORK_DIR/nanobot/$f" "$SITE/$f"
    echo "  ✓ $f"
done

echo "Restarting nanobot..."
systemctl restart nanobot
sleep 2
systemctl status nanobot --no-pager | head -5

echo "Done! 🐈"
