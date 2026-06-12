#!/usr/bin/env bash
# Sync local Code-RL-Ground to gx10-a (and optionally pull remote changes back).
#   ./scripts/sync_gx10.sh        push local -> gx10-a
#   ./scripts/sync_gx10.sh pull   pull gx10-a -> local
set -euo pipefail

LOCAL="/Users/adityanarayan/Desktop/Training/Code-RL-Ground/"
REMOTE="gx10-a:~/adityaN/Code-RL-Ground/"
EXCLUDES=(--exclude '.venv/' --exclude '__pycache__/' --exclude '.git/'
          --exclude 'node_modules/' --exclude '.DS_Store' --exclude 'install.log'
          --exclude 'checkpoints/' --exclude 'cache/' --exclude 'logs/' --exclude '.cache/')

if [[ "${1:-push}" == "pull" ]]; then
    rsync -az -v "${EXCLUDES[@]}" "$REMOTE" "$LOCAL"
else
    rsync -az -v "${EXCLUDES[@]}" "$LOCAL" "$REMOTE"
fi
