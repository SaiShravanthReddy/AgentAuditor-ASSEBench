#!/bin/bash
# Runs one or more AgentAuditor pipeline stages against a specific model's credentials, without
# manually copying .env_20b / .env_120b over .env each time.
#
# Usage: ./run_with_env.sh <20b|120b> <dataset> <stage> [<stage> ...]
# Example: ./run_with_env.sh 120b cnfinbench-pooled preprocess cluster demo infer_emb infer eval
#
# AgentAuditor/__main__.py's load_dotenv() call is hardcoded to read a literal `.env` in the repo
# root - there's no built-in way to point it at a different file per run. This script copies
# .env_<model> over .env before each stage instead. If an existing .env differs from the one about
# to be copied in, it's backed up to .env.bak first (last-backup-wins, not a full history) so a
# manually-configured .env is never silently lost.
set -euo pipefail

MODEL="${1:?Usage: run_with_env.sh <20b|120b> <dataset> <stage> [<stage> ...]}"
DATASET="${2:?Usage: run_with_env.sh <20b|120b> <dataset> <stage> [<stage> ...]}"
shift 2
STAGES=("$@")
if [ ${#STAGES[@]} -eq 0 ]; then
    echo "Usage: run_with_env.sh <20b|120b> <dataset> <stage> [<stage> ...]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SRC_ENV=".env_${MODEL}"
if [ ! -f "$SRC_ENV" ]; then
    echo "No such env file: $SCRIPT_DIR/$SRC_ENV (expected .env_20b or .env_120b)" >&2
    exit 1
fi

if [ -f .env ] && ! cmp -s .env "$SRC_ENV"; then
    cp .env .env.bak
    echo "Existing .env differed from $SRC_ENV - backed up to .env.bak"
fi
cp "$SRC_ENV" .env
echo "Active model config: $MODEL ($SRC_ENV -> .env)"

for STAGE in "${STAGES[@]}"; do
    echo "=== python -m AgentAuditor $DATASET $STAGE ==="
    python -m AgentAuditor "$DATASET" "$STAGE"
done
