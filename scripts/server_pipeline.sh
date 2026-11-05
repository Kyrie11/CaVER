#!/usr/bin/env bash
# New safe default: CBC first, bounded observation exports, no giant .jsonl.
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/cbc_womd_pipeline.sh "$@"
