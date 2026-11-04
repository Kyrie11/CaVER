#!/usr/bin/env bash
# Mandatory real-data smoke; does not require tf_example/Waymax.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p outputs
python -m caver.cli doctor --output outputs/doctor.json
python -m caver.cli womd-check --format scenario --split training --decode 3 --output outputs/womd_training_check.json
python -m caver.cli womd-check --format scenario --split validation --decode 3 --output outputs/womd_validation_check.json
if [[ "${CHECK_WAYMAX:-0}" == '1' ]]; then
  python -m caver.cli womd-check --format tf_example --split training --decode 3 --output outputs/womd_tf_training_check.json
  python -m caver.cli waymax-read --split training --limit 2 --output outputs/waymax_scenes.jsonl
fi
