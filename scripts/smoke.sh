#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p outputs
python -m pytest -q
python -m caver.cli fourcar-challenge --output outputs/fourcar_challenge.json
python -m caver.cli collect --source synthetic --split train --scenes 5 --source-worlds 2 --audit-worlds 3 --output outputs/train.jsonl
python -m caver.cli train-cbc --input outputs/train.jsonl --epochs 6 --output outputs/cbc.pt
python -m caver.cli collect --source synthetic --split calibration --scenes 6 --source-worlds 2 --audit-worlds 3 --output outputs/calibration.jsonl
python -m caver.cli calibrate --input outputs/calibration.jsonl --checkpoint outputs/cbc.pt --alpha .1 --output outputs/calibration.json
python -m caver.cli collect --source synthetic --split test --scenes 4 --source-worlds 2 --audit-worlds 3 --output outputs/test.jsonl
python -m caver.cli audit-splits --train outputs/train.jsonl --calibration outputs/calibration.jsonl --test outputs/test.jsonl --output outputs/split_audit.json
python -m caver.cli rq-audit --input outputs/test.jsonl --checkpoint outputs/cbc.pt --output outputs/rq_audit.json
python -m caver.cli evaluate --input outputs/test.jsonl --checkpoint outputs/cbc.pt --calibration outputs/calibration.json --output outputs/evaluation.json
python -m caver.cli tiv-audit --input outputs/train.jsonl --output outputs/tiv_queries.jsonl --head outputs/tiv_head.pt --epochs 4
printf '\nSynthetic-only pipeline complete. See outputs/*.json. This is NOT evidence of real-road safety.\n'
