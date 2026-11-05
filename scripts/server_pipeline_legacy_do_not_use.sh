#!/usr/bin/env bash
# Research-only WOMD pipeline. Scenario protobuf is the sole authoritative
# corpus. Partition official training by conservative map group; reserve the
# entire official validation split for final untouched evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p outputs

# Edit via environment, not code patches. These are small PILOT budgets;
# do not report this experiment as benchmark-wide performance.
EXPORT_TRAIN="${EXPORT_TRAIN:-30000}"
EXPORT_VALIDATION="${EXPORT_VALIDATION:-12000}"
POLICY_SCENES="${POLICY_SCENES:-8000}"
TRAIN_SCENES="${TRAIN_SCENES:-100}"
CAL_SCENES="${CAL_SCENES:-80}"
TEST_SCENES="${TEST_SCENES:-80}"
SOURCE_WORLDS="${SOURCE_WORLDS:-8}"
AUDIT_WORLDS="${AUDIT_WORLDS:-64}"
CBC_EPOCHS="${CBC_EPOCHS:-80}"

python -m pytest -q
python -m caver.cli doctor --output outputs/server_preflight.json
python -m caver.cli womd-check --format scenario --split training --decode 3 --output outputs/womd_training_preflight.json
python -m caver.cli womd-check --format scenario --split validation --decode 3 --output outputs/womd_validation_preflight.json
# Waymax and tf_example are *not prerequisites* for this Scenario-based run.
python -m caver.cli nuplan-check --output outputs/nuplan_inventory.json
python -m caver.cli reactsim-check --output outputs/reactsim_inventory.json

python -m caver.cli womd-export --format scenario --split training --limit "$EXPORT_TRAIN" --output outputs/womd_training.jsonl
python -m caver.cli womd-export --format scenario --split validation --limit "$EXPORT_VALIDATION" --output outputs/womd_validation.jsonl

# Train the observational response model on train ONLY. Unseen calibration
# groups must never enter supervised policy fitting or CBC training.
python -m caver.cli train-policy --input outputs/womd_training.jsonl --womd-train-partition \
    --limit "$POLICY_SCENES" --epochs 15 --output outputs/logged_policy.pt
python -m caver.cli collect --source womd-jsonl --input outputs/womd_training.jsonl --split train \
    --scenes "$TRAIN_SCENES" --source-worlds "$SOURCE_WORLDS" --audit-worlds "$AUDIT_WORLDS" \
    --source-family learned_logged --policy-checkpoint outputs/logged_policy.pt --output outputs/train.jsonl
python -m caver.cli collect --source womd-jsonl --input outputs/womd_training.jsonl --split calibration \
    --scenes "$CAL_SCENES" --source-worlds "$SOURCE_WORLDS" --audit-worlds "$AUDIT_WORLDS" \
    --source-family learned_logged --policy-checkpoint outputs/logged_policy.pt --output outputs/calibration.jsonl
python -m caver.cli collect --source womd-jsonl --input outputs/womd_validation.jsonl --split test \
    --scenes "$TEST_SCENES" --source-worlds "$SOURCE_WORLDS" --audit-worlds "$AUDIT_WORLDS" \
    --source-family learned_logged --policy-checkpoint outputs/logged_policy.pt --output outputs/test.jsonl

# Halt on any exact scene-ID or heuristic common-log/map overlap.
python -m caver.cli audit-splits --train outputs/train.jsonl \
    --calibration outputs/calibration.jsonl --test outputs/test.jsonl --output outputs/split_audit.json
python -m caver.cli train-cbc --input outputs/train.jsonl --epochs "$CBC_EPOCHS" --output outputs/cbc.pt
python -m caver.cli calibrate --input outputs/calibration.jsonl --checkpoint outputs/cbc.pt \
    --alpha .1 --output outputs/calibration.json
python -m caver.cli rq-audit --input outputs/test.jsonl --checkpoint outputs/cbc.pt --output outputs/rq_audit.json
python -m caver.cli evaluate --input outputs/test.jsonl --checkpoint outputs/cbc.pt \
    --calibration outputs/calibration.json --output outputs/evaluation.json
python -m caver.cli tiv-audit --input outputs/train.jsonl --output outputs/tiv_queries.jsonl \
    --head outputs/tiv_head.pt --epochs 60
printf '\nFINISHED: OFFLINE SURROGATE EXPERIMENT ONLY, not official reactive closed-loop validation.\n'
