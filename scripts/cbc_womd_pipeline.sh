#!/usr/bin/env bash
# Canonical small-output WOMD Scenario-first pilot. Never exports the raw
# 9-second WOMD future to the model, nor writes a giant scene JSONL file.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
MODE="${MODE:-pilot}"
OUT="${CBC_OUT:-outputs/cbc_womd_${MODE}}"
case "$MODE" in
 pilot)
  : "${POLICY_SCENES:=300}" "${TRAIN_SCENES:=10}" "${CAL_SCENES:=6}" "${TEST_SCENES:=10}"
  : "${MAX_SCAN:=25000}" "${MAX_SHARDS:=65}" "${PER_SHARD:=400}"
  : "${SOURCE_WORLDS:=2}" "${AUDIT_WORLDS:=5}" "${CBC_EPOCHS:=6}" "${GRAPH_EPOCHS:=2}"
  : "${BOOTSTRAP:=100}" ;;
 study)
  : "${POLICY_SCENES:=3000}" "${TRAIN_SCENES:=120}" "${CAL_SCENES:=40}" "${TEST_SCENES:=120}"
  : "${MAX_SCAN:=110000}" "${MAX_SHARDS:=380}" "${PER_SHARD:=550}"
  : "${SOURCE_WORLDS:=8}" "${AUDIT_WORLDS:=96}" "${CBC_EPOCHS:=65}" "${GRAPH_EPOCHS:=20}"
  : "${BOOTSTRAP:=1000}" ;;
 *) echo 'MODE must be pilot or study' >&2;exit 2 ;;
esac
mkdir -p "$OUT"
python -m pytest -q | tee "$OUT/pytest_summary.txt"
python -m caver.cli doctor --output "$OUT/doctor.json" >/dev/null
python -m caver.cli womd-check --format scenario --split training --decode 1 \
 --output "$OUT/womd_training_check.json" >/dev/null
python -m caver.cli womd-check --format scenario --split validation --decode 1 \
 --output "$OUT/womd_validation_check.json" >/dev/null
# Policy-pretraining observation export: current plus past, nearest nearby
# agents, NO map polyline dump; partition train only.
python -m caver.cli womd-prep --split training --partition train --purpose policy \
 --limit "$POLICY_SCENES" --max-scan "$MAX_SCAN" --max-shards "$MAX_SHARDS" \
 --per-shard "$PER_SHARD" --max-output-mb 45 --output "$OUT/womd_policy_train.jsonl.gz" >/dev/null
# Each corridor export has <=45 MiB compressed (not full training-set dump).
for entry in "train training $TRAIN_SCENES 17" "calibration training $CAL_SCENES 17" "test validation $TEST_SCENES 23";do
  read -r partition split num seed <<< "$entry"
  python -m caver.cli womd-prep --split "$split" --partition "$partition" --purpose corridor \
    --limit "$num" --max-scan "$MAX_SCAN" --max-shards "$MAX_SHARDS" \
    --per-shard "$PER_SHARD" --shard-seed "$seed" --max-output-mb 45 \
    --output "$OUT/womd_${partition}_corridor.jsonl.gz" >/dev/null
done
python -m caver.cli train-policy --input "$OUT/womd_policy_train.jsonl.gz" \
 --womd-train-partition --limit "$POLICY_SCENES" --epochs 12 --device "${POLICY_DEVICE:-auto}" --output "$OUT/policy_observed.pt"
for partition in train calibration test;do
   n="$TRAIN_SCENES"; [[ "$partition" == calibration ]] && n="$CAL_SCENES"
   [[ "$partition" == test ]] && n="$TEST_SCENES"
   python -m caver.cli collect --source womd-jsonl \
      --input "$OUT/womd_${partition}_corridor.jsonl.gz" --split "$partition" \
      --scenes "$n" --source-worlds "$SOURCE_WORLDS" --audit-worlds "$AUDIT_WORLDS" \
      --source-family learned_logged --policy-checkpoint "$OUT/policy_observed.pt" \
      --output "$OUT/${partition}_records.jsonl.gz"
done
python -m caver.cli audit-splits --train "$OUT/train_records.jsonl.gz" \
 --calibration "$OUT/calibration_records.jsonl.gz" --test "$OUT/test_records.jsonl.gz" \
 --output "$OUT/split_audit.json" >/dev/null
if [[ "${RECORDS_ONLY:-0}" == "1" ]];then
  echo "WOMD records-only finished: $OUT/{train,calibration,test}_records.jsonl.gz"
  echo "Run scripts/cbc_train_dual_gpu.sh with the three records files (train/test/out/calibration)"
  exit 0
fi
for v in full no_events anchor_full;do
  python -m caver.cli train-cbc --input "$OUT/train_records.jsonl.gz" --variant "$v" \
    --epochs "$CBC_EPOCHS" --output "$OUT/cbc_${v}.pt"
done
python -m caver.cli train-cbc --input "$OUT/train_records.jsonl.gz" --variant anchor_graph \
    --epochs "$GRAPH_EPOCHS" --output "$OUT/cbc_anchor_graph.pt"
python -m caver.cli cbc-study --input "$OUT/test_records.jsonl.gz" \
    --model full="$OUT/cbc_full.pt" --model no_events="$OUT/cbc_no_events.pt" \
    --model anchor_full="$OUT/cbc_anchor_full.pt" --model anchor_graph="$OUT/cbc_anchor_graph.pt" \
    --contrast .20 --detect .05 --bootstrap "$BOOTSTRAP" --output "$OUT/cbc_study.json" >/dev/null
python -m caver.cli cbc-boundary-study --development "$OUT/train_records.jsonl.gz" \
 --test "$OUT/test_records.jsonl.gz" --output "$OUT/cbc_boundary_study.json" >/dev/null
python -m caver.cli rq-audit --input "$OUT/test_records.jsonl.gz" --checkpoint "$OUT/cbc_anchor_graph.pt" \
 --output "$OUT/rq_audit.json" >/dev/null
python -m caver.cli cbc-gate --study "$OUT/cbc_study.json" \
 --boundary "$OUT/cbc_boundary_study.json" --output "$OUT/cbc_gate.json" >/dev/null
python scripts/format_cbc_results.py --study "$OUT/cbc_study.json" \
  --output-prefix "$OUT/cbc_summary_table" >/dev/null
printf '\nWOMD Scenario CBC-first completed. ONLY reports+manifests should be uploaded.\n'
printf 'Results: %s/cbc_gate.json (not a real counterfactual causal label).\n' "$OUT"
