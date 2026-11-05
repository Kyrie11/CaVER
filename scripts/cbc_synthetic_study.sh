#!/usr/bin/env bash
# CBC-first falsification on frozen, explicitly synthetic stress families.
# Not an empirical human counterfactual benchmark or physical certificate.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
MODE="${MODE:-pilot}"
OUT="${CBC_OUT:-outputs/cbc_synthetic_${MODE}}"
case "$MODE" in
  pilot)
    : "${TRAIN_SCENES:=12}" "${CAL_SCENES:=0}" "${TEST_SCENES:=12}" "${SOURCE_WORLDS:=2}" "${AUDIT_WORLDS:=6}"
    : "${CBC_EPOCHS:=8}" "${GRAPH_EPOCHS:=3}" "${BOOTSTRAP:=100}" ;;
  study)
    : "${TRAIN_SCENES:=100}" "${CAL_SCENES:=40}" "${TEST_SCENES:=120}" "${SOURCE_WORLDS:=8}" "${AUDIT_WORLDS:=96}"
    : "${CBC_EPOCHS:=65}" "${GRAPH_EPOCHS:=20}" "${BOOTSTRAP:=1000}" ;;
  *) echo 'MODE must be pilot or study' >&2;exit 2 ;;
esac
mkdir -p "$OUT"
CHECKPOINT_ARGS=()
printf 'CBC synthetic %s: %s train scenes, %s held-out scenes; reference K=2*%s\n' "$MODE" "$TRAIN_SCENES" "$TEST_SCENES" "$AUDIT_WORLDS"
python -m pytest -q | tee "$OUT/pytest_summary.txt"
python -m caver.cli fourcar-challenge --output "$OUT/fourcar_challenge.json" >/dev/null
python -m caver.cli collect --source synthetic-stress --split train --scenes "$TRAIN_SCENES" \
    --source-worlds "$SOURCE_WORLDS" --audit-worlds "$AUDIT_WORLDS" \
    --source-family stress_source --output "$OUT/train_records.jsonl.gz"
if [[ "$CAL_SCENES" -gt 0 ]];then
python -m caver.cli collect --source synthetic-stress --split calibration --scenes "$CAL_SCENES" \
    --source-worlds "$SOURCE_WORLDS" --audit-worlds "$AUDIT_WORLDS" \
    --source-family stress_source --output "$OUT/calibration_records.jsonl.gz"
fi
python -m caver.cli collect --source synthetic-stress --split test --scenes "$TEST_SCENES" \
    --source-worlds "$SOURCE_WORLDS" --audit-worlds "$AUDIT_WORLDS" \
    --source-family stress_source --output "$OUT/test_records.jsonl.gz"
CAL_ARGS=()
EXPECTED_CAL_ARGS=()
if [[ "$CAL_SCENES" -gt 0 ]];then
  CAL_ARGS=(--calibration "$OUT/calibration_records.jsonl.gz")
  EXPECTED_CAL_ARGS=(--expected-calibration "$CAL_SCENES")
fi
python scripts/cbc_contract_audit.py --train "$OUT/train_records.jsonl.gz" --test "$OUT/test_records.jsonl.gz" \
  "${CAL_ARGS[@]}" "${EXPECTED_CAL_ARGS[@]}" \
  --expected-train "$TRAIN_SCENES" --expected-test "$TEST_SCENES" \
  --expected-source-worlds "$SOURCE_WORLDS" --expected-audit-worlds "$AUDIT_WORLDS" \
  --output "$OUT/contract_audit.json" >/dev/null
cat > "$OUT/manifest.json" <<EOF
{"schema":"cbc_synthetic_manifest_v2","mode":"$MODE","requested_train_scenes":$TRAIN_SCENES,"requested_calibration_scenes":$CAL_SCENES,"requested_test_scenes":$TEST_SCENES,"source_worlds":$SOURCE_WORLDS,"audit_worlds_per_family":$AUDIT_WORLDS,"reference_worlds_per_action":$((2*AUDIT_WORLDS)),"reference_families":["stress_reference","stress_shifted"],"records_only":${RECORDS_ONLY:-0}}
EOF
if [[ "${RECORDS_ONLY:-0}" == "1" ]];then
  echo "Synthetic records-only finished: $OUT/{train,test}_records.jsonl.gz"
  exit 0
fi
for v in full no_events anchor_full;do
  python -m caver.cli train-cbc --input "$OUT/train_records.jsonl.gz" --variant "$v" \
    --epochs "$CBC_EPOCHS" --output "$OUT/cbc_${v}.pt"
done
python -m caver.cli train-cbc --input "$OUT/train_records.jsonl.gz" --variant anchor_graph \
    --epochs "$GRAPH_EPOCHS" --output "$OUT/cbc_anchor_graph.pt"
for cp in "$OUT"/cbc_*.pt; do CHECKPOINT_ARGS+=(--checkpoint "$cp"); done
python scripts/cbc_contract_audit.py --train "$OUT/train_records.jsonl.gz" --test "$OUT/test_records.jsonl.gz" \
  "${CAL_ARGS[@]}" "${EXPECTED_CAL_ARGS[@]}" \
  --expected-train "$TRAIN_SCENES" --expected-test "$TEST_SCENES" \
  --expected-source-worlds "$SOURCE_WORLDS" --expected-audit-worlds "$AUDIT_WORLDS" \
  "${CHECKPOINT_ARGS[@]}" --output "$OUT/contract_audit.json" >/dev/null
python -m caver.cli cbc-study --input "$OUT/test_records.jsonl.gz" \
    --model full="$OUT/cbc_full.pt" --model no_events="$OUT/cbc_no_events.pt" \
    --model anchor_full="$OUT/cbc_anchor_full.pt" --model anchor_graph="$OUT/cbc_anchor_graph.pt" \
    --contrast .20 --detect .05 --bootstrap "$BOOTSTRAP" --output "$OUT/cbc_study.json" >/dev/null
BOUNDARY_DEV="$OUT/train_records.jsonl.gz"
if [[ "$CAL_SCENES" -gt 0 ]];then BOUNDARY_DEV="$OUT/calibration_records.jsonl.gz";fi
python -m caver.cli cbc-boundary-study --development "$BOUNDARY_DEV" \
    --test "$OUT/test_records.jsonl.gz" --output "$OUT/cbc_boundary_study.json" >/dev/null
python -m caver.cli rq-audit --input "$OUT/test_records.jsonl.gz" --checkpoint "$OUT/cbc_anchor_graph.pt" \
    --output "$OUT/rq_audit.json" >/dev/null
python -m caver.cli cbc-gate --study "$OUT/cbc_study.json" \
    --boundary "$OUT/cbc_boundary_study.json" --output "$OUT/cbc_gate.json" >/dev/null
python scripts/format_cbc_results.py --study "$OUT/cbc_study.json" \
  --output-prefix "$OUT/cbc_summary_table" >/dev/null
printf '\nDone: %s/cbc_study.json, cbc_boundary_study.json and cbc_gate.json\n' "$OUT"
printf 'These are SYNTHETIC reference model comparisons, not physical safety certifications.\n'
