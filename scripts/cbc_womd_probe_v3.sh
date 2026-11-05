#!/usr/bin/env bash
# For a controlled candidate-generation comparison only. No production claims.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
OLD="${OLD:-outputs/cbc_womd_study}"
OUT="${OUT:-outputs/cbc_womd_adaptive_pilot}"
TRAIN="${TRAIN_SCENES:-15}"
TEST="${TEST_SCENES:-15}"
SOURCE_WORLDS="${SOURCE_WORLDS:-3}"
AUDIT_WORLDS="${AUDIT_WORLDS:-10}"
mkdir -p "$OUT"
if [[ ! -f "$OLD/policy_observed.pt" ]];then
  echo "Missing $OLD/policy_observed.pt; first run canonical WOMD study (see README)" >&2
  exit 2
fi
for p in train test;do
  DATA="$OLD/womd_${p}_corridor.jsonl.gz"
  [[ -f "$DATA" ]] || { echo "Missing $DATA" >&2;exit 2;}
  python scripts/cbc_candidate_design.py --input "$DATA" --output "$OUT/${p}_candidate_design_v3.json"
  python scripts/inspect_womd_context.py --input "$DATA" --output "$OUT/${p}_womd_context_v3.json"
  scenes="$TRAIN";[[ "$p" == test ]] && scenes="$TEST"
  python -m caver.cli collect --source womd-jsonl --input "$DATA" \
    --split "$p" --scenes "$scenes" --source-worlds "$SOURCE_WORLDS" \
    --audit-worlds "$AUDIT_WORLDS" --source-family learned_logged \
    --policy-checkpoint "$OLD/policy_observed.pt" \
    --candidate-policy observed-adaptive \
    --output "$OUT/${p}_records.jsonl.gz" >"$OUT/collect_${p}.log" 2>&1
done
# Deliberately small MC is a debugging probe, NOT evidence adequate for claims.
python -m caver.cli cbc-study --input "$OUT/test_records.jsonl.gz" \
  --output "$OUT/cbc_study.json" --bootstrap 100 >/dev/null
python -m caver.cli cbc-gate --study "$OUT/cbc_study.json" \
 --output "$OUT/cbc_gate.json" >/dev/null
python scripts/package_diagnostics.py "$OUT" --output "$OUT/CaVER_CBC_v3_womd_probe.zip" \
 --log "$OUT/collect_test.log"
echo "Adaptive pilot done: $OUT/CaVER_CBC_v3_womd_probe.zip"
echo 'Never treat this small-reference experiment as real/physical evidence.'
