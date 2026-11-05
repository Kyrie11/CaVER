#!/usr/bin/env bash
# Reuse the already-computed outputs on the user's server. No new rollouts.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
OUT="${OUT:-outputs/cbc_phase0_v3}"
WOMD="${WOMD:-outputs/cbc_womd_study}"
SYN="${SYN:-outputs/cbc_synthetic_study}"
mkdir -p "$OUT"
python -m pytest -q | tee "$OUT/pytest_summary.txt"
for entry in "synthetic:$SYN" "womd:$WOMD";do
  label=${entry%%:*};folder=${entry#*:}
  if [[ -f "$folder/cbc_study.json" ]];then
    python -m caver.cli cbc-gate --study "$folder/cbc_study.json" \
      --boundary "$folder/cbc_boundary_study.json" \
      --output "$OUT/${label}_cbc_gate_v3.json" >/dev/null
  else
    echo "Missing old report: $folder/cbc_study.json" >&2
  fi
done
for part in train test;do
  DATA="$WOMD/womd_${part}_corridor.jsonl.gz"
  if [[ -f "$DATA" ]];then
    python scripts/inspect_womd_context.py --input "$DATA" --output "$OUT/${part}_womd_context_v3.json"
    python scripts/cbc_candidate_design.py --input "$DATA" --output "$OUT/${part}_candidate_design_v3.json"
  else
    echo "No cached $DATA - this stage needs the source .jsonl.gz on your server" >&2
  fi
done
python scripts/package_diagnostics.py "$OUT" --output "$OUT/CaVER_CBC_v3_phase0_bundle.zip"
echo 'Phase0 done. Upload outputs/cbc_phase0_v3/CaVER_CBC_v3_phase0_bundle.zip'
