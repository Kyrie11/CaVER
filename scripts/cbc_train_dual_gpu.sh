#!/usr/bin/env bash
# Re-train from already-prepared simulator records; no regeneration of WOMD.
# Usage: bash scripts/cbc_train_dual_gpu.sh TRAIN_RECORDS TEST_RECORDS OUT_FOLDER [DEV_RECORDS]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
TRAIN=${1:?Usage: TRAIN_RECORDS TEST_RECORDS OUT_FOLDER [DEV_RECORDS]}
TEST=${2:?Usage: TRAIN_RECORDS TEST_RECORDS OUT_FOLDER [DEV_RECORDS]}
OUT=${3:?Usage: TRAIN_RECORDS TEST_RECORDS OUT_FOLDER [DEV_RECORDS]}
DEV=${4:-$TRAIN}
mkdir -p "$OUT"
if [[ "$DEV" == "$TRAIN" ]];then
  echo "WARNING: Boundary development set defaults to CBC training data; screen is exploratory." >&2
  echo "For stronger evidence supply a separately held-out CAL_RECORDS as the fourth argument." >&2
else
  python -m caver.cli audit-splits --train "$TRAIN" --calibration "$DEV" --test "$TEST" \
    --output "$OUT/split_audit.json" >/dev/null
fi
for f in "$TRAIN" "$TEST" "$DEV";do [[ -f "$f" ]] || { echo "Missing $f" >&2;exit 1;};done
if [[ -f "$PWD/scripts/cbc_contract_audit.py" ]];then
  if [[ "$DEV" == "$TRAIN" ]];then
    python scripts/cbc_contract_audit.py --train "$TRAIN" --test "$TEST" \
      --output "$OUT/contract_audit_pretrain.json" >/dev/null
  else
    python scripts/cbc_contract_audit.py --train "$TRAIN" --test "$TEST" --calibration "$DEV" \
      --output "$OUT/contract_audit_pretrain.json" >/dev/null
  fi
fi
EPOCHS="${EPOCHS:-65}"
GRAPH_EPOCHS="${GRAPH_EPOCHS:-$EPOCHS}"
BOOTSTRAP="${BOOTSTRAP:-2000}"
CONTRAST_WEIGHT="${CONTRAST_WEIGHT:-0.10}"
RANK_WEIGHT="${RANK_WEIGHT:-0.10}"
SEED="${SEED:-11}"
if [[ "${ALLOW_UNMATCHED_EPOCHS:-0}" != "1" && "$GRAPH_EPOCHS" != "$EPOCHS" ]];then
  echo "ERROR: anchor_full and anchor_graph must use the same optimizer updates; set GRAPH_EPOCHS=EPOCHS or ALLOW_UNMATCHED_EPOCHS=1." >&2
  exit 2
fi
GPU_COUNT=$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)
if [[ "$GPU_COUNT" -ge 2 ]];then
  echo "Training in two parallel processes on distinct CUDA devices 0 and 1"
  DEVICE=cuda:0
else
  echo "Using a single process with automatic torch device ($GPU_COUNT GPUs available)"
  DEVICE=auto
fi
train_one() {
  local variant=$1 name=$2 weight=$3 epochs=$4 gpu=$5
  if [[ "$GPU_COUNT" -ge 2 ]];then
    CUDA_VISIBLE_DEVICES="$gpu" python -m caver.cli train-cbc --input "$TRAIN" \
       --variant "$variant" --epochs "$epochs" --device "$DEVICE" \
       --contrast-weight "$weight" --rank-weight "$RANK_WEIGHT" --seed "$SEED" --output "$OUT/cbc_${name}.pt" \
       >"$OUT/train_${name}.log" 2>&1
  else
    python -m caver.cli train-cbc --input "$TRAIN" --variant "$variant" \
       --epochs "$epochs" --device "$DEVICE" --contrast-weight "$weight" --rank-weight "$RANK_WEIGHT" --seed "$SEED" \
       --output "$OUT/cbc_${name}.pt" >"$OUT/train_${name}.log" 2>&1
  fi
}
train_pair() {
  if [[ "$GPU_COUNT" -ge 2 ]];then
    train_one "$1" "$2" "$3" "$4" 0 & local p0=$!
    train_one "$5" "$6" "$7" "$8" 1 & local p1=$!
    local fail=0
    wait "$p0" || fail=1
    wait "$p1" || fail=1
    if [[ "$fail" != 0 ]];then echo 'A CUDA training job failed - see train_*.log' >&2;exit 1;fi
  else
    train_one "$1" "$2" "$3" "$4" 0
    train_one "$5" "$6" "$7" "$8" 0
  fi
}
# Four original capacity-matched controls, followed by a separate contrastive ablation.
train_pair full full 0 "$EPOCHS" no_events no_events 0 "$EPOCHS"
train_pair anchor_full anchor_full 0 "$EPOCHS" anchor_graph anchor_graph 0 "$GRAPH_EPOCHS"
train_pair anchor_graph anchor_graph_contrast "$CONTRAST_WEIGHT" "$GRAPH_EPOCHS" \
           full full_contrast "$CONTRAST_WEIGHT" "$EPOCHS"
python -m caver.cli cbc-study --input "$TEST" \
  --model "full=$OUT/cbc_full.pt" --model "no_events=$OUT/cbc_no_events.pt" \
  --model "anchor_full=$OUT/cbc_anchor_full.pt" \
  --model "anchor_graph=$OUT/cbc_anchor_graph.pt" \
  --model "anchor_graph_contrast=$OUT/cbc_anchor_graph_contrast.pt" \
  --model "full_contrast=$OUT/cbc_full_contrast.pt" \
  --contrast .20 --detect .05 --bootstrap "$BOOTSTRAP" \
  --output "$OUT/cbc_study.json" >/dev/null
python -m caver.cli cbc-boundary-study --development "$DEV" --test "$TEST" \
 --output "$OUT/cbc_boundary_study.json" >/dev/null
python -m caver.cli cbc-gate --study "$OUT/cbc_study.json" \
 --boundary "$OUT/cbc_boundary_study.json" --output "$OUT/cbc_gate.json" >/dev/null
python scripts/format_cbc_results.py --study "$OUT/cbc_study.json" \
 --output-prefix "$OUT/cbc_summary_table" >/dev/null
python scripts/package_diagnostics.py "$OUT" \
 --output "$OUT/CaVER_CBC_v3_dual_gpu_reports.zip"
echo "CBC v3 done: $OUT/CaVER_CBC_v3_dual_gpu_reports.zip"
