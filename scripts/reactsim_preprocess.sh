#!/usr/bin/env bash
# Run only inside the official ReactSim Python 3.9 preprocessing environment,
# after reviewing the included official build_cache.py and accepting data terms.
set -euo pipefail
ROOT="${REACTSIM_ROOT:-/data0/senzeyu2/dataset/ReactSim-Bench}"
REPO="$ROOT/repo"
[[ -f "$REPO/reactsim_data_pipeline/preprocess.sh" ]] || {
  echo 'Please stage the official ReactSim-Bench repository first.' >&2; exit 2;
}
cd "$REPO/reactsim_data_pipeline"
python -c 'import sys; assert sys.version_info[:2] == (3,9), "Use the official Python 3.9 preprocessing environment"'
python -m pip install -r requirements.txt
bash preprocess.sh
# Official script currently writes to data/test/nuplan_processed/.
# Make compatibility symlink without moving or overwriting processed outputs.
if [[ -d "$REPO/data/test/nuplan_processed" && ! -e "$REPO/data/nuplan_processed" ]]; then
  ln -s "$REPO/data/test/nuplan_processed" "$REPO/data/nuplan_processed"
fi
