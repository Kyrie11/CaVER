#!/usr/bin/env bash
# Stage the official ReactSim-Bench resources without unpickling their content.
# Does NOT run the official behavior-world-model simulation or preprocessing.
set -euo pipefail
ROOT="${REACTSIM_ROOT:-/data0/senzeyu2/dataset/ReactSim-Bench}"
DB_ROOT="${NUPLAN_DB_ROOT:-/data0/senzeyu2/dataset/nuplan/data/cache}"
MAP_ROOT="${NUPLAN_MAP_ROOT:-/data0/senzeyu2/dataset/nuplan/maps}"
REPO="$ROOT/repo"

command -v git >/dev/null || { echo 'git not installed' >&2; exit 2; }
[[ -d "$DB_ROOT" && -d "$MAP_ROOT" ]] || {
  echo "nuPlan DB or maps missing: $DB_ROOT $MAP_ROOT" >&2; exit 2;
}
python -m pip install --upgrade huggingface_hub
command -v hf >/dev/null || { echo 'HuggingFace CLI hf unavailable' >&2; exit 2; }
mkdir -p "$ROOT/data/custom_behaviors" "$ROOT/ckpts" "$ROOT/data/nuplan/nuplan-v1.1/splits/trainval"

# Official HF dataset PKLs are in the REPOSITORY ROOT, not a nested folder.
hf download zzzy0729/ReactSim-Bench-Data --repo-type dataset \
  --include '*.pkl' --local-dir "$ROOT/data/custom_behaviors"
hf download zzzy0729/ReactSim-Bench-Models --local-dir "$ROOT/ckpts"
if [[ ! -d "$REPO/.git" ]]; then
  [[ ! -e "$REPO" ]] || { echo "Existing non-Git folder: $REPO" >&2; exit 2; }
  git clone https://github.com/Thinklab-SJTU/ReactSim-Bench.git "$REPO"
else
  echo "Keeping existing repository revision in $REPO; manually inspect / git pull when desired"
fi

link_if_safe() {
  local target="$1" link="$2"
  if [[ -L "$link" ]]; then
    [[ "$(readlink -f "$link")" == "$(readlink -f "$target")" ]] || {
      echo "Refusing to override incorrect link $link" >&2; exit 2;
    }
  elif [[ -e "$link" ]]; then
    echo "Refusing to replace existing path $link" >&2; exit 2;
  else
    ln -s "$target" "$link"
  fi
}
link_if_safe "$MAP_ROOT" "$ROOT/data/nuplan/maps"
mkdir -p "$REPO/data"
link_if_safe "$ROOT/data/nuplan" "$REPO/data/nuplan"
link_if_safe "$ROOT/data/custom_behaviors" "$REPO/data/custom_behaviors"
link_if_safe "$ROOT/ckpts" "$REPO/ckpts"

DB_LINKS="$ROOT/data/nuplan/nuplan-v1.1/splits/trainval"
linked=0
for split in public_set_boston_train public_set_pitts_train public_set_sg_train public_set_vegas_train public_set_val; do
  [[ -d "$DB_ROOT/$split" ]] || continue
  while IFS= read -r -d '' file; do
    link_if_safe "$file" "$DB_LINKS/$(basename "$file")"
    linked=$((linked+1))
  done < <(find "$DB_ROOT/$split" -type f -name '*.db' -print0)
done
[[ $linked -gt 0 ]] || { echo 'No nuPlan DB files were found in the declared train/val directories' >&2; exit 2; }

printf '\nReactSim files staged: %s\nDB links inspected: %s\n' "$ROOT" "$linked"
printf 'NEXT (separate Python 3.9 env): cd "%s/reactsim_data_pipeline" && pip install -r requirements.txt && bash preprocess.sh\n' "$REPO"
printf 'WARNING: Official preprocess.sh writes data/test/nuplan_processed, and only existing matched nuPlan DB logs can be processed.\n'
