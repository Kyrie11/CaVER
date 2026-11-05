#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Optional arguments: preexisting result folders. No dataset/checkpoint upload.
if [[ "$#" -eq 0 ]];then
  set -- outputs/cbc_synthetic_pilot outputs/cbc_womd_pilot
fi
python scripts/package_diagnostics.py "$@" --output outputs/CaVER_CBC_report_bundle.zip
