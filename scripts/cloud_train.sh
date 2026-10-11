#!/usr/bin/env bash
# Train the RED CELL swarm on an AMD cloud GPU (ROCm). Run inside tmux so it survives disconnects:
#   tmux new -s train 'bash scripts/cloud_train.sh'
# (no tmux? use: nohup bash scripts/cloud_train.sh > train.out 2>&1 &)
# Extra args are passed to train.py, e.g.  bash scripts/cloud_train.sh --updates 800
set -euo pipefail
cd "$(dirname "$0")/.."

# AMD cloud images keep PyTorch in /opt/venv; fall back to whatever python3 is on PATH
PY=${PY:-$( [ -x /opt/venv/bin/python ] && echo /opt/venv/bin/python || echo python3 )}
if ! "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "PyTorch with GPU support not found. Most AMD cloud images ship it preinstalled."
  echo "Otherwise install the ROCm build from https://pytorch.org/get-started/locally/ and re-run."
  exit 1
fi
"$PY" -c "import numpy" 2>/dev/null || "$PY" -m pip install -q numpy   # scipy is optional

"$PY" scripts/check_rocm.py
"$PY" -m redcell.terrain          # builds data/maps.npz (deterministic, same as the laptop's)

# The shipped Veteran: trained from random initial flight (--follow-bias 0), 160 updates (~16 min on a W7900D).
# Run one GPU job at a time: two concurrent jobs crashed the ROCm driver (hipErrorLaunchFailure).
OUT=checkpoints/amd_v3
mkdir -p "$OUT"
"$PY" -u train.py \
  --envs 4096 --rollout 64 --minibatch 4096 --updates 160 --follow-bias 0 --seed 1 \
  --mine-every 10 --mine-fresh 4096 --mine-mutate 4096 --mine-seeds 4 --pool-size 2048 --mine-chunk 8192 \
  --save-every 10 --amp --out "$OUT" "$@" 2>&1 | tee "$OUT/train.log"

"$PY" scripts/select_difficulties.py "$OUT" --seeds 24 --n 512
"$PY" scripts/recommend.py
echo "Done. Copy models/ (with models/timeline/), data/difficulties.json, data/recommended.json and data/training_metrics.jsonl back to the laptop."
