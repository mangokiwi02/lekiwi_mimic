#!/usr/bin/env bash
# Train + evaluate BC conditions on the Far task from the same initial states (eval seed fixed in eval_policy.py).
#   scripts/run_ablation.sh NAME DATA [train_bc args...]
# Example:
#   scripts/run_ablation.sh dart300_bv datasets/dart/far_gen.hdf5 --max_demos 300 --base_vel
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=$1
DATA=$2
shift 2
EPISODES=${EPISODES:-128}
SEED=${SEED:-0}
mkdir -p outputs/ablation logs/ablation
CK=outputs/ablation/${NAME}_s${SEED}.pt
/c/lkm/Scripts/python.exe scripts/train_bc.py --data "$DATA" --seed "$SEED" --out "$CK" "$@" > "logs/ablation/${NAME}_s${SEED}.train.log" 2>&1
scripts/run.sh scripts/eval_policy.py --headless --trace --ckpt "$CK" --num_episodes "$EPISODES" --num_envs 32 \
    --out "outputs/ablation/${NAME}_s${SEED}.far_eval.json" > "logs/ablation/${NAME}_s${SEED}.eval.log" 2>&1
grep '"success_rate"' "logs/ablation/${NAME}_s${SEED}.eval.log" | sed "s/^/${NAME} seed ${SEED}:/"
