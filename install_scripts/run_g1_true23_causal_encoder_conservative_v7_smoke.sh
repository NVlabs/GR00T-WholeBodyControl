#!/usr/bin/env bash
set -euo pipefail

repo_root="/mnt/z/codex/GR00T-WholeBodyControl"
run_dir="/root/g1_true23_runs/causal_encoder_conservative_v7_lr1e9_smoke50"
log_dir="/root/g1_true23_run_logs"
log_path="${log_dir}/causal_encoder_conservative_v7_lr1e9_smoke50.log"

if [[ -e "${run_dir}" || -e "${log_path}" ]]; then
  echo "refusing to reuse causal encoder v7 smoke outputs" >&2
  exit 2
fi
mkdir -p "${log_dir}"
cd "${repo_root}"
exec >>"${log_path}" 2>&1
echo "session_start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exec /root/.venvs/g1_true23_mjlab/bin/python -u \
  -m gear_sonic.scripts.train_g1_23dof_mjlab_causal_encoder_conservative_v7 \
  train \
  --run-dir "${run_dir}" \
  --num-envs 64 \
  --iterations 50 \
  --session-updates 50 \
  --save-interval 1 \
  --learning-rate 0.000000001 \
  --seed 20260803
