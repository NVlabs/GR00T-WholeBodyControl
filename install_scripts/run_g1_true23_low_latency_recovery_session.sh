#!/usr/bin/env bash
set -euo pipefail

repo_root="/mnt/z/codex/GR00T-WholeBodyControl"
run_dir="/root/g1_true23_runs/low_latency_recovery_stand_transition_dance_v2"
start_update="${1:?start update required}"
stop_update="${2:?stop update required}"

if ! [[ "${start_update}" =~ ^[0-9]+$ && "${stop_update}" =~ ^[0-9]+$ ]]; then
  echo "start/stop updates must be non-negative integers" >&2
  exit 2
fi
if (( stop_update <= start_update )); then
  echo "stop update must exceed start update" >&2
  exit 2
fi

mkdir -p "${run_dir}"
log_path="${run_dir}/train_session_$(printf '%04d' "${start_update}")_$(printf '%04d' "${stop_update}").log"
if [[ -e "${log_path}" ]]; then
  echo "refusing to overwrite session log: ${log_path}" >&2
  exit 2
fi

resume_args=()
if (( start_update > 0 )); then
  resume_path="${run_dir}/checkpoints/model_${start_update}.pt"
  if [[ ! -f "${resume_path}" ]]; then
    echo "resume checkpoint missing: ${resume_path}" >&2
    exit 2
  fi
  resume_args=(--resume "${resume_path}")
fi

cd "${repo_root}"
exec >>"${log_path}" 2>&1
echo "session_start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "session_updates=${start_update}:${stop_update}"
exec /root/.venvs/g1_true23_mjlab/bin/python -u \
  -m gear_sonic.scripts.train_g1_23dof_mjlab_low_latency_recovery \
  train \
  --run-dir "${run_dir}" \
  --num-envs 128 \
  --iterations 5001 \
  --session-updates "$((stop_update - start_update))" \
  --save-interval 250 \
  --learning-rate 0.00005 \
  --seed 20260803 \
  "${resume_args[@]}"
