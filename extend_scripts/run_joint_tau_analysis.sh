#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <dataset-directory> <output-directory>" >&2
    exit 2
fi

dataset_dir=$1
output_dir=$2
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
plot_script="${script_dir}/plot_g1_joint_tau.py"
python_bin=${PYTHON_BIN:-python3}

if [[ ! -d "$dataset_dir" ]]; then
    echo "Dataset directory does not exist: $dataset_dir" >&2
    exit 1
fi

if [[ ! -f "$plot_script" ]]; then
    echo "Plot script does not exist: $plot_script" >&2
    exit 1
fi

if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Python executable not found: $python_bin" >&2
    exit 1
fi

mapfile -d '' episode_files < <(
    find "$dataset_dir" -type f -name 'episode_*.parquet' -print0 | sort -z
)

if (( ${#episode_files[@]} == 0 )); then
    echo "No episode_*.parquet files found in: $dataset_dir" >&2
    exit 1
fi

mkdir -p -- "$output_dir"
echo "Found ${#episode_files[@]} episodes"

for index in "${!episode_files[@]}"; do
    episode_file=${episode_files[$index]}
    episode_name=$(basename -- "$episode_file" .parquet)
    episode_output="${output_dir}/${episode_name}"

    mkdir -p -- "$episode_output"
    printf '[%d/%d] %s\n' "$((index + 1))" "${#episode_files[@]}" "$episode_name"

    "$python_bin" "$plot_script" \
        --parquet-file "$episode_file" \
        --output "${episode_output}/joint_tau.png" \
        --csv "${episode_output}/joint_tau.csv" \
        --title "Unitree G1 joint tau_est — ${episode_name}"
done

echo "Done: $output_dir"
