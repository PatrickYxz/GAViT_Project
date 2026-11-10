#!/usr/bin/env bash
# Isolated, inference-only diagnostic. Internal worker writes the exit marker.
set -u

pair_bundle=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
pair_work=${GAVIT_PAIR_WORK:-/home/featurize/work}
pair_repo=${GAVIT_PAIR_REPO:-/home/featurize/work/GAViT_Project}
pair_python=${GAVIT_PAIR_PYTHON:-python}

if [[ "${1:-}" == "--worker" ]]; then
    pair_output=${2:?Missing worker output directory}
    if [[ ! -d "$pair_output" || -e "$pair_output/diagnostic.exit" ]]; then
        printf 'Invalid or already used output: %s\n' "$pair_output" >&2
        exit 1
    fi
    trap 'pair_rc=$?; printf "diagnostic_exit=%s\n" "$pair_rc" > "$pair_output/diagnostic.exit"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    "$pair_python" -u "$pair_bundle/tools/diagnose_bigearth_pair.py" \
        --repo "$pair_repo" \
        --data-dir "$pair_repo/bigearth_files/splits" \
        --swin-ckpt "$pair_work/swin44_3080_20260910/swin.pth" \
        --gavit-ckpt "$pair_work/gavit44_3080_20260909/gavit.pth" \
        --out "$pair_output/diagnosis" \
        --samples 4096 --seed 42 --min-focus-support 64 \
        --focus-classes 11,14,16 --examples-per-bucket 2 \
        --batch-size 32 --workers 2 --device cuda \
        --expected-counts 237871,122342,119825
    exit $?
fi

if [[ $# -ne 0 ]]; then
    printf 'Usage: bash %s\n' "${BASH_SOURCE[0]}" >&2
    exit 2
fi
pair_output=$(mktemp -d "$pair_work/pair44_diagnostic_XXXXXX") || exit 1
nohup bash "$pair_bundle/jobs/run_bigearth_pair_seed44.sh" --worker "$pair_output" \
    > "$pair_output/diagnostic.log" 2>&1 < /dev/null &
pair_pid=$!
printf '%s\n' "$pair_pid" > "$pair_output/diagnostic.pid"
printf 'PID=%s\nOutput=%s\n' "$pair_pid" "$pair_output"
printf 'Check startup and progress:\ntail -f %s/diagnostic.log\n' "$pair_output"
printf 'Check completion:\ncat %s/diagnostic.exit\n' "$pair_output"
printf 'Results after success:\n%s/diagnosis/results.zip\n' "$pair_output"
