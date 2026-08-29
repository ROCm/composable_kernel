#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Live test (pure bash): rocm-smi vs amd-smi normalized output must agree.

set -euo pipefail

failures=0

require_tools() {
    local missing=0
    for tool in rocm-smi amd-smi; do
        if ! command -v "${tool}" &>/dev/null; then
            echo "ERROR: ${tool} not found on PATH"
            missing=1
        fi
    done
    if [ "${missing}" -ne 0 ]; then
        exit 1
    fi
}

run_smi() {
    local tool=$1
    shift
    "${tool}" "$@" 2>/dev/null
}

# Parse GPU ids from rocm-smi --showid (GPU[0], GPU[1], ...)
parse_rocm_gpu_ids() {
    run_smi rocm-smi --showid \
        | grep -oE 'GPU\[[0-9]+\]' \
        | sed 's/GPU\[//;s/\]//' \
        | sort -nu \
        | paste -sd, -
}

# Parse GPU ids from amd-smi list (GPU: 0, GPU: 1, ...)
parse_amd_gpu_ids() {
    run_smi amd-smi list \
        | grep '^GPU:' \
        | sed 's/^GPU:[[:space:]]*//' \
        | sort -nu \
        | paste -sd, -
}

parse_rocm_product() {
    run_smi rocm-smi --showproductname \
        | sed -n 's/.*Card Series:[[:space:]]*\(.*\)/\1/p' \
        | head -1
}

parse_amd_product() {
    run_smi amd-smi static \
        | sed -n 's/.*MARKET_NAME:[[:space:]]*\(.*\)/\1/p' \
        | head -1
}

parse_rocm_gfx() {
    run_smi rocm-smi --showproductname \
        | sed -n 's/.*GFX Version:[[:space:]]*\([^[:space:]]*\).*/\1/p' \
        | head -1
}

parse_amd_gfx() {
    run_smi amd-smi static \
        | sed -n 's/.*TARGET_GRAPHICS_VERSION:[[:space:]]*\([^[:space:]]*\).*/\1/p' \
        | head -1
}

parse_rocm_driver() {
    run_smi rocm-smi --showdriverversion \
        | sed -n 's/.*Driver version:[[:space:]]*\([^[:space:]]*\).*/\1/p' \
        | head -1
}

parse_amd_driver() {
    local ver
    ver=$(run_smi amd-smi version \
        | sed -n 's/.*amdgpu version:[[:space:]]*\([^[:space:]|]*\).*/\1/p' \
        | head -1)
    if [ -n "${ver}" ]; then
        echo "${ver}"
        return
    fi
    run_smi amd-smi static -d \
        | sed -n 's/.*VERSION:[[:space:]]*\([^[:space:]]*\).*/\1/p' \
        | head -1
}

compare_field() {
    local name=$1
    local rocm_val=$2
    local amd_val=$3
    if [ "${rocm_val}" = "${amd_val}" ]; then
        printf '  %-8s  rocm-smi=%s  amd-smi=%s  OK\n' "${name}" "${rocm_val}" "${amd_val}"
    else
        printf '  %-8s  rocm-smi=%s  amd-smi=%s  FAIL\n' "${name}" "${rocm_val}" "${amd_val}"
        failures=$((failures + 1))
    fi
}

require_tools

echo "rocm-smi vs amd-smi (live comparison, pure bash):"
echo "---------------------------------------------------"

compare_field "gpu_ids" "$(parse_rocm_gpu_ids)" "$(parse_amd_gpu_ids)"
compare_field "product" "$(parse_rocm_product)" "$(parse_amd_product)"
compare_field "gfx"     "$(parse_rocm_gfx)"     "$(parse_amd_gfx)"
compare_field "driver"  "$(parse_rocm_driver)"  "$(parse_amd_driver)"

echo ""
if [ "${failures}" -ne 0 ]; then
    echo "${failures} field(s) mismatched."
    exit 1
fi
echo "All fields match."
