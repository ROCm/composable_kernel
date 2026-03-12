#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# exit when a command exits with non-zero status; also when an unbound variable is referenced
set -eu
# pipefail is supported by many shells, not supported by sh and dash
set -o pipefail 2>/dev/null | true
# when treating a string as a sequence, do not split on spaces
IFS=$(printf '\n\t')

# clean the build system files
find . -name CMakeFiles     -type d -exec rm -rfv {} +
find . -name CMakeCache.txt -type f -exec rm -rv  {} +

if [ $# -ge 1 ]; then
    MY_PROJECT_SOURCE="$1"
    shift 1
else
    MY_PROJECT_SOURCE=".."
fi

GPU_TARGETS="gfx908;gfx90a;gfx942"

if [ $# -ge 1 ]; then
    case "$1" in
        gfx*)
            GPU_TARGETS="$1"
            shift 1
            echo "GPU targets provided: $GPU_TARGETS"
            REST_ARGS=("$@")
            ;;
        *)
            REST_ARGS=("$@")
            ;;
    esac
else
    REST_ARGS=("$@")
fi

cmake "${MY_PROJECT_SOURCE}" --preset dev -DGPU_TARGETS="$GPU_TARGETS" "${REST_ARGS[@]}"
