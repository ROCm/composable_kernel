#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Usage: cmake-ck-dev.sh [--minimal|--preset=NAME] [SOURCE_DIR] [GPU_TARGET] [CMAKE_ARGS...]
#
# Flags (can appear anywhere):
#   --minimal              Use dev-minimal preset (fast ~5s vs ~150s configure)
#   --preset=NAME          Use custom CMake preset
#
# Positional arguments:
#   SOURCE_DIR             Source directory (default: ..)
#   GPU_TARGET             GPU target like gfx90a (default: gfx908;gfx90a;gfx942)
#   CMAKE_ARGS             Additional arguments passed to cmake
#
# Examples:
#   cmake-ck-dev.sh                              # Default build
#   cmake-ck-dev.sh --minimal .. gfx90a          # Fast iteration build
#   cmake-ck-dev.sh .. gfx90a --minimal          # Flags can go anywhere
#   cmake-ck-dev.sh --preset=dev-gfx942 ..       # Custom preset

# exit when a command exits with non-zero status; also when an unbound variable is referenced
set -eu
# pipefail is supported by many shells, not supported by sh and dash
set -o pipefail 2>/dev/null | true
# when treating a string as a sequence, do not split on spaces
IFS=$(printf '\n\t')

# clean the build system files
find . -name CMakeFiles     -type d -exec rm -rfv {} +
find . -name CMakeCache.txt -type f -exec rm -rv  {} +

# Default preset
PRESET="dev"
POSITIONAL_ARGS=()

# Parse all arguments, extracting flags and preserving positional args
while [ $# -gt 0 ]; do
    case "$1" in
        --minimal)
            PRESET="dev-minimal"
            echo "Using minimal preset (fast configure ~5s vs ~150s)"
            shift
            ;;
        --preset=*)
            PRESET="${1#--preset=}"
            echo "Using preset: $PRESET"
            shift
            ;;
        *)
            # Preserve positional arguments
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore positional arguments
set -- "${POSITIONAL_ARGS[@]}"

# Parse positional arguments
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

cmake "${MY_PROJECT_SOURCE}" --preset "$PRESET" -DGPU_TARGETS="$GPU_TARGETS" "${REST_ARGS[@]}"
