#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Build the FMHA Python extension with a few kernels per API, then import it.
# This catches C++ API changes that break the bindings; no GPU is required.
# Objects and tools go to the package's build/ directory.
# Usage: check_python_build.sh [tools_dir]   (GPU_ARCHS defaults to gfx1250)

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_DIR="$SCRIPT_DIR/../python"
CHECK_DIR=$(realpath -m "${1:-$PYTHON_DIR/build/check-tools}")
PYTHON=${PYTHON:-python3}

FWD_FILTER="fmha_fwd_d128_fp16_batch_*_npad_nlogits_nbias_nmask_lse_ndropout_*"
FWD_FILTER+=",fmha_fwd_splitkv_combine_d128_fp16_batch_*_psdv_*"
FWD_FILTER+="@fmha_fwd_splitkv_d128_fp16_batch_*_psskddv_nlogits_nbias_nmask_lse_nsquant_npagedkv_*"
FWD_FILTER+=",fmha_fwd_appendkv_d128_fp16_*_vr_psskddv"
BWD_FILTER="fmha_bwd_dot_do_o_d128_fp16_*_batch_*_npad*@fmha_bwd_convert_dq_d128_fp16_*_batch_*"
BWD_FILTER+="@fmha_bwd_d128_fp16_batch_*_npad_nbias_ndbias_nmask_ndropout_ndeterministic*"

# Install build tools into a private user site; the image keeps providing PyTorch.
export PYTHONUSERBASE="$CHECK_DIR/python"
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PIP_CACHE_DIR="$CHECK_DIR/pip-cache"
export PIP_BREAK_SYSTEM_PACKAGES=1
"$PYTHON" -m pip install --user --quiet 'cmake>=3.21' ninja 'setuptools>=64' wheel

export GPU_ARCHS=${GPU_ARCHS:-gfx1250}
export MAX_JOBS=${MAX_JOBS:-$(nproc)}
export CK_TILE_FMHA_CMAKE_ARGS="-DFMHA_FWD_INSTANCE_FILTER=$FWD_FILTER -DFMHA_BWD_INSTANCE_FILTER=$BWD_FILTER"
"$PYTHON" -m pip install --user --no-build-isolation --no-deps -v "$PYTHON_DIR"

# Python loads extensions with RTLD_NOW, so this also reports unresolved symbols.
cd "$CHECK_DIR"
"$PYTHON" -c "import ck_tile_fmha, ck_tile_fmha._C as C; print('Loaded', C.__file__)"
