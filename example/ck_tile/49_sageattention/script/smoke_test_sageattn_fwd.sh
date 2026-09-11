#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# SageAttention forward smoke tests - structure mirrors
#   example/ck_tile/01_fmha/script/smoke_test_fwd.sh
#
# Run from the ComposableKernel *build* directory (after ninja), same as FMHA:
#   cd build && ninja tile_example_sageattn_fwd
#   bash ../example/ck_tile/49_sageattention/script/smoke_test_sageattn_fwd.sh
#
# Optional: VERBOSE=1 enables bash -x. CURR_FAILS_FILE / KNOWN_FAILS_FILE override fail logs.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXE_NAME=tile_example_sageattn_fwd
EXE="$(find . -name "$EXE_NAME" -type f 2>/dev/null | head -n 1)"
KNAME=1
GPU_arch=${GPU_arch:-}
if [ -z "$GPU_arch" ]; then
    GPU_arch=$(rocminfo 2>/dev/null | grep -E 'Name:\s+gfx' | head -n1 | awk '{print $2}' || echo "unknown")
fi

export CK_WARMUP=0
export CK_REPEAT=1

CURR_FAILS_FILE=${CURR_FAILS_FILE:-"sageattn_fwd_fails_${GPU_arch}.txt"}
rm -f "$CURR_FAILS_FILE"
touch "$CURR_FAILS_FILE"
KNOWN_FAILS_FILE=${KNOWN_FAILS_FILE:-"$SCRIPT_DIR/sageattn_fwd_known_fails_${GPU_arch}.txt"}

COMMON_ARGS='-v=1 -warmup=0 -repeat=1'

if [ -z "${EXE:-}" ] || [ ! -x "$EXE" ]; then
    echo "ERROR: $EXE_NAME not found under cwd ($(pwd)). Build with: ninja $EXE_NAME" >&2
    exit 1
fi

run_exe() {
    set +e
    $EXE "$@"
    local ret=$?
    if [ $ret -ne 0 ]; then
        echo "$EXE_NAME $*" >>"$CURR_FAILS_FILE"
    fi
    set -e
}

# Core FP8xBF16 cases aligned with FMHA smoke_test_fwd.sh (lines 80-87): batch/group shapes,
# masks, GQA, short seqlen, k-only pad. Sweeps blockscale (2) vs per-warp (3) and layouts.
run_fp8bf16_smoke() {
    local qscale
    local perm
    for qscale in 2 3; do
        for perm in 0 1; do
            run_exe -prec=fp8bf16 -init=3 -qscale=$qscale -iperm=$perm -operm=$perm -vlayout=r \
                -kname=$KNAME $COMMON_ARGS -mode=0 -b=2 -h=2 -h_k=1 -d=128 -d_v=128 -s=55 -s_k=256 \
                -mask=1
            run_exe -prec=fp8bf16 -init=3 -qscale=$qscale -iperm=$perm -operm=$perm -vlayout=r \
                -kname=$KNAME $COMMON_ARGS -mode=0 -b=1 -h=3 -d=128 -s=100 -s_k=51 -mask=0
            run_exe -prec=fp8bf16 -init=3 -qscale=$qscale -iperm=$perm -operm=$perm -vlayout=r \
                -kname=$KNAME $COMMON_ARGS -mode=0 -b=2 -h=1 -d=128 -d_v=128 -s=99 -s_k=256 \
                -mask=1
            run_exe -prec=fp8bf16 -init=3 -qscale=$qscale -iperm=$perm -operm=$perm -vlayout=r \
                -kname=$KNAME $COMMON_ARGS -mode=0 -b=1 -h=2 -h_k=1 -d=128 -s=1024 -s_k=256 \
                -mask=2
            run_exe -prec=fp8bf16 -init=3 -qscale=$qscale -iperm=$perm -operm=$perm -vlayout=r \
                -kname=$KNAME $COMMON_ARGS -mode=0 -b=2 -h=1 -d=128 -d_v=128 -s=3 -s_k=99 -mask=2
            run_exe -prec=fp8bf16 -init=3 -qscale=$qscale -iperm=$perm -operm=$perm -vlayout=r \
                -kname=$KNAME $COMMON_ARGS -mode=0 -b=3 -h=2 -h_k=1 -d=128 -s=200 -s_k=520 \
                -mask=t:128,30
            run_exe -prec=fp8bf16 -init=3 -qscale=$qscale -iperm=$perm -operm=$perm -vlayout=r \
                -kname=$KNAME $COMMON_ARGS -mode=0 -b=2 -h=1 -d=128 -s=99 -s_k=32 -mask=b:4,35
            run_exe -prec=fp8bf16 -init=3 -qscale=$qscale -iperm=$perm -operm=$perm -vlayout=r \
                -kname=$KNAME $COMMON_ARGS -mode=0 -b=1 -h=2 -h_k=1 -d=128 -s=33 -s_k=0 -mask=2
            run_exe -prec=fp8bf16 -init=3 -qscale=$qscale -iperm=$perm -operm=$perm -vlayout=r \
                -kname=$KNAME $COMMON_ARGS -mode=0 -b=1 -h=2 -h_k=1 -d=128 -s=1 -s_k=10 \
                -s_kpad=32 -mask=2
        done
    done
}

# Extra FP8: explicit causal string, xformer window, per-tensor / per-thread quant, V col-major.
run_fp8bf16_extras() {
    run_exe -prec=fp8bf16 -init=3 -qscale=3 -iperm=0 -operm=0 -vlayout=r -kname=$KNAME \
        $COMMON_ARGS -mode=0 -b=4 -h=8 -d=128 -s=1024 -s_k=1024 -mask=t:-1,0
    run_exe -prec=fp8bf16 -init=3 -qscale=3 -iperm=1 -operm=1 -vlayout=c -kname=$KNAME \
        $COMMON_ARGS -mode=0 -b=2 -h=4 -d=128 -s=256 -s_k=256 -mask=t
    run_exe -prec=fp8bf16 -init=3 -qscale=3 -iperm=0 -operm=0 -vlayout=r -kname=$KNAME \
        $COMMON_ARGS -mode=0 -b=1 -h=2 -d=128 -s=256 -s_k=256 -mask=xt:64
    run_exe -prec=fp8bf16 -init=3 -qscale=1 -iperm=0 -operm=0 -vlayout=r -kname=$KNAME \
        $COMMON_ARGS -mode=0 -b=1 -h=2 -d=128 -s=128 -s_k=128 -mask=0
    run_exe -prec=fp8bf16 -init=3 -qscale=4 -iperm=0 -operm=0 -vlayout=r -kname=$KNAME \
        $COMMON_ARGS -mode=0 -b=1 -h=2 -d=128 -s=64 -s_k=64 -mask=0
}

# Group mode + physical padding (same intent as FMHA run_padding_smoke_tests, Sage-only flags).
run_group_and_padding_smoke() {
    run_exe -prec=fp8bf16 -init=3 -qscale=3 -iperm=0 -operm=0 -vlayout=r -kname=$KNAME \
        $COMMON_ARGS -mode=1 -b=3 -h=2 -h_k=1 -d=128 -s=50,60,40 -s_k=128,256,192 -mask=1
    # group + PERTHREAD: block_scale_seqstart_* must be allocated (same as bs/pw)
    run_exe -prec=fp8bf16 -init=3 -qscale=4 -iperm=0 -operm=0 -vlayout=r -kname=$KNAME \
        $COMMON_ARGS -mode=1 -b=3 -h=2 -h_k=1 -d=128 -s=50,60,40 -s_k=128,256,192 -mask=1
    run_exe -prec=fp8bf16 -init=3 -qscale=3 -iperm=0 -operm=0 -vlayout=r -kname=$KNAME \
        $COMMON_ARGS -mode=1 -b=4 -h=8 -h_k=8 -d=128 -s=1024,768,512,256 -s_k=1024,768,512,256 \
        -mask=0 -s_qpad=1152,896,576,320 -s_kpad=1152,896,576,320
    run_exe -prec=fp8bf16 -init=3 -qscale=3 -iperm=0 -operm=0 -vlayout=r -kname=$KNAME \
        $COMMON_ARGS -mode=0 -b=4 -h=8 -d=128 -s=1024 -s_k=1024 -mask=0 \
        -q_eff_lens=960,512,384,256 -kv_eff_lens=960,512,384,256
}

# BF16 (no quant): pipeline sanity only; not a shipped Sage mode (see example --help prec).
run_bf16_pipeline_smoke() {
    run_exe -prec=bf16 -init=1 -qscale=n -iperm=0 -operm=0 -vlayout=r -kname=$KNAME \
        $COMMON_ARGS -mode=0 -b=2 -h=2 -d=128 -s=128 -s_k=128 -mask=1
    run_exe -prec=bf16 -init=1 -qscale=n -iperm=1 -operm=1 -vlayout=r -kname=$KNAME \
        $COMMON_ARGS -mode=0 -b=1 -h=4 -h_k=1 -d=128 -s=256 -s_k=128 -mask=t:32,32
}

# int8 / int4 x fp8xbf16 (hdim divisible by 8 for int4)
run_int_quant_smoke() {
    run_exe -prec=i8fp8bf16 -init=3 -qscale=3 -iperm=0 -operm=0 -vlayout=r -kname=$KNAME \
        $COMMON_ARGS -mode=0 -b=2 -h=2 -d=128 -s=128 -s_k=128 -mask=1
    run_exe -prec=i4fp8bf16 -init=3 -qscale=3 -iperm=0 -operm=0 -vlayout=r -kname=$KNAME \
        $COMMON_ARGS -mode=0 -b=1 -h=2 -d=128 -s=128 -s_k=128 -mask=t
}

if [ "${VERBOSE:-0}" = 1 ]; then
    set -x
fi

run_fp8bf16_smoke
run_fp8bf16_extras
run_group_and_padding_smoke
run_bf16_pipeline_smoke
run_int_quant_smoke

set +x

new_fails_count=0
known_fails_count=0
if [ -f "$KNOWN_FAILS_FILE" ]; then
    echo "Comparing current fails ($CURR_FAILS_FILE) against known fails ($KNOWN_FAILS_FILE):"
    while IFS= read -r line; do
        if grep -Fxq "$line" "$KNOWN_FAILS_FILE"; then
            echo "Known fail: $line"
            known_fails_count=$((known_fails_count + 1))
        else
            echo "New fail: $line"
            new_fails_count=$((new_fails_count + 1))
        fi
    done <"$CURR_FAILS_FILE"
else
    new_fails_count=$(wc -l <"$CURR_FAILS_FILE")
    echo "No known fails file, all fails ($new_fails_count) are new:"
    if [ "$new_fails_count" -gt 0 ]; then
        cat "$CURR_FAILS_FILE"
    fi
fi
echo "New fails count: $new_fails_count; Known fails count: $known_fails_count"
exit $((new_fails_count != 0))
