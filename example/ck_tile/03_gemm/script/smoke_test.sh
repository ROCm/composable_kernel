#!/bin/bash
EXE="$(find . -name tile_example_gemm_basic -type f | head -n 1)"
KNAME=1

export CK_WARMUP=0
export CK_REPEAT=1

COMMON_ARGS='-v_gpu=1 -warmup=0 -repeat=1'

run_fp16_tests() {
    for batch in 1 2 ; do
    for m in 128 1024 ;  do
    for n in 128 2048 ; do
    for k in 32 64 ; do

    $EXE -b=$batch -m=$m -n=$n -k=$k -stride_a=0 -stride_b=0 -stride_c=0 -v_cpu=0 -v_gpu=1 -e=1e-5 -prec=fp16 $COMMON_ARGS
    done ; done ;  done ; done;
}

set -x

run_fp16_tests

set +x