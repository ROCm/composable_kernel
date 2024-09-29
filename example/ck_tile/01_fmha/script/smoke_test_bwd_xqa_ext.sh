#!/bin/sh
# TODO: run this script from CK root or build directory
EXE="$(find . -name tile_example_fmha_bwd -type f | head -n 1)"
KNAME=1

export CK_WARMUP=0
export CK_REPEAT=1

COMMON_ARGS='-v=1'
set -x
for prec in "fp16" "bf16" ; do
for perm in 0 1 ; do
for hdim in 128 ; do
for mask in 0 1 ; do

$EXE -prec=$prec -b=2 -h=4 -h_k=2 -d=$hdim -s=512 -iperm=$perm -operm=$perm -mask=$mask -ext_asm=1 -asm_atomic_fp32=0 -asm_rtz_cvt=1 -mode=0 -kname=$KNAME $COMMON_ARGS
$EXE -prec=$prec -b=1 -h=3 -h_k=1 -d=$hdim -s=768 -iperm=$perm -operm=$perm -mask=$mask -ext_asm=1 -asm_atomic_fp32=0 -asm_rtz_cvt=1 -mode=0 -kname=$KNAME $COMMON_ARGS

done
done
done
done
set +x
