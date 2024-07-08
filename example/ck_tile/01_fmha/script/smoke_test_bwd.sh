#!/bin/sh
# TODO: run this script from CK root
BUILD=build
EXE=$BUILD/bin/tile_example_fmha_bwd
KNAME=1

export CK_WARMUP=0
export CK_REPEAT=1

COMMON_ARGS='-v=1'

for prec in "fp16" "bf16" ; do
for perm in 0 1 ; do
for hdim in 32 64 128 ; do
for mode in 0 1 ; do
for bias in "n" "e" "a"; do
for dbias in 0 1 ; do
for p_drop in 0.0 0.2; do

$EXE -prec=$prec -b=1 -h=4 -h_k=2 -d=$hdim -s=259 -bias=$bias -dbias=$dbias -p_drop=$p_drop -iperm=$perm -operm=$perm -v=1 -mode=$mode -kname=$KNAME $COMMON_ARGS
$EXE -prec=$prec -b=2 -h=2 -d=$hdim -s=516 -s_k=253 -bias=$bias -dbias=$dbias -p_drop=$p_drop -iperm=$perm -operm=$perm -v=1 -mode=$mode -kname=$KNAME $COMMON_ARGS
$EXE -prec=$prec -b=1 -h=4 -h_k=1 -d=$hdim -s=500 -s_k=251 -bias=$bias -dbias=$dbias -p_drop=$p_drop -iperm=$perm -operm=$perm -mask=1 -v=1 -mode=$mode -kname=$KNAME $COMMON_ARGS
$EXE -prec=$prec -b=1 -h=2 -d=$hdim -s=900 -s_k=258 -bias=$bias -dbias=$dbias -p_drop=$p_drop -iperm=$perm -operm=$perm -mask=2 -v=1 -mode=$mode -kname=$KNAME $COMMON_ARGS
$EXE -prec=$prec -b=2 -h=1 -d=$hdim -s=987 -s_k=219 -bias=$bias -dbias=$dbias -p_drop=$p_drop -iperm=$perm -operm=$perm -mask=t:128,30 -v=1 -mode=$mode -kname=$KNAME $COMMON_ARGS
$EXE -prec=$prec -b=2 -h=3 -h_k=1 -d=$hdim -s=244 -s_k=499 -bias=$bias -dbias=$dbias -p_drop=$p_drop -iperm=$perm -operm=$perm -mask=b:4,35 -v=1 -mode=$mode -kname=$KNAME $COMMON_ARGS

done
done
done
done
done
done
done
