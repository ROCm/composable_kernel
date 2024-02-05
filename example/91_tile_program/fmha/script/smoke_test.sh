#!/bin/sh
# TODO: run this script from CK root
BUILD=build
EXE=$BUILD/bin/example_fmha_fwd
KNAME=1

export CK_WARMUP=0
export CK_REPEAT=1

COMMON_ARGS='-v=1 -warmup=0 -repeat=1'
mode=0

for prec in "fp16" "bf16" ; do
# for mode in 1 0 ; do
for perm in 0 1 ; do
for vlayout in "r" "c" ; do
for hdim in 128 64 32 256 ; do
for lse in 0 1 ; do
for bias in 0 1 ; do

$EXE -prec=$prec -mode=$mode -b=1 -h=1 -d=$hdim -s=1024 -bias=$bias -lse=$lse -iperm=$perm -operm=$perm -vlayout=$vlayout -kname=$KNAME $COMMON_ARGS
$EXE -prec=$prec -mode=$mode -b=2 -h=2 -h_k=1 -d=16, -d_v=$hdim -s=55 -s_k=256 -bias=$bias -lse=$lse -iperm=$perm -operm=$perm -vlayout=$vlayout -kname=$KNAME $COMMON_ARGS
$EXE -prec=$prec -mode=$mode -b=1 -h=3 -d=$hdim -s=100 -s_k=51 -bias=$bias -lse=$lse -iperm=$perm -operm=$perm -vlayout=$vlayout -kname=$KNAME $COMMON_ARGS
$EXE -prec=$prec -mode=$mode -b=1 -h=1 -d=16 -d_v=$hdim -s=99 -s_k=256 -bias=$bias -lse=$lse -iperm=$perm -operm=$perm -mask=1 -vlayout=$vlayout -kname=$KNAME $COMMON_ARGS
$EXE -prec=$prec -mode=$mode -b=1 -h=1 -d=$hdim -s=1024 -s_k=256 -bias=$bias -lse=$lse -iperm=$perm -operm=$perm -mask=2 -vlayout=$vlayout -kname=$KNAME $COMMON_ARGS
$EXE -prec=$prec -mode=$mode -b=1 -h=1 -d=$hdim -s=256 -s_k=512 -bias=$bias -lse=$lse -iperm=$perm -operm=$perm -mask=g:128,32 -vlayout=$vlayout -kname=$KNAME $COMMON_ARGS

done
done
done
done
done
done
#done
