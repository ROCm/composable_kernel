#!/bin/sh
# TODO: run this script from CK root
BUILD=build
EXE=$BUILD/bin/example_fmha_fwd
KNAME=1

for prec in "fp16" "bf16" ; do
for perm in 0 1 ; do
for vlayout in "r" "c" ; do
for hdim in 128 64 256 32 ; do
for lse in 0 1 ; do
for bias in 0 1 ; do

$EXE -prec=$prec -b=1 -h=1 -d=$hdim -s=1024 -bias=$bias -lse=$lse -iperm=$perm -operm=$perm -vlayout=$vlayout -v=1 -kname=$KNAME
$EXE -prec=$prec -b=1 -h=4 -h_k=2 -d=$hdim -s=256 -bias=$bias -lse=$lse -iperm=$perm -operm=$perm -vlayout=$vlayout -v=1 -kname=$KNAME
$EXE -prec=$prec -b=2 -h=2 -h_k=1 -d=16, -d_v=$hdim -s=55 -s_k=256 -bias=$bias -lse=$lse -iperm=$perm -operm=$perm -vlayout=$vlayout -v=1 -kname=$KNAME
$EXE -prec=$prec -b=1 -h=2 -d=$hdim -s=100 -s_k=512 -bias=$bias -lse=$lse -iperm=$perm -operm=$perm -vlayout=$vlayout -v=1 -kname=$KNAME
$EXE -prec=$prec -b=1 -h=1 -d=$hdim -s=99 -s_k=256 -bias=$bias -lse=$lse -iperm=$perm -operm=$perm -mask=1 -vlayout=$vlayout -v=1 -kname=$KNAME
$EXE -prec=$prec -b=1 -h=1 -d=$hdim -s=1024 -s_k=256 -bias=$bias -lse=$lse -iperm=$perm -operm=$perm -mask=2 -vlayout=$vlayout -v=1 -kname=$KNAME
$EXE -prec=$prec -b=1 -h=1 -d=$hdim -s=256 -s_k=512 -bias=$bias -lse=$lse -iperm=$perm -operm=$perm -mask=g:128,32 -vlayout=$vlayout -v=1 -kname=$KNAME

done
done
done
done
done
done
