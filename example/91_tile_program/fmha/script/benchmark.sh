#!/bin/sh
# TODO: run this script from CK root
BUILD=build
EXE=$BUILD/bin/example_fmha_fwd
VALID=0

for prec in "fp16" "bf16" ; do
for perm in 0 1 ; do
for hdim in 128 64 256 ; do

$EXE -prec=$prec -b=32 -h=16 -d=$hdim -s=512   -iperm=$perm -operm=$perm -v=$VALID ; sleep 3
$EXE -prec=$prec -b=16 -h=16 -d=$hdim -s=1024  -iperm=$perm -operm=$perm -v=$VALID ; sleep 3
$EXE -prec=$prec -b=8  -h=16 -d=$hdim -s=2048  -iperm=$perm -operm=$perm -v=$VALID ; sleep 3
$EXE -prec=$prec -b=4  -h=16 -d=$hdim -s=4096  -iperm=$perm -operm=$perm -v=$VALID ; sleep 3
$EXE -prec=$prec -b=2  -h=16 -d=$hdim -s=8192  -iperm=$perm -operm=$perm -v=$VALID ; sleep 3
$EXE -prec=$prec -b=1  -h=16 -d=$hdim -s=16384 -iperm=$perm -operm=$perm -v=$VALID ; sleep 3

done
done
done
