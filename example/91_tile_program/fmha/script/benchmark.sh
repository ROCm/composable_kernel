#!/bin/sh
# TODO: run this script from CK root
BUILD=build
EXE=$BUILD/bin/example_fmha_fwd
VALID=0

for prec in "fp16" "bf16" ; do
for perm in 0 1 ; do

$EXE -prec=$prec -b=32 -h=16 -d=128 -s=512   -iperm=$perm -operm=$perm -v=$VALID
$EXE -prec=$prec -b=16 -h=16 -d=128 -s=1024  -iperm=$perm -operm=$perm -v=$VALID
$EXE -prec=$prec -b=8  -h=16 -d=128 -s=2048  -iperm=$perm -operm=$perm -v=$VALID
$EXE -prec=$prec -b=4  -h=16 -d=128 -s=4096  -iperm=$perm -operm=$perm -v=$VALID
$EXE -prec=$prec -b=2  -h=16 -d=128 -s=8192  -iperm=$perm -operm=$perm -v=$VALID
$EXE -prec=$prec -b=1  -h=16 -d=128 -s=16384 -iperm=$perm -operm=$perm -v=$VALID

$EXE -prec=$prec -b=32 -h=32 -d=64 -s=512   -iperm=$perm -operm=$perm -v=$VALID
$EXE -prec=$prec -b=16 -h=32 -d=64 -s=1024  -iperm=$perm -operm=$perm -v=$VALID
$EXE -prec=$prec -b=8  -h=32 -d=64 -s=2048  -iperm=$perm -operm=$perm -v=$VALID
$EXE -prec=$prec -b=4  -h=32 -d=64 -s=4096  -iperm=$perm -operm=$perm -v=$VALID
$EXE -prec=$prec -b=2  -h=32 -d=64 -s=8192  -iperm=$perm -operm=$perm -v=$VALID
$EXE -prec=$prec -b=1  -h=32 -d=64 -s=16384 -iperm=$perm -operm=$perm -v=$VALID

done
done
