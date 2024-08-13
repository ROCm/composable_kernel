#!/bin/sh
# TODO: run this script from CK root
BUILD=build
EXE=$BUILD/bin/tile_example_fmha_bwd
VALID=0

for prec in "fp16" "bf16" ; do
for perm in 0 1 ; do
for hdim in 32 64 128 ; do

nhead=$((2048 / $hdim))     # follow fav2 setup
$EXE -prec=$prec -b=32 -h=$nhead -d=$hdim -s=512   -iperm=$perm -operm=$perm -kname=1 -v=$VALID ; sleep 3
$EXE -prec=$prec -b=16 -h=$nhead -d=$hdim -s=1024  -iperm=$perm -operm=$perm -kname=1 -v=$VALID ; sleep 3
$EXE -prec=$prec -b=8  -h=$nhead -d=$hdim -s=2048  -iperm=$perm -operm=$perm -kname=1 -v=$VALID ; sleep 3
$EXE -prec=$prec -b=4  -h=$nhead -d=$hdim -s=4096  -iperm=$perm -operm=$perm -kname=1 -v=$VALID ; sleep 3
$EXE -prec=$prec -b=2  -h=$nhead -d=$hdim -s=8192  -iperm=$perm -operm=$perm -kname=1 -v=$VALID ; sleep 3
$EXE -prec=$prec -b=1  -h=$nhead -d=$hdim -s=16384 -iperm=$perm -operm=$perm -kname=1 -v=$VALID ; sleep 3

done
done
done
