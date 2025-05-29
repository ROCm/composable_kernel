#!/bin/sh
# TODO: run this script from CK root or build directory
EXE="$(find . -name tile_example_fmha_bwd -type f | head -n 1)"
VALID=0

for batch in 1 2 4 8 16 32; do
for hdim in 64 96 128 ; do
for perm in 0 1 ; do
for mask in 0 1 ; do
for v3 in 0 1 ; do

nhead=$((2048 / $hdim))     # follow fav2 setup
seqlen=$((16384 / $batch))

set -x
$EXE -prec="bf16" -b=$batch -h=$nhead -d=$hdim -s=$seqlen -iperm=$perm -operm=$perm -kname=1 -bwd_v3=$v3 -v3_atomic_fp32=1 -mask=$mask -mode=1 -v3_bf16_cvt=0 -v=$VALID ;
$EXE -prec="bf16" -b=$batch -h=$nhead -d=$hdim -s=$seqlen -iperm=$perm -operm=$perm -kname=1 -bwd_v3=$v3 -v3_atomic_fp32=1 -mask=$mask -mode=1 -v3_bf16_cvt=1 -v=$VALID ;
$EXE -prec="bf16" -b=$batch -h=$nhead -d=$hdim -s=$seqlen -iperm=$perm -operm=$perm -kname=1 -bwd_v3=$v3 -v3_atomic_fp32=1 -mask=$mask -mode=1 -v3_bf16_cvt=2 -v=$VALID ; 
$EXE -prec="fp16" -b=$batch -h=$nhead -d=$hdim -s=$seqlen -iperm=$perm -operm=$perm -kname=1 -bwd_v3=$v3 -v3_atomic_fp32=1 -mask=$mask -mode=1 -v=$VALID ; 
set +x

done
done
done
done
done

