#!/bin/sh
# TODO: run this script from CK root or build directory
EXE="$(find . -name tile_example_fmha_bwd -type f | head -n 1)"

for seqlen in 63 127 200; do
for prec in "bf16" "fp16" ; do
for perm in 0 1 ; do
for hdim in 64 80 96 120 128 ; do
for mask in 0 1 ; do
for rdm in 0 1 2 ; do #valid for bf16. Pls set CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT in config.hpp to the corresponding value and re-test if a small number of slight mimatchs occurred

set -x
$EXE -prec=$prec -b=3 -h=3 -d=$hdim -s=$seqlen  -iperm=$perm -operm=$perm -mask=$mask -v3_bf16_cvt=$rdm -kname=1 -bwd_v3=1 -v3_atomic_fp32=1 -mode=1 -v=1 ;
set +x

done
done
done
done
done
done
