#!/bin/sh
# TODO: run this script from CK root or build directory
EXE="$(find . -name tile_example_fmha_fwd -type f | head -n 1)"
VALID=1
prec="fp16"
bsz=1
hdim=160
sk=16384

#
nhead=40
nheadkv=2
for seqlen in 4028 2048 1024 512; do
$EXE -prec=$prec -b=$bsz -h=$nhead -h_k=$nheadkv -d=$hdim -s=$seqlen -s_k=$sk  -kname=1 -v=$VALID ; sleep 1
done


nheadkv=1
for nhead in 20 10 5;do
for seqlen in 4028 2048 1024 512; do
$EXE -prec=$prec -b=$bsz -h=$nhead -h_k=$nheadkv -d=$hdim -s=$seqlen -s_k=$sk  -kname=1 -v=$VALID ; sleep 1
done
done
