#!/bin/bash

BUILD=build
EXE=$BUILD/bin/tile_example_hstu_attention

dtype="bf16"
hdim=128
num_batch=32
num_head=4

for seqlen in 1024 2048 4096 8192 16384 32768; do
    set -x 

    ## no causal
    $EXE -v=0 -prec=$dtype -b=$num_batch -jagged=0 -nhead=$num_head -hdim_qk=$hdim -hdim_v=$hdim -seqlens=$seqlen -causal=0 -local_len=0 -context_len=0 -minfull_len=0 -targets=0 -perf=1

    ## has causal
    $EXE -v=0 -prec=$dtype -b=$num_batch -jagged=0 -nhead=$num_head -hdim_qk=$hdim -hdim_v=$hdim -seqlens=$seqlen -causal=1 -local_len=0 -context_len=0 -minfull_len=0 -targets=0 -perf=1
    set +x
done 
