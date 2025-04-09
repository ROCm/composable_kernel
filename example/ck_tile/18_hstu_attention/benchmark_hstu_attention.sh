#!/bin/bash

BUILD=build
EXE=$BUILD/bin/tile_example_hstu_attention

for dtype in "fp16" "bf16"; do
    set -x 

    ## jagged is true
    $EXE -v=0 -prec=$dtype -b=80 -jagged=1 -nhead=8 -hdim_qk=128 -hdim_v=128 -seqlen=1000 -causal=1 -local_len=5 -context_len=8 -minfull_len=7 -targets=8 -perf=1

    ## jagged is false
    $EXE -v=0 -prec=$dtype -b=80 -jagged=0 -nhead=8 -hdim_qk=128 -hdim_v=128 -seqlen=1000 -causal=1 -local_len=5 -context_len=8 -minfull_len=7 -targets=8 -perf=1

    set +x
done
