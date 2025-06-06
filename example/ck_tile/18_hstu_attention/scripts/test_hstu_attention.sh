#!/bin/bash

BUILD=build
EXE=$BUILD/bin/tile_example_hstu_attention

for dtype in "fp16" "bf16"; do
    set -x 

    ## no masking batched
    $EXE -v=1 -prec=$dtype -b=10 -jagged=0 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=256 -causal=0 -local_len=0 -context_len=0 -minfull_len=0 -targets=0

    ## no masking jagged
    $EXE -v=1 -prec=$dtype -b=10 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310 -causal=0 -local_len=0 -context_len=0 -minfull_len=0 -targets=0

    ## batched causal
    $EXE -v=1 -prec=$dtype -b=10 -jagged=0 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=256 -causal=1 -local_len=0 -context_len=0 -minfull_len=0 -targets=0

    ## jagged causal
    $EXE -v=1 -prec=$dtype -b=10 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310 -causal=1 -local_len=0 -context_len=0 -minfull_len=0 -targets=0

    ## batched causal+local
    $EXE -v=1 -prec=$dtype -b=10 -jagged=0 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=256 -causal=1 -local_len=5 -context_len=0 -minfull_len=0 -targets=0

    ## jagged causal+local 
    $EXE -v=1 -prec=$dtype -b=10 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310 -causal=1 -local_len=5 -context_len=0 -minfull_len=0 -targets=0

    ## batched causal+local+context
    $EXE -v=1 -prec=$dtype -b=10 -jagged=0 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=256 -causal=1 -local_len=5 -context_len=8 -minfull_len=7 -targets=0

    ## jagged causal+local+context
    $EXE -v=1 -prec=$dtype -b=10 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310 -causal=1 -local_len=5 -context_len=8 -minfull_len=7 -targets=0

    ## batched causal+local+context+target
    $EXE -v=1 -prec=$dtype -b=10 -jagged=0 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=256 -causal=1 -local_len=5 -context_len=8 -minfull_len=7 -targets=8

    ## jagged causal+local+context+target
    $EXE -v=1 -prec=$dtype -b=10 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310 -causal=1 -local_len=5 -context_len=8 -minfull_len=7 -targets=8

    set +x
done
