#!/bin/bash

BUILD=build
EXE=$BUILD/bin/tile_example_hstu_attention

attn_scale=0
if [ $# -ge 1 ]; then
    attn_scale=$1
fi

for dtype in "fp16" "bf16"; do
    set -x 

    ## no masking batched
    $EXE -v=1 -prec=$dtype -b=10 -jagged=0 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=256 -causal=0 -local_len=0 -context_len=0 -minfull_len=0 -targets=0 -attn_scale=$attn_scale

    ## no masking jagged
    $EXE -v=1 -prec=$dtype -b=10 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310 -causal=0 -local_len=0 -context_len=0 -minfull_len=0 -targets=0 -attn_scale=$attn_scale

    ## batched causal
    $EXE -v=1 -prec=$dtype -b=10 -jagged=0 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=256 -causal=1 -local_len=0 -context_len=0 -minfull_len=0 -targets=0 -attn_scale=$attn_scale

    ## jagged causal
    $EXE -v=1 -prec=$dtype -b=10 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310 -causal=1 -local_len=0 -context_len=0 -minfull_len=0 -targets=0 -attn_scale=$attn_scale

    ## batched causal+local
    $EXE -v=1 -prec=$dtype -b=10 -jagged=0 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=256 -causal=1 -local_len=5 -context_len=0 -minfull_len=0 -targets=0 -attn_scale=$attn_scale

    ## jagged causal+local 
    $EXE -v=1 -prec=$dtype -b=10 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310 -causal=1 -local_len=5 -context_len=0 -minfull_len=0 -targets=0 -attn_scale=$attn_scale

    ## batched causal+local+context
    $EXE -v=1 -prec=$dtype -b=10 -jagged=0 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=256 -causal=1 -local_len=5 -context_len=8 -minfull_len=7 -targets=0 -attn_scale=$attn_scale

    ## jagged causal+local+context
    $EXE -v=1 -prec=$dtype -b=10 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310 -causal=1 -local_len=5 -context_len=8 -minfull_len=7 -targets=0 -attn_scale=$attn_scale

    ## batched causal+local+context+target
    $EXE -v=1 -prec=$dtype -b=10 -jagged=0 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=256 -causal=1 -local_len=5 -context_len=8 -minfull_len=7 -targets=8 -attn_scale=$attn_scale

    ## jagged causal+local+context+target
    $EXE -v=1 -prec=$dtype -b=10 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310 -causal=1 -local_len=5 -context_len=8 -minfull_len=7 -targets=8 -attn_scale=$attn_scale

    ## jagged no-causal+local+context+target
    $EXE -v=1 -prec=$dtype -b=10 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310 -causal=0 -local_len=5 -context_len=8 -minfull_len=7 -targets=8 -attn_scale=$attn_scale

    ## jagged causal+local+target (minfull_len > max_uih_len)
    $EXE -v=1 -prec=$dtype -b=10 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310 -causal=1 -local_len=5 -context_len=0 -minfull_len=290 -targets=8 -attn_scale=$attn_scale

    ## jagged causal+local+context+target (minfull_len > max_uih_len)
    $EXE -v=1 -prec=$dtype -b=10 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310 -causal=1 -local_len=5 -context_len=8 -minfull_len=290 -targets=8 -attn_scale=$attn_scale

    ## jagged no-causal+local+context+target (minfull_len > max_uih_len)
    $EXE -v=1 -prec=$dtype -b=10 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310 -causal=0 -local_len=5 -context_len=3 -minfull_len=290 -targets=8 -attn_scale=$attn_scale
    set +x
done
