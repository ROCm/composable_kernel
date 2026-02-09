#!/bin/bash

BUILD=build
EXE="$BUILD/bin/tile_example_hstu_attention"

for T in "fp16" "bf16"; do
    set -x

    ## no masking batched
    $EXE -v=1 -prec=$T -b=10 -jagged=0 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=256 -seqlens_kv=300 -causal=0 -local_len=0 -context_len=0 -minfull_len=0 -targets=0 -attn_scale=0 -norm_dist=0

    ## no masking jagged
    $EXE -v=1 -prec=$T -b=10 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310 -seqlens_kv=300 -causal=0 -local_len=0 -context_len=0 -minfull_len=0 -targets=0 -attn_scale=0 -norm_dist=0

    ## batched causal
    $EXE -v=1 -prec=$T -b=10 -jagged=0 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=256 -seqlens_kv=300 -causal=1 -local_len=0 -context_len=0 -minfull_len=0 -targets=0 -attn_scale=0 -norm_dist=0

    ## jagged causal
    $EXE -v=1 -prec=$T -b=10 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310 -seqlens_kv=300 -causal=1 -local_len=0 -context_len=0 -minfull_len=0 -targets=0 -attn_scale=0 -norm_dist=0

    ## batched causal+local
    $EXE -v=1 -prec=$T -b=10 -jagged=0 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=256 -seqlens_kv=300 -causal=1 -local_len=5 -context_len=0 -minfull_len=0 -targets=0 -attn_scale=0 -norm_dist=0

    ## jagged causal+local
    $EXE -v=1 -prec=$T -b=10 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310 -seqlens_kv=300 -causal=1 -local_len=5 -context_len=0 -minfull_len=0 -targets=0 -attn_scale=0 -norm_dist=0

    ## batched causal+local+context
    $EXE -v=1 -prec=$T -b=10 -jagged=0 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=256 -seqlens_kv=300 -causal=1 -local_len=5 -context_len=8 -minfull_len=7 -targets=0 -attn_scale=0 -norm_dist=0

    ## jagged causal+local+context
    $EXE -v=1 -prec=$T -b=10 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310 -seqlens_kv=300 -causal=1 -local_len=5 -context_len=8 -minfull_len=7 -targets=0 -attn_scale=0 -norm_dist=0

    ## batched causal+local+context+target
    $EXE -v=1 -prec=$T -b=10 -jagged=0 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=256 -seqlens_kv=300 -causal=1 -local_len=5 -context_len=8 -minfull_len=7 -targets=8 -attn_scale=0 -norm_dist=0

    ## jagged causal+local+context+target
    $EXE -v=1 -prec=$T -b=10 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310 -seqlens_kv=300 -causal=1 -local_len=5 -context_len=8 -minfull_len=7 -targets=8 -attn_scale=0 -norm_dist=0

    ## jagged no-causal+local+context+target
    $EXE -v=1 -prec=$T -b=10 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310 -seqlens_kv=300 -causal=0 -local_len=5 -context_len=8 -minfull_len=7 -targets=8 -attn_scale=0 -norm_dist=0

    ## jagged causal+local+target (minfull_len > max_uih_len)
    $EXE -v=1 -prec=$T -b=10 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310 -seqlens_kv=300 -causal=1 -local_len=5 -context_len=0 -minfull_len=290 -targets=8 -attn_scale=0 -norm_dist=0

    ## jagged causal+local+context+target (minfull_len > max_uih_len)
    $EXE -v=1 -prec=$T -b=10 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310 -seqlens_kv=300 -causal=1 -local_len=5 -context_len=8 -minfull_len=290 -targets=8 -attn_scale=0 -norm_dist=0

    ## jagged no-causal+local+context+target (minfull_len > max_uih_len)
    $EXE -v=1 -prec=$T -b=10 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlens=300,300,290,280,310 -seqlens_kv=300 -causal=0 -local_len=5 -context_len=3 -minfull_len=290 -targets=8 -attn_scale=0 -norm_dist=0
    set +x
done

## This case is used to verify the masking when seqlen_kv > seqlen_q by comparing the saved mask tensor with the output of test_pytorch_hstu_mask_v2.py
$EXE -v=1 -prec=bf16 -b=3 -jagged=1 -nhead=1 -hdim_qk=128 -hdim_v=128 -seqlens=52,55,58 -seqlens_kv=70,76,80 -causal=1 -local_len=0 -context_len=0 -minfull_len=0 -targets=4,5,6 -attn_scale=0 -norm_dist=0 -save_mask=1
