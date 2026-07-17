#!/bin/bash
## This script can be used the verifying the using of WarpGemm 32x32x16 which is used by hdim64 + softmax

BUILD=build
EXE="$BUILD/bin/tile_example_hstu_attention_bwd -softmax=0 -p_drop=0.2"

attn_scale=1.0
ndist=1

dtype="fp16"

for hdim in 96 64; do
    set -x 

    ## no masking batched
    $EXE -v=1 -prec=$dtype -b=10 -jagged=0 -nhead=4 -hdim_qk=$hdim -hdim_v=$hdim -seqlens=256 -causal=0 -local_len=0 -context_len=0 -minfull_len=0 -targets=0 -attn_scale=$attn_scale -norm_dist=$ndist

    ## no masking jagged
    $EXE -v=1 -prec=$dtype -b=10 -jagged=1 -nhead=4 -hdim_qk=$hdim -hdim_v=$hdim -seqlens=300,300,290,280,310 -causal=0 -local_len=0 -context_len=0 -minfull_len=0 -targets=0 -attn_scale=$attn_scale -norm_dist=$ndist

    ## batched causal
    $EXE -v=1 -prec=$dtype -b=10 -jagged=0 -nhead=4 -hdim_qk=$hdim -hdim_v=$hdim -seqlens=256 -causal=1 -local_len=0 -context_len=0 -minfull_len=0 -targets=0 -attn_scale=$attn_scale -norm_dist=$ndist

    ## jagged causal
    $EXE -v=1 -prec=$dtype -b=10 -jagged=1 -nhead=4 -hdim_qk=$hdim -hdim_v=$hdim -seqlens=300,300,290,280,310 -causal=1 -local_len=0 -context_len=0 -minfull_len=0 -targets=0 -attn_scale=$attn_scale -norm_dist=$ndist

    ## batched causal+local
    $EXE -v=1 -prec=$dtype -b=10 -jagged=0 -nhead=4 -hdim_qk=$hdim -hdim_v=$hdim -seqlens=256 -causal=1 -local_len=5 -context_len=0 -minfull_len=0 -targets=0 -attn_scale=$attn_scale -norm_dist=$ndist

    ## jagged causal+local 
    $EXE -v=1 -prec=$dtype -b=10 -jagged=1 -nhead=4 -hdim_qk=$hdim -hdim_v=$hdim -seqlens=300,300,290,280,310 -causal=1 -local_len=5 -context_len=0 -minfull_len=0 -targets=0 -attn_scale=$attn_scale -norm_dist=$ndist

    ## batched causal+local+context
    $EXE -v=1 -prec=$dtype -b=10 -jagged=0 -nhead=4 -hdim_qk=$hdim -hdim_v=$hdim -seqlens=256 -causal=1 -local_len=5 -context_len=8 -minfull_len=7 -targets=0 -attn_scale=$attn_scale -norm_dist=$ndist

    ## jagged causal+local+context
    $EXE -v=1 -prec=$dtype -b=10 -jagged=1 -nhead=4 -hdim_qk=$hdim -hdim_v=$hdim -seqlens=300,300,290,280,310 -causal=1 -local_len=5 -context_len=8 -minfull_len=7 -targets=0 -attn_scale=$attn_scale -norm_dist=$ndist

    ## batched causal+local+context+target
    $EXE -v=1 -prec=$dtype -b=10 -jagged=0 -nhead=4 -hdim_qk=$hdim -hdim_v=$hdim -seqlens=256 -causal=1 -local_len=5 -context_len=8 -minfull_len=7 -targets=8 -attn_scale=$attn_scale -norm_dist=$ndist

    ## jagged causal+local+context+target
    $EXE -v=1 -prec=$dtype -b=10 -jagged=1 -nhead=4 -hdim_qk=$hdim -hdim_v=$hdim -seqlens=300,300,290,280,310 -causal=1 -local_len=5 -context_len=8 -minfull_len=7 -targets=8 -attn_scale=$attn_scale -norm_dist=$ndist

    ## jagged no-causal+local+context+target
    $EXE -v=1 -prec=$dtype -b=10 -jagged=1 -nhead=4 -hdim_qk=$hdim -hdim_v=$hdim -seqlens=300,300,290,280,310 -causal=0 -local_len=5 -context_len=8 -minfull_len=7 -targets=8 -attn_scale=$attn_scale -norm_dist=$ndist

    ## jagged causal+local+target (minfull_len > max_uih_len)
    $EXE -v=1 -prec=$dtype -b=10 -jagged=1 -nhead=4 -hdim_qk=$hdim -hdim_v=$hdim -seqlens=300,300,290,280,310 -causal=1 -local_len=5 -context_len=0 -minfull_len=290 -targets=8 -attn_scale=$attn_scale -norm_dist=$ndist

    ## jagged causal+local+context+target (minfull_len > max_uih_len)
    $EXE -v=1 -prec=$dtype -b=10 -jagged=1 -nhead=4 -hdim_qk=$hdim -hdim_v=$hdim -seqlens=300,300,290,280,310 -causal=1 -local_len=5 -context_len=8 -minfull_len=290 -targets=8 -attn_scale=$attn_scale -norm_dist=$ndist

    ## jagged no-causal+local+context+target (minfull_len > max_uih_len)
    $EXE -v=1 -prec=$dtype -b=10 -jagged=1 -nhead=4 -hdim_qk=$hdim -hdim_v=$hdim -seqlens=300,300,290,280,310 -causal=0 -local_len=5 -context_len=3 -minfull_len=290 -targets=8 -attn_scale=$attn_scale -norm_dist=$ndist

    set +x
done
