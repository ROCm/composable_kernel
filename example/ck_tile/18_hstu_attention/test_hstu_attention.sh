#!/bin/bash

## no masking batched
bin/tile_example_hstu_attention -v=1 -prec=fp16 -b=10 -jagged=0 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlen=256 -causal=0 -local_len=0 -context_len=0 -minfull_len=0 -targets=0

## no masking jagged
bin/tile_example_hstu_attention -v=1 -prec=fp16 -b=10 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlen=300,300,290,280,310 -causal=0 -local_len=0 -context_len=0 -minfull_len=0 -targets=0

## batched causal
bin/tile_example_hstu_attention -v=1 -prec=fp16 -b=10 -jagged=0 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlen=256 -causal=1 -local_len=0 -context_len=0 -minfull_len=0 -targets=0

## jagged causal
bin/tile_example_hstu_attention -v=1 -prec=fp16 -b=10 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlen=300,300,290,280,310 -causal=1 -local_len=0 -context_len=0 -minfull_len=0 -targets=0

## batched causal+local
bin/tile_example_hstu_attention -v=1 -prec=fp16 -b=10 -jagged=0 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlen=256 -causal=1 -local_len=5 -context_len=8 -minfull_len=7 -targets=8

## jagged causal+local 
bin/tile_example_hstu_attention -v=1 -prec=fp16 -b=10 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlen=300,300,290,280,310 -causal=1 -local_len=5 -context_len=8 -minfull_len=7 -targets=8

