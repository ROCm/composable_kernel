#!/bin/bash

BUILD=build
EXE=$BUILD/bin/tile_example_hstu_attention

$EXE -v=1 -prec=fp16 -b=3 -jagged=1 -nhead=1 -hdim_qk=128 -hdim_v=128 -seqlens=49,52,55 -causal=1 -local_len=4 -context_len=3 -minfull_len=0 -targets=4,5,6 -save_mask=1
mv ck_hstu_mask.dat ck_hstu_mask_0.dat

$EXE -v=1 -prec=fp16 -b=3 -jagged=1 -nhead=1 -hdim_qk=128 -hdim_v=128 -seqlens=49,52,55 -causal=1 -local_len=4 -context_len=3 -minfull_len=6 -targets=4,5,6 -save_mask=1
mv ck_hstu_mask.dat ck_hstu_mask_1.dat
