#!/bin/bash

BUILD=build
EXE=$BUILD/bin/tile_example_hstu_attention

$EXE -v=1 -prec=fp16 -b=3 -jagged=1 -nhead=1 -hdim_qk=128 -hdim_v=128 -seqlen=56,60,64 -causal=1 -local_len=4 -context_len=3 -minfull_len=0 -targets=4,5,6 -save_mask=1 
