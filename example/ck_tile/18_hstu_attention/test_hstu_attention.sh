#!/bin/bash

bin/tile_example_hstu_attention -v=1 -prec=bf16 -b=10 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlen=750,730,733 -causal=1 -local_len=5 -context_len=6 -minfull_len=6
