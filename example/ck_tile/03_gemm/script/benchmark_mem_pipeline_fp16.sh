#!/bin/sh
EXE="$(find . -name tile_example_gemm_universal -type f | head -n 1)"
VALID=2

for b_matrix_layout in "C"; do
    for m in "512" "1024" "2048" "8192"; do
        for n in "512" "1024" "8192"; do
            for k in "512" "1024" "8192"; do
                $EXE -prec=fp16 -m=$m -n=$n -k=$k -a_layout="R" -b_layout="$b_matrix_layout" -c_layout="R" -v=$VALID
            done
        done
    done
done
