#!/bin/sh
EXE="$(find . -name tile_example_gemm_universal -type f | head -n 1)"
VALID=0

a_layouts=("R")
b_layouts=("R" "C")
c_layouts=("R")

matrix_dim_combinations_1=(64 512 1024 2048)
matrix_dim_combinations_2=(512 1024 2048)


for a in "${a_layouts[@]}"; do
    for b in "${b_layouts[@]}"; do
        for c in "${c_layouts[@]}"; do
            for m in "${matrix_dim_combinations_1[@]}"; do
                for n in "${matrix_dim_combinations_2[@]}"; do
                    for k in "${matrix_dim_combinations_1[@]}"; do
                        $EXE -prec=fp16 -b=1 -m=$m -n=$n -k=$k -a_layout="$a" -b_layout="$b" -c_layout="$c" -v=$VALID
                    done
                done
            done
        done
    done
done
