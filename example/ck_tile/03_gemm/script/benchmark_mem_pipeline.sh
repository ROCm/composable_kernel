#!/bin/sh
EXE="$(find . -name tile_example_gemm_universal -type f | head -n 1)"
VALID=0
a_layouts=("R" "C")
b_layouts=("R" "C")
c_layouts=("R" "C")

m_values=(64 512 1024 2048)
n_values=(64 512 1024 2048)
k_values=(64 512 1024 2048)

for a in "${a_layouts[@]}"; do
    for b in "${b_layouts[@]}"; do
        for c in "${c_layouts[@]}"; do
            for m in "${m_values[@]}"; do
                for n in "${n_values[@]}"; do
                    for k in "${k_values[@]}"; do
                        $EXE -prec=fp16 -b=1 -m=$m -n=$n -k=$k -a_layout="$a" -b_layout="$b" -c_layout="$c" -v=$VALID
                    done
                done
            done
        done
    done
done
