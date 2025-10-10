#!/bin/bash

EXE="$(find . -name tile_example_fp4_uint8_gemm -type f | head -n 1)"
VALID=1

N1=5888
K1=3072
N2=3072
K2=2944


#m_values=(1 16 32 64 128 256 512 1024 4096 16384)
m_values=(1 16 64 256 512 1024 4096 16384)

for m in "${m_values[@]}"; do
    #echo "Running tests for m=$m"
    
    # echo "Running test with m=$m, n=$N1, k=$K1"
    $EXE -prec=pk_fp4_t -m=$m -n=$N1 -k=$K1 -v=1
    
    # echo "Running test with m=$m, n=$N2, k=$K2"
    $EXE -prec=pk_fp4_t -m=$m -n=$N2 -k=$K2 -v=1
    
    # echo "Finished tests for m=$m"
done
