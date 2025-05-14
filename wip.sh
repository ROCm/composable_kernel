bin/ckProfiler  gemm_multiply_multiply_weight_preshuffle 1 0 0 2 0 1 8192 16384 5120 -1 -1 0 0 -1 1 20 50 512 



echo
echo
echo
bash -xc "bin/ckProfiler  gemm_multiply_multiply_weight_preshuffle 1 0 0 2 0 1  8 131072 5120 -1 -1 0 0 -1 1 20 50 512 | grep Best"
bash -xc "bin/ckProfiler  gemm_multiply_multiply_weight_preshuffle 1 0 0 2 0 1  16 131072 5120 -1 -1 0 0 -1 1 20 50 512 | grep Best"
bash -xc "bin/ckProfiler  gemm_multiply_multiply_weight_preshuffle 1 0 0 2 0 1  32 131072 5120 -1 -1 0 0 -1 1 20 50 512 | grep Best"
bash -xc "bin/ckProfiler  gemm_multiply_multiply_weight_preshuffle 1 0 0 2 0 1  64 131072 5120 -1 -1 0 0 -1 1 20 50 512 | grep Best"
bash -xc "bin/ckProfiler  gemm_multiply_multiply_weight_preshuffle 1 0 0 2 0 1  128 131072 5120 -1 -1 0 0 -1 1 20 50 512 | grep Best"
bash -xc "bin/ckProfiler  gemm_multiply_multiply_weight_preshuffle 1 0 0 2 0 1  256 131072 5120 -1 -1 0 0 -1 1 20 50 512 | grep Best"
bash -xc "bin/ckProfiler  gemm_multiply_multiply_weight_preshuffle 1 0 0 2 0 1  512 131072 5120 -1 -1 0 0 -1 1 20 50 512 | grep Best"
echo
bash -xc "bin/ckProfiler  gemm_multiply_multiply_weight_preshuffle 1 0 0 2 0 1  1024 1024 5120 -1 -1 0 0 -1 1 20 50 512 | grep Best"
bash -xc "bin/ckProfiler  gemm_multiply_multiply_weight_preshuffle 1 0 0 2 0 1  2048 1024 5120 -1 -1 0 0 -1 1 20 50 512 | grep Best"
bash -xc "bin/ckProfiler  gemm_multiply_multiply_weight_preshuffle 1 0 0 2 0 1  4096 1024 5120 -1 -1 0 0 -1 1 20 50 512 | grep Best"
bash -xc "bin/ckProfiler  gemm_multiply_multiply_weight_preshuffle 1 0 0 2 0 1  16384 1024 5120 -1 -1 0 0 -1 1 20 50 512 | grep Best"
bash -xc "bin/ckProfiler  gemm_multiply_multiply_weight_preshuffle 1 0 0 2 0 1  32768 1024 5120 -1 -1 0 0 -1 1 20 50 512 | grep Best"
echo
bash -xc "bin/ckProfiler  gemm_multiply_multiply_weight_preshuffle 1 0 0 2 0 1  32 16384 5120 -1 -1 0 0 -1 1 20 50 512 | grep Best"
bash -xc "bin/ckProfiler  gemm_multiply_multiply_weight_preshuffle 1 0 0 2 0 1  64 16384 5120 -1 -1 0 0 -1 1 20 50 512 | grep Best"
bash -xc "bin/ckProfiler  gemm_multiply_multiply_weight_preshuffle 1 0 0 2 0 1  128 16384 5120 -1 -1 0 0 -1 1 20 50 512 | grep Best"
bash -xc "bin/ckProfiler  gemm_multiply_multiply_weight_preshuffle 1 0 0 2 0 1  4096 16384 5120 -1 -1 0 0 -1 1 20 50 512 | grep Best"
bash -xc "bin/ckProfiler  gemm_multiply_multiply_weight_preshuffle 1 0 0 2 0 1  8192 16384 5120 -1 -1 0 0 -1 1 20 50 512 | grep Best"
