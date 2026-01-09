#!/bin/bash

# BF16 表现最差的前 30 个 Case 测试脚本 (Forward + Backward)
# 按平均 TFLOPS 从低到高排序

BINARY="./build/bin/tile_example_grouped_gemm"

# 检查二进制文件是否存在
if [ ! -f "$BINARY" ]; then
    echo "Error: $BINARY not found!"
    echo "Please build the example first with: make tile_example_grouped_gemm -j\$(nproc)"
    exit 1
fi

echo "========================================================================================================"
echo "Running BF16 Worst 30 Cases - Grouped GEMM Benchmark (Forward + Backward)"
echo "========================================================================================================"
echo ""

# 生成重复参数的函数
repeat_param() {
    local val=$1
    local count=$2
    local result=""
    for ((i=0; i<count; i++)); do
        if [ -z "$result" ]; then
            result="$val"
        else
            result="$result,$val"
        fi
    done
    echo "$result"
}

# 运行单个测试 (Forward + Backward)
# 用法: run_test rank testid case B M N K
run_test() {
    local rank=$1
    local testid=$2
    local case_name=$3
    local B=$4
    local M=$5
    local N=$6
    local K=$7
    
    echo "========================================================================================================"
    echo "Rank $rank: $case_name (TestID=$testid)"
    echo "  B=$B, M=$M, N=$N, K=$K"
    echo "========================================================================================================"
    
    # ==================== Forward ====================
    # Forward: (M, K) @ (K, N) = (M, N)
    local fwd_Ms=$(repeat_param $M $B)
    local fwd_Ns=$(repeat_param $N $B)
    local fwd_Ks=$(repeat_param $K $B)
    local strides=$(repeat_param 0 $B)
    
    echo ""
    echo "[Forward] GEMM: M=$M, N=$N, K=$K"
    echo "  Command: $BINARY -Ms=$fwd_Ms -Ns=$fwd_Ns -Ks=$fwd_Ks -stride_As=$strides -stride_Bs=$strides -stride_Cs=$strides -group_count=$B -prec=bf16 -validate=1"
    $BINARY -Ms=$fwd_Ms -Ns=$fwd_Ns -Ks=$fwd_Ks -stride_As=$strides -stride_Bs=$strides -stride_Cs=$strides -group_count=$B -prec=bf16 -validate=1
    
    # ==================== Backward grad_A ====================
    # grad_A = grad_Y @ W^T
    # (M, N) @ (N, K) = (M, K)
    # GEMM: M=M, N=K, K=N
    local bwd_a_Ms=$(repeat_param $M $B)
    local bwd_a_Ns=$(repeat_param $K $B)
    local bwd_a_Ks=$(repeat_param $N $B)
    
    echo ""
    echo "[Backward grad_A] GEMM: M=$M, N=$K, K=$N"
    echo "  Command: $BINARY -Ms=$bwd_a_Ms -Ns=$bwd_a_Ns -Ks=$bwd_a_Ks -stride_As=$strides -stride_Bs=$strides -stride_Cs=$strides -group_count=$B -prec=bf16 -validate=1"
    $BINARY -Ms=$bwd_a_Ms -Ns=$bwd_a_Ns -Ks=$bwd_a_Ks -stride_As=$strides -stride_Bs=$strides -stride_Cs=$strides -group_count=$B -prec=bf16 -validate=1
    
    # ==================== Backward grad_B ====================
    # grad_B = X^T @ grad_Y
    # (K, M) @ (M, N) = (K, N)
    # GEMM: M=K, N=N, K=M
    local bwd_b_Ms=$(repeat_param $K $B)
    local bwd_b_Ns=$(repeat_param $N $B)
    local bwd_b_Ks=$(repeat_param $M $B)
    
    echo ""
    echo "[Backward grad_B] GEMM: M=$K, N=$N, K=$M"
    echo "  Command: $BINARY -Ms=$bwd_b_Ms -Ns=$bwd_b_Ns -Ks=$bwd_b_Ks -stride_As=$strides -stride_Bs=$strides -stride_Cs=$strides -group_count=$B -prec=bf16 -validate=1"
    $BINARY -Ms=$bwd_b_Ms -Ns=$bwd_b_Ns -Ks=$bwd_b_Ks -stride_As=$strides -stride_Bs=$strides -stride_Cs=$strides -group_count=$B -prec=bf16 -validate=1
    
    echo ""
}

# 运行所有30个测试用例
run_test 1   62  "DeepSeek-V2-Lite-Down"    2   512   2048  1408
run_test 2   61  "DeepSeek-V2-Lite-GateUP"  2   512   2816  2048
run_test 3   72  "DeepSeek-V2-Lite-Down"    4   512   2048  1408
run_test 4   64  "DeepSeek-V2-Lite-Down"    2   1024  2048  1408
run_test 5   162 "Mixtral-8x7B-Down"        1   512   4096  14336
run_test 6   102 "Qwen3-30B-A3B-Down"       4   512   2048  2048
run_test 7   71  "DeepSeek-V2-Lite-GateUP"  4   512   2816  2048
run_test 8   172 "Mixtral-8x22B-Down"       1   512   6144  16384
run_test 9   63  "DeepSeek-V2-Lite-GateUP"  2   1024  2816  2048
run_test 10  92  "Grok-2-Down"              1   512   8192  16384
run_test 11  101 "Qwen3-30B-A3B-GateUP"     4   512   4096  2048
run_test 12  82  "DeepSeek-V2-Lite-Down"    8   512   2048  1408
run_test 13  112 "Qwen3-30B-A3B-Down"       8   512   2048  2048
run_test 14  74  "DeepSeek-V2-Lite-Down"    4   1024  2048  1408
run_test 15  164 "Mixtral-8x7B-Down"        1   1024  4096  14336
run_test 16  104 "Qwen3-30B-A3B-Down"       4   1024  2048  2048
run_test 17  66  "DeepSeek-V2-Lite-Down"    2   2048  2048  1408
run_test 18  32  "DeepSeek-V2-Down"         5   512   5120  1536
run_test 19  132 "Qwen3-235B-A22B-Down"     4   512   4096  4096
run_test 20  81  "DeepSeek-V2-Lite-GateUP"  8   512   2816  2048
run_test 21  31  "DeepSeek-V2-GateUP"       5   512   3072  5120
run_test 22  42  "DeepSeek-V2-Down"         10  512   5120  1536
run_test 23  161 "Mixtral-8x7B-GateUP"      1   512   28672 4096
run_test 24  73  "DeepSeek-V2-Lite-GateUP"  4   1024  2816  2048
run_test 25  171 "Mixtral-8x22B-GateUP"     1   512   32768 6144
run_test 26  174 "Mixtral-8x22B-Down"       1   1024  6144  16384
run_test 27  84  "DeepSeek-V2-Lite-Down"    8   1024  2048  1408
run_test 28  34  "DeepSeek-V2-Down"         5   1024  5120  1536
run_test 29  65  "DeepSeek-V2-Lite-GateUP"  2   2048  2816  2048
run_test 30  83  "DeepSeek-V2-Lite-GateUP"  8   1024  2816  2048

echo "========================================================================================================"
echo "All tests completed!"
echo "========================================================================================================"
