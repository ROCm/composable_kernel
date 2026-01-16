#!/bin/bash

# BF16 表现最差的前 30 个 Case 测试脚本 (Forward + Backward)
# 测试三种 config: compute_v3, memory_interwave, memory_intrawave

BINARY="./build/bin/tile_example_grouped_gemm"

# 检查二进制文件是否存在
if [ ! -f "$BINARY" ]; then
    echo "Error: $BINARY not found!"
    echo "Please build the example first with: make tile_example_grouped_gemm -j\$(nproc)"
    exit 1
fi

# 可选参数: 指定要测试的 config (默认全部测试)
# 用法: ./run_worst_30_bf16_cases.sh [config]
#   config: compute_v3, compute_v3_32x128, compute_v3_128x128, memory_intrawave, all (默认)
#   所有 config 都会测试 kbatch=1 和 kbatch=2
TEST_CONFIG=${1:-all}

echo "========================================================================================================"
echo "Running BF16 Worst 30 Cases - Grouped GEMM Benchmark (Forward + Backward)"
echo "Config: $TEST_CONFIG"
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

# 运行单个 GEMM 测试
# 用法: run_gemm config kbatch a_layout b_layout Ms Ns Ks strides B label
run_gemm() {
    local config=$1
    local kbatch=$2
    local a_layout=$3
    local b_layout=$4
    local Ms=$5
    local Ns=$6
    local Ks=$7
    local strides=$8
    local B=$9
    local label=${10}
    
    local config_arg=""
    local kbatch_arg="-kbatch=$kbatch"
    
    if [ "$config" = "compute_v3" ]; then
        config_arg=""  # default
    else
        config_arg="-config=$config"
    fi
    
    local config_name="$config"
    if [ "$kbatch" = "2" ]; then
        config_name="${config}_kb2"
    fi
    
    echo "  [$config_name] $label"
    $BINARY -Ms=$Ms -Ns=$Ns -Ks=$Ks -stride_As=$strides -stride_Bs=$strides -stride_Cs=$strides \
        -group_count=$B -prec=bf16 -validate=0 -a_layout=$a_layout -b_layout=$b_layout $config_arg $kbatch_arg 2>&1 | grep -E "Config|Perf"
}

# 运行单个测试 (Forward + Backward) 对比三种 config
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
    
    local fwd_Ms=$(repeat_param $M $B)
    local fwd_Ns=$(repeat_param $N $B)
    local fwd_Ks=$(repeat_param $K $B)
    local strides=$(repeat_param 0 $B)
    
    local bwd_a_Ms=$(repeat_param $M $B)
    local bwd_a_Ns=$(repeat_param $K $B)
    local bwd_a_Ks=$(repeat_param $N $B)
    
    local bwd_b_Ms=$(repeat_param $K $B)
    local bwd_b_Ns=$(repeat_param $N $B)
    local bwd_b_Ks=$(repeat_param $M $B)
    
    # 确定要测试的 configs
    local configs=""
    if [ "$TEST_CONFIG" = "all" ]; then
        configs="compute_v3 compute_v3_32x128 compute_v3_128x128 memory_intrawave"
    else
        configs="$TEST_CONFIG"
    fi
    
    # 测试每个 config 的 kbatch=1 和 kbatch=2
    for cfg in $configs; do
        for kbatch in 1 2; do
            local cfg_name="$cfg"
            if [ "$kbatch" = "2" ]; then
                cfg_name="${cfg}_kb2"
            fi
            
            echo ""
            echo "--- Config: $cfg_name ---"
            
            echo "[Forward] M=$M, N=$N, K=$K (a_layout=R, b_layout=C)"
            run_gemm $cfg $kbatch R C "$fwd_Ms" "$fwd_Ns" "$fwd_Ks" "$strides" $B "Forward"
            
            echo "[Backward grad_A] M=$M, N=$K, K=$N (a_layout=R, b_layout=R)"
            run_gemm $cfg $kbatch R R "$bwd_a_Ms" "$bwd_a_Ns" "$bwd_a_Ks" "$strides" $B "grad_A"
            
            echo "[Backward grad_B] M=$K, N=$N, K=$M (a_layout=C, b_layout=R)"
            run_gemm $cfg $kbatch C R "$bwd_b_Ms" "$bwd_b_Ns" "$bwd_b_Ks" "$strides" $B "grad_B"
        done
    done
    
    echo ""
}

# 运行所有30个测试用例
run_test 1   62  "DeepSeek-V2-Lite-Down"    2   1024  2048  1408
run_test 2   61  "DeepSeek-V2-Lite-GateUP"  2   1024  2816  2048
run_test 3   72  "DeepSeek-V2-Lite-Down"    4   1024  2048  1408
run_test 4   64  "DeepSeek-V2-Lite-Down"    2   2048  2048  1408
run_test 5   162 "Mixtral-8x7B-Down"        1   1024  4096  14336
run_test 6   102 "Qwen3-30B-A3B-Down"       4   1024  2048  2048
run_test 7   71  "DeepSeek-V2-Lite-GateUP"  4   1024  2816  2048
run_test 8   82  "DeepSeek-V2-Lite-Down"    8   1024  2048  1408
run_test 9   32  "DeepSeek-V2-Down"         5   1024  5120  1536
run_test 10  63  "DeepSeek-V2-Lite-GateUP"  2   2048  2816  2048
run_test 11  74  "DeepSeek-V2-Lite-Down"    4   2048  2048  1408
run_test 12  172 "Mixtral-8x22B-Down"       1   1024  6144  16384
run_test 13  161 "Mixtral-8x7B-GateUP"      1   1024  28672 4096
run_test 14  81  "DeepSeek-V2-Lite-GateUP"  8   1024  2816  2048
run_test 15  66  "DeepSeek-V2-Lite-Down"    2   4096  2048  1408
run_test 16  42  "DeepSeek-V2-Down"         10  1024  5120  1536
run_test 17  101 "Qwen3-30B-A3B-GateUP"     4   1024  4096  2048
run_test 18  73  "DeepSeek-V2-Lite-GateUP"  4   2048  2816  2048
run_test 19  112 "Qwen3-30B-A3B-Down"       8   1024  2048  2048
run_test 20  122 "Qwen3-30B-A3B-Down"       16  1024  2048  2048
run_test 21  84  "DeepSeek-V2-Lite-Down"    8   2048  2048  1408
run_test 22  52  "DeepSeek-V2-Down"         20  1024  5120  1536
run_test 23  182 "MoE-1T-Down"              7   1024  8192  1920
run_test 24  34  "DeepSeek-V2-Down"         5   2048  5120  1536
run_test 25  76  "DeepSeek-V2-Lite-Down"    4   4096  2048  1408
run_test 26  92  "Grok-2-Down"              1   1024  8192  16384
run_test 27  65  "DeepSeek-V2-Lite-GateUP"  2   4096  2816  2048
run_test 28  68  "DeepSeek-V2-Lite-Down"    2   8192  2048  1408
run_test 29  80  "DeepSeek-V2-Lite-Down"    4   16384 2048  1408
run_test 30  171 "Mixtral-8x22B-GateUP"     1   1024  32768 6144

echo "========================================================================================================"
echo "All tests completed!"
echo "========================================================================================================"
