#!/usr/bin/env python3
"""
BF16 最差 30 Case: Primus-Turbo vs CK Grouped GEMM 对比脚本
从 validation_results.log 提取 CK 测试结果，与 Primus-Turbo 原始数据对比
"""

import re
import sys

# Primus-Turbo 原始基准数据
PT_DATA = {
    1:  ("DeepSeek-V2-Lite-Down",   2,   512,  2048,  1408,  88.31,  77.83),
    2:  ("DeepSeek-V2-Lite-GateUP", 2,   512,  2816,  2048,  128.56, 111.88),
    3:  ("DeepSeek-V2-Lite-Down",   4,   512,  2048,  1408,  169.94, 153.80),
    4:  ("DeepSeek-V2-Lite-Down",   2,   1024, 2048,  1408,  171.98, 155.41),
    5:  ("Mixtral-8x7B-Down",       1,   512,  4096,  14336, 109.59, 235.06),
    6:  ("Qwen3-30B-A3B-Down",      4,   512,  2048,  2048,  180.66, 167.65),
    7:  ("DeepSeek-V2-Lite-GateUP", 4,   512,  2816,  2048,  235.38, 165.31),
    8:  ("Mixtral-8x22B-Down",      1,   512,  6144,  16384, 162.22, 245.13),
    9:  ("DeepSeek-V2-Lite-GateUP", 2,   1024, 2816,  2048,  240.26, 185.67),
    10: ("Grok-2-Down",             1,   512,  8192,  16384, 213.48, 249.60),
    11: ("Qwen3-30B-A3B-GateUP",    4,   512,  4096,  2048,  302.33, 181.72),
    12: ("DeepSeek-V2-Lite-Down",   8,   512,  2048,  1408,  274.33, 212.22),
    13: ("Qwen3-30B-A3B-Down",      8,   512,  2048,  2048,  289.45, 219.50),
    14: ("DeepSeek-V2-Lite-Down",   4,   1024, 2048,  1408,  282.12, 232.32),
    15: ("Mixtral-8x7B-Down",       1,   1024, 4096,  14336, 212.73, 337.73),
    16: ("Qwen3-30B-A3B-Down",      4,   1024, 2048,  2048,  297.60, 253.24),
    17: ("DeepSeek-V2-Lite-Down",   2,   2048, 2048,  1408,  293.07, 262.01),
    18: ("DeepSeek-V2-Down",        5,   512,  5120,  1536,  378.06, 180.62),
    19: ("Qwen3-235B-A22B-Down",    4,   512,  4096,  4096,  330.68, 239.15),
    20: ("DeepSeek-V2-Lite-GateUP", 8,   512,  2816,  2048,  350.89, 223.60),
    21: ("DeepSeek-V2-GateUP",      5,   512,  3072,  5120,  310.56, 265.50),
    22: ("DeepSeek-V2-Down",        10,  512,  5120,  1536,  354.81, 238.12),
    23: ("Mixtral-8x7B-GateUP",     1,   512,  28672, 4096,  449.17, 144.20),
    24: ("DeepSeek-V2-Lite-GateUP", 4,   1024, 2816,  2048,  364.11, 241.49),
    25: ("Mixtral-8x22B-GateUP",    1,   512,  32768, 6144,  457.38, 179.06),
    26: ("Mixtral-8x22B-Down",      1,   1024, 6144,  16384, 292.92, 346.51),
    27: ("DeepSeek-V2-Lite-Down",   8,   1024, 2048,  1408,  395.28, 245.58),
    28: ("DeepSeek-V2-Down",        5,   1024, 5120,  1536,  367.96, 276.79),
    29: ("DeepSeek-V2-Lite-GateUP", 2,   2048, 2816,  2048,  376.12, 270.01),
    30: ("DeepSeek-V2-Lite-GateUP", 8,   1024, 2816,  2048,  337.25, 310.81),
}


def parse_log(log_file):
    """解析 validation_results.log 提取 CK TFLOPS 数据"""
    with open(log_file, 'r') as f:
        content = f.read()
    
    # 提取所有 TFlops 值
    tflops_pattern = r'Perf:.*?(\d+\.?\d*) TFlops'
    tflops_values = [float(x) for x in re.findall(tflops_pattern, content)]
    
    # 每个 rank 有 3 个值: Forward, Backward_A, Backward_B
    ck_data = {}
    for rank in range(1, 31):
        idx = (rank - 1) * 3
        if idx + 2 < len(tflops_values):
            ck_data[rank] = (
                tflops_values[idx],      # Forward
                tflops_values[idx + 1],  # Backward grad_A
                tflops_values[idx + 2],  # Backward grad_B
            )
    
    # 统计信息
    correct_count = content.count("correct")
    fail_count = content.count("fail")
    tile_256 = content.count("256x256 tile")
    tile_128 = content.count("256x128 tile")
    
    return ck_data, correct_count, fail_count, tile_256, tile_128


def harmonic_mean(a, b):
    """计算调和平均: 2 / (1/a + 1/b)
    
    这是正确的方式来合并两个 TFLOPS 值，因为:
    Combined_TFLOPS = Total_FLOPs / Total_Time
                    = (FLOPs_A + FLOPs_B) / (Time_A + Time_B)
                    = 2*FLOPs / (FLOPs/TFLOPS_A + FLOPs/TFLOPS_B)  (当 FLOPs_A = FLOPs_B 时)
                    = 2 / (1/TFLOPS_A + 1/TFLOPS_B)
    """
    if a <= 0 or b <= 0:
        return 0.0
    return 2.0 / (1.0/a + 1.0/b)


def print_comparison(ck_data):
    """打印对比表格"""
    sep = "=" * 195
    line = "-" * 195
    
    print(sep)
    print("BF16 最差 30 Case: Primus-Turbo vs CK Grouped GEMM 完整对比")
    print(sep)
    print(f"{'Rank':<5} {'Case':<28} {'B':<3} {'M':<5} {'N':<6} {'K':<6} | {'CK_Fwd':>8} {'CK_BwdA':>8} {'CK_BwdB':>8} {'CK_Bwd':>8} | {'PT_Fwd':>8} {'PT_Bwd':>8} | {'Δ Fwd':>8} {'Δ Bwd':>8}")
    print(line)
    
    total_ck_fwd = total_ck_bwd = 0
    total_pt_fwd = total_pt_bwd = 0
    
    for rank in range(1, 31):
        case, B, M, N, K, pt_fwd, pt_bwd = PT_DATA[rank]
        
        if rank in ck_data:
            ck_fwd, ck_bwd_a, ck_bwd_b = ck_data[rank]
        else:
            ck_fwd = ck_bwd_a = ck_bwd_b = 0.0
        
        # 使用调和平均计算综合 backward TFLOPS (正确的合并方式)
        ck_bwd_combined = harmonic_mean(ck_bwd_a, ck_bwd_b)
        
        delta_fwd = ck_fwd - pt_fwd
        delta_bwd = ck_bwd_combined - pt_bwd
        
        total_ck_fwd += ck_fwd
        total_ck_bwd += ck_bwd_combined
        total_pt_fwd += pt_fwd
        total_pt_bwd += pt_bwd
        
        print(f"{rank:<5} {case:<28} {B:<3} {M:<5} {N:<6} {K:<6} | {ck_fwd:>8.2f} {ck_bwd_a:>8.2f} {ck_bwd_b:>8.2f} {ck_bwd_combined:>8.2f} | {pt_fwd:>8.2f} {pt_bwd:>8.2f} | {delta_fwd:>+8.2f} {delta_bwd:>+8.2f}")
    
    print(line)
    
    avg_ck_fwd = total_ck_fwd / 30
    avg_ck_bwd = total_ck_bwd / 30
    avg_pt_fwd = total_pt_fwd / 30
    avg_pt_bwd = total_pt_bwd / 30
    
    print(f"{'平均':<38} | {avg_ck_fwd:>8.2f} {avg_ck_bwd:>26.2f} | {avg_pt_fwd:>8.2f} {avg_pt_bwd:>8.2f} | {avg_ck_fwd-avg_pt_fwd:>+8.2f} {avg_ck_bwd-avg_pt_bwd:>+8.2f}")
    print(sep)
    
    return avg_ck_fwd, avg_ck_bwd, avg_pt_fwd, avg_pt_bwd


def print_summary(avg_ck_fwd, avg_ck_bwd, avg_pt_fwd, avg_pt_bwd, 
                  correct_count, fail_count, tile_256, tile_128):
    """打印总结"""
    sep = "=" * 100
    
    print()
    print(sep)
    print("性能对比总结")
    print(sep)
    print(f"Forward  平均: PT = {avg_pt_fwd:.2f} TFLOPS → CK = {avg_ck_fwd:.2f} TFLOPS ({(avg_ck_fwd/avg_pt_fwd-1)*100:>+.1f}%)")
    print(f"Backward 平均: PT = {avg_pt_bwd:.2f} TFLOPS → CK = {avg_ck_bwd:.2f} TFLOPS ({(avg_ck_bwd/avg_pt_bwd-1)*100:>+.1f}%)")
    
    avg_pt = (avg_pt_fwd + avg_pt_bwd) / 2
    avg_ck = (avg_ck_fwd + avg_ck_bwd) / 2
    print(f"综合平均:      PT = {avg_pt:.2f} TFLOPS → CK = {avg_ck:.2f} TFLOPS ({(avg_ck/avg_pt-1)*100:>+.1f}%)")
    print(sep)
    
    print()
    print(f"精度验证: {correct_count} 通过, {fail_count} 失败")
    print(f"Tile 配置: 256x256 使用 {tile_256} 次, 256x128 使用 {tile_128} 次")


def main():
    log_file = sys.argv[1] if len(sys.argv) > 1 else "validation_results.log"
    
    try:
        ck_data, correct_count, fail_count, tile_256, tile_128 = parse_log(log_file)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {log_file}")
        print("用法: python3 compare_results.py [validation_results.log]")
        sys.exit(1)
    
    avg_ck_fwd, avg_ck_bwd, avg_pt_fwd, avg_pt_bwd = print_comparison(ck_data)
    print_summary(avg_ck_fwd, avg_ck_bwd, avg_pt_fwd, avg_pt_bwd,
                  correct_count, fail_count, tile_256, tile_128)


if __name__ == "__main__":
    main()

