#!/usr/bin/env python3
import re

with open('config_comparison_new.log', 'r') as f:
    content = f.read()

# 按 Rank 分割
rank_blocks = re.split(r'={80,}\nRank', content)[1:]

# 所有 config 名称
all_configs = ['compute_v3', 'compute_v3_kb2', 'compute_v3_32x128', 'compute_v3_32x128_kb2', 
               'compute_v3_128x128', 'compute_v3_128x128_kb2', 'memory_intrawave', 'memory_intrawave_kb2']

print("=" * 220)
print("8 种 Config 性能对比 (TFLOPS) - 含 kbatch=2")
print("=" * 220)

# 统计
wins = {p: {c: 0 for c in all_configs} for p in ['Forward', 'grad_A', 'grad_B']}

for block in rank_blocks:
    header = re.search(r'(\d+): (.+?) \(TestID=\d+\)\s+B=(\d+), M=(\d+), N=(\d+), K=(\d+)', block)
    if not header:
        continue
    rank, case, B, M, N, K = header.groups()
    
    # 按 config 分割
    config_blocks = re.split(r'--- Config: (\S+) ---', block)
    results = {}
    
    for i in range(1, len(config_blocks), 2):
        config_name = config_blocks[i]
        config_content = config_blocks[i+1] if i+1 < len(config_blocks) else ""
        results[config_name] = {}
        
        fwd_match = re.search(r'\[Forward\].*?Perf:\s+([\d.]+) ms, ([\d.]+) TFlops', config_content, re.DOTALL)
        if fwd_match:
            results[config_name]['Forward'] = float(fwd_match.group(2))
        
        grada_match = re.search(r'\[Backward grad_A\].*?Perf:\s+([\d.]+) ms, ([\d.]+) TFlops', config_content, re.DOTALL)
        if grada_match:
            results[config_name]['grad_A'] = float(grada_match.group(2))
        
        gradb_match = re.search(r'\[Backward grad_B\].*?Perf:\s+([\d.]+) ms, ([\d.]+) TFlops', config_content, re.DOTALL)
        if gradb_match:
            results[config_name]['grad_B'] = float(gradb_match.group(2))
    
    # 打印每个 rank 的结果
    print(f"\nRank {rank}: {case} (B={B}, M={M}, N={N}, K={K})")
    print("-" * 180)
    print(f"{'Pass':<8} | {'v3':>7} {'v3_k2':>7} | {'32x128':>7} {'32_k2':>7} | {'128x128':>8} {'128_k2':>8} | {'intra':>7} {'intra_k2':>8} | {'Best':>18}")
    print("-" * 180)
    
    for pass_name in ['Forward', 'grad_A', 'grad_B']:
        vals = {}
        for cfg in all_configs:
            vals[cfg] = results.get(cfg, {}).get(pass_name, 0)
        
        best_val = max(vals.values()) if vals.values() else 0
        best_cfg = [k for k, v in vals.items() if v == best_val][0] if best_val > 0 else 'N/A'
        
        if best_val > 0:
            wins[pass_name][best_cfg] += 1
        
        v3 = vals.get('compute_v3', 0)
        v3_k2 = vals.get('compute_v3_kb2', 0)
        c32 = vals.get('compute_v3_32x128', 0)
        c32_k2 = vals.get('compute_v3_32x128_kb2', 0)
        c128 = vals.get('compute_v3_128x128', 0)
        c128_k2 = vals.get('compute_v3_128x128_kb2', 0)
        intra = vals.get('memory_intrawave', 0)
        intra_k2 = vals.get('memory_intrawave_kb2', 0)
        
        short_best = best_cfg.replace('compute_v3_', '').replace('memory_', '')
        print(f"{pass_name:<8} | {v3:>7.1f} {v3_k2:>7.1f} | {c32:>7.1f} {c32_k2:>7.1f} | {c128:>8.1f} {c128_k2:>8.1f} | {intra:>7.1f} {intra_k2:>8.1f} | {short_best:>18}")

print("\n" + "=" * 120)
print("胜率统计 (30 cases)")
print("=" * 120)
print(f"{'Pass':<10} | {'v3':>6} {'v3_k2':>6} | {'32x128':>7} {'32_k2':>6} | {'128x128':>8} {'128_k2':>7} | {'intra':>6} {'intra_k2':>8}")
print("-" * 120)
for pass_name in ['Forward', 'grad_A', 'grad_B']:
    w = wins[pass_name]
    print(f"{pass_name:<10} | {w['compute_v3']:>6} {w['compute_v3_kb2']:>6} | {w['compute_v3_32x128']:>7} {w['compute_v3_32x128_kb2']:>6} | {w['compute_v3_128x128']:>8} {w['compute_v3_128x128_kb2']:>7} | {w['memory_intrawave']:>6} {w['memory_intrawave_kb2']:>8}")

total = {c: sum(wins[p][c] for p in wins) for c in all_configs}
print("-" * 120)
print(f"{'Total':<10} | {total['compute_v3']:>6} {total['compute_v3_kb2']:>6} | {total['compute_v3_32x128']:>7} {total['compute_v3_32x128_kb2']:>6} | {total['compute_v3_128x128']:>8} {total['compute_v3_128x128_kb2']:>7} | {total['memory_intrawave']:>6} {total['memory_intrawave_kb2']:>8}")
print("=" * 120)
