#!/usr/bin/env python3

import os
import argparse
import sys
import pandas as pd
import csv
import matplotlib
from collections import defaultdict
import numpy as np

matplotlib.use('Agg')  # Use a non-interactive backend
from matplotlib import pyplot as plt

def parse_cli_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Analyze convolution test results.")
    parser.add_argument("--perf-file", type=str, required=True, help="Path to the perf results.")
    parser.add_argument("--baseline-perf-file", type=str, required=True, help="Path to the baseline perf results.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save output plots.")
    parser.add_argument("--kernel", type=str, required=True, default="", help="Kernel for which the performance is studied.")
    
    
    args, unknown_args = parser.parse_known_args()
    
    if unknown_args:
        print(f"Unknown arguments: {unknown_args}", file=sys.stderr)
        sys.exit(1)
    
    return args

def extract_perf_data(line):
    import re
    tflops_pattern = r'([\d.]+)\s+TFlops'
    tflops_match = re.search(tflops_pattern, line)
    
    if not tflops_match:
        return None, None
    
    tflops = float(tflops_match.group(1))
    
    # Use a pattern that captures everything between "GB/s, " and ", SplitK"
    # This pattern handles nested brackets by using a recursive-like approach
    kernel_pattern = r'GB/s,\s+(.*?),\s+SplitK(?:\s+\-?\d+)?'
    kernel_match = re.search(kernel_pattern, line)
    
    if kernel_match:
        kernel_name = kernel_match.group(1).strip()
        return tflops, kernel_name
    else:
        return None, None

def get_tflops_per_kernel(file):
  res = defaultdict(list)
  with open(file, 'r') as f:
    lines = f.readlines()
    n_lines = len(lines)
    for i, line in enumerate(lines):
      #print(f"Processing line {i + 1}/{n_lines}")
      tflops, kernel_name = extract_perf_data(line)
      assert tflops is not None, f"Failed to extract TFlops from line: {line.strip()}"
      assert kernel_name is not None, f"Failed to extract kernel name from line: {line.strip()}"
      res[kernel_name].append(tflops)

  return res

def filter_by_kernel(perf_results, kernel):
    """Filter performance results by kernel name."""
    if kernel == "all":
        return perf_results
    if kernel:
        return {k: v for k, v in perf_results.items() if kernel == k.split('<')[0]}
    return perf_results

def plot_perf_difference(perf_difference, output_dir, label=""):
    """Plot the performance differences as a histogram with statistics."""
    import numpy as np
    
    mean_val = np.mean(perf_difference)
    median_val = np.median(perf_difference)
    std_val = np.std(perf_difference)
    min_val = np.min(perf_difference)
    max_val = np.max(perf_difference)
    p25 = np.percentile(perf_difference, 25)
    p75 = np.percentile(perf_difference, 75)
    count = len(perf_difference)
    
    bin_width = 1
    min_edge = np.floor(min_val / bin_width) * bin_width
    max_edge = np.ceil(max_val / bin_width) * bin_width
    bin_edges = np.arange(min_edge, max_edge + bin_width, bin_width)
    
    plt.figure(figsize=(12, 6))
    
    below_100 = [x for x in perf_difference if x < 100]
    above_100 = [x for x in perf_difference if x >= 100]
    
    if below_100:
        counts_below, _ = np.histogram(below_100, bins=bin_edges)
    else:
        counts_below = np.zeros(len(bin_edges) - 1)
        
    if above_100:
        counts_above, _ = np.histogram(above_100, bins=bin_edges)
    else:
        counts_above = np.zeros(len(bin_edges) - 1)
    
    if below_100:
        plt.hist(below_100, bins=bin_edges, color='red', 
                alpha=0.7, edgecolor='black', label='Below 100%')
    
    if above_100:
        plt.hist(above_100, bins=bin_edges, color='green', 
                alpha=0.7, edgecolor='black', label='Above 100%')
    
    total_counts = counts_below + counts_above
    
    for i in range(len(bin_edges) - 1):
        if total_counts[i] > 0:
            bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
            
            plt.text(
                bin_center,                 
                total_counts[i] + 0.5,     
                f'{int(total_counts[i])}',  
                ha='center',                
                va='bottom',                
                fontweight='bold',          
                fontsize=9                 
            )
    
    stats_text = (f"Statistics:\n"
                  f"Count: {count}\n"
                  f"Mean: {mean_val:.2f}%\n"
                  f"Median: {median_val:.2f}%\n"
                  f"Std Dev: {std_val:.2f}%\n"
                  f"Min: {min_val:.2f}%\n"
                  f"Max: {max_val:.2f}%\n"
                  f"25th Percentile: {p25:.2f}%\n"
                  f"75th Percentile: {p75:.2f}%")
    
    title = label
    size = 12
    plt.title(title, 
              fontsize=size, fontweight='bold')
    plt.xlabel('Performance (%)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(bin_edges)
    plt.text(0.02, 0.97, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.axvline(x=100, color='black', linestyle='--', alpha=0.9, linewidth=2, 
                label='100% Threshold')
    
    below_count = len(below_100)
    above_count = len(above_100)
    below_percent = (below_count / count) * 100 if count > 0 else 0
    above_percent = (above_count / count) * 100 if count > 0 else 0

    legend =plt.legend([
        f'Below 100% ({below_count}, {below_percent:.1f}%)',
        f'Above 100% ({above_count}, {above_percent:.1f}%)',
        '100% Threshold'
    ])
    legend.set_bbox_to_anchor((0.225, 0.65))
    
    plt.tight_layout()
 
    file_name = os.path.join(output_dir, f'performance_{label}.png')
    plt.savefig(file_name, dpi=150)
    print(f"Saved performance chart to: {file_name}")
    
    plt.close()

def plot_perf(perf, baseline_perf, kernel, output_dir):
    """Plot the performance difference between the current and baseline results."""

    perf_difference = []
    for k in perf:
        if k in baseline_perf:
          perf_list = perf[k]
          baseline_perf_list = baseline_perf[k]
          if len(perf_list) != len(baseline_perf_list):
              raise ValueError(f"Performance lists for kernel {k} have different lengths: {len(perf_list)} vs {len(baseline_perf_list)}")
          
          for i in range(len(perf_list)):
              diff = 100.0 * (perf_list[i] / baseline_perf_list[i])
              perf_difference.append(diff)
        else:
          raise ValueError(f"Kernel {k} not found in baseline performance data.")

    plot_perf_difference(perf_difference, output_dir, label=kernel)

def main():
    args = parse_cli_args()

    perf = get_tflops_per_kernel(args.perf_file)
    baseline_perf = get_tflops_per_kernel(args.baseline_perf_file)

    print(f"Found in total {len(perf)} different kernels in perf file.")
    print(f"Found in total {len(baseline_perf)} different kernels in baseline perf file.")

    kernel_perf = filter_by_kernel(perf, args.kernel)
    baseline_kernel_perf = filter_by_kernel(baseline_perf, args.kernel)

    print(f"Found {len(kernel_perf)} instances of {args.kernel} in perf file.")
    print(f"Found {len(baseline_kernel_perf)} instances of {args.kernel} in baseline perf file.")

    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    plot_perf(kernel_perf, baseline_kernel_perf, args.kernel, args.output_dir)

if __name__ == "__main__":
    main()