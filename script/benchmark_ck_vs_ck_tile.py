#!/usr/bin/env python3

import os
import argparse
import subprocess
import sys
import matplotlib.pyplot as plt
# Non-interactive backend for matplotlib
plt.switch_backend('Agg')
import numpy as np

def data_type_str_to_profiler_arg(data_type_str):
    """Convert data type string to profiler argument"""
    data_type_map = {
        "fp16": "1",
        "fp32": "0",
        "int8": "3"
    }
    return data_type_map.get(data_type_str.lower(), 1)  # Default to fp16 if unknown

def parse_cli_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run CK and CK Tile convolution profilers.")
    parser.add_argument("--input-file", type=str, dest="input_file", required=False, help="Path to the file containing test cases.")
    parser.add_argument("--log-to-stdout", action="store_true", help="Log profiler output to stdout instead of /dev/null.")
    parser.add_argument("--bin-path", type=str, dest="bin_path", required=False, help="Path to the CK/CK Tile profiler executables.")
    parser.add_argument("--results-path", type=str, dest="results_path", required=False, help="Path to store profiler results.", default=".")
    parser.add_argument("--analyze-file", type=str, dest="analyze_file", required=False, help="Path to store analysis results.", default="")
    parser.add_argument("--data-type", type=str, dest="data_type", required=False, help="Data type for the profiler (e.g., fp16, fp32).", default="fp16")
    parser.add_argument("--start-idx", type=int, dest="start_idx", required=False, help="Start index for test cases.", default=0)
    parser.add_argument("--end-idx", type=int, dest="end_idx", required=False, help="End index for test cases.", default=-1)

    args, unknown_args = parser.parse_known_args()
    
    if unknown_args:
        print(f"Unknown arguments: {unknown_args}", file=sys.stderr)
        sys.exit(1)
    
    return args

class ProfilerType:
    CK = 1
    CK_TILE = 2

def run_ck_profiler_cmd(cmd_args, profiler_type, bin_path, results_file, log_to_stdout=False):
    profiler = "ckTileProfiler" if profiler_type == ProfilerType.CK_TILE else "ckProfiler"
    profiler_path = os.path.join(bin_path, profiler)
    cmd = [profiler_path] + cmd_args
    cmd_str = ' '.join(cmd)

    # Environment variable to specify results file
    env = os.environ.copy()
    env["CK_PROFILER_LOG_FILE"] = results_file
    env["CK_TILE_PROFILER_LOG_FILE"] = results_file

    timeoutInSec = 300 * 60 # 300 minutes timeout
    if log_to_stdout:
      subprocess.run(cmd, timeout=timeoutInSec, env=env) 
    else:
      with open(os.devnull, 'w') as devnull:
        try:
          subprocess.run(cmd, stdout=devnull, stderr=devnull, timeout=timeoutInSec, env=env)
        except subprocess.TimeoutExpired:
          print(f"Command '{cmd_str}' timed out after {timeoutInSec} seconds.", file=sys.stderr)

def get_profiler_commands(file):
  profiler_commands = []
  with open(file, 'r') as f:
    lines = f.readlines()
    lines = lines[1:]  # Skip the header line
    lines = list(dict.fromkeys(lines))
    for line in lines:
        line = line.strip()
        cmd = [x.strip() for x in line.split(',') if x.strip() and x.strip() != '']
        profiler_commands.append(cmd)
  return profiler_commands

def run_analysis(results_file):
  """Analyze benchmark results and create performance comparison plots"""
    
  # Parse the results file
  test_cases = []
  current_case = {}
  
  with open(results_file, 'r') as f:
      lines = f.readlines()
  
  i = 0
  while i < len(lines):
      line = lines[i].strip()
      
      # Look for grouped_conv_* command lines
      if line.startswith('grouped_conv_'):
          current_case = {'command': line}
          i += 1
          
          # Parse CK Tile results
          while i < len(lines) and not lines[i].strip().startswith('CK Tile best configuration:'):
              i += 1
          
          if i < len(lines):
              i += 1  # Skip "CK Tile best configuration:" line
              if i < len(lines) and lines[i].strip().startswith('name:'):
                  current_case['ck_tile_name'] = lines[i].strip().replace('name:', '').strip()
                  i += 1
              if i < len(lines) and lines[i].strip().startswith('avg_time:'):
                  current_case['ck_tile_time'] = float(lines[i].strip().replace('avg_time:', '').strip())
                  i += 1
              if i < len(lines) and lines[i].strip().startswith('SplitK:'):
                  current_case['ck_tile_splitk'] = lines[i].strip().replace('SplitK:', '').strip()
                  i += 1
              if i < len(lines) and lines[i].strip().startswith('all_pass'):
                  current_case['ck_tile_all_pass'] = lines[i].strip().replace('all_pass', '').strip()
                  i += 1
          
          # Parse CK results
          while i < len(lines) and not lines[i].strip().startswith('CK best configuration:'):
              i += 1
          
          if i < len(lines):
              i += 1  # Skip "CK best configuration:" line
              if i < len(lines) and lines[i].strip().startswith('name:'):
                  current_case['ck_name'] = lines[i].strip().replace('name:', '').strip()
                  i += 1
              if i < len(lines) and lines[i].strip().startswith('avg_time:'):
                  current_case['ck_time'] = float(lines[i].strip().replace('avg_time:', '').strip())
                  i += 1
              if i < len(lines) and lines[i].strip().startswith('SplitK:'):
                  current_case['ck_splitk'] = lines[i].strip().replace('SplitK:', '').strip()
                  i += 1
          
          # Only add if we have both CK and CK Tile results
          if all(key in current_case for key in ['ck_tile_time', 'ck_time']):
              # Skip cases where CK Tile failed (time = 0)
              if current_case['ck_tile_time'] > 0:
                  test_cases.append(current_case)
      else:
          i += 1
  
  print(f"Found {len(test_cases)} valid test cases for analysis")
  
  # Calculate performance ratios (CK Tile performance relative to CK, where 100% = parity)
  performance_ratios = []
  ck_times = []
  ck_tile_times = []
  case_labels = []

  workbook = xlsxwriter.Workbook('conv_perf.xlsx')
  worksheet = workbook.add_worksheet()

  header_format = workbook.add_format()
  header_format.set_bold()

  offset = 4

  worksheet.write(offset, 0, "command", header_format)
  worksheet.set_column(0, 0, 66)
  worksheet.write(offset, 1, "CK Time", header_format)
  worksheet.set_column(1, 1, 11)
  worksheet.write(offset, 2, "CK Tile Time", header_format)
  worksheet.set_column(2, 2, 11)
  worksheet.write(offset, 3, "CK / CK Tile", header_format)
  worksheet.set_column(3, 3, 11)
  worksheet.write(offset, 4, "All pass", header_format)
  worksheet.set_column(4, 4, 11)
  worksheet.write(offset, 5, "CK best kernel", header_format)
  worksheet.set_column(5, 5, 25)
  worksheet.write(offset, 6, "CK splitk", header_format)
  worksheet.set_column(6, 6, 15)
  worksheet.write(offset, 7, "CK tile best kernel", header_format)
  worksheet.set_column(7, 7, 25)
  worksheet.write(offset, 8, "CK tile splitk", header_format)
  worksheet.set_column(8, 8, 15)

  offset += 1

  num_of_ck_tile_slower = 0

  for i, case in enumerate(test_cases):
      worksheet.write(i + offset, 0, case['command'])
      worksheet.write(i + offset, 1, case['ck_time'])
      worksheet.write(i + offset, 2, case['ck_tile_time'])

      format = workbook.add_format()
      ratio = case['ck_time'] / case['ck_tile_time']

      if ratio < 1.0:
        format.set_bg_color('red')
        num_of_ck_tile_slower += 1
      else:
        format.set_bg_color('green')

      all_pass = case['ck_tile_all_pass']

      worksheet.write(i + offset, 3, ratio, format)

      format2 = workbook.add_format()
      format2.set_bg_color('green' if all_pass == "true" else 'red')
      worksheet.write(i + offset, 4, all_pass, format2)
      worksheet.write(i + offset, 5, case['ck_name'])
      worksheet.write(i + offset, 6, case['ck_splitk'])
      worksheet.write(i + offset, 7, case['ck_tile_name'])
      worksheet.write(i + offset, 8, case['ck_tile_splitk'])

      ck_time = case['ck_time']
      ck_tile_time = case['ck_tile_time']
      
      # Performance ratio: CK_time / CK_Tile_time * 100%
      # >100% means CK Tile is faster, <100% means CK is faster
    #   ratio = (ck_time / ck_tile_time) * 100
    #   performance_ratios.append(ratio)
    #   ck_times.append(ck_time)
    #   ck_tile_times.append(ck_tile_time)
      
    #   # Create a short label for the test case
    #   cmd_parts = case['command'].split()
    #   if len(cmd_parts) >= 8:
    #       label = f"G{cmd_parts[8]}_N{cmd_parts[9]}_K{cmd_parts[10]}_C{cmd_parts[11]}"
    #   else:
    #       label = f"Case_{i+1}"
    #   case_labels.append(label)

  worksheet.write(0, 0, f"all cases: {len(test_cases)}")
  worksheet.write(1, 0, f"ck tile slower: {num_of_ck_tile_slower}")
  worksheet.write(2, 0, f"ck tile slower: {(num_of_ck_tile_slower / len(test_cases) * 100):2.1f}%")

  workbook.close()
  return
      
  
  max_cases_to_detailed_plot = 10
  if len(test_cases) < max_cases_to_detailed_plot:
    # Create performance comparison plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Performance ratio bar chart
    x_pos = np.arange(len(case_labels))
    colors = ['green' if ratio >= 100 else 'red' for ratio in performance_ratios]
    
    bars = ax1.bar(x_pos, performance_ratios, color=colors, alpha=0.7)
    ax1.set_xlabel('Test Cases')
    ax1.set_ylabel('CK Tile Performance (% of CK)')
    ax1.set_title('CK Tile vs CK Performance Comparison\n(>100% = CK Tile Faster, <100% = CK Faster)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(case_labels, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, ratio in zip(bars, performance_ratios):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{ratio:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Absolute timing comparison
    x_pos_offset = np.arange(len(case_labels))
    width = 0.35
    
    bars1 = ax2.bar(x_pos_offset - width/2, ck_times, width, label='CK', color='blue', alpha=0.7)
    bars2 = ax2.bar(x_pos_offset + width/2, ck_tile_times, width, label='CK Tile', color='orange', alpha=0.7)
    
    ax2.set_xlabel('Test Cases')
    ax2.set_ylabel('Average Time (seconds)')
    ax2.set_title('Absolute Performance Comparison: CK vs CK Tile')
    ax2.set_xticks(x_pos_offset)
    ax2.set_xticklabels(case_labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Use log scale for better visualization
    
    plt.tight_layout()
    
    # Save the plot
    output_file = results_file.replace('.txt', '_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Performance analysis plot saved to: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    faster_count = sum(1 for ratio in performance_ratios if ratio > 100)
    slower_count = len(performance_ratios) - faster_count
    
    print(f"Total test cases: {len(test_cases)}")
    print(f"CK Tile faster: {faster_count} ({faster_count/len(test_cases)*100:.1f}%)")
    print(f"CK faster: {slower_count} ({slower_count/len(test_cases)*100:.1f}%)")
    print(f"Average CK Tile performance: {np.mean(performance_ratios):.1f}% of CK")
    print(f"Median CK Tile performance: {np.median(performance_ratios):.1f}% of CK")
    print(f"Best CK Tile performance: {np.max(performance_ratios):.1f}% of CK")
    print(f"Worst CK Tile performance: {np.min(performance_ratios):.1f}% of CK")
    
    # Show the plot
    plt.show()
  else:
    # Plot the histogram of the performance ratios
    plt.figure(figsize=(10, 6))

    min_ratio = min(performance_ratios)
    max_ratio = max(performance_ratios)

    bin_width = 5

    # Extend range to ensure we capture all data
    bin_start = int(min_ratio // bin_width) * bin_width
    bin_end = int(max_ratio // bin_width) * bin_width
    bin_edges = np.arange(bin_start, bin_end, bin_width)

    # Create the histogram data
    counts, bins = np.histogram(performance_ratios, bins=bin_edges)

    # Color bars based on whether they're above or below 100%
    colors = []
    for i in range(len(counts)):
        bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
        if bin_center < 100:
            colors.append('red')
        else:
            colors.append('green')

    # Plot the histogram with custom colors
    plt.bar(bin_edges[:-1], counts, width=bin_width, color=colors, edgecolor='black', 
            alpha=0.7, align='edge')

    plt.xlabel('CK Tile Performance (% of CK)')
    plt.ylabel('Number of Test Cases')
    plt.title('CK Tile vs CK Performance Distribution\n(>100% = CK Tile Faster, <100% = CK Faster)')

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='CK Faster (<100%)'),
        Patch(facecolor='green', alpha=0.7, label='CK Tile Faster (>100%)')
    ]
    plt.legend(handles=legend_elements)

    plt.grid(True, alpha=0.3)

    # Set x-axis to show percentage marks at logical intervals
    x_ticks = np.arange(int(min_ratio//10)*10, int(max_ratio//10)*10 + 20, 10)
    plt.xticks(x_ticks)

    # Set y-axis to integer positions only
    max_count = max(counts) if len(counts) > 0 else 1
    y_ticks = np.arange(0, max_count + 1, 2)
    plt.yticks(y_ticks)

    # Save the histogram
    output_file = results_file.replace('.txt', '_analysis_histogram.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Performance analysis histogram saved to: {output_file}")
    plt.close()

    # Collect aggregated statistics for cases where CK is faster
    print("\n" + "="*80)
    print("CK FASTER TEST CASES - AGGREGATED STATISTICS")
    print("="*80)

    ck_faster_cases = []
    ck_faster_ratios = []
    ck_faster_kernels = {}  # Track which CK kernels are winning
    ck_tile_losing_kernels = {}  # Track which CK Tile kernels are losing

    for i, case in enumerate(test_cases):
        ratio = performance_ratios[i]
        if ratio < 100:
            ck_faster_cases.append(case)
            ck_faster_ratios.append(ratio)
            
            # Count CK kernels that are winning
            ck_kernel = case.get('ck_name', 'N/A')
            if ck_kernel not in ck_faster_kernels:
                    ck_faster_kernels[ck_kernel] = {'count': 0, 'ratios': []}
            ck_faster_kernels[ck_kernel]['count'] += 1
            ck_faster_kernels[ck_kernel]['ratios'].append(ratio)
            
            # Count CK Tile kernels that are losing
            ck_tile_kernel = case.get('ck_tile_name', 'N/A')
            if ck_tile_kernel not in ck_tile_losing_kernels:
                    ck_tile_losing_kernels[ck_tile_kernel] = {'count': 0, 'ratios': []}
            ck_tile_losing_kernels[ck_tile_kernel]['count'] += 1
            ck_tile_losing_kernels[ck_tile_kernel]['ratios'].append(ratio)

    if ck_faster_cases:
        print(f"Number of cases where CK is faster: {len(ck_faster_cases)}/{len(test_cases)} ({len(ck_faster_cases)/len(test_cases)*100:.1f}%)")
        print(f"Average CK performance advantage: {100 - np.mean(ck_faster_ratios):.1f}%")
        print(f"Median CK performance advantage: {100 - np.median(ck_faster_ratios):.1f}%")
        print(f"Best CK performance advantage: {100 - np.min(ck_faster_ratios):.1f}%")
        print(f"Worst CK performance advantage: {100 - np.max(ck_faster_ratios):.1f}%")
        
        print(f"\nTop CK kernels that outperform CK Tile:")
        sorted_ck_kernels = sorted(ck_faster_kernels.items(), key=lambda x: x[1]['count'], reverse=True)
        for kernel, stats in sorted_ck_kernels[:5]:  # Show top 5
                avg_advantage = 100 - np.mean(stats['ratios'])
                print(f"  {kernel}: {stats['count']} wins, avg advantage: {avg_advantage:.1f}%")
        
        print(f"\nCK Tile kernels that lose most often:")
        sorted_ck_tile_kernels = sorted(ck_tile_losing_kernels.items(), key=lambda x: x[1]['count'], reverse=True)
        for kernel, stats in sorted_ck_tile_kernels[:5]:  # Show top 5
                avg_disadvantage = np.mean(stats['ratios'])
                print(f"  {kernel}: {stats['count']} losses, avg performance: {avg_disadvantage:.1f}% of CK")
    else:
        print("No cases found where CK is faster than CK Tile.")

def main():
  args = parse_cli_args()

  if (args.analyze_file):
    print(f"Analyzing results from file: {args.analyze_file}")
    run_analysis(args.analyze_file)
    return
  else:
    print(f"Running profilers using test cases from file: {args.input_file}")
    profiler_commands = get_profiler_commands(args.input_file)
    print(f"Got {len(profiler_commands)} unique commands to run.")

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    # Get computer host name
    hostname = os.uname().nodename.split('.')[0]

    results_file = os.path.join(args.results_path, f"ck_results_{hostname}_{args.data_type}_{os.getpid()}.txt")

    data_type_arg = data_type_str_to_profiler_arg(args.data_type)

    commands = profiler_commands[args.start_idx:args.end_idx]
    for i, cmd in enumerate(commands):
        # Set the correct data type based on user input
        cmd[1] = data_type_arg

        # Set layout arg to Input[N, Hi, Wi, G, C], Weight[G, K, Y, X, C], Output[N, Ho, Wo, G, K]
        cmd[2] = '1'

        # We don't want to run verification. We assume CK already works correctly.
        cmd[3] = '0'  # Set verification flag to 0 (no verification)

        cmd_concatenated_str = ' '.join(cmd)
        print(f"\n####################################################################################################################")
        print(f"Running command {i + 1}/{len(commands)}: {cmd_concatenated_str}")
        print(f"######################################################################################################################")
  
        # Print the command to the output file
        with open(results_file, 'a') as f:
          f.write(cmd_concatenated_str + "\n")

        run_ck_profiler_cmd(cmd, ProfilerType.CK, args.bin_path, results_file, args.log_to_stdout)
  
if __name__ == "__main__":
    main()