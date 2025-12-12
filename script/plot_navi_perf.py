#!/usr/bin/env python3

import os
import argparse
import subprocess
import sys
import matplotlib.pyplot as plt
# Non-interactive backend for matplotlib
plt.switch_backend('Agg')
import numpy as np

def parse_cli_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run CK and CK Tile convolution profilers.")
    parser.add_argument("--input-file-int8", type=str, dest="input_file_int8", required=False, help="Path to the file containing test results for int8.")
    parser.add_argument("--input-file-fp16", type=str, dest="input_file_fp16", required=False, help="Path to the file containing test results for fp16.")
    
    args, unknown_args = parser.parse_known_args()
    
    if unknown_args:
        print(f"Unknown arguments: {unknown_args}", file=sys.stderr)
        sys.exit(1)
    
    return args

def parse_times(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
        ang_time_lines = lines[3::4]  # Every 4th line starting from line 3
        avg_times = [float(line.strip().split("avg_time: ")[-1]) for line in ang_time_lines]
        commnds = lines[0::4]  # Every 4th line starting from line 0

        # Create a dictionary of commands to their average times
        cmd_time_dict = {}
        for cmd, time in zip(commnds, avg_times):
            cmd_time_dict[cmd.strip()] = time
    
    return cmd_time_dict

def plot_perf(times_int8, times_fp16, output_file):
    #n_samples = min(len(times_int8), len(times_fp16))

    # From two dictionaries, extract the values where the key is present in both dictionaries
    speedup_percentage = []
    for cmd in times_int8:
        print(cmd)
        print(f"Times int8: {times_int8[cmd]}")
        # TODO: WE need account for the different data types in the commands
        if cmd in times_fp16:
            time_int8 = times_int8[cmd]
            time_fp16 = times_fp16[cmd]
            print(f"int8 time: {time_int8}, fp16 time: {time_fp16}")
            if time_fp16 > 0:
                speedup = (time_fp16 - time_int8) / time_fp16 * 100
                speedup_percentage.append(speedup)

    n_samples = len(speedup_percentage)
    x = np.arange(n_samples)
    plt.figure(figsize=(10, 6))
    plt.plot(x, speedup_percentage, marker='o')
    plt.title('Speedup of int8 over fp16')
    plt.xlabel('Sample Index')
    plt.ylabel('Speedup (%)')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close() 

def main():
  args = parse_cli_args()

  times_int8 = parse_times(args.input_file_int8)
  times_fp16 = parse_times(args.input_file_fp16)

  #avg_times_int8 = np.mean(np.array(times_int8.items()))
  #avg_times_fp16 = np.mean(np.array(times_fp16.items()))

  print(f"Got {len(times_int8)} int8 samples and {len(times_fp16)} fp16 samples.")

#   print(f"Average time for int8: {avg_times_int8} ms")
#   print(f"Average time for fp16: {avg_times_fp16} ms")
#   print(f"Speedup (int8 over fp16): {avg_times_fp16 / avg_times_int8:.2f}x")

  output_plot_file = "navi_perf_int8_vs_fp16.png"
  output_path = os.path.join(os.getcwd(), output_plot_file)
  plot_perf(times_int8, times_fp16, output_path)
  print(f"Performance plot saved to {output_path}")
  
if __name__ == "__main__":
    main()