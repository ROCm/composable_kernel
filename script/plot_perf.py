#!/usr/bin/env python3

import os
import argparse
import sys
import matplotlib.pyplot as plt
# Non-interactive backend for matplotlib
plt.switch_backend('Agg')
import numpy as np

def parse_cli_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run CK and CK Tile convolution profilers.")
    parser.add_argument("--input-file-int8", type=str, dest="input_file_int8", required=True, help="Path to the file containing test results for int8.")
    parser.add_argument("--input-file-fp16", type=str, dest="input_file_fp16", required=True, help="Path to the file containing test results for fp16.")
    parser.add_argument("--label", type=str, dest="label", default="", help="Label for the plot.")
    parser.add_argument("--device-type", type=str, dest="device_type", default="", help="Device type.")
    parser.add_argument("--output-dir", type=str, dest="output_dir", default="", help="Output directory for plots.")
    
    args, unknown_args = parser.parse_known_args()
    
    if unknown_args:
        print(f"Unknown arguments: {unknown_args}", file=sys.stderr)
        sys.exit(1)
    
    return args

def parse_fwd_conv_igemm_params(cmd):
    args = cmd.strip().split(" ")
    
    # Args: 0-6: are control variables
    dim = int(args[7])
    print(f"Convolution dimension: {dim}D")
    if dim == 2:
        G = int(args[8])
        N = int(args[9])
        K = int(args[10])
        C = int(args[11])
        Y = int(args[12])
        X = int(args[13])
        H = int(args[14])
        W = int(args[15])
    else:
        raise ValueError(f"Unsupported convolution dimension: {dim}")

    Gemm_M = N * H * W
    Gemm_N = K
    Gemm_K = C * Y * X

    return Gemm_M, Gemm_N, Gemm_K

def calculate_arithmetic_intensity(cmd):
    Gemm_M, Gemm_N, Gemm_K = parse_fwd_conv_igemm_params(cmd)

    # Parse data type
    args = cmd.strip().split(" ")
    data_type = args[1]
    if data_type == "1": # fp16
        data_type_size = 2  # bytes
    elif data_type == "3": # int8
        data_type_size = 1  # bytes
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    # Total FLOPs for GEMM: 2 * M * N * K
    total_flops = 2 * Gemm_M * Gemm_N * Gemm_K

    # Total data movement in bytes: A + B + C
    # A: M x K, B: K x N, C: M x N
    total_data_movement = (Gemm_M * Gemm_K + Gemm_K * Gemm_N + Gemm_M * Gemm_N) * data_type_size

    arithmetic_intensity = total_flops / total_data_movement  # FLOPs per byte

    return arithmetic_intensity

class PerfData:
    def __init__(self, avg_time, arithmetic_intensity, tflops):
        self.avg_time = avg_time
        self.arithmetic_intensity = arithmetic_intensity
        self.tflops = tflops

def parse_times(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
        avg_time_lines = lines[2::4]  # Every 4th line starting from line 2
        avg_times = [float(line.strip().split("avg_time: ")[-1]) for line in avg_time_lines]
        tflops_lines = lines[3::4]  # Every 4th line starting from line 3
        tflops = [float(line.strip().split("flops: ")[-1]) for line in tflops_lines]
        commnds = lines[0::4]  # Every 4th line starting from line 0

        # Create a dictionary of commands to their average times
        cmd_time_dict = {}
        for cmd, time, flops in zip(commnds, avg_times, tflops):
            arithmetic_intensity = calculate_arithmetic_intensity(cmd)
            cmd_parts = cmd.strip().split(" ")
            # Reconstruct the command without the data type part
            cmd_parts[1] = "<data_type>"
            cmd_reconstructed = " ".join(cmd_parts)
            cmd_time_dict[cmd_reconstructed] = PerfData(avg_time=time, arithmetic_intensity=arithmetic_intensity, tflops=flops)
    
    return cmd_time_dict

def plot_roofline(times_int8, times_fp16, output_file, device_type):
    if device_type == "MI300X":
        peak_int8_tflops = 2614.9
        peak_fp16_tflops = 1307.4
        peak_bandwidth = 5.3  # HBM bandwidth 5.3 TB/s
    elif device_type == "RX9070XT":
        peak_int8_tflops = 779.0
        peak_fp16_tflops = 195.0
        peak_bandwidth = 0.705  # GDDR bandwidth 0.705 TB/s
    else:
        raise ValueError(f"Unsupported device type: {device_type}")

    arithmetic_intensity = []
    perf_in_tflops_int8 = []
    perf_in_tflops_fp16 = []
    for cmd in times_int8:
        if cmd in times_fp16:
            tflops_int8 = times_int8[cmd].tflops
            tflops_fp16 = times_fp16[cmd].tflops
            perf_in_tflops_int8.append(tflops_int8)
            perf_in_tflops_fp16.append(tflops_fp16)
            arithmetic_int = times_int8[cmd].arithmetic_intensity
            arithmetic_intensity.append(arithmetic_int)              

    plt.figure(figsize=(10, 6))
    ai = np.logspace(-1, 4, 100)
    int8_perf = np.minimum(peak_int8_tflops, ai * peak_bandwidth)
    fp16_perf = np.minimum(peak_fp16_tflops, ai * peak_bandwidth)
    plt.loglog(ai, int8_perf, label='int8 Roofline', color='blue')
    plt.loglog(ai, fp16_perf, label='fp16 Roofline', color='orange')

    # Plot data points
    plt.scatter(arithmetic_intensity, perf_in_tflops_int8, label='int8 Performance', color='cyan', marker='o', alpha=0.7)
    plt.scatter(arithmetic_intensity, perf_in_tflops_fp16, label='fp16 Performance', color='red', marker='o', alpha=0.7)

    plt.xlabel('Arithmetic Intensity (FLOPs/Byte)')
    plt.ylabel('Performance (TFLOPs)')
    plt.title(f'Roofline Model on {device_type}' if device_type else 'Roofline Model')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(output_file)
    plt.close()

def plot_perf(times_int8, times_fp16, output_file, device_type):
    
    # Exclude cases which run too fast, i.e., runtime < 0.01 ms
    eps = 0.01

    speedup_percentage = []
    arithmetic_intensity = []
    for cmd in times_int8:
        if cmd in times_fp16:
            time_int8 = times_int8[cmd].avg_time
            time_fp16 = times_fp16[cmd].avg_time
            if time_fp16 > 0 and time_int8 > eps:
                speedup = (time_fp16 - time_int8) / time_fp16 * 100
                arithmetic_int = times_int8[cmd].arithmetic_intensity
                speedup_percentage.append(speedup)
                arithmetic_intensity.append(arithmetic_int)
                print("-----------------------------")
                print(f"int8 time: {time_int8}, fp16 time: {time_fp16}, arithmetic intensity: {arithmetic_int:.2f}")
                if speedup > 0:
                    print(f"\033[92mSpeedup for command {cmd}: {speedup:.2f}%\033[0m")
                else:
                    print(f"\033[91mNegative speedup (slowdown) for command {cmd}: {speedup:.2f}%\033[0m")
                # Filter out extreme speedup values
                # if abs(speedup) < 1000:
                #     speedup_percentage.append(speedup)
                #     arithmetic_intensity.append(arithmetic_int)
                # else:
                #     print(f"\033[93mWARNING: Skipping extreme speedup value: {speedup:.2f}% for command {cmd}\033[0m")

    title = f"Speedup of int8 over fp16 on {device_type}" if device_type else "Speedup of int8 over fp16"

    n_samples = len(speedup_percentage)
    x = np.arange(n_samples)
    
    plt.figure(figsize=(14, 8))
    scatter = plt.scatter(x, arithmetic_intensity, c=speedup_percentage, 
                         cmap='RdBu_r', s=100, alpha=0.8, 
                         vmin=-50, vmax=50, edgecolors='black', linewidth=0.5)
    
    plt.colorbar(scatter, label='Speedup (%)')
    plt.yscale('log')
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Arithmetic Intensity (FLOPs/Byte)')
    plt.grid(True, alpha=0.3, which='both')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

def main():
  args = parse_cli_args()

  times_int8 = parse_times(args.input_file_int8)
  times_fp16 = parse_times(args.input_file_fp16)

  print(f"Got {len(times_int8)} int8 samples and {len(times_fp16)} fp16 samples.")

  output_base_dir = args.output_dir if args.output_dir else os.getcwd()
  if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

  output_plot_file = f"navi_perf_int8_vs_fp16_{args.label}.png" if args.label else "navi_perf_int8_vs_fp16.png" 
  output_path = os.path.join(output_base_dir, output_plot_file)
  plot_perf(times_int8, times_fp16, output_path, args.device_type)

  print()
  print(f"Performance plot saved to {output_path}")

  output_plot_file = f"roofline_{args.label}.png" if args.label else "roofline.png" 
  output_path = os.path.join(output_base_dir, output_plot_file)
  plot_roofline(times_int8, times_fp16, output_path, args.device_type)
  
if __name__ == "__main__":
    main()