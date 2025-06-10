#!/usr/bin/env python3

import os
import argparse
import subprocess
import sys
import pandas as pd
from convert_miopen_driver_to_profiler import get_parser, init_const_args, process_miopen_driver_name, \
  get_ck_grouped_conv_fwd_cmd, get_ck_grouped_conv_bwd_data_cmd, get_ck_grouped_conv_bwd_weight_cmd

def parse_cli_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run CK convolution profiler.")
    parser.add_argument("--csv-file", type=str, dest="csv_file", required=True, help="Path to the CSV file containing test cases.")
    parser.add_argument("--fwd-only", action="store_true", help="Run only forward convolution.")
    parser.add_argument("--bwd-data-only", action="store_true", help="Run only backward data convolution.")
    parser.add_argument("--bwd-weight-only", action="store_true", help="Run only backward weight convolution.")
    parser.add_argument("--start", type=int, default=None, help="Start index for processing shapes in the CSV file.")
    parser.add_argument("--end", type=int, default=None, help="End index for processing shapes in the CSV file. If None, process all shapes.")
    parser.add_argument("--no-verification", action="store_true", help="Disable verification in the CK profiler.")
    parser.add_argument("--log-to-stdout", action="store_true", help="Log profiler output to stdout instead of /dev/null.")
    
    args, unknown_args = parser.parse_known_args()
    
    if unknown_args:
        print(f"Unknown arguments: {unknown_args}", file=sys.stderr)
        sys.exit(1)
    
    return args

def parse_profiler_command(args, fwd_only=False, bwd_data_only=False, bwd_weight_only=False):
  # MIOpen get number of channel per all groups, CK profiler get number of
  # channel per group
  args.in_channels = int(args.in_channels / args.group_count)
  args.out_channels = int(args.out_channels / args.group_count)

  cmd = None
  if fwd_only:
      args.forw = 1
      cmd = get_ck_grouped_conv_fwd_cmd(args)
  if bwd_data_only:
      args.forw = 2
      cmd = get_ck_grouped_conv_bwd_data_cmd(args)
  if bwd_weight_only:
      args.forw = 4
      cmd = get_ck_grouped_conv_bwd_weight_cmd(args)

  return cmd

def get_profiler_commands(csv_file, no_verification=False, fwd_only=False, bwd_data_only=False, bwd_weight_only=False):
  if not os.path.isfile(csv_file):
      print(f"Error: The specified CSV file '{csv_file}' does not exist.", file=sys.stderr)
      sys.exit(1)

  df = pd.read_csv(csv_file)
  shapes = df['Shape'].tolist()
  parser = get_parser()
  commands = []

  for i, line in enumerate(shapes):
    try:
      args, unknown = parser.parse_known_args(line.split())

      init_const_args(args)
      process_miopen_driver_name(args, unknown)
      assert len(unknown) == 4 and unknown[0] == "--fil_layout" and unknown[2] == "--out_layout" and unknown[1] == unknown[3], \
        f"Error: Unknown arguments do not match: {unknown}"
      assert unknown[1] == args.in_layout, \
        f"Error: Input layout does not match unknown arguments: {unknown[1]} != {args.in_layout}"
      
      if no_verification:
         args.verify = 0

      # Ensure we run always the timing.
      args.time = 1

      command = parse_profiler_command(args, 
                                       fwd_only=fwd_only, 
                                       bwd_data_only=bwd_data_only, 
                                       bwd_weight_only=bwd_weight_only)
      if command is not None:

        commands.append(command)
    except AttributeError as e:
      print(f"Error processing line {i}: {line}. Skipping the line.")
      continue

  return commands

def run_ck_profiler_cmd(cmd, log_to_stdout=False):
    cmd_concatenated_str = ""
    for arg in cmd:
        cmd_concatenated_str += arg + " "
    working_dir = os.path.dirname(os.path.abspath(__file__))
    pid = os.getpid()
    env_vars = os.environ.copy()
    env_vars["CK_PROFILER_DISABLED_OPS"] = "DeviceGroupedConvBwdWeight_Dl;DeviceGroupedConvBwdWeight_Xdl_CShuffleV3"
    env_vars["CK_PROFILER_OUTPUT_FILE"] = f"{working_dir}/conv_profiler_output_{pid}.csv"
    if log_to_stdout:
      subprocess.run(cmd, env=env_vars, stdout=devnull) 
    else:
      with open(os.devnull, 'w') as devnull:
        subprocess.run(cmd, env=env_vars, stdout=devnull, stderr=devnull)

def main():
  args = parse_cli_args()
  profiler_commands = get_profiler_commands(args.csv_file, args.no_verification, args.fwd_only, args.bwd_data_only, args.bwd_weight_only)
  print(f"Got {len(profiler_commands)} commands in total to run.")
  if args.start is not None:
      end = len(profiler_commands)
      if args.end is not None:
        end = min(args.end, end)
      profiler_commands = profiler_commands[args.start-1:end]

  for i, cmd in enumerate(profiler_commands):
      print(f"Running command {i + 1}/{len(profiler_commands)}: {cmd}")
      run_ck_profiler_cmd(cmd, args.log_to_stdout)
  
if __name__ == "__main__":
    main()