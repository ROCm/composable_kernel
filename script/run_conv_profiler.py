#!/usr/bin/env python3

import os
import argparse
import subprocess
import sys

def parse_cli_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run CK convolution profiler.")
    parser.add_argument("--csv-file", type=str, dest="csv_file", required=True, help="Path to the CSV file containing test cases.")
    parser.add_argument("--log-to-stdout", action="store_true", help="Log profiler output to stdout instead of /dev/null.")
    parser.add_argument("--start", type=int, default=None, help="Start index for the commands to run (1-based).")
    parser.add_argument("--end", type=int, default=None, help="End index for the commands to run (1-based).")
    parser.add_argument("--disabled-ops", type=str, default=None, help="Comma-separated list of disabled operations for the profiler.")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID for the profiler run.")
    
    args, unknown_args = parser.parse_known_args()
    
    if unknown_args:
        print(f"Unknown arguments: {unknown_args}", file=sys.stderr)
        sys.exit(1)
    
    return args

def run_ck_profiler_cmd(cmd, disabled_ops, run_id, log_to_stdout=False):
    working_dir = os.path.dirname(os.path.abspath(__file__))
    pid = os.getpid()
    env_vars = os.environ.copy()
    disabled_ops_str = ""
    for op in disabled_ops:
      disabled_ops_str += op + ";"
    disabled_ops_str = disabled_ops_str.strip(';')
    env_vars["CK_PROFILER_DISABLED_OPS"] = disabled_ops_str
    env_vars["CK_PROFILER_OUTPUT_DIR"] = working_dir
    run_id_str = f"_{run_id}" if run_id else ""
    env_vars["CK_PROFILER_OUTPUT_FILE"] = f"{working_dir}/conv_profiler_output{run_id_str}_{pid}.csv"
    if log_to_stdout:
      subprocess.run(cmd, env=env_vars) 
    else:
      with open(os.devnull, 'w') as devnull:
        subprocess.run(cmd, env=env_vars, stdout=devnull)

def get_profiler_commands(csv_file):
  profiler_commands = []
  with open(csv_file, 'r') as f:
    for line in f:
        line = line.strip()
        cmd = line.split(',')
        profiler_commands.append(cmd)
  return profiler_commands

def main():
  args = parse_cli_args()
  profiler_commands = get_profiler_commands(args.csv_file)
  print(f"Got {len(profiler_commands)} commands in total to run.")

  if args.start is not None:
      end = len(profiler_commands)
      if args.end is not None:
        end = min(args.end, end)
      profiler_commands = profiler_commands[args.start-1:end]

  disabled_ops = []
  if args.disabled_ops:
      disabled_ops = args.disabled_ops.split(',')
      disabled_ops = [op.strip() for op in disabled_ops if op.strip()]

  for i, cmd in enumerate(profiler_commands):
      cmd_concatenated_str = ""
      for arg in cmd:
          cmd_concatenated_str += arg + " "
      cmd_concatenated_str = cmd_concatenated_str.strip()
      print(f"\n##################################################################################################################################")
      print(f"Running command {i + 1}/{len(profiler_commands)}: {cmd_concatenated_str}")
      print(f"##################################################################################################################################")
      run_ck_profiler_cmd(cmd, disabled_ops, args.run_id, args.log_to_stdout)
  
if __name__ == "__main__":
    main()