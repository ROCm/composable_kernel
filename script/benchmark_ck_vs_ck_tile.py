#!/usr/bin/env python3

import os
import argparse
import subprocess
import sys

def parse_cli_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run CK and CK Tile convolution profilers.")
    parser.add_argument("--file", type=str, dest="file", required=True, help="Path to the file containing test cases.")
    parser.add_argument("--log-to-stdout", action="store_true", help="Log profiler output to stdout instead of /dev/null.")
    parser.add_argument("--bin-path", type=str, dest="bin_path", required=True, help="Path to the CK/CK Tile profiler executables.")
    
    args, unknown_args = parser.parse_known_args()
    
    if unknown_args:
        print(f"Unknown arguments: {unknown_args}", file=sys.stderr)
        sys.exit(1)
    
    return args

class ProfilerType:
    CK = 1
    CK_TILE = 2

def run_ck_profiler_cmd(cmd_args, profiler_type, bin_path, log_to_stdout=False):
    profiler = "ckTileProfiler" if profiler_type == ProfilerType.CK_TILE else "ckProfiler"
    profiler_path = os.path.join(bin_path, profiler)
    cmd = [profiler_path] + cmd_args
    cmd_str = ' '.join(cmd)

    if log_to_stdout:
      subprocess.run(cmd) 
    else:
      with open(os.devnull, 'w') as devnull:
        timeoutInSec = 300
        try:
          subprocess.run(cmd, stdout=devnull, timeout=timeoutInSec)
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
        cmd = [x.strip() for x in line.split(' ') if x.strip() and x.strip() != '']
        profiler_commands.append(cmd)
  return profiler_commands

def main():
  args = parse_cli_args()
  profiler_commands = get_profiler_commands(args.file)
  print(f"Got {len(profiler_commands)} unique commands to run.")

  for i, cmd in enumerate(profiler_commands):
      cmd_concatenated_str = ' '.join(cmd)
      print(f"\n####################################################################################################################")
      print(f"Running command {i + 1}/{len(profiler_commands)}: {cmd_concatenated_str}")
      print(f"######################################################################################################################")
      run_ck_profiler_cmd(cmd, ProfilerType.CK_TILE, args.bin_path, args.log_to_stdout)
      run_ck_profiler_cmd(cmd, ProfilerType.CK, args.bin_path, args.log_to_stdout)
  
if __name__ == "__main__":
    main()