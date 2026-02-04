#!/usr/bin/env python3

import os
import subprocess
import sys
import argparse

profiler_commands = [
  "1         1       1              0      1         0     1           2     32  32  4     4    3  3  200  200   1   1   1   1   1   1    1   1", 
  "1         1       1              0      1         0     1           2     32  32  8     8    3  3  200  200   2   2   1   1   1   1    1   1", 
  "1         1       1              0      1         0     1           2     32  32  8     8    3  3  100  100   1   2   1   1   1   1    1   1", 
  "1         1       1              0      1         0     1           2     1   32  2376  256  3  3  100  100   1   1   1   1   1   1    1   1", 
  "1         1       1              0      1         0     1           2     1   32  256   256  3  3  100  100   1   1   1   1   1   1    1   1"
  ]

baseline_instances = [
  "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 64, 64, 32, Default, 16, 16, 2, 2, 4, 4, 4, 1, 1, 1>", 
  "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_DirectLoad<128, 16, 64, 64, Default, 16, 16, 1, 2, 8, 8, 4, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, 1>",
  "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_DirectLoad<256, 256, 32, 64, Default, 32, 32, 2, 1, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, 1>", 
  "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 32, Default, 32, 32, 4, 2, 8, 8, 8, 1, 1, 1>", 
  "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 256, 128, 32, OddC, 32, 32, 4, 2, 8, 8, 8, 1, 1, 1>"
]

improved_instances = [
  "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 128, 32, 32, Default, 32, 32, 2, 1, 4, 4, 1, 1, 1, BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1, 8>",
  "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 32, Default, 32, 32, 4, 2, 8, 8, 1, 1, 1, 8>",
  "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 64, 32, Default, 32, 32, 4, 2, 8, 8, 1, 1, 1, 8>",
  "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 64, Default, 32, 32, 4, 4, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3, 1>",
  "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<256, 256, 256, 64, Default, 32, 32, 4, 4, 8, 8, 8, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3, 1>"
]

def main():
  # Parse command-line arguments
  parser = argparse.ArgumentParser(description='Run CK profiler with best instances for given conv shapes.')
  parser.add_argument('--profiler-path', type=str, required=True, help='Path to the profiler binary')
  parser.add_argument('--baseline', action='store_true', 
                      help='Run baseline instances (default: run improved instances)')
  parser.add_argument("--print-stdout", action='store_true', help='Print CK profiler output to stdout')
  args = parser.parse_args()

  instances_to_run = baseline_instances if args.baseline else improved_instances
  instance_type = "baseline" if args.baseline else "improved"

  print(f"Running {instance_type} instances...\n")

  ck_profiler_path = args.profiler_path
  if not os.path.isfile(ck_profiler_path):
      print(f"Error: Profiler binary not found at {ck_profiler_path}")
      sys.exit(1)

  for i in range(len(profiler_commands)):
        command = profiler_commands[i]
        instance = instances_to_run[i]
        profiler_args = [x for x in command.split()]
        profiler_args.append(instance)

        print(f"Running profiler for {instance_type} instance {i+1}/{len(profiler_commands)}:")
        print(instance)
        res = subprocess.run([ck_profiler_path] + ["grouped_conv_fwd"] + profiler_args, check=True, timeout=300, 
                       capture_output=True, text=True)
        if args.print_stdout:
            print(res.stdout)
        print()

if __name__ == "__main__":
    main()
