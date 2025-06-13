#!/usr/bin/env python3

import sys
import os
import argparse
import pandas as pd
import csv

from convert_miopen_driver_to_profiler import get_parser, init_const_args, process_miopen_driver_name, \
  get_ck_grouped_conv_fwd_cmd, get_ck_grouped_conv_bwd_data_cmd, get_ck_grouped_conv_bwd_weight_cmd

def parse_cli_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run CK convolution profiler.")
    parser.add_argument("--fremont-csv-file", type=str, dest="fremont_csv_file", required=True, help="Path to the CSV file containing Fremont test cases.")
    parser.add_argument("--ktn-csv-file", type=str, dest="ktn_csv_file", required=True, help="Path to the CSV file containing KTN test cases.")
    parser.add_argument("--fwd-only", action="store_true", help="Run only forward convolution.")
    parser.add_argument("--bwd-data-only", action="store_true", help="Run only backward data convolution.")
    parser.add_argument("--bwd-weight-only", action="store_true", help="Run only backward weight convolution.")
    parser.add_argument("--no-verification", action="store_true", help="Disable verification in the CK profiler.")
    parser.add_argument("--full-set", action="store_true", help="Create a full set of tests. By default only a subset of the raw data is used.")
    parser.add_argument("--output-path", type=str, dest="output_path", default=".", help="Path to save the output files. Default is current directory.")

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

def parse_fremont_profiler_commands(csv_file, no_verification=False, fwd_only=False, bwd_data_only=False, bwd_weight_only=False):
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

def process_miopen_driver(args, unknown):
    if "convint8" in unknown:
        args.data_type = 'int8'
    elif "convbfp16" in unknown:
        args.data_type = 'bfp16'
    elif "convfp16" in unknown:
        args.data_type = 'fp16'
    elif "conv" in unknown:
        args.data_type = 'fp32'
    else:
        print('Not supported driver (supported: conv, convfp16, convint8,'
              ' convbfp16).')
        exit(1)

def parse_ktn_command(csv_file, no_verification=False, fwd_only=False, bwd_data_only=False, bwd_weight_only=False):
  if not os.path.isfile(csv_file):
      print(f"Error: The specified CSV file '{csv_file}' does not exist.", file=sys.stderr)
      sys.exit(1)

  df = pd.read_csv(csv_file)

  # Remove the KTN commands where column 'Group Size' has value 1
  df = df[df['Group Size'] != 1]
  
  commands = []
  parser = get_parser()
  for i, cmd in enumerate(df['Command']):
    cmd = cmd.strip()
    args, _ = parser.parse_known_args(cmd)

    init_const_args(args)
    process_miopen_driver(args, cmd.split()[0])
    
    if no_verification:
        args.verify = 0
    else:
      args.verify = 1         

    # Ensure we run always the timing.
    args.time = 1

    command = parse_profiler_command(args, 
                                      fwd_only=fwd_only, 
                                      bwd_data_only=bwd_data_only, 
                                      bwd_weight_only=bwd_weight_only)
    if command is not None:
      commands.append(command)

  return commands

def main():
    args = parse_cli_args()

    # Initialize random seed for reproducibility
    seed = 42
    n_fremont_shapes = 1000
    n_ktn_shapes = 1000

    fremont_commands = parse_fremont_profiler_commands(args.fremont_csv_file, 
                                                       no_verification=args.no_verification, 
                                                       fwd_only=args.fwd_only, 
                                                       bwd_data_only=args.bwd_data_only, 
                                                       bwd_weight_only=args.bwd_weight_only)

    ktn_commands = parse_ktn_command(args.ktn_csv_file,
                                     no_verification=args.no_verification, 
                                     fwd_only=args.fwd_only, 
                                     bwd_data_only=args.bwd_data_only, 
                                     bwd_weight_only=args.bwd_weight_only)

    # Create a DataFrame to hold the commands
    commands_fremont_df = pd.DataFrame({
        'Command': fremont_commands,
    })

    commands_ktn_df = pd.DataFrame({
        'Command': ktn_commands,
    })

    # Take a randomly sampled subset of Fremont commands
    if not args.full_set:
        commands_fremont_df = commands_fremont_df.sample(n=min(n_fremont_shapes, len(commands_fremont_df)), random_state=seed)
        commands_ktn_df = commands_ktn_df.sample(n=min(n_ktn_shapes, len(commands_ktn_df)), random_state=seed)

    output_file = os.path.join(args.output_path, "ck_profiler_commands.csv")
    with open(output_file, "w") as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for command in commands_fremont_df['Command']:
            csv_writer.writerow(command)        
        for command in commands_ktn_df['Command']:
            csv_writer.writerow(command)
    print(f"Commands saved to {output_file}")

if __name__ == "__main__":
    main()