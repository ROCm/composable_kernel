#!/usr/bin/env python3

import sys
import argparse
import os

def parse_miopen_command(miopen_cmd):
    """Parse MIOpen driver command and extract parameters"""
    # Remove 'convbfp16' or similar prefix and split into arguments
    parts = miopen_cmd.strip().split()
    if not parts:
        return None
    
    # Skip the command name (convbfp16, convfp16, etc.)
    args = parts[1:] if parts[0].startswith('conv') else parts
    
    params = {}
    i = 0
    while i < len(args):
        if args[i].startswith('-'):
            key = args[i]
            if i + 1 < len(args) and not args[i + 1].startswith('-'):
                params[key] = args[i + 1]
                i += 2
            else:
                params[key] = True
                i += 1
        else:
            i += 1
    
    return params

def determine_operation_type(params):
    """Determine the operation type based on MIOpen parameters"""
    # TODO: Current data is for bwd weight.
    return "grouped_conv_bwd_data"
    #return "grouped_conv_bwd_weight"
    #return "grouped_conv_fwd"#"grouped_conv_bwd_weight"

def convert_miopen_to_ck_profiler(miopen_cmd):
    """Convert MIOpen driver command to CK profiler command"""
    params = parse_miopen_command(miopen_cmd)
    if not params:
        return None
    
    # Determine operation type
    operation = determine_operation_type(params)
    
    data_type    = 2 #2 for bwd data 2 FOR FWD 5 FOR BWD WEI # BF16
    layout       = 1 #1 FIR BWD DATA 1 FOR FWD 2 FOR BWE WEI # channels last
    verification = 1 # with verification
    init_method  = 2 # uniform data
    print_output = 0 # no print output
    time_kernel  = 1 # time kernel
    n_dim        = 2 # 2D convolution by default  
    
    # Build CK profiler command
    ck_cmd = [operation, str(data_type), str(layout), str(verification),
              str(init_method), str(print_output), str(time_kernel), str(n_dim)]
    
    # Add tensor dimensions
    G = params.get('-g', '1')
    N = params.get('-n', '1')
    K = params.get('-k', '1')
    C = params.get('-c', '1')
    
    ck_cmd.extend([G, N, K, C])
    
    Y = params.get('-y', '1')
    X = params.get('-x', '1')
    ck_cmd.extend([Y, X])
    
    # Input dimensions
    Hi = params.get('-H', '1')
    Wi = params.get('-W', '1')
    ck_cmd.extend([Hi, Wi])
    
    # Stride
    stride_h = params.get('-u', '1')
    stride_w = params.get('-v', '1')
    ck_cmd.extend([stride_h, stride_w])
    
    # Dilation
    dilation_h = params.get('-l', '1')
    dilation_w = params.get('-j', '1')
    ck_cmd.extend([dilation_h, dilation_w])
    
    # Padding
    pad_h = params.get('-p', '0')
    pad_w = params.get('-q', '0')
    ck_cmd.extend([pad_h, pad_w, pad_h, pad_w])  # Assuming symmetric padding

    # Split-K
    split_k = "all"
    ck_cmd.append(split_k)
    
    return ' '.join(ck_cmd)

def convert_file(input_file, output_file):
    """Convert MIOpen commands from input file and save CK profiler commands to output file"""
    converted_commands = []
    failed_conversions = []
    
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        # Skip header line if present
        start_idx = 0
        if lines and (lines[0].strip().lower() == 'shape' or 'conv' not in lines[0].lower()):
            start_idx = 1
        
        for line_num, line in enumerate(lines[start_idx:], start_idx + 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            ck_cmd = convert_miopen_to_ck_profiler(line)
            if ck_cmd:
                converted_commands.append(ck_cmd)
            else:
                failed_conversions.append((line_num, line))
        
        # Write converted commands to output file
        with open(output_file, 'w') as f:
          for cmd in converted_commands:
              f.write(cmd + '\n')
        
        print(f"Conversion completed successfully!")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print(f"Converted {len(converted_commands)} commands")
        
        if failed_conversions:
            print(f"\nFailed to convert {len(failed_conversions)} commands:")
            for line_num, line in failed_conversions[:5]:  # Show first 5 failures
                print(f"  Line {line_num}: {line[:80]}...")
            if len(failed_conversions) > 5:
                print(f"  ... and {len(failed_conversions) - 5} more")
        
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        return False
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False

def generate_output_filename(input_file):
    """Generate output filename based on input filename"""
    base_name = os.path.splitext(input_file)[0]
    return f"{base_name}_ck_profiler.txt"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Convert MIOpen driver commands to CK profiler commands',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python3 convert_miopen_driver_to_profiler.py miopen_commands.txt
  python3 convert_miopen_driver_to_profiler.py miopen_commands.txt -o ck_commands.txt
  python3 convert_miopen_driver_to_profiler.py --validate miopen_commands.txt"""
    )
    
    parser.add_argument('input_file', 
                       help='Input file containing MIOpen driver commands')
    parser.add_argument('-o', '--output', 
                       help='Output file for CK profiler commands (default: auto-generated)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate converted commands (dry run)')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    print(args.input_file)
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)

    print(args.input_file)
    
    # Generate output filename if not provided
    output_file = args.output if args.output else generate_output_filename(args.input_file)
    
    # Validate mode - just show what would be converted
    if args.validate:
        print("Validation mode - showing first 5 conversions:")
        try:
            with open(args.input_file, 'r') as f:
                lines = f.readlines()[:6]  # Header + 5 commands
            
            start_idx = 1 if lines and 'conv' not in lines[0].lower() else 0
            
            for i, line in enumerate(lines[start_idx:start_idx+5]):
                line = line.strip()
                if not line:
                    continue
                
                ck_cmd = convert_miopen_to_ck_profiler(line)
                print(f"\n{i+1}. MIOpen: {line[:80]}{'...' if len(line) > 80 else ''}")
                print(f"   CK:     {ck_cmd if ck_cmd else 'CONVERSION FAILED'}")
            
            print(f"\nWould write to: {output_file}")
        except Exception as e:
            print(f"Validation error: {e}")
        
        return
    
    # Perform the actual conversion
    success = convert_file(args.input_file, output_file)
    
    if success:
        print(f"\nConversion completed! You can now use the converted commands with:")
        print(f"  # For individual command testing:")
        print(f"  ./profiler/ck_tile/ckTileProfiler <command_from_{os.path.basename(output_file)}>")
        print(f"  ")
        print(f"  # For batch processing, you can create a script that reads from {os.path.basename(output_file)}")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()