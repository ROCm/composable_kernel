#!/usr/bin/env python3
"""
Generate Systematic Convolution Test Cases

Creates comprehensive parameter sweeps for convolution testing.
Complements MIOpen-captured cases with systematic coverage.

Usage:
    python3 generate_systematic_cases.py --count 5000 --ndim 2 --output systematic_2d.csv
    python3 generate_systematic_cases.py --count 1000 --ndim 3 --output systematic_3d.csv
    python3 generate_systematic_cases.py --preset resnet --output resnet_cases.csv
"""

import argparse
import csv
import random
import itertools

def generate_systematic_2d(target_count):
    """Generate systematic 2D convolution parameter combinations"""
    
    # Parameter ranges
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    groups_list = [1, 2, 4, 8, 16, 32]
    
    # Channel pairs (input, output)
    channel_pairs = [
        (3, 32), (3, 64), (32, 64), (64, 128), (128, 256), 
        (256, 512), (512, 1024), (64, 32), (128, 64), (256, 128)
    ]
    
    kernel_sizes = [(1, 1), (3, 3), (5, 5), (7, 7), (1, 3), (3, 1)]
    input_sizes = [(7, 7), (14, 14), (28, 28), (56, 56), (112, 112), (224, 224)]
    stride_configs = [(1, 1), (2, 2), (1, 2), (2, 1), (3, 3)]
    dilation_configs = [(1, 1), (2, 2), (1, 2), (2, 1)]
    
    # Generate all combinations
    all_combinations = list(itertools.product(
        batch_sizes, groups_list, channel_pairs, kernel_sizes,
        input_sizes, stride_configs, dilation_configs
    ))
    
    # Shuffle and limit
    random.shuffle(all_combinations)
    combinations = all_combinations[:target_count]
    
    test_cases = []
    for i, (batch, groups, (in_ch, out_ch), (kh, kw), (ih, iw), (sh, sw), (dh, dw)) in enumerate(combinations):
        
        # Adjust channels for grouped convolution
        if groups > 1:
            in_ch = ((in_ch // groups) * groups) if in_ch > groups else groups
            out_ch = ((out_ch // groups) * groups) if out_ch > groups else groups
        
        # Calculate padding for "same" padding when possible
        ph = max(0, (kh - 1) // 2)
        pw = max(0, (kw - 1) // 2)
        
        # Calculate output size
        oh = (ih + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        ow = (iw + 2 * pw - dw * (kw - 1) - 1) // sw + 1
        
        # Skip invalid configurations
        if oh <= 0 or ow <= 0:
            continue
            
        test_case = {
            'NDim': 2,
            'Groups': groups,
            'BatchSize': batch,
            'OutChannels': out_ch,
            'InChannels': in_ch,
            'KernelH': kh, 'KernelW': kw,
            'InputH': ih, 'InputW': iw,
            'OutputH': oh, 'OutputW': ow,
            'StrideH': sh, 'StrideW': sw,
            'DilationH': dh, 'DilationW': dw,
            'LeftPadH': ph, 'LeftPadW': pw,
            'RightPadH': ph, 'RightPadW': pw,
            'TestName': f'Systematic_2D_{i+1}'
        }
        test_cases.append(test_case)
        
        if len(test_cases) >= target_count:
            break
    
    return test_cases

def generate_systematic_3d(target_count):
    """Generate systematic 3D convolution parameter combinations"""
    
    # Smaller ranges for 3D due to memory constraints
    batch_sizes = [1, 2, 4, 8, 16]
    groups_list = [1, 2, 4, 8]
    
    channel_pairs = [
        (3, 32), (32, 64), (64, 128), (128, 256), (64, 32), (128, 64)
    ]
    
    kernel_sizes = [(1, 1, 1), (3, 3, 3), (1, 3, 3), (3, 1, 1), (2, 2, 2)]
    input_sizes = [(8, 8, 8), (16, 16, 16), (4, 28, 28), (8, 14, 14), (2, 56, 56)]
    stride_configs = [(1, 1, 1), (2, 2, 2), (1, 2, 2), (2, 1, 1)]
    dilation_configs = [(1, 1, 1), (2, 2, 2), (1, 1, 2)]
    
    all_combinations = list(itertools.product(
        batch_sizes, groups_list, channel_pairs, kernel_sizes,
        input_sizes, stride_configs, dilation_configs
    ))
    
    random.shuffle(all_combinations)
    combinations = all_combinations[:target_count]
    
    test_cases = []
    for i, (batch, groups, (in_ch, out_ch), (kd, kh, kw), (id, ih, iw), (sd, sh, sw), (dd, dh, dw)) in enumerate(combinations):
        
        if groups > 1:
            in_ch = ((in_ch // groups) * groups) if in_ch > groups else groups
            out_ch = ((out_ch // groups) * groups) if out_ch > groups else groups
        
        # Calculate padding
        pd = max(0, (kd - 1) // 2)  
        ph = max(0, (kh - 1) // 2)
        pw = max(0, (kw - 1) // 2)
        
        # Calculate output size
        od = (id + 2 * pd - dd * (kd - 1) - 1) // sd + 1
        oh = (ih + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        ow = (iw + 2 * pw - dw * (kw - 1) - 1) // sw + 1
        
        if od <= 0 or oh <= 0 or ow <= 0:
            continue
            
        test_case = {
            'NDim': 3,
            'Groups': groups,
            'BatchSize': batch,
            'OutChannels': out_ch,
            'InChannels': in_ch,
            'KernelD': kd, 'KernelH': kh, 'KernelW': kw,
            'InputD': id, 'InputH': ih, 'InputW': iw,
            'OutputD': od, 'OutputH': oh, 'OutputW': ow,
            'StrideD': sd, 'StrideH': sh, 'StrideW': sw,
            'DilationD': dd, 'DilationH': dh, 'DilationW': dw,
            'LeftPadD': pd, 'LeftPadH': ph, 'LeftPadW': pw,
            'RightPadD': pd, 'RightPadH': ph, 'RightPadW': pw,
            'TestName': f'Systematic_3D_{i+1}'
        }
        test_cases.append(test_case)
        
        if len(test_cases) >= target_count:
            break
    
    return test_cases

def generate_preset_cases(preset_name):
    """Generate predefined model architectures"""
    
    if preset_name == 'resnet':
        return [
            # ResNet50 layers
            {'NDim': 2, 'Groups': 1, 'BatchSize': 32, 'OutChannels': 64, 'InChannels': 3,
             'KernelH': 7, 'KernelW': 7, 'InputH': 224, 'InputW': 224, 'OutputH': 112, 'OutputW': 112,
             'StrideH': 2, 'StrideW': 2, 'DilationH': 1, 'DilationW': 1,
             'LeftPadH': 3, 'LeftPadW': 3, 'RightPadH': 3, 'RightPadW': 3, 'TestName': 'ResNet_conv1'},
            
            {'NDim': 2, 'Groups': 1, 'BatchSize': 32, 'OutChannels': 64, 'InChannels': 64,
             'KernelH': 1, 'KernelW': 1, 'InputH': 56, 'InputW': 56, 'OutputH': 56, 'OutputW': 56,
             'StrideH': 1, 'StrideW': 1, 'DilationH': 1, 'DilationW': 1,
             'LeftPadH': 0, 'LeftPadW': 0, 'RightPadH': 0, 'RightPadW': 0, 'TestName': 'ResNet_block1_conv1'},
             
            {'NDim': 2, 'Groups': 1, 'BatchSize': 32, 'OutChannels': 64, 'InChannels': 64,
             'KernelH': 3, 'KernelW': 3, 'InputH': 56, 'InputW': 56, 'OutputH': 56, 'OutputW': 56,
             'StrideH': 1, 'StrideW': 1, 'DilationH': 1, 'DilationW': 1,
             'LeftPadH': 1, 'LeftPadW': 1, 'RightPadH': 1, 'RightPadW': 1, 'TestName': 'ResNet_block1_conv2'},
        ]
    
    elif preset_name == 'mobilenet':
        return [
            # MobileNet layers
            {'NDim': 2, 'Groups': 1, 'BatchSize': 32, 'OutChannels': 32, 'InChannels': 3,
             'KernelH': 3, 'KernelW': 3, 'InputH': 224, 'InputW': 224, 'OutputH': 112, 'OutputW': 112,
             'StrideH': 2, 'StrideW': 2, 'DilationH': 1, 'DilationW': 1,
             'LeftPadH': 1, 'LeftPadW': 1, 'RightPadH': 1, 'RightPadW': 1, 'TestName': 'MobileNet_conv1'},
             
            {'NDim': 2, 'Groups': 32, 'BatchSize': 32, 'OutChannels': 32, 'InChannels': 32,
             'KernelH': 3, 'KernelW': 3, 'InputH': 112, 'InputW': 112, 'OutputH': 112, 'OutputW': 112,
             'StrideH': 1, 'StrideW': 1, 'DilationH': 1, 'DilationW': 1,
             'LeftPadH': 1, 'LeftPadW': 1, 'RightPadH': 1, 'RightPadW': 1, 'TestName': 'MobileNet_dw1'},
             
            {'NDim': 2, 'Groups': 1, 'BatchSize': 32, 'OutChannels': 64, 'InChannels': 32,
             'KernelH': 1, 'KernelW': 1, 'InputH': 112, 'InputW': 112, 'OutputH': 112, 'OutputW': 112,
             'StrideH': 1, 'StrideW': 1, 'DilationH': 1, 'DilationW': 1,
             'LeftPadH': 0, 'LeftPadW': 0, 'RightPadH': 0, 'RightPadW': 0, 'TestName': 'MobileNet_pw1'},
        ]
    
    return []

def write_csv(test_cases, output_file):
    """Write test cases to CSV file"""
    if not test_cases:
        print("No test cases to write")
        return
    
    # Determine headers based on first test case
    sample = test_cases[0]
    if sample['NDim'] == 2:
        headers = ['NDim', 'Groups', 'BatchSize', 'OutChannels', 'InChannels',
                  'KernelH', 'KernelW', 'InputH', 'InputW', 'OutputH', 'OutputW',
                  'StrideH', 'StrideW', 'DilationH', 'DilationW', 
                  'LeftPadH', 'LeftPadW', 'RightPadH', 'RightPadW', 'TestName']
    else:
        headers = ['NDim', 'Groups', 'BatchSize', 'OutChannels', 'InChannels',
                  'KernelD', 'KernelH', 'KernelW', 'InputD', 'InputH', 'InputW', 
                  'OutputD', 'OutputH', 'OutputW', 'StrideD', 'StrideH', 'StrideW',
                  'DilationD', 'DilationH', 'DilationW', 
                  'LeftPadD', 'LeftPadH', 'LeftPadW', 'RightPadD', 'RightPadH', 'RightPadW', 'TestName']
    
    print(f"Writing {len(test_cases)} test cases to {output_file}")
    
    with open(output_file, 'w', newline='') as csvfile:
        csvfile.write(f"# Systematic {sample['NDim']}D Convolution Test Cases\n")
        csvfile.write(f"# Generated {len(test_cases)} test cases\n")
        
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        
        for case in test_cases:
            writer.writerow(case)

def main():
    parser = argparse.ArgumentParser(description='Generate systematic convolution test cases')
    
    parser.add_argument('--count', type=int, default=1000,
                       help='Number of test cases to generate')
    parser.add_argument('--ndim', type=int, choices=[2, 3], default=2,
                       help='Convolution dimension (2D or 3D)')
    parser.add_argument('--preset', choices=['resnet', 'mobilenet'],
                       help='Generate preset model architectures')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible generation')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    if args.preset:
        test_cases = generate_preset_cases(args.preset)
        print(f"Generated {len(test_cases)} {args.preset} preset cases")
    else:
        if args.ndim == 2:
            test_cases = generate_systematic_2d(args.count)
        else:
            test_cases = generate_systematic_3d(args.count)
        print(f"Generated {len(test_cases)} systematic {args.ndim}D cases")
    
    if test_cases:
        write_csv(test_cases, args.output)
        print("Generation completed!")
    else:
        print("ERROR: No test cases generated")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())