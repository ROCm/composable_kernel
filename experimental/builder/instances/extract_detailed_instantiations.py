#!/usr/bin/env python3
"""
Extract detailed template instantiations for forward convolution device operations.
This script parses the actual template parameters for each device operation instance.
"""

import os
import re
from pathlib import Path
from collections import defaultdict

def find_files(base_paths, extensions):
    """Find all files with given extensions in base paths."""
    files = []
    for base_path in base_paths:
        path_obj = Path(base_path)
        if path_obj.is_dir():
            for ext in extensions:
                files.extend(path_obj.rglob(f'*{ext}'))
    return files

def extract_device_op_instantiations(content, device_op_name):
    """Extract complete device operation instantiations."""
    instantiations = []
    lines = content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this line contains the device op
        if device_op_name + '<' in line:
            # Start collecting the full instantiation
            full_instantiation = line
            bracket_count = line.count('<') - line.count('>')
            
            # Continue to next lines if brackets aren't balanced
            j = i + 1
            while bracket_count > 0 and j < len(lines):
                full_instantiation += '\n' + lines[j]
                bracket_count += lines[j].count('<') - lines[j].count('>')
                j += 1
            
            # Clean up and store
            instantiation_clean = full_instantiation.strip()
            if instantiation_clean:
                instantiations.append({
                    'line_start': i + 1,
                    'line_end': j,
                    'text': instantiation_clean
                })
            
            i = j
        else:
            i += 1
    
    return instantiations

def main():
    # Device operations we're interested in (from previous analysis)
    target_device_ops = [
        'DeviceConv2dFwdXdl_C_Shuffle_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K',
        'DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K',
        'DeviceConvFwd',
        'DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK',
        'DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle',
        'DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3',
        'DeviceGroupedConvFwdMultipleD_Wmma_CShuffle',
        'DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor',
    ]
    
    # Define paths to search
    header_paths = [
        'library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd',
        'library/include/ck/library/tensor_operation_instance/gpu',
    ]
    
    source_paths = [
        'library/src/tensor_operation_instance/gpu/grouped_conv1d_fwd',
        'library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd',
        'library/src/tensor_operation_instance/gpu/grouped_conv3d_fwd',
        'library/src/tensor_operation_instance/gpu/conv2d_fwd',
    ]
    
    # Find all relevant files
    header_files = find_files(header_paths, ['.hpp', '.h', '.inc'])
    source_files = find_files(source_paths, ['.cpp', '.hpp', '.inc', '.in'])
    
    all_files = list(set(header_files + source_files))
    
    print(f"Analyzing {len(all_files)} files for detailed instantiations...\n")
    
    # Process each device operation
    results = {}
    
    for device_op in target_device_ops:
        print(f"Processing {device_op}...")
        
        device_op_data = {
            'name': device_op,
            'files_with_instantiations': {},
            'total_instantiations': 0
        }
        
        for file_path in all_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if device_op in content:
                    instantiations = extract_device_op_instantiations(content, device_op)
                    
                    if instantiations:
                        rel_path = str(file_path).replace(os.getcwd() + '/', '')
                        device_op_data['files_with_instantiations'][rel_path] = instantiations
                        device_op_data['total_instantiations'] += len(instantiations)
            except Exception as e:
                print(f"  Error processing {file_path}: {e}")
        
        results[device_op] = device_op_data
        print(f"  Found {device_op_data['total_instantiations']} instantiations in {len(device_op_data['files_with_instantiations'])} files")
    
    # Generate detailed report
    report = []
    report.append("=" * 100)
    report.append("DETAILED FORWARD CONVOLUTION DEVICE OPERATIONS AND TEMPLATE INSTANTIATIONS")
    report.append("=" * 100)
    report.append("")
    report.append(f"Generated for: ck::tensor_operation::device namespace")
    report.append("")
    
    # Summary
    report.append("SUMMARY")
    report.append("-" * 100)
    total_all_instantiations = sum(data['total_instantiations'] for data in results.values())
    report.append(f"Total Device Operation Types: {len(target_device_ops)}")
    report.append(f"Total Template Instantiations: {total_all_instantiations}")
    report.append("")
    
    for device_op in target_device_ops:
        data = results[device_op]
        report.append(f"  • {device_op}")
        report.append(f"    - Instantiations: {data['total_instantiations']}")
        report.append(f"    - Files: {len(data['files_with_instantiations'])}")
    
    report.append("")
    report.append("=" * 100)
    report.append("")
    
    # Detailed section for each device op
    for device_op in target_device_ops:
        data = results[device_op]
        
        report.append("=" * 100)
        report.append(f"DEVICE OPERATION: {device_op}")
        report.append("=" * 100)
        report.append("")
        report.append(f"Total Template Instantiations: {data['total_instantiations']}")
        report.append(f"Number of Files: {len(data['files_with_instantiations'])}")
        report.append("")
        
        # List files and show sample instantiations
        for file_path, instantiations in sorted(data['files_with_instantiations'].items()):
            report.append("-" * 100)
            report.append(f"File: {file_path}")
            report.append(f"Instantiations in this file: {len(instantiations)}")
            report.append("")
            
            # Show up to 3 sample instantiations from each file
            num_samples = min(3, len(instantiations))
            if num_samples > 0:
                report.append("Sample instantiations:")
                for idx, inst in enumerate(instantiations[:num_samples]):
                    report.append(f"  [{idx+1}] Line {inst['line_start']}:")
                    # Truncate very long instantiations
                    text = inst['text']
                    if len(text) > 500:
                        text = text[:500] + "..."
                    report.append(f"      {text}")
                
                if len(instantiations) > num_samples:
                    report.append(f"  ... and {len(instantiations) - num_samples} more instantiations")
            
            report.append("")
        
        report.append("")
    
    # Save detailed report
    report_text = '\n'.join(report)
    
    output_file = 'forward_conv_device_ops_detailed_report.txt'
    with open(output_file, 'w') as f:
        f.write(report_text)
    
    print(f"\n{'='*60}")
    print(f"Detailed report saved to: {output_file}")
    print(f"Total instantiations found: {total_all_instantiations}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
