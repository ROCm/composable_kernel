#!/usr/bin/env python3
"""
Extract ALL template instantiations for forward convolution device operations.
Generates both human-readable .txt and programmatic JSON outputs.
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import json

def find_files(base_paths, extensions):
    """Find all files with given extensions in base paths."""
    files = []
    for base_path in base_paths:
        path_obj = Path(base_path)
        if path_obj.is_dir():
            for ext in extensions:
                files.extend(path_obj.rglob(f'*{ext}'))
    return files

def extract_complete_instantiation(lines, start_idx, device_op_name):
    """Extract a complete device operation instantiation with balanced brackets."""
    if start_idx >= len(lines):
        return None, start_idx
    
    line = lines[start_idx]
    
    # Check if this line contains the device op
    if device_op_name + '<' not in line:
        return None, start_idx
    
    # Start collecting the full instantiation
    full_lines = [line]
    bracket_count = line.count('<') - line.count('>')
    
    # Continue to next lines until brackets are balanced
    idx = start_idx + 1
    while bracket_count > 0 and idx < len(lines):
        full_lines.append(lines[idx])
        bracket_count += lines[idx].count('<') - lines[idx].count('>')
        idx += 1
    
    # Join and clean up
    full_text = '\n'.join(full_lines)
    
    return {
        'line_start': start_idx + 1,
        'line_end': idx,
        'full_text': full_text.strip(),
        'compressed_text': ' '.join(full_text.split())  # Single line version
    }, idx

def extract_all_instantiations(content, device_op_name):
    """Extract ALL device operation instantiations from content."""
    instantiations = []
    lines = content.split('\n')
    
    idx = 0
    while idx < len(lines):
        inst, next_idx = extract_complete_instantiation(lines, idx, device_op_name)
        if inst:
            instantiations.append(inst)
            idx = next_idx
        else:
            idx += 1
    
    return instantiations

def parse_template_parameters(instantiation_text):
    """Parse template parameters from instantiation text (simplified)."""
    # This is a simplified parser - actual parsing would need full C++ template parser
    # For now, just extract key information
    params = {}
    
    # Extract data types
    type_patterns = {
        'BF16': r'\bBF16\b',
        'F16': r'\bF16\b',
        'F32': r'\bF32\b',
        'TF32': r'\bTF32\b',
        'INT8': r'\bint8_t\b',
        'F8': r'\bF8\b',
        'BF8': r'\bBF8\b',
    }
    
    for type_name, pattern in type_patterns.items():
        if re.search(pattern, instantiation_text):
            params[f'uses_{type_name}'] = True
    
    # Extract block sizes if visible
    block_size_match = re.search(r',\s*(\d{2,3}),\s*(\d{2,3}),\s*(\d{2,3}),\s*(\d{1,3}),', instantiation_text)
    if block_size_match:
        params['block_size'] = block_size_match.group(1)
        params['m_per_block'] = block_size_match.group(2)
        params['n_per_block'] = block_size_match.group(3)
        params['k_per_block'] = block_size_match.group(4)
    
    return params

def main():
    # Device operations to extract
    target_device_ops = [
        'DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle',
        'DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3',
        'DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K',
        'DeviceConv2dFwdXdl_C_Shuffle_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K',
        'DeviceGroupedConvFwdMultipleD_Wmma_CShuffle',
        'DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK',
        'DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor',
        'DeviceConvFwd',
    ]
    
    # Define paths
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
    
    # Find all files
    header_files = find_files(header_paths, ['.hpp', '.h', '.inc'])
    source_files = find_files(source_paths, ['.cpp', '.hpp', '.inc', '.in'])
    all_files = list(set(header_files + source_files))
    
    print(f"Analyzing {len(all_files)} files for ALL instantiations...\n")
    
    # Collect data
    all_results = {}
    
    for device_op in target_device_ops:
        print(f"Extracting all instantiations for {device_op}...")
        
        device_op_data = {
            'device_operation_name': device_op,
            'files': {},
            'total_instantiations': 0
        }
        
        for file_path in all_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if device_op in content:
                    instantiations = extract_all_instantiations(content, device_op)
                    
                    if instantiations:
                        rel_path = str(file_path).replace(os.getcwd() + '/', '')
                        
                        # Process each instantiation
                        processed_insts = []
                        for inst in instantiations:
                            processed_inst = {
                                'line_start': inst['line_start'],
                                'line_end': inst['line_end'],
                                'full_text': inst['full_text'],
                                'compressed_text': inst['compressed_text'],
                                'template_params': parse_template_parameters(inst['full_text'])
                            }
                            processed_insts.append(processed_inst)
                        
                        device_op_data['files'][rel_path] = processed_insts
                        device_op_data['total_instantiations'] += len(instantiations)
            
            except Exception as e:
                print(f"  Error processing {file_path}: {e}")
        
        all_results[device_op] = device_op_data
        print(f"  Found {device_op_data['total_instantiations']} instantiations in {len(device_op_data['files'])} files")
    
    # Generate comprehensive TXT report
    print("\nGenerating comprehensive TXT report...")
    txt_report = []
    txt_report.append("=" * 120)
    txt_report.append("COMPLETE FORWARD CONVOLUTION DEVICE OPERATIONS AND ALL TEMPLATE INSTANTIATIONS")
    txt_report.append("=" * 120)
    txt_report.append("")
    txt_report.append(f"Namespace: ck::tensor_operation::device")
    txt_report.append(f"Total Device Operation Types: {len(target_device_ops)}")
    total_insts = sum(data['total_instantiations'] for data in all_results.values())
    txt_report.append(f"Total Template Instantiations: {total_insts}")
    txt_report.append("")
    
    # Summary table
    txt_report.append("SUMMARY OF DEVICE OPERATIONS")
    txt_report.append("-" * 120)
    for device_op in target_device_ops:
        data = all_results[device_op]
        txt_report.append(f"{data['total_instantiations']:4d} instantiations | {len(data['files']):3d} files | {device_op}")
    txt_report.append("")
    txt_report.append("=" * 120)
    txt_report.append("")
    
    # Detailed section for each device op - ALL instantiations
    for device_op in target_device_ops:
        data = all_results[device_op]
        
        txt_report.append("")
        txt_report.append("=" * 120)
        txt_report.append(f"DEVICE OPERATION: {device_op}")
        txt_report.append("=" * 120)
        txt_report.append("")
        txt_report.append(f"Total Instantiations: {data['total_instantiations']}")
        txt_report.append(f"Number of Files: {len(data['files'])}")
        txt_report.append("")
        
        # List ALL instantiations in each file
        for file_path, instantiations in sorted(data['files'].items()):
            txt_report.append("-" * 120)
            txt_report.append(f"FILE: {file_path}")
            txt_report.append(f"Instantiations: {len(instantiations)}")
            txt_report.append("")
            
            for idx, inst in enumerate(instantiations, 1):
                txt_report.append(f"[{idx}] Lines {inst['line_start']}-{inst['line_end']}:")
                txt_report.append("")
                # Include full text with proper indentation
                for line in inst['full_text'].split('\n'):
                    txt_report.append(f"    {line}")
                txt_report.append("")
            
            txt_report.append("")
        
        txt_report.append("")
    
    # Save TXT report
    txt_output = 'forward_conv_all_instantiations.txt'
    with open(txt_output, 'w') as f:
        f.write('\n'.join(txt_report))
    print(f"✓ TXT report saved: {txt_output}")
    
    # Generate JSON for programmatic use
    print("Generating JSON for programmatic use...")
    json_data = {
        'metadata': {
            'description': 'Forward convolution device operations and template instantiations',
            'namespace': 'ck::tensor_operation::device',
            'total_device_operations': len(target_device_ops),
            'total_instantiations': total_insts
        },
        'device_operations': {}
    }
    
    for device_op in target_device_ops:
        data = all_results[device_op]
        
        json_device_op = {
            'name': device_op,
            'total_instantiations': data['total_instantiations'],
            'total_files': len(data['files']),
            'instantiations_by_file': {}
        }
        
        for file_path, instantiations in data['files'].items():
            file_insts = []
            for inst in instantiations:
                file_insts.append({
                    'line_start': inst['line_start'],
                    'line_end': inst['line_end'],
                    'instantiation_text': inst['full_text'],
                    'instantiation_compressed': inst['compressed_text'],
                    'parsed_parameters': inst['template_params']
                })
            
            json_device_op['instantiations_by_file'][file_path] = {
                'count': len(file_insts),
                'instantiations': file_insts
            }
        
        json_data['device_operations'][device_op] = json_device_op
    
    # Save JSON
    json_output = 'forward_conv_all_instantiations.json'
    with open(json_output, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"✓ JSON file saved: {json_output}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"TXT File:  {txt_output}")
    print(f"JSON File: {json_output}")
    print(f"Total instantiations extracted: {total_insts}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
