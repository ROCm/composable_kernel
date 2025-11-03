#!/usr/bin/env python3
"""
Extract ALL backward convolution device operations and template instantiations.
Handles both backward weight and backward data convolutions.
Generates human-readable .txt and programmatic JSON outputs.
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
        'compressed_text': ' '.join(full_text.split())
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
    """Parse template parameters from instantiation text."""
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

def find_device_operations(all_files):
    """Find all device operations matching backward convolution patterns."""
    device_ops = set()
    
    # Patterns for backward operations
    patterns = [
        r'(Device\w*Conv\w*Bwd\w*Weight\w*)<',
        r'(Device\w*Conv\w*Bwd\w*Data\w*)<',
        r'(Device\w*ConvBwd\w*)<',
    ]
    
    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            for pattern in patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    device_ops.add(match.group(1))
        except:
            pass
    
    return sorted(list(device_ops))

def main():
    print("="*80)
    print("EXTRACTING BACKWARD CONVOLUTION DEVICE OPERATIONS")
    print("="*80)
    print()
    
    # Define paths
    header_paths = [
        'library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight',
        'library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data',
        'library/include/ck/library/tensor_operation_instance/gpu',
    ]
    
    source_paths = [
        'library/src/tensor_operation_instance/gpu/grouped_conv1d_bwd_weight',
        'library/src/tensor_operation_instance/gpu/grouped_conv2d_bwd_weight',
        'library/src/tensor_operation_instance/gpu/grouped_conv3d_bwd_weight',
        'library/src/tensor_operation_instance/gpu/grouped_convnd_bwd_weight',
        'library/src/tensor_operation_instance/gpu/conv1d_bwd_data',
        'library/src/tensor_operation_instance/gpu/conv2d_bwd_data',
        'library/src/tensor_operation_instance/gpu/conv3d_bwd_data',
        'library/src/tensor_operation_instance/gpu/grouped_conv1d_bwd_data',
        'library/src/tensor_operation_instance/gpu/grouped_conv2d_bwd_data',
        'library/src/tensor_operation_instance/gpu/grouped_conv3d_bwd_data',
    ]
    
    # Find all files
    header_files = find_files(header_paths, ['.hpp', '.h', '.inc'])
    source_files = find_files(source_paths, ['.cpp', '.hpp', '.inc', '.in'])
    all_files = list(set(header_files + source_files))
    
    print(f"Found {len(all_files)} files to analyze")
    
    # Discover all backward device operations
    print("Discovering device operations...")
    target_device_ops = find_device_operations(all_files)
    print(f"Found {len(target_device_ops)} device operation types")
    for op in target_device_ops:
        print(f"  - {op}")
    print()
    
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
                pass
        
        all_results[device_op] = device_op_data
        print(f"  Found {device_op_data['total_instantiations']} instantiations in {len(device_op_data['files'])} files")
    
    # Separate by type
    bwd_weight_ops = {k: v for k, v in all_results.items() if 'Weight' in k or 'Wgt' in k}
    bwd_data_ops = {k: v for k, v in all_results.items() if 'Data' in k}
    
    print(f"\nBackward Weight Operations: {len(bwd_weight_ops)}")
    print(f"Backward Data Operations: {len(bwd_data_ops)}")
    
    # Generate TXT report
    print("\nGenerating comprehensive TXT report...")
    txt_report = generate_txt_report(all_results, bwd_weight_ops, bwd_data_ops)
    
    txt_output = 'backward_conv_all_instantiations.txt'
    with open(txt_output, 'w') as f:
        f.write(txt_report)
    print(f"✓ TXT report saved: {txt_output}")
    
    # Generate JSON
    print("Generating JSON for programmatic use...")
    json_data = generate_json_data(all_results, bwd_weight_ops, bwd_data_ops)
    
    json_output = 'backward_conv_all_instantiations.json'
    with open(json_output, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"✓ JSON file saved: {json_output}")
    
    # Generate markdown summary
    print("Generating markdown summary...")
    md_content = generate_markdown_summary(all_results, bwd_weight_ops, bwd_data_ops)
    
    md_output = 'BACKWARD_CONVOLUTION_DEVICE_OPS_SUMMARY.md'
    with open(md_output, 'w') as f:
        f.write(md_content)
    print(f"✓ Markdown summary saved: {md_output}")
    
    # Final summary
    total_insts = sum(data['total_instantiations'] for data in all_results.values())
    print(f"\n{'='*80}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Device Operations Found: {len(all_results)}")
    print(f"  - Backward Weight: {len(bwd_weight_ops)}")
    print(f"  - Backward Data: {len(bwd_data_ops)}")
    print(f"Total Instantiations: {total_insts}")
    print(f"\nFiles generated:")
    print(f"  - {txt_output}")
    print(f"  - {json_output}")
    print(f"  - {md_output}")
    print(f"{'='*80}")

def generate_txt_report(all_results, bwd_weight_ops, bwd_data_ops):
    """Generate comprehensive TXT report."""
    report = []
    report.append("=" * 120)
    report.append("COMPLETE BACKWARD CONVOLUTION DEVICE OPERATIONS AND ALL TEMPLATE INSTANTIATIONS")
    report.append("=" * 120)
    report.append("")
    report.append(f"Namespace: ck::tensor_operation::device")
    report.append(f"Total Device Operation Types: {len(all_results)}")
    total_insts = sum(data['total_instantiations'] for data in all_results.values())
    report.append(f"Total Template Instantiations: {total_insts}")
    report.append("")
    
    # Summary
    report.append("SUMMARY OF DEVICE OPERATIONS")
    report.append("-" * 120)
    report.append("")
    report.append("BACKWARD WEIGHT OPERATIONS:")
    for device_op, data in sorted(bwd_weight_ops.items(), key=lambda x: x[1]['total_instantiations'], reverse=True):
        report.append(f" {data['total_instantiations']:4d} instantiations | {len(data['files']):3d} files | {device_op}")
    report.append("")
    report.append("BACKWARD DATA OPERATIONS:")
    for device_op, data in sorted(bwd_data_ops.items(), key=lambda x: x[1]['total_instantiations'], reverse=True):
        report.append(f" {data['total_instantiations']:4d} instantiations | {len(data['files']):3d} files | {device_op}")
    report.append("")
    report.append("=" * 120)
    report.append("")
    
    # Detailed sections
    for category_name, ops_dict in [("BACKWARD WEIGHT", bwd_weight_ops), ("BACKWARD DATA", bwd_data_ops)]:
        report.append("")
        report.append("=" * 120)
        report.append(f"{category_name} OPERATIONS")
        report.append("=" * 120)
        report.append("")
        
        for device_op in sorted(ops_dict.keys()):
            data = all_results[device_op]
            
            report.append("=" * 120)
            report.append(f"DEVICE OPERATION: {device_op}")
            report.append("=" * 120)
            report.append("")
            report.append(f"Total Instantiations: {data['total_instantiations']}")
            report.append(f"Number of Files: {len(data['files'])}")
            report.append("")
            
            for file_path, instantiations in sorted(data['files'].items()):
                report.append("-" * 120)
                report.append(f"FILE: {file_path}")
                report.append(f"Instantiations: {len(instantiations)}")
                report.append("")
                
                for idx, inst in enumerate(instantiations, 1):
                    report.append(f"[{idx}] Lines {inst['line_start']}-{inst['line_end']}:")
                    report.append("")
                    for line in inst['full_text'].split('\n'):
                        report.append(f"    {line}")
                    report.append("")
                
                report.append("")
            
            report.append("")
    
    return '\n'.join(report)

def generate_json_data(all_results, bwd_weight_ops, bwd_data_ops):
    """Generate JSON data for programmatic use."""
    total_insts = sum(data['total_instantiations'] for data in all_results.values())
    
    json_data = {
        'metadata': {
            'description': 'Backward convolution device operations and template instantiations',
            'namespace': 'ck::tensor_operation::device',
            'total_device_operations': len(all_results),
            'backward_weight_operations': len(bwd_weight_ops),
            'backward_data_operations': len(bwd_data_ops),
            'total_instantiations': total_insts
        },
        'backward_weight_operations': {},
        'backward_data_operations': {}
    }
    
    # Process backward weight operations
    for device_op, data in bwd_weight_ops.items():
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
        
        json_data['backward_weight_operations'][device_op] = json_device_op
    
    # Process backward data operations
    for device_op, data in bwd_data_ops.items():
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
        
        json_data['backward_data_operations'][device_op] = json_device_op
    
    return json_data

def generate_markdown_summary(all_results, bwd_weight_ops, bwd_data_ops):
    """Generate markdown summary document."""
    total_insts = sum(data['total_instantiations'] for data in all_results.values())
    bwd_weight_insts = sum(data['total_instantiations'] for data in bwd_weight_ops.values())
    bwd_data_insts = sum(data['total_instantiations'] for data in bwd_data_ops.values())
    
    md = []
    md.append("# Backward Convolution Device Operations - Comprehensive Summary")
    md.append("")
    md.append("**Generated:** October 29, 2025  ")
    md.append("**Namespace:** `ck::tensor_operation::device`  ")
    md.append("**Location:** `/library/include` and `/library/src` directories")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Overview")
    md.append("")
    md.append("This document provides a comprehensive list of all device operations used for backward convolutions (both weight gradients and data gradients) in the Composable Kernel library.")
    md.append("")
    md.append("### Total Statistics")
    md.append(f"- **Unique Device Operation Types:** {len(all_results)}")
    md.append(f"- **Backward Weight Operations:** {len(bwd_weight_ops)} types ({bwd_weight_insts} instantiations)")
    md.append(f"- **Backward Data Operations:** {len(bwd_data_ops)} types ({bwd_data_insts} instantiations)")
    md.append(f"- **Total Template Instantiations:** {total_insts}")
    md.append("")
    md.append("---")
    md.append("")
    
    # Backward Weight Table
    md.append("## Backward Weight Device Operations")
    md.append("")
    md.append("| # | Device Operation Name | Instantiations | Files | Primary Location |")
    md.append("|---|----------------------|----------------|-------|------------------|")
    
    for idx, (device_op, data) in enumerate(sorted(bwd_weight_ops.items(), key=lambda x: x[1]['total_instantiations'], reverse=True), 1):
        primary_file = sorted(data['files'].keys())[0] if data['files'] else "N/A"
        primary_file_short = primary_file.split('/')[-1] if '/' in primary_file else primary_file
        md.append(f"| {idx} | `{device_op}` | {data['total_instantiations']} | {len(data['files'])} | `{primary_file_short}` |")
    
    md.append(f"| **Total** | **Backward Weight Operations** | **{bwd_weight_insts}** | **{sum(len(d['files']) for d in bwd_weight_ops.values())}** | |")
    md.append("")
    md.append("---")
    md.append("")
    
    # Backward Data Table
    md.append("## Backward Data Device Operations")
    md.append("")
    md.append("| # | Device Operation Name | Instantiations | Files | Primary Location |")
    md.append("|---|----------------------|----------------|-------|------------------|")
    
    for idx, (device_op, data) in enumerate(sorted(bwd_data_ops.items(), key=lambda x: x[1]['total_instantiations'], reverse=True), 1):
        primary_file = sorted(data['files'].keys())[0] if data['files'] else "N/A"
        primary_file_short = primary_file.split('/')[-1] if '/' in primary_file else primary_file
        md.append(f"| {idx} | `{device_op}` | {data['total_instantiations']} | {len(data['files'])} | `{primary_file_short}` |")
    
    md.append(f"| **Total** | **Backward Data Operations** | **{bwd_data_insts}** | **{sum(len(d['files']) for d in bwd_data_ops.values())}** | |")
    md.append("")
    md.append(f"### Grand Total: {total_insts} Template Instantiations across {len(all_results)} Device Operation Types")
    md.append("")
    md.append("---")
    md.append("")
    
    # Output files section
    md.append("## Output Files")
    md.append("")
    md.append("### Complete Instantiation Files")
    md.append("")
    md.append("1. **`backward_conv_all_instantiations.txt`**")
    md.append("   - COMPLETE listing of ALL template instantiations")
    md.append("   - Human-readable format with line numbers")
    md.append("   - Organized by operation type, then by file")
    md.append("")
    md.append("2. **`backward_conv_all_instantiations.json`**")
    md.append("   - Structured JSON for programmatic instantiation generation")
    md.append("   - Separated into `backward_weight_operations` and `backward_data_operations`")
    md.append("   - Each instantiation includes full text and parsed parameters")
    md.append("   - Ready for automated code generation")
    md.append("")
    md.append("3. **`BACKWARD_CONVOLUTION_DEVICE_OPS_SUMMARY.md`** (This file)")
    md.append("   - Executive summary with tables")
    md.append("   - Quick reference for all backward operations")
    md.append("")
    
    return '\n'.join(md)

if __name__ == '__main__':
    main()
