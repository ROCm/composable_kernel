#!/usr/bin/env python3
"""
Extract all forward convolution device operations and their template instantiations
from CK library headers and source files.
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import json

# Device operation patterns to search for
DEVICE_OP_PATTERNS = [
    r'(Device\w*Conv\w*Fwd\w*)<',  # Generic pattern for device ops
]

def find_files(base_paths, extensions):
    """Find all files with given extensions in base paths."""
    files = []
    for base_path in base_paths:
        path_obj = Path(base_path)
        if path_obj.is_dir():
            for ext in extensions:
                files.extend(path_obj.rglob(f'*{ext}'))
    return files

def extract_device_operations(file_path):
    """Extract device operations from a file."""
    device_ops = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Search for device operation class names
        for pattern in DEVICE_OP_PATTERNS:
            matches = re.finditer(pattern, content)
            for match in matches:
                device_op_name = match.group(1)
                # Store the device operation with file location
                device_ops.append({
                    'name': device_op_name,
                    'file': str(file_path),
                    'line_context': None  # We'll extract context later if needed
                })
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return device_ops

def extract_template_instantiation(content, device_op_name):
    """Extract a single template instantiation for analysis."""
    # Look for the full template instantiation
    pattern = rf'{re.escape(device_op_name)}<[^>]*(?:<[^>]*>)*[^>]*>'
    
    # This is complex due to nested templates, so let's use a simpler approach
    # Just find lines with the device op name
    lines = content.split('\n')
    instantiations = []
    
    for i, line in enumerate(lines):
        if device_op_name in line and '<' in line:
            # Try to capture the full instantiation (might span multiple lines)
            instantiation_text = line.strip()
            
            # Count template brackets to see if we have a complete instantiation
            open_brackets = instantiation_text.count('<')
            close_brackets = instantiation_text.count('>')
            
            # If brackets are balanced, we have a complete line
            if open_brackets > 0:
                instantiations.append({
                    'line_number': i + 1,
                    'text': instantiation_text[:200]  # Truncate for readability
                })
    
    return instantiations

def main():
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
    
    all_paths = header_paths + source_paths
    
    # Find all relevant files
    header_files = find_files(header_paths, ['.hpp', '.h'])
    source_files = find_files(source_paths, ['.cpp', '.hpp', '.inc', '.in'])
    
    all_files = list(set(header_files + source_files))
    
    print(f"Found {len(all_files)} files to analyze")
    
    # Collect all device operations
    all_device_ops = []
    for file_path in all_files:
        ops = extract_device_operations(file_path)
        all_device_ops.extend(ops)
    
    # Group by device operation name
    ops_by_name = defaultdict(list)
    for op in all_device_ops:
        ops_by_name[op['name']].append(op)
    
    # Count unique device operations
    unique_ops = sorted(ops_by_name.keys())
    
    print(f"\nFound {len(unique_ops)} unique device operation types:")
    for op_name in unique_ops:
        count = len(ops_by_name[op_name])
        print(f"  {op_name}: {count} occurrences")
    
    # Now extract detailed instantiations for each device op
    device_op_details = {}
    
    for device_op_name in unique_ops:
        instantiation_files = set()
        total_instantiations = 0
        
        # Re-scan files that contain this device op
        for file_path in all_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if device_op_name in content:
                    instantiations = extract_template_instantiation(content, device_op_name)
                    if instantiations:
                        instantiation_files.add(str(file_path))
                        total_instantiations += len(instantiations)
            except Exception as e:
                pass
        
        device_op_details[device_op_name] = {
            'files': sorted(list(instantiation_files)),
            'total_instantiations': total_instantiations
        }
    
    # Generate report
    report = []
    report.append("=" * 80)
    report.append("FORWARD CONVOLUTION DEVICE OPERATIONS REPORT")
    report.append("=" * 80)
    report.append("")
    report.append(f"Total Unique Device Operations: {len(unique_ops)}")
    report.append("")
    
    for device_op_name in unique_ops:
        details = device_op_details[device_op_name]
        report.append("-" * 80)
        report.append(f"Device Operation: {device_op_name}")
        report.append(f"Total Template Instantiations: {details['total_instantiations']}")
        report.append(f"Number of Files: {len(details['files'])}")
        report.append("")
        report.append("Files containing this device operation:")
        for file_path in details['files']:
            # Make path relative to workspace
            rel_path = file_path.replace(os.getcwd() + '/', '')
            report.append(f"  - {rel_path}")
        report.append("")
    
    # Save report
    report_text = '\n'.join(report)
    
    with open('forward_convolution_device_ops_report.txt', 'w') as f:
        f.write(report_text)
    
    print(f"\nReport saved to: forward_convolution_device_ops_report.txt")
    
    # Also save as JSON for programmatic access
    json_data = {
        'summary': {
            'total_device_operations': len(unique_ops),
            'device_operation_names': unique_ops
        },
        'details': device_op_details
    }
    
    with open('forward_convolution_device_ops_data.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"JSON data saved to: forward_convolution_device_ops_data.json")

if __name__ == '__main__':
    main()
