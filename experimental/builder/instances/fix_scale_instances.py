#!/usr/bin/env python3
"""
Script to fix elementwise_operation in forward_conv_structured_instantiations.json
For instances where source_file contains "_scale_", change elementwise_operation to "SCALE"
"""

import json
import sys

def fix_scale_instances(json_file):
    # Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Counter for modified instances
    modified_count = 0
    
    # Iterate through all instantiations
    for inst in data.get('instantiations', []):
        source_file = inst.get('source_file', '')
        
        # Check if source_file contains "_scale_"
        if '_scale_' in source_file:
            # Check if elementwise_operation needs to be changed
            if inst.get('signature', {}).get('elementwise_operation') == 'PASS_THROUGH':
                inst['signature']['elementwise_operation'] = 'SCALE'
                modified_count += 1
                print(f"Modified instance id={inst.get('id')}: {source_file}")
    
    # Write back to the file
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nTotal instances modified: {modified_count}")
    return modified_count

if __name__ == '__main__':
    json_file = 'forward_conv_structured_instantiations.json'
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    
    try:
        count = fix_scale_instances(json_file)
        sys.exit(0 if count > 0 else 1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
