#!/usr/bin/env python3
"""
Script to move elementwise_operation back to signature section
while keeping device_operation in algorithm section.
"""

import json
import sys

def refactor_instantiation(inst: dict) -> dict:
    """Move elementwise_operation back to signature, keep device_operation in algorithm"""
    
    # Extract elementwise_operation from algorithm
    elementwise_op = inst["algorithm"].pop("elementwise_operation", "PASS_THROUGH")
    
    # Add to signature
    inst["signature"]["elementwise_operation"] = elementwise_op
    
    return inst

def refactor_json(input_file: str, output_file: str):
    """Main refactoring function"""
    
    print(f"Loading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Refactoring {len(data['instantiations'])} instantiations...")
    
    # Refactor each instantiation
    for inst in data["instantiations"]:
        refactor_instantiation(inst)
    
    # Update schema documentation
    if "schemas" in data:
        # Add elementwise_operation back to signature schema
        if "signature" in data["schemas"]:
            sig_schema = data["schemas"]["signature"]
            # Add after data_type
            new_sig_schema = {}
            for key, value in sig_schema.items():
                new_sig_schema[key] = value
                if key == "data_type":
                    new_sig_schema["elementwise_operation"] = "enum (BIAS, BIAS_CLAMP, BIAS_BNORM_CLAMP, BILINEAR, CLAMP, SCALE, PASS_THROUGH)"
            data["schemas"]["signature"] = new_sig_schema
        
        # Remove elementwise_operation from algorithm schemas (keep device_operation)
        for algo_schema_key in ["algorithm_xdl", "algorithm_wmma"]:
            if algo_schema_key in data["schemas"]:
                data["schemas"][algo_schema_key].pop("elementwise_operation", None)
    
    print(f"Writing to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    input_file = "experimental/builder/instances/forward_conv_structured_instantiations.json"
    output_file = "experimental/builder/instances/forward_conv_structured_instantiations.json"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    refactor_json(input_file, output_file)
