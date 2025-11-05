#!/usr/bin/env python3
"""
Script to refactor forward_conv_structured_instantiations.json
to move elementwise_operation and device_operation from signature to algorithm.
"""

import json
import sys

def refactor_instantiation(inst: dict) -> dict:
    """Move elementwise_operation and device_operation from signature to algorithm"""
    
    # Extract fields from signature
    elementwise_op = inst["signature"].pop("elementwise_operation", "PASS_THROUGH")
    device_op = inst["signature"].pop("device_operation", "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle")
    
    # Add to algorithm at the beginning for better readability
    algorithm = inst["algorithm"]
    new_algorithm = {
        "elementwise_operation": elementwise_op,
        "device_operation": device_op
    }
    
    # Copy rest of algorithm fields
    for key, value in algorithm.items():
        new_algorithm[key] = value
    
    inst["algorithm"] = new_algorithm
    
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
        # Update signature schema - remove elementwise_operation and device_operation
        if "signature" in data["schemas"]:
            sig_schema = data["schemas"]["signature"]
            sig_schema.pop("elementwise_operation", None)
            sig_schema.pop("device_operation", None)
        
        # Add these fields to algorithm schemas
        for algo_schema_key in ["algorithm_xdl", "algorithm_wmma"]:
            if algo_schema_key in data["schemas"]:
                algo_schema = data["schemas"][algo_schema_key]
                # Add at the beginning of the description
                if "elementwise_operation" not in algo_schema:
                    # Insert before other fields
                    new_schema = {
                        "elementwise_operation": "enum (BIAS, BIAS_CLAMP, BIAS_BNORM_CLAMP, BILINEAR, CLAMP, SCALE, PASS_THROUGH)",
                        "device_operation": "string (DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle, DeviceGroupedConvFwdMultipleD_Wmma_CShuffle, etc.)"
                    }
                    new_schema.update(algo_schema)
                    data["schemas"][algo_schema_key] = new_schema
    
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
