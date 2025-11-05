#!/usr/bin/env python3
"""
Script to fix instantiation fields based on C++ concept requirements:
1. Remove redundant fields from V3 instantiations
2. Remove unnecessary fields from standard XDL instantiations
3. Add back WMMA instantiations
"""

import json
import re
from typing import List, Dict, Any

def parse_sequence(seq_str: str) -> List[int]:
    """Parse sequence like 'S<4, 16, 1>' to list [4, 16, 1]"""
    match = re.search(r'S<(.+?)>', seq_str.strip())
    if match:
        return [int(x.strip()) for x in match.group(1).split(',')]
    return []

def safe_get_seq_elem(seq: List[int], idx: int, default: int) -> int:
    """Safely get element from sequence with default"""
    return seq[idx] if len(seq) > idx else default

def map_data_type(dtype: str) -> str:
    """Map C++ data type to enum value"""
    dtype = dtype.strip()
    type_map = {
        'F32': 'FP32',
        'F16': 'FP16', 
        'BF16': 'BF16',
        'F8': 'FP8',
        'BF8': 'FP8',
        'int8_t': 'I8',
        'int32_t': 'I32',
        'I8': 'I8',
        'TF32': 'FP32'
    }
    return type_map.get(dtype, dtype)

def parse_instantiation_compressed(inst_text: str) -> List[str]:
    """Extract template parameters from compressed instantiation string"""
    match = re.search(r'<(.+)>,$', inst_text.strip())
    if not match:
        match = re.search(r'<(.+)>$', inst_text.strip())
    
    if not match:
        return []
    
    content = match.group(1)
    
    # Split by comma, handling nested templates
    params = []
    current = []
    depth = 0
    
    for char in content + ',':
        if char == '<':
            depth += 1
            current.append(char)
        elif char == '>':
            depth -= 1
            current.append(char)
        elif char == ',' and depth == 0:
            param = ''.join(current).strip()
            if param:
                params.append(param)
            current = []
        else:
            current.append(char)
    
    return params

def parse_wmma_params(params: List[str]) -> Dict[str, Any]:
    """Parse DeviceGroupedConvFwdMultipleD_Wmma_CShuffle parameters"""
    
    # Parse sequences
    seq_26 = parse_sequence(params[26]) if len(params) > 26 else []
    seq_27 = parse_sequence(params[27]) if len(params) > 27 else []
    seq_28 = parse_sequence(params[28]) if len(params) > 28 else []
    seq_33 = parse_sequence(params[33]) if len(params) > 33 else []
    seq_34 = parse_sequence(params[34]) if len(params) > 34 else []
    seq_35 = parse_sequence(params[35]) if len(params) > 35 else []
    seq_43 = parse_sequence(params[43]) if len(params) > 43 else []
    
    result = {
        "signature": {
            "spatial_dim": 2,
            "direction": "FORWARD",
            "layout": {
                "input": "GNHWC",
                "weight": "GKYXC",
                "output": "GNHWK"
            },
            "data_type": {
                "input": map_data_type(params[5]) if len(params) > 5 else "FP16",
                "weight": map_data_type(params[6]) if len(params) > 6 else "FP16",
                "accumulator": map_data_type(params[7]) if len(params) > 7 else "FP32",
                "shuffle": map_data_type(params[8]) if len(params) > 8 else "FP16",
                "output": map_data_type(params[10]) if len(params) > 10 else "FP16"
            },
            "elementwise_operation": "PASS_THROUGH"
        },
        "algorithm": {
            "device_operation": "DeviceGroupedConvFwdMultipleD_Wmma_CShuffle",
            "algorithm_type": "WMMA",
            "thread_block": {
                "block_size": int(params[17]) if len(params) > 17 else 128,
                "tile_size": {
                    "m": int(params[18]) if len(params) > 18 else 64,
                    "n": int(params[19]) if len(params) > 19 else 64,
                    "k": int(params[20]) if len(params) > 20 else 32
                }
            },
            "gridwise_wmma_gemm": {
                "k1": int(params[21]) if len(params) > 21 else 8,
                "m_per_wmma": int(params[22]) if len(params) > 22 else 16,
                "n_per_wmma": int(params[23]) if len(params) > 23 else 16,
                "m_wmma_per_wave": int(params[24]) if len(params) > 24 else 2,
                "n_wmma_per_wave": int(params[25]) if len(params) > 25 else 2,
                "pipeline_version": "V1"
            },
            "block_transfer": {
                "block_transfer_a": {
                    "k0": safe_get_seq_elem(seq_26, 0, 4),
                    "m_n": safe_get_seq_elem(seq_26, 1, 32),
                    "k1": safe_get_seq_elem(seq_26, 2, 1)
                },
                "block_transfer_b": {
                    "k0": safe_get_seq_elem(seq_33, 0, 4),
                    "m_n": safe_get_seq_elem(seq_33, 1, 32),
                    "k1": safe_get_seq_elem(seq_33, 2, 1)
                },
                "thread_cluster_dims_c": {
                    "m_block": safe_get_seq_elem(seq_43, 0, 1),
                    "m_wave_per_xdl": safe_get_seq_elem(seq_43, 1, 32),
                    "n_block": safe_get_seq_elem(seq_43, 2, 1),
                    "n_wave_per_xdl": safe_get_seq_elem(seq_43, 3, 8)
                },
                "lds_transfer_a": {
                    "src_vector_dim": int(params[29]) if len(params) > 29 else 2,
                    "src_scalar_per_vector": int(params[30]) if len(params) > 30 else 8,
                    "lds_dst_scalar_per_vector": int(params[31]) if len(params) > 31 else 8,
                    "is_direct_load": False,
                    "lds_padding": True
                },
                "lds_transfer_b": {
                    "src_vector_dim": int(params[36]) if len(params) > 36 else 2,
                    "src_scalar_per_vector": int(params[37]) if len(params) > 37 else 8,
                    "lds_dst_scalar_per_vector": int(params[38]) if len(params) > 38 else 8,
                    "is_direct_load": False,
                    "lds_padding": True
                },
                "epilogue_c": {
                    "m_per_wave_per_shuffle": int(params[40]) if len(params) > 40 else 1,
                    "n_per_wave_per_shuffle": int(params[41]) if len(params) > 41 else 1,
                    "scalar_per_vector": int(params[43]) if len(params) > 43 else 8
                },
                "block_transfer_access_order_a": {
                    "order": seq_27 if seq_27 else [1, 0, 2]
                },
                "block_transfer_access_order_b": {
                    "order": seq_34 if seq_34 else [1, 0, 2]
                },
                "src_access_order_a": {
                    "order": seq_28 if seq_28 else [1, 0, 2]
                },
                "src_access_order_b": {
                    "order": seq_35 if seq_35 else [1, 0, 2]
                }
            },
            "fwd_specialization": "DEFAULT",
            "gemm_specialization": "MNKPadding",
            "num_gemm_k_prefetch_stages": int(params[16]) if len(params) > 16 else 1,
            "loop_scheduler": "DEFAULT"
        }
    }
    
    return result

def fix_json_structure():
    """Main function to fix the JSON structure"""
    
    print("Loading forward_conv_structured_instantiations.json...")
    with open('experimental/builder/instances/forward_conv_structured_instantiations.json', 'r') as f:
        data = json.load(f)
    
    print("Loading forward_conv_all_instantiations.json for WMMA instances...")
    with open('experimental/builder/instances/forward_conv_all_instantiations.json', 'r') as f:
        all_data = json.load(f)
    
    # Fix existing instantiations
    print(f"Fixing {len(data['instantiations'])} existing instantiations...")
    for inst in data['instantiations']:
        device_op = inst['algorithm']['device_operation']
        
        if 'V3' in device_op:
            # Remove redundant fields from V3
            inst['algorithm'].pop('num_gemm_k_prefetch_stages', None)
            inst['algorithm'].pop('loop_scheduler', None)
            inst['algorithm'].pop('num_groups_to_merge', None)
        else:
            # Remove unnecessary fields from standard XDL
            inst['algorithm'].pop('lds_extra', None)
            inst['algorithm'].pop('block_gemm', None)
    
    # Add WMMA instantiations
    print("Adding WMMA instantiations...")
    wmma_count = 0
    inst_id = len(data['instantiations'])
    
    if "DeviceGroupedConvFwdMultipleD_Wmma_CShuffle" in all_data["device_operations"]:
        dev_op = all_data["device_operations"]["DeviceGroupedConvFwdMultipleD_Wmma_CShuffle"]
        
        for file_path, file_data in dev_op["instantiations_by_file"].items():
            for inst in file_data["instantiations"]:
                params = parse_instantiation_compressed(inst["instantiation_compressed"])
                if params:
                    try:
                        structured = parse_wmma_params(params)
                        structured["id"] = inst_id
                        structured["source_file"] = file_path
                        structured["line"] = inst["line_start"]
                        data['instantiations'].append(structured)
                        inst_id += 1
                        wmma_count += 1
                    except Exception as e:
                        print(f"  Warning: Failed to parse WMMA instantiation at line {inst['line_start']}: {e}")
    
    # Update metadata
    data['metadata']['total_instantiations'] = len(data['instantiations'])
    data['metadata']['notes'] = "Each instantiation is represented as a JSON object with signature and algorithm fields that map to C++ structs satisfying ConvSignatureDescriptor and ConvAlgorithmDescriptor concepts. The algorithm structure differs based on algorithm_type (XDL vs WMMA) and device_operation. V3 instances do not include num_gemm_k_prefetch_stages, loop_scheduler, and num_groups_to_merge. Standard XDL instances do not include lds_extra and block_gemm."
    
    print(f"\nFixed {len(data['instantiations'])} total instantiations")
    print(f"Added {wmma_count} WMMA instantiations")
    
    print("\nWriting to forward_conv_structured_instantiations.json...")
    with open('experimental/builder/instances/forward_conv_structured_instantiations.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    fix_json_structure()
