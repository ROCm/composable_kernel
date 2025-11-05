#!/usr/bin/env python3
"""
Script to convert template instantiations from forward_conv_all_instantiations.json
to structured JSON format that maps to C++ concept structures.
"""

import json
import re
from typing import Dict, List, Any, Optional

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

def parse_xdl_cshuffle_params(params: List[str]) -> Dict[str, Any]:
    """Parse DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle template parameters"""
    
    # Parse sequences with safety
    seq_27 = parse_sequence(params[27]) if len(params) > 27 else []
    seq_28 = parse_sequence(params[28]) if len(params) > 28 else []
    seq_29 = parse_sequence(params[29]) if len(params) > 29 else []
    seq_34 = parse_sequence(params[34]) if len(params) > 34 else []
    seq_35 = parse_sequence(params[35]) if len(params) > 35 else []
    seq_36 = parse_sequence(params[36]) if len(params) > 36 else []
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
                "input": map_data_type(params[5]) if len(params) > 5 else "FP32",
                "weight": map_data_type(params[6]) if len(params) > 6 else "FP32",
                "accumulator": map_data_type(params[7]) if len(params) > 7 else "FP32",
                "shuffle": map_data_type(params[8]) if len(params) > 8 else "FP32",
                "output": map_data_type(params[10]) if len(params) > 10 else "FP32"
            },
            "elementwise_operation": "PASS_THROUGH",
            "device_operation": "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle"
        },
        "algorithm": {
            "algorithm_type": "XDL",
            "thread_block": {
                "block_size": int(params[17]) if len(params) > 17 else 256,
                "tile_size": {
                    "m": int(params[18]) if len(params) > 18 else 128,
                    "n": int(params[19]) if len(params) > 19 else 128,
                    "k": int(params[20]) if len(params) > 20 else 32
                }
            },
            "gridwise_xdl_gemm": {
                "ak1": int(params[21]) if len(params) > 21 else 8,
                "bk1": int(params[22]) if len(params) > 22 else 8,
                "m_per_xdl": int(params[23]) if len(params) > 23 else 32,
                "n_per_xdl": int(params[24]) if len(params) > 24 else 32,
                "m_xdl_per_wave": int(params[25]) if len(params) > 25 else 2,
                "n_xdl_per_wave": int(params[26]) if len(params) > 26 else 2
            },
            "block_transfer": {
                "block_transfer_a": {
                    "k0": safe_get_seq_elem(seq_27, 0, 4),
                    "m_n": safe_get_seq_elem(seq_27, 1, 64),
                    "k1": safe_get_seq_elem(seq_27, 2, 1)
                },
                "block_transfer_b": {
                    "k0": safe_get_seq_elem(seq_34, 0, 4),
                    "m_n": safe_get_seq_elem(seq_34, 1, 64),
                    "k1": safe_get_seq_elem(seq_34, 2, 1)
                },
                "thread_cluster_dims_c": {
                    "m_block": safe_get_seq_elem(seq_43, 0, 1),
                    "m_wave_per_xdl": safe_get_seq_elem(seq_43, 1, 32),
                    "n_block": safe_get_seq_elem(seq_43, 2, 1),
                    "n_wave_per_xdl": safe_get_seq_elem(seq_43, 3, 8)
                },
                "lds_transfer_a": {
                    "src_vector_dim": int(params[30]) if len(params) > 30 else 2,
                    "src_scalar_per_vector": int(params[31]) if len(params) > 31 else 8,
                    "lds_dst_scalar_per_vector": int(params[32]) if len(params) > 32 else 8,
                    "is_direct_load": False,
                    "lds_padding": True
                },
                "lds_transfer_b": {
                    "src_vector_dim": int(params[37]) if len(params) > 37 else 2,
                    "src_scalar_per_vector": int(params[38]) if len(params) > 38 else 8,
                    "lds_dst_scalar_per_vector": int(params[39]) if len(params) > 39 else 8,
                    "is_direct_load": False,
                    "lds_padding": True
                },
                "epilogue_c": {
                    "m_per_wave_per_shuffle": int(params[41]) if len(params) > 41 else 1,
                    "n_per_wave_per_shuffle": int(params[42]) if len(params) > 42 else 1,
                    "scalar_per_vector": int(params[44]) if len(params) > 44 else 8
                },
                "block_transfer_access_order_a": {
                    "order": seq_28 if seq_28 else [1, 0, 2]
                },
                "block_transfer_access_order_b": {
                    "order": seq_35 if seq_35 else [1, 0, 2]
                },
                "src_access_order_a": {
                    "order": seq_29 if seq_29 else [1, 0, 2]
                },
                "src_access_order_b": {
                    "order": seq_36 if seq_36 else [1, 0, 2]
                }
            },
            "block_gemm": {
                "pipeline_version": "V1",
                "scheduler": "INTRAWAVE"
            },
            "fwd_specialization": "DEFAULT",
            "gemm_specialization": "MNKPadding",
            "num_gemm_k_prefetch_stages": int(params[16]) if len(params) > 16 else 1,
            "loop_scheduler": "DEFAULT",
            "num_groups_to_merge": 1,
            "lds_extra": {
                "a_block_lds_extra_m": int(params[33]) if len(params) > 33 else 1,
                "b_block_lds_extra_n": int(params[40]) if len(params) > 40 else 1
            }
        }
    }
    
    return result

def parse_xdl_cshuffle_v3_params(params: List[str]) -> Dict[str, Any]:
    """Parse DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 parameters
    
    V3 has different parameter layout - no num_gemm_k_prefetch_stages at index 16,
    and has scheduler/pipeline_version at the end.
    """
    
    # Parse sequences - indices are shifted by -1 compared to regular version  
    seq_26 = parse_sequence(params[26]) if len(params) > 26 else []
    seq_27 = parse_sequence(params[27]) if len(params) > 27 else []
    seq_28 = parse_sequence(params[28]) if len(params) > 28 else []
    seq_33 = parse_sequence(params[33]) if len(params) > 33 else []
    seq_34 = parse_sequence(params[34]) if len(params) > 34 else []
    seq_35 = parse_sequence(params[35]) if len(params) > 35 else []
    seq_42 = parse_sequence(params[42]) if len(params) > 42 else []
    
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
                "input": map_data_type(params[5]) if len(params) > 5 else "FP32",
                "weight": map_data_type(params[6]) if len(params) > 6 else "FP32",
                "accumulator": map_data_type(params[7]) if len(params) > 7 else "FP32",
                "shuffle": map_data_type(params[8]) if len(params) > 8 else "FP32",
                "output": map_data_type(params[10]) if len(params) > 10 else "FP32"
            },
            "elementwise_operation": "PASS_THROUGH",
            "device_operation": "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3"
        },
        "algorithm": {
            "algorithm_type": "XDL",
            "thread_block": {
                "block_size": int(params[16]) if len(params) > 16 else 256,
                "tile_size": {
                    "m": int(params[17]) if len(params) > 17 else 128,
                    "n": int(params[18]) if len(params) > 18 else 128,
                    "k": int(params[19]) if len(params) > 19 else 32
                }
            },
            "gridwise_xdl_gemm": {
                "ak1": int(params[20]) if len(params) > 20 else 8,
                "bk1": int(params[21]) if len(params) > 21 else 8,
                "m_per_xdl": int(params[22]) if len(params) > 22 else 32,
                "n_per_xdl": int(params[23]) if len(params) > 23 else 32,
                "m_xdl_per_wave": int(params[24]) if len(params) > 24 else 2,
                "n_xdl_per_wave": int(params[25]) if len(params) > 25 else 2
            },
            "block_transfer": {
                "block_transfer_a": {
                    "k0": safe_get_seq_elem(seq_26, 0, 4),
                    "m_n": safe_get_seq_elem(seq_26, 1, 64),
                    "k1": safe_get_seq_elem(seq_26, 2, 1)
                },
                "block_transfer_b": {
                    "k0": safe_get_seq_elem(seq_33, 0, 4),
                    "m_n": safe_get_seq_elem(seq_33, 1, 64),
                    "k1": safe_get_seq_elem(seq_33, 2, 1)
                },
                "thread_cluster_dims_c": {
                    "m_block": safe_get_seq_elem(seq_42, 0, 1),
                    "m_wave_per_xdl": safe_get_seq_elem(seq_42, 1, 32),
                    "n_block": safe_get_seq_elem(seq_42, 2, 1),
                    "n_wave_per_xdl": safe_get_seq_elem(seq_42, 3, 8)
                },
                "lds_transfer_a": {
                    "src_vector_dim": int(params[29]) if len(params) > 29 else 2,
                    "src_scalar_per_vector": int(params[30]) if len(params) > 30 else 8,
                    "lds_dst_scalar_per_vector": int(params[31]) if len(params) > 31 else 8,
                    "is_direct_load": int(params[32]) == 1 if len(params) > 32 else False,
                    "lds_padding": True
                },
                "lds_transfer_b": {
                    "src_vector_dim": int(params[36]) if len(params) > 36 else 2,
                    "src_scalar_per_vector": int(params[37]) if len(params) > 37 else 8,
                    "lds_dst_scalar_per_vector": int(params[38]) if len(params) > 38 else 8,
                    "is_direct_load": int(params[39]) == 1 if len(params) > 39 else False,
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
            "block_gemm": {
                "pipeline_version": "V1",
                "scheduler": "INTRAWAVE"
            },
            "fwd_specialization": "DEFAULT",
            "gemm_specialization": "MNKPadding",
            "num_gemm_k_prefetch_stages": 1,
            "loop_scheduler": "DEFAULT",
            "num_groups_to_merge": 1
        }
    }
    
    # V3 has scheduler and pipeline version at the end
    if len(params) > 44:
        scheduler_str = params[44]
        if "Interwave" in scheduler_str or "INTERWAVE" in scheduler_str.upper():
            result["algorithm"]["block_gemm"]["scheduler"] = "INTERWAVE"
        
    if len(params) > 45:
        version_str = params[45]
        version_map = {
            'v1': 'V1', 'v2': 'V2', 'v3': 'V3', 'v4': 'V4', 'v5': 'V5'
        }
        for k, v in version_map.items():
            if k in version_str.lower():
                result["algorithm"]["block_gemm"]["pipeline_version"] = v
                break
    
    return result

def parse_xdl_cshuffle_params_with_lds_extra(params: List[str]) -> Dict[str, Any]:
    """Parse standard XDL_CShuffle with lds_extra field"""
    
    # Parse sequences with safety
    seq_27 = parse_sequence(params[27]) if len(params) > 27 else []
    seq_28 = parse_sequence(params[28]) if len(params) > 28 else []
    seq_29 = parse_sequence(params[29]) if len(params) > 29 else []
    seq_34 = parse_sequence(params[34]) if len(params) > 34 else []
    seq_35 = parse_sequence(params[35]) if len(params) > 35 else []
    seq_36 = parse_sequence(params[36]) if len(params) > 36 else []
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
                "input": map_data_type(params[5]) if len(params) > 5 else "FP32",
                "weight": map_data_type(params[6]) if len(params) > 6 else "FP32",
                "accumulator": map_data_type(params[7]) if len(params) > 7 else "FP32",
                "shuffle": map_data_type(params[8]) if len(params) > 8 else "FP32",
                "output": map_data_type(params[10]) if len(params) > 10 else "FP32"
            },
            "elementwise_operation": "PASS_THROUGH",
            "device_operation": "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle"
        },
        "algorithm": {
            "algorithm_type": "XDL",
            "thread_block": {
                "block_size": int(params[17]) if len(params) > 17 else 256,
                "tile_size": {
                    "m": int(params[18]) if len(params) > 18 else 128,
                    "n": int(params[19]) if len(params) > 19 else 128,
                    "k": int(params[20]) if len(params) > 20 else 32
                }
            },
            "gridwise_xdl_gemm": {
                "ak1": int(params[21]) if len(params) > 21 else 8,
                "bk1": int(params[22]) if len(params) > 22 else 8,
                "m_per_xdl": int(params[23]) if len(params) > 23 else 32,
                "n_per_xdl": int(params[24]) if len(params) > 24 else 32,
                "m_xdl_per_wave": int(params[25]) if len(params) > 25 else 2,
                "n_xdl_per_wave": int(params[26]) if len(params) > 26 else 2
            },
            "block_transfer": {
                "block_transfer_a": {
                    "k0": safe_get_seq_elem(seq_27, 0, 4),
                    "m_n": safe_get_seq_elem(seq_27, 1, 64),
                    "k1": safe_get_seq_elem(seq_27, 2, 1)
                },
                "block_transfer_b": {
                    "k0": safe_get_seq_elem(seq_34, 0, 4),
                    "m_n": safe_get_seq_elem(seq_34, 1, 64),
                    "k1": safe_get_seq_elem(seq_34, 2, 1)
                },
                "thread_cluster_dims_c": {
                    "m_block": safe_get_seq_elem(seq_43, 0, 1),
                    "m_wave_per_xdl": safe_get_seq_elem(seq_43, 1, 32),
                    "n_block": safe_get_seq_elem(seq_43, 2, 1),
                    "n_wave_per_xdl": safe_get_seq_elem(seq_43, 3, 8)
                },
                "lds_transfer_a": {
                    "src_vector_dim": int(params[30]) if len(params) > 30 else 2,
                    "src_scalar_per_vector": int(params[31]) if len(params) > 31 else 8,
                    "lds_dst_scalar_per_vector": int(params[32]) if len(params) > 32 else 8,
                    "is_direct_load": False,
                    "lds_padding": True
                },
                "lds_transfer_b": {
                    "src_vector_dim": int(params[37]) if len(params) > 37 else 2,
                    "src_scalar_per_vector": int(params[38]) if len(params) > 38 else 8,
                    "lds_dst_scalar_per_vector": int(params[39]) if len(params) > 39 else 8,
                    "is_direct_load": False,
                    "lds_padding": True
                },
                "epilogue_c": {
                    "m_per_wave_per_shuffle": int(params[41]) if len(params) > 41 else 1,
                    "n_per_wave_per_shuffle": int(params[42]) if len(params) > 42 else 1,
                    "scalar_per_vector": int(params[44]) if len(params) > 44 else 8
                },
                "block_transfer_access_order_a": {
                    "order": seq_28 if seq_28 else [1, 0, 2]
                },
                "block_transfer_access_order_b": {
                    "order": seq_35 if seq_35 else [1, 0, 2]
                },
                "src_access_order_a": {
                    "order": seq_29 if seq_29 else [1, 0, 2]
                },
                "src_access_order_b": {
                    "order": seq_36 if seq_36 else [1, 0, 2]
                }
            },
            "block_gemm": {
                "pipeline_version": "V1",
                "scheduler": "INTRAWAVE"
            },
            "fwd_specialization": "DEFAULT",
            "gemm_specialization": "MNKPadding",
            "num_gemm_k_prefetch_stages": int(params[16]) if len(params) > 16 else 1,
            "loop_scheduler": "DEFAULT",
            "num_groups_to_merge": 1,
            "lds_extra": {
                "a_block_lds_extra_m": int(params[33]) if len(params) > 33 else 1,
                "b_block_lds_extra_n": int(params[40]) if len(params) > 40 else 1
            }
        }
    }
    
    return result

def parse_wmma_cshuffle_params(params: List[str]) -> Dict[str, Any]:
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
            "elementwise_operation": "PASS_THROUGH",
            "device_operation": "DeviceGroupedConvFwdMultipleD_Wmma_CShuffle"
        },
        "algorithm": {
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
                "n_wmma_per_wave": int(params[25]) if len(params) > 25 else 2
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
            "block_gemm": {
                "pipeline_version": "V1",
                "scheduler": "INTRAWAVE"
            },
            "fwd_specialization": "DEFAULT",
            "gemm_specialization": "MNKPadding",
            "num_gemm_k_prefetch_stages": int(params[16]) if len(params) > 16 else 1,
            "loop_scheduler": "DEFAULT",
            "num_groups_to_merge": 1,
            "lds_extra": {
                "a_block_lds_extra_m": int(params[33]) if len(params) > 33 else 1,
                "b_block_lds_extra_n": int(params[40]) if len(params) > 40 else 1
            }
        }
    }
    
    return result

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

def convert_instantiations(input_file: str, output_file: str):
    """Main conversion function"""
    
    print(f"Loading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    instantiations = []
    inst_id = 0
    
    # Process DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle
    print("Processing DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle instantiations...")
    if "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle" in data["device_operations"]:
        dev_op = data["device_operations"]["DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle"]
        
        for file_path, file_data in dev_op["instantiations_by_file"].items():
            for inst in file_data["instantiations"]:
                params = parse_instantiation_compressed(inst["instantiation_compressed"])
                if params:
                    try:
                        structured = parse_xdl_cshuffle_params_with_lds_extra(params)
                        structured["id"] = inst_id
                        structured["source_file"] = file_path
                        structured["line"] = inst["line_start"]
                        instantiations.append(structured)
                        inst_id += 1
                    except Exception as e:
                        print(f"  Warning: Failed to parse instantiation at line {inst['line_start']}: {e}")
    
    print(f"  Added {inst_id} XDL_CShuffle instantiations")
    
    # Process DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3
    print("Processing DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 instantiations...")
    start_id = inst_id
    if "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3" in data["device_operations"]:
        dev_op = data["device_operations"]["DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3"]
        
        for file_path, file_data in dev_op["instantiations_by_file"].items():
            for inst in file_data["instantiations"]:
                params = parse_instantiation_compressed(inst["instantiation_compressed"])
                if params:
                    try:
                        structured = parse_xdl_cshuffle_v3_params(params)
                        structured["id"] = inst_id
                        structured["source_file"] = file_path
                        structured["line"] = inst["line_start"]
                        instantiations.append(structured)
                        inst_id += 1
                    except Exception as e:
                        print(f"  Warning: Failed to parse V3 instantiation at line {inst['line_start']}: {e}")
    
    print(f"  Added {inst_id - start_id} XDL_CShuffle_V3 instantiations")
    
    # Process DeviceGroupedConvFwdMultipleD_Wmma_CShuffle
    print("Processing DeviceGroupedConvFwdMultipleD_Wmma_CShuffle instantiations...")
    start_id = inst_id
    if "DeviceGroupedConvFwdMultipleD_Wmma_CShuffle" in data["device_operations"]:
        dev_op = data["device_operations"]["DeviceGroupedConvFwdMultipleD_Wmma_CShuffle"]
        
        for file_path, file_data in dev_op["instantiations_by_file"].items():
            for inst in file_data["instantiations"]:
                params = parse_instantiation_compressed(inst["instantiation_compressed"])
                if params:
                    try:
                        structured = parse_wmma_cshuffle_params(params)
                        structured["id"] = inst_id
                        structured["source_file"] = file_path
                        structured["line"] = inst["line_start"]
                        instantiations.append(structured)
                        inst_id += 1
                    except Exception as e:
                        print(f"  Warning: Failed to parse instantiation at line {inst['line_start']}: {e}")
    
    print(f"  Added {inst_id - start_id} WMMA_CShuffle instantiations")
    
    # Create output structure with separate schemas for XDL and WMMA
    output_data = {
        "metadata": {
            "description": "Forward convolution kernel instantiations structured for C++ concept compliance",
            "version": "1.0",
            "total_instantiations": len(instantiations),
            "notes": "Each instantiation is represented as a JSON object with signature and algorithm fields that map to C++ structs satisfying ConvSignatureDescriptor and ConvAlgorithmDescriptor concepts. The algorithm structure differs based on algorithm_type (XDL vs WMMA). Note: V3 instances do not include lds_extra field as this is encoded in the is_direct_load flags."
        },
        "instantiations": instantiations,
        "schemas": {
            "signature": {
                "description": "Common signature schema for all instantiations",
                "spatial_dim": "integer (1, 2, or 3)",
                "direction": "enum (FORWARD, BACKWARD_DATA, BACKWARD_WEIGHT)",
                "layout": {
                    "input": "string (e.g., GNHWC, NGCHW, etc.)",
                    "weight": "string (e.g., GKYXC, GKCYX, etc.)",
                    "output": "string (e.g., GNHWK, NGKHW, etc.)"
                },
                "data_type": {
                    "input": "enum (FP32, FP16, BF16, FP8, I8, I32, U8)",
                    "weight": "enum (FP32, FP16, BF16, FP8, I8, I32, U8)",
                    "accumulator": "enum (FP32, FP16, BF16, FP8, I8, I32, U8)",
                    "shuffle": "enum (FP32, FP16, BF16, FP8, I8, I32, U8)",
                    "output": "enum (FP32, FP16, BF16, FP8, I8, I32, U8)"
                },
                "elementwise_operation": "enum (BIAS, BIAS_CLAMP, BIAS_BNORM_CLAMP, BILINEAR, CLAMP, SCALE, PASS_THROUGH)",
                "device_operation": "string (DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle, DeviceGroupedConvFwdMultipleD_Wmma_CShuffle, etc.)"
            },
            "algorithm_xdl": {
                "description": "Algorithm schema for XDL-based operations (algorithm_type = 'XDL')",
                "algorithm_type": "string literal 'XDL'",
                "gridwise_xdl_gemm": {
                    "ak1": "integer - A matrix K dimension vectorization",
                    "bk1": "integer - B matrix K dimension vectorization",
                    "m_per_xdl": "integer - M dimension per XDL operation",
                    "n_per_xdl": "integer - N dimension per XDL operation",
                    "m_xdl_per_wave": "integer - Number of XDL operations in M direction per wave",
                    "n_xdl_per_wave": "integer - Number of XDL operations in N direction per wave"
                },
                "note": "Use gridwise_xdl_gemm for XDL-based convolution operations"
            },
            "algorithm_wmma": {
                "description": "Algorithm schema for WMMA-based operations (algorithm_type = 'WMMA')",
                "algorithm_type": "string literal 'WMMA'",
                "gridwise_wmma_gemm": {
                    "k1": "integer - K dimension vectorization",
                    "m_per_wmma": "integer - M dimension per WMMA operation (typically 16)",
                    "n_per_wmma": "integer - N dimension per WMMA operation (typically 16)",
                    "m_wmma_per_wave": "integer - Number of WMMA operations in M direction per wave",
                    "n_wmma_per_wave": "integer - Number of WMMA operations in N direction per wave"
                },
                "note": "Use gridwise_wmma_gemm for WMMA-based convolution operations"
            }
        }
    }
    
    print(f"\nTotal instantiations converted: {len(instantiations)}")
    print(f"Writing to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    import sys
    
    input_file = "experimental/builder/instances/forward_conv_all_instantiations.json"
    output_file = "experimental/builder/instances/forward_conv_structured_instantiations.json"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    convert_instantiations(input_file, output_file)
