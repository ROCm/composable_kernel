#!/usr/bin/env python3
"""
Generate C++ source files with ConvBuilder instantiations from JSON.

This script reads forward_conv_structured_instantiations.json and generates
batched C++ files containing compile-time ConvBuilder instantiations for all
device operation entries.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple

def map_data_type(dt: str) -> str:
    """Map JSON data type string to C++ DataType enum"""
    type_map = {
        'FP32': 'DataType::FP32',
        'FP16': 'DataType::FP16',
        'BF16': 'DataType::BF16',
        'FP8': 'DataType::FP8',
        'I8': 'DataType::I8',
        'I32': 'DataType::I32',
        'U8': 'DataType::U8'
    }
    return type_map.get(dt, f'DataType::{dt}')

def map_layout(layout_str: Any, spatial_dim: int) -> str:
    """Map JSON layout string to C++ GroupConvLayout"""
    if spatial_dim == 1:
        layout_map = {
            'NWGC_GKXC_NWGK': 'GroupConvLayout1D::NWGC_GKXC_NWGK',
            'NGCW_GKXC_NGKW': 'GroupConvLayout1D::NGCW_GKXC_NGKW',
            'GNWC_GKXC_GNWK': 'GroupConvLayout1D::GNWC_GKXC_GNWK',
            'NGCW_GKCX_NGKW': 'GroupConvLayout1D::NGCW_GKCX_NGKW'
        }
        mapped = layout_map.get(layout_str, "/* UNKNOWN */")
        return mapped
    elif spatial_dim == 2:
        layout_map = {
            'GNHWC_GKYXC_GNHWK': 'GroupConvLayout2D::GNHWC_GKYXC_GNHWK',
            'NHWGC_GKYXC_NHWGK': 'GroupConvLayout2D::NHWGC_GKYXC_NHWGK',
            'NGCHW_GKYXC_NGKHW': 'GroupConvLayout2D::NGCHW_GKYXC_NGKHW',
            'NGCHW_GKCYX_NGKHW': 'GroupConvLayout2D::NGCHW_GKCYX_NGKHW'
        }
        layout_key = f"{layout_str['input']}_{layout_str['weight']}_{layout_str['output']}"
        mapped = layout_map.get(layout_key, "/* UNKNOWN */")
        return mapped
    elif spatial_dim == 3:
        layout_map = {
            'GNDHWC_GKZYXC_GNDHWK': 'GroupConvLayout3D::GNDHWC_GKZYXC_GNDHWK',
            'NDHWGC_GKZYXC_NDHWGK': 'GroupConvLayout3D::NDHWGC_GKZYXC_NDHWGK',
            'NGCDHW_GKCZYX_NGKDHW': 'GroupConvLayout3D::NGCDHW_GKCZYX_NGKDHW'
        }
        layout_key = f"{layout_str['input']}_{layout_str['weight']}_{layout_str['output']}"
        mapped = layout_map.get(layout_key, "/* UNKNOWN */")
        return mapped
    return 'GroupConvLayout{}'

def map_direction(direction: str) -> str:
    """Map JSON direction to C++ ConvDirection enum"""
    return f'ConvDirection::{direction}'

def map_elementwise_op(op: str) -> str:
    """Map JSON elementwise operation to C++ ElementwiseOperation enum"""
    return f'ElementwiseOperation::{op}'

def map_fwd_specialization(spec: str) -> str:
    """Map JSON fwd_specialization to C++ ConvFwdSpecialization enum"""
    spec_map = {
        'DEFAULT': 'ConvFwdSpecialization::DEFAULT',
        'FILTER_1X1_PAD0': 'ConvFwdSpecialization::FILTER_1X1_PAD0',
        'FILTER_1X1_STRIDE1_PAD0': 'ConvFwdSpecialization::FILTER_1X1_STRIDE1_PAD0',
        'FILTER_3x3': 'ConvFwdSpecialization::FILTER_3x3'
    }
    return spec_map.get(spec, f'ConvFwdSpecialization::{spec}')

def map_gemm_specialization(spec: str) -> str:
    """Map JSON gemm_specialization to C++ GemmSpecialization enum"""
    spec_map = {
        'Default': 'GemmSpecialization::Default',
        'MNKPadding': 'GemmSpecialization::MNKPadding',
        'MPadding': 'GemmSpecialization::MPadding',
        'NPadding': 'GemmSpecialization::NPadding',
        'KPadding': 'GemmSpecialization::KPadding'
    }
    return spec_map.get(spec, f'GemmSpecialization::{spec}')

def map_loop_scheduler(sched: str) -> str:
    """Map JSON loop_scheduler to C++ LoopScheduler enum"""
    sched_map = {
        'DEFAULT': 'PipelineScheduler::DEFAULT',
        'INTERWAVE': 'PipelineScheduler::INTERWAVE'
    }
    return sched_map.get(sched, f'LoopScheduler::{sched}')

def map_block_gemm_pipeline_version(ver: str) -> str:
    """Map JSON block_gemm pipeline_version to C++ BlockGemmPipelineVersion enum"""
    return f'PipelineVersion::{ver}'

def map_block_gemm_scheduler(sched: str) -> str:
    """Map JSON block_gemm scheduler to C++ BlockGemmPipelineScheduler enum"""
    return f'PipelineScheduler::{sched}'

def map_gridwise_gemm_pipeline_version(ver: str) -> str:
    """Map JSON gridwise_gemm pipeline_version to C++ PipelineVersion enum"""
    return f'PipelineVersion::{ver}'

def format_array(arr: List) -> str:
    """Format array for C++ initialization"""
    return '{' + ', '.join(map(str, arr)) + '}'

def generate_signature_struct(sig: Dict, idx: int) -> str:
    """Generate C++ constexpr ConvSignature struct"""
    layout = sig['layout']
    data_type = sig['data_type']
    
    parts = [
        f"constexpr ConvSignature sig_{idx} = {{",
        f"    .spatial_dim = {sig['spatial_dim']},",
        f"    .direction = {map_direction(sig['direction'])},",
        f"    .layout = {map_layout(layout, sig['spatial_dim'])},",
        f"    .data_type = {map_data_type(data_type['input'])},",
        f"    .elementwise_operation = {map_elementwise_op(sig['elementwise_operation'])}",
        "};"
    ]
    return '\n'.join(parts)

def generate_xdl_v3_algorithm_struct(algo: Dict, idx: int) -> Tuple[str, str]:
    """Generate C++ constexpr algorithm struct for V3 XDL"""
    bt = algo['block_transfer']
    
    parts = [
        f"constexpr ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 algo_{idx} = {{",
        "    .thread_block = {",
        f"        .block_size = {algo['thread_block']['block_size']},",
        "        .tile_size = {",
        f"            .m = {algo['thread_block']['tile_size']['m']},",
        f"            .n = {algo['thread_block']['tile_size']['n']},",
        f"            .k = {algo['thread_block']['tile_size']['k']}",
        "        }",
        "    },",
        "    .gridwise_gemm = {",
        f"        .ak1 = {algo['gridwise_xdl_gemm']['ak1']},",
        f"        .bk1 = {algo['gridwise_xdl_gemm']['bk1']},",
        f"        .m_per_xdl = {algo['gridwise_xdl_gemm']['m_per_xdl']},",
        f"        .n_per_xdl = {algo['gridwise_xdl_gemm']['n_per_xdl']},",
        f"        .m_xdl_per_wave = {algo['gridwise_xdl_gemm']['m_xdl_per_wave']},",
        f"        .n_xdl_per_wave = {algo['gridwise_xdl_gemm']['n_xdl_per_wave']}",
        "    },",
        "    .block_transfer = {",
        "        .block_transfer_a = {",
        f"            .k0 = {bt['block_transfer_a']['k0']},",
        f"            .m_n = {bt['block_transfer_a']['m_n']},",
        f"            .k1 = {bt['block_transfer_a']['k1']}",
        "        },",
        "        .block_transfer_b = {",
        f"            .k0 = {bt['block_transfer_b']['k0']},",
        f"            .m_n = {bt['block_transfer_b']['m_n']},",
        f"            .k1 = {bt['block_transfer_b']['k1']}",
        "        },",
        "        .thread_cluster_dims_c = {",
        f"            .m_block = {bt['thread_cluster_dims_c']['m_block']},",
        f"            .m_wave_per_xdl = {bt['thread_cluster_dims_c']['m_wave_per_xdl']},",
        f"            .n_block = {bt['thread_cluster_dims_c']['n_block']},",
        f"            .n_wave_per_xdl = {bt['thread_cluster_dims_c']['n_wave_per_xdl']}",
        "        },",
        "        .lds_transfer_a = {",
        f"            .src_vector_dim = {bt['lds_transfer_a']['src_vector_dim']},",
        f"            .src_scalar_per_vector = {bt['lds_transfer_a']['src_scalar_per_vector']},",
        f"            .lds_dst_scalar_per_vector = {bt['lds_transfer_a']['lds_dst_scalar_per_vector']},",
        f"            .is_direct_load = {str(bt['lds_transfer_a']['is_direct_load']).lower()},",
        f"            .lds_padding = {str(bt['lds_transfer_a']['lds_padding']).lower()}",
        "        },",
        "        .lds_transfer_b = {",
        f"            .src_vector_dim = {bt['lds_transfer_b']['src_vector_dim']},",
        f"            .src_scalar_per_vector = {bt['lds_transfer_b']['src_scalar_per_vector']},",
        f"            .lds_dst_scalar_per_vector = {bt['lds_transfer_b']['lds_dst_scalar_per_vector']},",
        f"            .is_direct_load = {str(bt['lds_transfer_b']['is_direct_load']).lower()},",
        f"            .lds_padding = {str(bt['lds_transfer_b']['lds_padding']).lower()}",
        "        },",
        "        .epilogue_c = {",
        f"            .m_per_wave_per_shuffle = {bt['epilogue_c']['m_per_wave_per_shuffle']},",
        f"            .n_per_wave_per_shuffle = {bt['epilogue_c']['n_per_wave_per_shuffle']},",
        f"            .scalar_per_vector = {bt['epilogue_c']['scalar_per_vector']}",
        "        },",
        f"        .block_transfer_access_order_a = {{.order = {format_array(bt['block_transfer_access_order_a']['order'])}}},",
        f"        .block_transfer_access_order_b = {{.order = {format_array(bt['block_transfer_access_order_b']['order'])}}},",
        f"        .src_access_order_a = {{.order = {format_array(bt['src_access_order_a']['order'])}}},",
        f"        .src_access_order_b = {{.order = {format_array(bt['src_access_order_b']['order'])}}}",
        "    },",
        f"    .fwd_specialization = {map_fwd_specialization(algo['fwd_specialization'])},",
        f"    .gemm_specialization = {map_gemm_specialization(algo['gemm_specialization'])},",
        "    .block_gemm = {",
        f"        .pipeline_version = {map_block_gemm_pipeline_version(algo['block_gemm']['pipeline_version'])},",
        f"        .scheduler = {map_block_gemm_scheduler(algo['block_gemm']['scheduler'])}",
        "    }",
        "};"
    ]
    
    return '\n'.join(parts), "ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3"

def generate_xdl_standard_algorithm_struct(algo: Dict, idx: int) -> Tuple[str, str]:
    """Generate C++ constexpr algorithm struct for standard XDL"""
    bt = algo['block_transfer']
    
    parts = [
        f"constexpr ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle algo_{idx} = {{",
        "    .thread_block = {",
        f"        .block_size = {algo['thread_block']['block_size']},",
        "        .tile_size = {",
        f"            .m = {algo['thread_block']['tile_size']['m']},",
        f"            .n = {algo['thread_block']['tile_size']['n']},",
        f"            .k = {algo['thread_block']['tile_size']['k']}",
        "        }",
        "    },",
        "    .gridwise_gemm = {",
        f"        .ak1 = {algo['gridwise_xdl_gemm']['ak1']},",
        f"        .bk1 = {algo['gridwise_xdl_gemm']['bk1']},",
        f"        .m_per_xdl = {algo['gridwise_xdl_gemm']['m_per_xdl']},",
        f"        .n_per_xdl = {algo['gridwise_xdl_gemm']['n_per_xdl']},",
        f"        .m_xdl_per_wave = {algo['gridwise_xdl_gemm']['m_xdl_per_wave']},",
        f"        .n_xdl_per_wave = {algo['gridwise_xdl_gemm']['n_xdl_per_wave']}",
        "    },",
        "    .block_transfer = {",
        "        .block_transfer_a = {",
        f"            .k0 = {bt['block_transfer_a']['k0']},",
        f"            .m_n = {bt['block_transfer_a']['m_n']},",
        f"            .k1 = {bt['block_transfer_a']['k1']}",
        "        },",
        "        .block_transfer_b = {",
        f"            .k0 = {bt['block_transfer_b']['k0']},",
        f"            .m_n = {bt['block_transfer_b']['m_n']},",
        f"            .k1 = {bt['block_transfer_b']['k1']}",
        "        },",
        "        .thread_cluster_dims_c = {",
        f"            .m_block = {bt['thread_cluster_dims_c']['m_block']},",
        f"            .m_wave_per_xdl = {bt['thread_cluster_dims_c']['m_wave_per_xdl']},",
        f"            .n_block = {bt['thread_cluster_dims_c']['n_block']},",
        f"            .n_wave_per_xdl = {bt['thread_cluster_dims_c']['n_wave_per_xdl']}",
        "        },",
        "        .lds_transfer_a = {",
        f"            .src_vector_dim = {bt['lds_transfer_a']['src_vector_dim']},",
        f"            .src_scalar_per_vector = {bt['lds_transfer_a']['src_scalar_per_vector']},",
        f"            .lds_dst_scalar_per_vector = {bt['lds_transfer_a']['lds_dst_scalar_per_vector']},",
        f"            .is_direct_load = {str(bt['lds_transfer_a']['is_direct_load']).lower()},",
        f"            .lds_padding = {str(bt['lds_transfer_a']['lds_padding']).lower()}",
        "        },",
        "        .lds_transfer_b = {",
        f"            .src_vector_dim = {bt['lds_transfer_b']['src_vector_dim']},",
        f"            .src_scalar_per_vector = {bt['lds_transfer_b']['src_scalar_per_vector']},",
        f"            .lds_dst_scalar_per_vector = {bt['lds_transfer_b']['lds_dst_scalar_per_vector']},",
        f"            .is_direct_load = {str(bt['lds_transfer_b']['is_direct_load']).lower()},",
        f"            .lds_padding = {str(bt['lds_transfer_b']['lds_padding']).lower()}",
        "        },",
        "        .epilogue_c = {",
        f"            .m_per_wave_per_shuffle = {bt['epilogue_c']['m_per_wave_per_shuffle']},",
        f"            .n_per_wave_per_shuffle = {bt['epilogue_c']['n_per_wave_per_shuffle']},",
        f"            .scalar_per_vector = {bt['epilogue_c']['scalar_per_vector']}",
        "        },",
        f"        .block_transfer_access_order_a = {{.order = {format_array(bt['block_transfer_access_order_a']['order'])}}},",
        f"        .block_transfer_access_order_b = {{.order = {format_array(bt['block_transfer_access_order_b']['order'])}}},",
        f"        .src_access_order_a = {{.order = {format_array(bt['src_access_order_a']['order'])}}},",
        f"        .src_access_order_b = {{.order = {format_array(bt['src_access_order_b']['order'])}}}",
        "    },",
        f"    .fwd_specialization = {map_fwd_specialization(algo['fwd_specialization'])},",
        f"    .gemm_specialization = {map_gemm_specialization(algo['gemm_specialization'])},",
        f"    .num_gemm_k_prefetch_stages = {algo['num_gemm_k_prefetch_stages']},",
        f"    .num_groups_to_merge = {algo['num_groups_to_merge']},",
        f"    .loop_scheduler = {map_loop_scheduler(algo['loop_scheduler'])}",
        "};"
    ]
    
    return '\n'.join(parts), "ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle"

def generate_wmma_standard_algorithm_struct(algo: Dict, idx: int) -> Tuple[str, str]:
    """Generate C++ constexpr algorithm struct for WMMA"""
    bt = algo['block_transfer']
    
    parts = [
        f"constexpr ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Wmma_CShuffle algo_{idx} = {{",
        "    .thread_block = {",
        f"        .block_size = {algo['thread_block']['block_size']},",
        "        .tile_size = {",
        f"            .m = {algo['thread_block']['tile_size']['m']},",
        f"            .n = {algo['thread_block']['tile_size']['n']},",
        f"            .k = {algo['thread_block']['tile_size']['k']}",
        "        }",
        "    },",
        "    .gridwise_gemm = {",
        f"        .k1 = {algo['gridwise_wmma_gemm']['k1']},",
        f"        .m_per_wmma = {algo['gridwise_wmma_gemm']['m_per_wmma']},",
        f"        .n_per_wmma = {algo['gridwise_wmma_gemm']['n_per_wmma']},",
        f"        .m_wmma_per_wave = {algo['gridwise_wmma_gemm']['m_wmma_per_wave']},",
        f"        .n_wmma_per_wave = {algo['gridwise_wmma_gemm']['n_wmma_per_wave']},",
        f"        .pipeline_version = {map_gridwise_gemm_pipeline_version(algo['gridwise_wmma_gemm']['pipeline_version'])}",
        "    },",
        "    .block_transfer = {",
        "        .block_transfer_a = {",
        f"            .k0 = {bt['block_transfer_a']['k0']},",
        f"            .m_n = {bt['block_transfer_a']['m_n']},",
        f"            .k1 = {bt['block_transfer_a']['k1']}",
        "        },",
        "        .block_transfer_b = {",
        f"            .k0 = {bt['block_transfer_b']['k0']},",
        f"            .m_n = {bt['block_transfer_b']['m_n']},",
        f"            .k1 = {bt['block_transfer_b']['k1']}",
        "        },",
        "        .thread_cluster_dims_c = {",
        f"            .m_block = {bt['thread_cluster_dims_c']['m_block']},",
        f"            .m_wave_per_xdl = {bt['thread_cluster_dims_c']['m_wave_per_xdl']},",
        f"            .n_block = {bt['thread_cluster_dims_c']['n_block']},",
        f"            .n_wave_per_xdl = {bt['thread_cluster_dims_c']['n_wave_per_xdl']}",
        "        },",
        "        .lds_transfer_a = {",
        f"            .src_vector_dim = {bt['lds_transfer_a']['src_vector_dim']},",
        f"            .src_scalar_per_vector = {bt['lds_transfer_a']['src_scalar_per_vector']},",
        f"            .lds_dst_scalar_per_vector = {bt['lds_transfer_a']['lds_dst_scalar_per_vector']},",
        f"            .is_direct_load = {str(bt['lds_transfer_a']['is_direct_load']).lower()},",
        f"            .lds_padding = {str(bt['lds_transfer_a']['lds_padding']).lower()}",
        "        },",
        "        .lds_transfer_b = {",
        f"            .src_vector_dim = {bt['lds_transfer_b']['src_vector_dim']},",
        f"            .src_scalar_per_vector = {bt['lds_transfer_b']['src_scalar_per_vector']},",
        f"            .lds_dst_scalar_per_vector = {bt['lds_transfer_b']['lds_dst_scalar_per_vector']},",
        f"            .is_direct_load = {str(bt['lds_transfer_b']['is_direct_load']).lower()},",
        f"            .lds_padding = {str(bt['lds_transfer_b']['lds_padding']).lower()}",
        "        },",
        "        .epilogue_c = {",
        f"            .m_per_wave_per_shuffle = {bt['epilogue_c']['m_per_wave_per_shuffle']},",
        f"            .n_per_wave_per_shuffle = {bt['epilogue_c']['n_per_wave_per_shuffle']},",
        f"            .scalar_per_vector = {bt['epilogue_c']['scalar_per_vector']}",
        "        },",
        f"        .block_transfer_access_order_a = {{.order = {format_array(bt['block_transfer_access_order_a']['order'])}}},",
        f"        .block_transfer_access_order_b = {{.order = {format_array(bt['block_transfer_access_order_b']['order'])}}},",
        f"        .src_access_order_a = {{.order = {format_array(bt['src_access_order_a']['order'])}}},",
        f"        .src_access_order_b = {{.order = {format_array(bt['src_access_order_b']['order'])}}}",
        "    },",
        f"    .fwd_specialization = {map_fwd_specialization(algo['fwd_specialization'])},",
        f"    .gemm_specialization = {map_gemm_specialization(algo['gemm_specialization'])},",
        f"    .num_gemm_k_prefetch_stages = {algo['num_gemm_k_prefetch_stages']},",
        f"    .loop_scheduler = {map_loop_scheduler(algo['loop_scheduler'])}",
        "};"
    ]
    
    return '\n'.join(parts), "ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Wmma_CShuffle"

def generate_cpp_file(batch_idx: int, instantiations: List[Dict], output_dir: Path) -> Tuple[Path, str]:
    """Generate a single C++ file for a batch of instantiations"""
    
    header = f"""// GENERATED FILE - DO NOT EDIT
// Generated from forward_conv_structured_instantiations.json
// Batch {batch_idx} of convolution builder instantiations

#include "ck_tile/builder/conv_builder.hpp"
#include "conv_signature_types.hpp"
#include "conv_algorithm_types.hpp"
#include "instance_registry.hpp"

namespace ck_tile::builder::generated::batch_{batch_idx} {{

using namespace ck_tile::builder;
using namespace ck_tile::builder::test;
using namespace ck_tile::builder::registry;

"""
    
    structs_code = []
    builder_typedefs = []
    
    for inst in instantiations:
        idx = inst['id']
        sig_code = generate_signature_struct(inst['signature'], idx)
        
        # Skip the scale-add instances, we cannot correctly generate those.
        if 'scaleadd' in inst['source_file']:
            print(f"\t\033[33m  Skipping scale-add instance {idx} from {inst['source_file']}:{inst['line']}\033[0m")
            continue

        # Skip small number fo instances that do not build correctly
        skip_indices = [570, 587, 609]
        if idx in skip_indices:
            print(f"\t\033[33m  Skipping instance {idx} from {inst['source_file']}:{inst['line']} since we cannot build it.\033[0m")
            continue

        # Determine algorithm type and generate appropriate struct
        device_op = inst['algorithm']['device_operation']
        algo_type = inst['algorithm']['algorithm_type']
        
        if 'V3' in device_op:
            algo_code, algo_type_name = generate_xdl_v3_algorithm_struct(inst['algorithm'], idx)
        elif algo_type == 'WMMA':
            algo_code, algo_type_name = generate_wmma_standard_algorithm_struct(inst['algorithm'], idx)
        else:  # Standard XDL
            algo_code, algo_type_name = generate_xdl_standard_algorithm_struct(inst['algorithm'], idx)
        
        structs_code.append(f"// Instantiation {idx} from {inst['source_file']}:{inst['line']}")
        structs_code.append(sig_code)
        structs_code.append(algo_code)
        structs_code.append("")
        
        builder_typedefs.append(f"using Builder_{idx} = ConvBuilder<sig_{idx}, algo_{idx}>;")
        builder_typedefs.append(f"static AutoRegister<Builder_{idx}> reg_{idx}(\"batch_{batch_idx}_instance_{idx}\");")
        builder_typedefs.append("")
    
    footer = f"""}} // namespace ck_tile::builder::generated::batch_{batch_idx}
"""
    
    content = header + '\n'.join(structs_code) + '\n// ConvBuilder instantiations\n' + '\n'.join(builder_typedefs) + '\n' + footer
    
    filename = output_dir / f'conv_instances_batch_{batch_idx:02d}.cpp'
    return filename, content

def generate_all_cpp_files(json_file: Path, output_dir: Path, batch_size: int = 50):
    """Generate all C++ files from JSON"""
    
    print(f"Loading JSON from {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    instantiations = data['instantiations']
    total = len(instantiations)
    
    print(f"Found {total} instantiations")
    print(f"Generating C++ files with batch size {batch_size}...")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate batched files
    num_batches = (total + batch_size - 1) // batch_size
    generated_files = []
    
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch = instantiations[start:end]
        
        filename, content = generate_cpp_file(batch_idx, batch, output_dir)
        
        with open(filename, 'w') as f:
            f.write(content)
        
        generated_files.append(filename)
        print(f"  Generated {filename.name} ({len(batch)} instantiations)")
    
    print(f"\nGenerated {len(generated_files)} files in {output_dir}")
    return generated_files

def generate_cmake_file(output_dir: Path, cpp_files: List[Path]):
    """Generate CMakeLists.txt for building generated instances"""
    
    file_list = '\n  '.join([f.name for f in cpp_files])
    
    cmake_content = f"""# GENERATED FILE - DO NOT EDIT
# CMakeLists.txt for ConvBuilder generated instantiations

cmake_minimum_required(VERSION 3.16)

# Add all generated instance files as a library
add_library(ck_builder_generated_instances STATIC
  {file_list}
)

target_include_directories(ck_builder_generated_instances PUBLIC
  ${{CMAKE_SOURCE_DIR}}/include
  ${{CMAKE_SOURCE_DIR}}/experimental/builder/include
  ${{CMAKE_SOURCE_DIR}}/experimental/builder/test/impl
)

target_compile_features(ck_builder_generated_instances PUBLIC cxx_std_20)

# Optionally add tests
if(BUILD_TESTING)
  add_subdirectory(test)
endif()
"""
    
    cmake_file = output_dir / 'CMakeLists.txt'
    with open(cmake_file, 'w') as f:
        f.write(cmake_content)
    
    print(f"Generated {cmake_file}")
    return cmake_file

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate ConvBuilder instantiations from JSON')
    parser.add_argument('--json', type=str,
                       default='./forward_conv_structured_instantiations.json',
                       help='Path to JSON file')
    parser.add_argument('--output', type=str,
                       default='../codegen',
                       help='Output directory for generated files')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Number of instantiations per C++ file')
    parser.add_argument('--cmake', action='store_true',
                       help='Generate CMakeLists.txt')
    
    args = parser.parse_args()
    
    json_file = Path(args.json)
    output_dir = Path(args.output)
    
    if not json_file.exists():
        print(f"Error: JSON file not found: {json_file}")
        return 1
    
    # Generate C++ files
    cpp_files = generate_all_cpp_files(json_file, output_dir, args.batch_size)
    
    # Generate CMakeLists.txt if requested
    if args.cmake:
        generate_cmake_file(output_dir, cpp_files)
    
    print("\n✓ Code generation complete!")
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
