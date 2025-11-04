#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Unified Codegen: Generate dispatcher-compatible wrappers from tile_engine kernels

This script scans tile_engine generated kernel headers and creates:
1. Dispatcher wrapper headers that register kernels
2. Automatic registration initialization code
3. Python-compatible kernel metadata

Usage:
    python generate_dispatcher_wrappers.py \
        --tile-engine-dir ../tile_engine/ops/gemm \
        --output-dir ./generated \
        --operation gemm
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class KernelMetadata:
    """Metadata extracted from tile_engine generated kernel"""
    name: str
    datatype: str
    layout: str
    pipeline: str
    epilogue: str
    scheduler: str
    pad_m: bool
    pad_n: bool
    pad_k: bool
    persistent: bool
    tile_m: int
    tile_n: int
    tile_k: int
    warp_m: int
    warp_n: int
    warp_k: int
    warp_tile_m: int
    warp_tile_n: int
    warp_tile_k: int
    block_size: int
    double_buffer: bool
    preshuffle: bool
    transpose_c: bool
    structured_sparsity: bool
    num_wave_groups: int
    header_path: str


def parse_kernel_name(name: str) -> Optional[Dict[str, str]]:
    """
    Parse kernel name to extract metadata
    Format: gemm_dtype_layout_pipeline_epilogue_scheduler_padM_padN_padK_persistent_tileconfig
    Example: gemm_fp16_rcr_compv4_cshuffle_intrawave_False_False_False_False_256x256x32_2x2x1_32x32x16
    """
    pattern = r'gemm_(\w+)_(\w+)_(\w+)_(\w+)_(\w+)_(True|False)_(True|False)_(True|False)_(True|False)_(\d+)x(\d+)x(\d+)_(\d+)x(\d+)x(\d+)_(\d+)x(\d+)x(\d+)'
    match = re.match(pattern, name)
    
    if not match:
        return None
    
    return {
        'datatype': match.group(1),
        'layout': match.group(2),
        'pipeline': match.group(3),
        'epilogue': match.group(4),
        'scheduler': match.group(5),
        'pad_m': match.group(6) == 'True',
        'pad_n': match.group(7) == 'True',
        'pad_k': match.group(8) == 'True',
        'persistent': match.group(9) == 'True',
        'tile_m': int(match.group(10)),
        'tile_n': int(match.group(11)),
        'tile_k': int(match.group(12)),
        'warp_m': int(match.group(13)),
        'warp_n': int(match.group(14)),
        'warp_k': int(match.group(15)),
        'warp_tile_m': int(match.group(16)),
        'warp_tile_n': int(match.group(17)),
        'warp_tile_k': int(match.group(18)),
    }


def scan_tile_engine_kernels(tile_engine_dir: Path) -> List[KernelMetadata]:
    """Scan tile_engine directory for generated kernel headers"""
    kernels = []
    
    # Look for generated kernel headers
    for header_file in tile_engine_dir.rglob("gemm_*.hpp"):
        kernel_name = header_file.stem
        
        # Parse kernel name
        metadata_dict = parse_kernel_name(kernel_name)
        if not metadata_dict:
            continue
        
        # Read header to extract additional metadata
        content = header_file.read_text()
        
        # Extract static constexpr values
        block_size = 256  # Default
        double_buffer = 'compv4' in metadata_dict['pipeline']
        preshuffle = False
        transpose_c = False
        structured_sparsity = False
        num_wave_groups = 1
        
        # Try to extract from header
        if 'BlockSize = ' in content:
            match = re.search(r'BlockSize\s*=\s*(\d+)', content)
            if match:
                block_size = int(match.group(1))
        
        if 'DoubleSmemBuffer' in content:
            match = re.search(r'DoubleSmemBuffer\s*=\s*(true|false)', content)
            if match:
                double_buffer = match.group(1) == 'true'
        
        if 'Preshuffle' in content:
            match = re.search(r'Preshuffle\s*=\s*(true|false)', content)
            if match:
                preshuffle = match.group(1) == 'true'
        
        metadata = KernelMetadata(
            name=kernel_name,
            datatype=metadata_dict['datatype'],
            layout=metadata_dict['layout'],
            pipeline=metadata_dict['pipeline'],
            epilogue=metadata_dict['epilogue'],
            scheduler=metadata_dict['scheduler'],
            pad_m=metadata_dict['pad_m'],
            pad_n=metadata_dict['pad_n'],
            pad_k=metadata_dict['pad_k'],
            persistent=metadata_dict['persistent'],
            tile_m=metadata_dict['tile_m'],
            tile_n=metadata_dict['tile_n'],
            tile_k=metadata_dict['tile_k'],
            warp_m=metadata_dict['warp_m'],
            warp_n=metadata_dict['warp_n'],
            warp_k=metadata_dict['warp_k'],
            warp_tile_m=metadata_dict['warp_tile_m'],
            warp_tile_n=metadata_dict['warp_tile_n'],
            warp_tile_k=metadata_dict['warp_tile_k'],
            block_size=block_size,
            double_buffer=double_buffer,
            preshuffle=preshuffle,
            transpose_c=transpose_c,
            structured_sparsity=structured_sparsity,
            num_wave_groups=num_wave_groups,
            header_path=str(header_file)
        )
        
        kernels.append(metadata)
    
    return kernels


def map_datatype(dt: str) -> str:
    """Map tile_engine datatype to dispatcher DataType enum"""
    mapping = {
        'fp16': 'DataType::FP16',
        'bf16': 'DataType::BF16',
        'fp32': 'DataType::FP32',
        'fp8': 'DataType::FP8',
        'bf8': 'DataType::BF8',
        'int8': 'DataType::INT8',
    }
    return mapping.get(dt, 'DataType::UNKNOWN')


def map_layout(layout_str: str, pos: int) -> str:
    """Map layout character to dispatcher LayoutTag enum"""
    layout_char = layout_str[pos] if pos < len(layout_str) else 'r'
    mapping = {
        'r': 'LayoutTag::RowMajor',
        'c': 'LayoutTag::ColMajor',
    }
    return mapping.get(layout_char, 'LayoutTag::RowMajor')


def map_pipeline(pipeline: str) -> str:
    """Map pipeline name to dispatcher Pipeline enum"""
    mapping = {
        'mem': 'Pipeline::Mem',
        'compv1': 'Pipeline::CompV1',
        'compv2': 'Pipeline::CompV2',
        'compv3': 'Pipeline::CompV3',
        'compv4': 'Pipeline::CompV4',
        'compv5': 'Pipeline::CompV5',
    }
    return mapping.get(pipeline, 'Pipeline::CompV4')


def map_scheduler(scheduler: str) -> str:
    """Map scheduler name to dispatcher Scheduler enum"""
    mapping = {
        'intrawave': 'Scheduler::Intrawave',
        'interwave': 'Scheduler::Interwave',
        'default': 'Scheduler::Auto',
    }
    return mapping.get(scheduler, 'Scheduler::Intrawave')


def map_epilogue(epilogue: str) -> str:
    """Map epilogue name to dispatcher Epilogue enum"""
    mapping = {
        'cshuffle': 'Epilogue::CShuffle',
        'default': 'Epilogue::Default',
        'none': 'Epilogue::None',
    }
    return mapping.get(epilogue, 'Epilogue::CShuffle')


def generate_wrapper_header(kernel: KernelMetadata, output_dir: Path) -> Path:
    """Generate dispatcher wrapper header for a single kernel"""
    
    wrapper_name = f"dispatcher_wrapper_{kernel.name}"
    output_file = output_dir / f"{wrapper_name}.hpp"
    
    # Determine output datatype (fp8/bf8 -> fp16)
    output_dtype = kernel.datatype
    if kernel.datatype in ['fp8', 'bf8']:
        output_dtype = 'fp16'
    
    content = f"""// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
// Auto-generated by generate_dispatcher_wrappers.py

#pragma once

#include "ck_tile/dispatcher.hpp"
#include "{kernel.header_path}"

namespace ck_tile {{
namespace dispatcher {{
namespace generated {{

/// Dispatcher wrapper for {kernel.name}
inline KernelInstancePtr make_{kernel.name}(std::uint16_t gfx_arch = 942)
{{
    return make_tile_kernel_instance<SelectedKernel>(
        {map_datatype(kernel.datatype)},      // dtype_a
        {map_datatype(kernel.datatype)},      // dtype_b
        {map_datatype(output_dtype)},         // dtype_c
        DataType::FP32,                       // dtype_acc
        {map_layout(kernel.layout, 0)},       // layout_a
        {map_layout(kernel.layout, 1)},       // layout_b
        {map_layout(kernel.layout, 2)},       // layout_c
        {map_pipeline(kernel.pipeline)},      // pipeline
        {map_scheduler(kernel.scheduler)},    // scheduler
        {map_epilogue(kernel.epilogue)},      // epilogue
        gfx_arch,                             // gfx_arch
        "{kernel.name}"                       // name
    );
}}

}} // namespace generated
}} // namespace dispatcher
}} // namespace ck_tile
"""
    
    output_file.write_text(content)
    return output_file


def generate_registration_header(kernels: List[KernelMetadata], output_dir: Path) -> Path:
    """Generate master registration header that includes all wrappers"""
    
    output_file = output_dir / "register_all_kernels.hpp"
    
    includes = "\n".join([
        f'#include "dispatcher_wrapper_{k.name}.hpp"'
        for k in kernels
    ])
    
    registrations = "\n        ".join([
        f'registry.register_kernel(generated::make_{k.name}(gfx_arch), priority);'
        for k in kernels
    ])
    
    content = f"""// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
// Auto-generated by generate_dispatcher_wrappers.py

#pragma once

#include "ck_tile/dispatcher.hpp"
{includes}

namespace ck_tile {{
namespace dispatcher {{

/// Register all tile_engine generated GEMM kernels with the dispatcher
/// @param gfx_arch Target GPU architecture (e.g., 942 for gfx942)
/// @param priority Registration priority for conflict resolution
inline void register_all_tile_gemm_kernels(
    std::uint16_t gfx_arch = 942,
    Registry::Priority priority = Registry::Priority::Normal)
{{
    auto& registry = Registry::instance();
    
    // Register all generated kernels
    {registrations}
}}

/// Get count of available tile_engine GEMM kernels
inline std::size_t get_tile_gemm_kernel_count()
{{
    return {len(kernels)};
}}

}} // namespace dispatcher
}} // namespace ck_tile
"""
    
    output_file.write_text(content)
    return output_file


def generate_kernel_metadata_json(kernels: List[KernelMetadata], output_dir: Path) -> Path:
    """Generate JSON metadata file for Python/external tools"""
    
    output_file = output_dir / "kernel_metadata.json"
    
    metadata_list = []
    for k in kernels:
        metadata_list.append({
            'name': k.name,
            'datatype': k.datatype,
            'layout': k.layout,
            'pipeline': k.pipeline,
            'epilogue': k.epilogue,
            'scheduler': k.scheduler,
            'tile': {
                'm': k.tile_m,
                'n': k.tile_n,
                'k': k.tile_k
            },
            'wave': {
                'm': k.warp_m,
                'n': k.warp_n,
                'k': k.warp_k
            },
            'warp_tile': {
                'm': k.warp_tile_m,
                'n': k.warp_tile_n,
                'k': k.warp_tile_k
            },
            'persistent': k.persistent,
            'double_buffer': k.double_buffer,
            'block_size': k.block_size,
            'header_path': k.header_path
        })
    
    with open(output_file, 'w') as f:
        json.dump(metadata_list, f, indent=2)
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Generate dispatcher wrappers from tile_engine kernels')
    parser.add_argument('--tile-engine-dir', type=Path, required=True,
                       help='Path to tile_engine ops directory')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Output directory for generated files')
    parser.add_argument('--operation', type=str, default='gemm',
                       help='Operation type (gemm, conv, etc.)')
    parser.add_argument('--gfx-arch', type=int, default=942,
                       help='Target GPU architecture')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning {args.tile_engine_dir} for {args.operation} kernels...")
    
    # Scan for kernels
    kernels = scan_tile_engine_kernels(args.tile_engine_dir)
    print(f"Found {len(kernels)} kernels")
    
    if not kernels:
        print("No kernels found. Make sure tile_engine has generated kernels.")
        return 1
    
    # Generate wrapper headers
    print(f"\nGenerating wrapper headers in {args.output_dir}...")
    for kernel in kernels:
        wrapper_file = generate_wrapper_header(kernel, args.output_dir)
        print(f"  Generated: {wrapper_file.name}")
    
    # Generate registration header
    print("\nGenerating registration header...")
    reg_file = generate_registration_header(kernels, args.output_dir)
    print(f"  Generated: {reg_file.name}")
    
    # Generate metadata JSON
    print("\nGenerating metadata JSON...")
    json_file = generate_kernel_metadata_json(kernels, args.output_dir)
    print(f"  Generated: {json_file.name}")
    
    print(f"\n✅ Code generation complete!")
    print(f"   Total kernels: {len(kernels)}")
    print(f"   Output directory: {args.output_dir}")
    print(f"\nTo use in your code:")
    print(f'  #include "{reg_file.name}"')
    print(f'  ck_tile::dispatcher::register_all_tile_gemm_kernels({args.gfx_arch});')
    
    return 0


if __name__ == '__main__':
    exit(main())

