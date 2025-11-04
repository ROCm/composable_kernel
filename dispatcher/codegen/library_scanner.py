#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Library Scanner - Discover Existing CK Library Kernels

Scans the CK library directory for existing kernel instances and generates
dispatcher wrappers for them. This allows reusing pre-compiled kernels
without regenerating them.

Inspired by ck4inductor's gen_ops_library() approach.
"""

import re
import subprocess
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from functools import lru_cache

log = logging.getLogger(__name__)


# ============================================================================
# Parsed Kernel Information
# ============================================================================

@dataclass
class ParsedKernel:
    """Information extracted from library kernel"""
    file_path: Path
    line_number: int
    kernel_type: str  # e.g., "GemmKernel", "DeviceGemm_Xdl_CShuffleV3"
    template_args: List[str]
    raw_line: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'file_path': str(self.file_path),
            'line_number': self.line_number,
            'kernel_type': self.kernel_type,
            'template_args': self.template_args,
            'raw_line': self.raw_line,
        }


# ============================================================================
# Library Scanner
# ============================================================================

class LibraryScanner:
    """Scan CK library for existing kernel instances"""
    
    def __init__(self, library_path: Path):
        self.library_path = Path(library_path)
        self.kernels: List[ParsedKernel] = []
        
    def scan_tile_gemm_kernels(self) -> List[ParsedKernel]:
        """
        Scan for CK Tile GEMM kernels
        
        Looks for patterns like:
        - ck_tile::GemmKernel<...>
        - using GemmKernel = ck_tile::GemmKernel<...>
        """
        log.info(f"Scanning for CK Tile GEMM kernels in: {self.library_path}")
        
        if not self.library_path.exists():
            log.error(f"Library path does not exist: {self.library_path}")
            return []
        
        patterns = [
            r'ck_tile::GemmKernel<',
            r'using\s+\w+\s*=\s*ck_tile::GemmKernel<',
        ]
        
        kernels = []
        for pattern in patterns:
            found = self._grep_pattern(pattern)
            kernels.extend(found)
        
        self.kernels = kernels
        log.info(f"Found {len(kernels)} CK Tile GEMM kernel instances")
        return kernels
    
    def scan_legacy_gemm_kernels(self) -> List[ParsedKernel]:
        """
        Scan for legacy CK library GEMM kernels
        
        Looks for patterns like:
        - DeviceGemm_Xdl_CShuffleV3<...>
        - DeviceGemm_Xdl_CShuffle<...>
        """
        log.info(f"Scanning for legacy GEMM kernels in: {self.library_path}")
        
        if not self.library_path.exists():
            log.error(f"Library path does not exist: {self.library_path}")
            return []
        
        patterns = [
            r'DeviceGemm_Xdl_CShuffleV3<',
            r'DeviceGemm_Xdl_CShuffle<',
        ]
        
        kernels = []
        for pattern in patterns:
            found = self._grep_pattern(pattern)
            kernels.extend(found)
        
        log.info(f"Found {len(kernels)} legacy GEMM kernel instances")
        return kernels
    
    def _grep_pattern(self, pattern: str) -> List[ParsedKernel]:
        """Use grep to find pattern in library"""
        try:
            result = subprocess.run(
                ['grep', '-inR', pattern, str(self.library_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0 and result.returncode != 1:
                log.warning(f"grep failed with code {result.returncode}")
                return []
            
            return self._parse_grep_output(result.stdout, pattern)
            
        except subprocess.TimeoutExpired:
            log.error("grep timed out")
            return []
        except FileNotFoundError:
            log.error("grep not found, falling back to Python search")
            return self._python_search(pattern)
        except Exception as e:
            log.error(f"grep failed: {e}")
            return []
    
    def _parse_grep_output(self, output: str, pattern: str) -> List[ParsedKernel]:
        """Parse grep output into ParsedKernel objects"""
        kernels = []
        
        for line in output.strip().split('\n'):
            if not line:
                continue
            
            try:
                # Format: file:line:content
                parts = line.split(':', 2)
                if len(parts) < 3:
                    continue
                
                file_path = Path(parts[0])
                line_number = int(parts[1])
                content = parts[2].strip()
                
                # Extract kernel type
                kernel_type = self._extract_kernel_type(content, pattern)
                
                # Extract template arguments (simplified)
                template_args = self._extract_template_args(content)
                
                kernel = ParsedKernel(
                    file_path=file_path,
                    line_number=line_number,
                    kernel_type=kernel_type,
                    template_args=template_args,
                    raw_line=content
                )
                
                kernels.append(kernel)
                
            except Exception as e:
                log.debug(f"Failed to parse line: {line[:100]}... Error: {e}")
                continue
        
        return kernels
    
    def _extract_kernel_type(self, content: str, pattern: str) -> str:
        """Extract kernel type from content"""
        # Look for pattern in content
        match = re.search(r'(\w+::\w+|\w+)<', content)
        if match:
            return match.group(1)
        return "Unknown"
    
    def _extract_template_args(self, content: str) -> List[str]:
        """
        Extract template arguments (simplified)
        
        This is a simplified version. Full parsing would require
        handling nested templates, which is complex.
        """
        # Find content between < and >
        match = re.search(r'<(.+)>', content)
        if not match:
            return []
        
        args_str = match.group(1)
        
        # Simple split by comma (doesn't handle nested templates well)
        # For production, would need proper C++ template parser
        args = [arg.strip() for arg in args_str.split(',')]
        
        return args
    
    def _python_search(self, pattern: str) -> List[ParsedKernel]:
        """Fallback: Python-based search if grep not available"""
        log.info("Using Python-based search (slower than grep)")
        
        kernels = []
        regex = re.compile(pattern)
        
        # Search all .hpp and .cpp files
        for ext in ['*.hpp', '*.cpp', '*.h']:
            for file_path in self.library_path.rglob(ext):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            if regex.search(line):
                                kernel = ParsedKernel(
                                    file_path=file_path,
                                    line_number=line_num,
                                    kernel_type=self._extract_kernel_type(line, pattern),
                                    template_args=self._extract_template_args(line),
                                    raw_line=line.strip()
                                )
                                kernels.append(kernel)
                except Exception as e:
                    log.debug(f"Failed to read {file_path}: {e}")
                    continue
        
        return kernels
    
    def filter_by_datatype(self, datatype: str) -> List[ParsedKernel]:
        """Filter kernels by datatype"""
        datatype_patterns = {
            'fp16': ['half_t', 'F16', 'fp16'],
            'bf16': ['bf16_t', 'BF16', 'bf16'],
            'fp32': ['float', 'F32', 'fp32'],
            'fp8': ['fp8_t', 'F8', 'fp8'],
            'bf8': ['bf8_t', 'BF8', 'bf8'],
            'int8': ['int8_t', 'I8', 'int8'],
        }
        
        patterns = datatype_patterns.get(datatype.lower(), [])
        if not patterns:
            log.warning(f"Unknown datatype: {datatype}")
            return []
        
        filtered = []
        for kernel in self.kernels:
            # Check if any pattern appears in template args or raw line
            if any(p in kernel.raw_line for p in patterns):
                filtered.append(kernel)
        
        log.info(f"Filtered to {len(filtered)} kernels with datatype {datatype}")
        return filtered
    
    def filter_by_layout(self, layout: str) -> List[ParsedKernel]:
        """Filter kernels by layout"""
        layout_patterns = {
            'r': ['RowMajor', 'Row'],
            'c': ['ColumnMajor', 'Col'],
        }
        
        filtered = []
        for kernel in self.kernels:
            # Check if layout pattern appears
            layout_match = all(
                any(layout_patterns.get(l, [l]) for p in layout_patterns.get(l, [l]) 
                    if p in kernel.raw_line)
                for l in layout
            )
            if layout_match:
                filtered.append(kernel)
        
        return filtered
    
    def export_to_json(self, output_path: Path):
        """Export discovered kernels to JSON"""
        import json
        
        data = {
            'library_path': str(self.library_path),
            'kernel_count': len(self.kernels),
            'kernels': [k.to_dict() for k in self.kernels]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        log.info(f"Exported {len(self.kernels)} kernels to {output_path}")
    
    def generate_summary(self) -> Dict:
        """Generate summary statistics"""
        summary = {
            'total_kernels': len(self.kernels),
            'kernel_types': {},
            'files': set(),
        }
        
        for kernel in self.kernels:
            # Count by type
            kernel_type = kernel.kernel_type
            summary['kernel_types'][kernel_type] = \
                summary['kernel_types'].get(kernel_type, 0) + 1
            
            # Track files
            summary['files'].add(str(kernel.file_path))
        
        summary['unique_files'] = len(summary['files'])
        summary['files'] = sorted(summary['files'])
        
        return summary


# ============================================================================
# Wrapper Generator for Library Kernels
# ============================================================================

class LibraryWrapperGenerator:
    """Generate dispatcher wrappers for library kernels"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_wrapper(self, kernel: ParsedKernel, kernel_name: str) -> Path:
        """
        Generate dispatcher wrapper for a library kernel
        
        Note: This is a simplified version. Full implementation would need
        to parse template arguments and map them to KernelKey fields.
        """
        wrapper_code = f"""// SPDX-License-Identifier: MIT
// Auto-generated dispatcher wrapper for library kernel
#pragma once

#include "ck_tile/dispatcher.hpp"
#include "{kernel.file_path.name}"

namespace ck_tile {{
namespace dispatcher {{
namespace library {{

// Wrapper for kernel found at:
// File: {kernel.file_path}
// Line: {kernel.line_number}
// Type: {kernel.kernel_type}

// TODO: Parse template arguments and create KernelKey
// For now, this is a placeholder

/*
inline KernelInstancePtr make_{kernel_name}(std::uint16_t gfx_arch = 942) {{
    KernelKey key;
    // TODO: Fill in key from parsed template arguments
    
    return std::make_shared<LibraryKernelInstance>(key, "{kernel_name}");
}}
*/

// Original kernel signature:
// {kernel.raw_line[:200]}...

}}}}
}}
"""
        
        wrapper_path = self.output_dir / f"library_wrapper_{kernel_name}.hpp"
        wrapper_path.write_text(wrapper_code)
        
        log.debug(f"Generated wrapper: {wrapper_path}")
        return wrapper_path


# ============================================================================
# Cached Library Scanning
# ============================================================================

@lru_cache(None)
def scan_default_library(library_path: Optional[Path] = None) -> LibraryScanner:
    """
    Scan default CK library location (cached)
    
    Args:
        library_path: Path to library, or None to auto-detect
    
    Returns:
        LibraryScanner with discovered kernels
    """
    if library_path is None:
        # Try to find library path
        possible_paths = [
            Path(__file__).parent.parent.parent / "library",
            Path(__file__).parent.parent.parent / "build" / "library",
            Path("/opt/rocm/composable_kernel/library"),
        ]
        
        for path in possible_paths:
            if path.exists():
                library_path = path
                break
        
        if library_path is None:
            log.warning("Could not find CK library path")
            return LibraryScanner(Path("."))
    
    scanner = LibraryScanner(library_path)
    scanner.scan_tile_gemm_kernels()
    return scanner


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Scan CK library for existing kernel instances')
    parser.add_argument('--library-path', type=Path, required=True,
                       help='Path to CK library directory')
    parser.add_argument('--output-dir', type=Path,
                       help='Output directory for wrappers')
    parser.add_argument('--export-json', type=Path,
                       help='Export discovered kernels to JSON')
    parser.add_argument('--datatype', type=str,
                       help='Filter by datatype (fp16, bf16, etc.)')
    parser.add_argument('--layout', type=str,
                       help='Filter by layout (rcr, rrr, etc.)')
    parser.add_argument('--summary', action='store_true',
                       help='Print summary statistics')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Scan library
    scanner = LibraryScanner(args.library_path)
    scanner.scan_tile_gemm_kernels()
    
    # Apply filters
    kernels = scanner.kernels
    if args.datatype:
        kernels = scanner.filter_by_datatype(args.datatype)
    if args.layout:
        kernels = scanner.filter_by_layout(args.layout)
    
    # Print summary
    if args.summary:
        summary = scanner.generate_summary()
        print(f"\nLibrary Scan Summary:")
        print(f"  Total kernels: {summary['total_kernels']}")
        print(f"  Unique files: {summary['unique_files']}")
        print(f"\nKernel types:")
        for ktype, count in summary['kernel_types'].items():
            print(f"    {ktype}: {count}")
    
    # Export to JSON
    if args.export_json:
        scanner.export_to_json(args.export_json)
    
    # Generate wrappers
    if args.output_dir:
        generator = LibraryWrapperGenerator(args.output_dir)
        for i, kernel in enumerate(kernels):
            kernel_name = f"library_kernel_{i}"
            generator.generate_wrapper(kernel, kernel_name)
        print(f"\nGenerated {len(kernels)} wrappers in {args.output_dir}")
    
    return 0


if __name__ == '__main__':
    exit(main())

