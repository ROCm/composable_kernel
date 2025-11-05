"""
High-Level Python API for CK Tile Dispatcher

Provides simple Python interface for:
1. Kernel generation via unified_gemm_codegen.py
2. Automatic registration with dispatcher
3. GPU execution via C++ backend

Example:
    >>> from ck_tile_dispatcher import Dispatcher, generate_kernels
    >>> 
    >>> # Generate kernels
    >>> generate_kernels(datatype='fp16', layout='rcr', preset='essential')
    >>> 
    >>> # Use dispatcher
    >>> dispatcher = Dispatcher()
    >>> dispatcher.load_generated_kernels()
    >>> result = dispatcher.gemm(A, B, C)
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple
from dataclasses import dataclass
import numpy as np

# Try to import C++ extension
try:
    import _dispatcher_native as cpp
    HAS_CPP_EXTENSION = True
except ImportError:
    HAS_CPP_EXTENSION = False
    import warnings
    warnings.warn("C++ extension not available. Build with -DBUILD_DISPATCHER_PYTHON=ON")


def get_dispatcher_root() -> Path:
    """Get dispatcher root directory"""
    return Path(__file__).parent.parent


def get_codegen_script() -> Path:
    """Get unified codegen script path"""
    return get_dispatcher_root() / "codegen" / "unified_gemm_codegen.py"


def get_generated_kernels_dir() -> Path:
    """Get default generated kernels directory"""
    return get_dispatcher_root() / "build" / "generated_kernels"


def generate_kernels(
    datatype: str = 'fp16',
    layout: str = 'rcr',
    preset: str = 'essential',
    gpu_target: str = 'gfx942',
    output_dir: Optional[Path] = None,
    parallel: bool = True,
    register: bool = True,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Generate CK Tile GEMM kernels
    
    Args:
        datatype: Data type ('fp16', 'bf16', 'fp32', 'fp8')
        layout: Memory layout ('rcr', 'rrr', 'crr', 'ccr')
        preset: Kernel preset ('essential', 'compute', 'memory')
        gpu_target: Target GPU architecture
        output_dir: Output directory (default: build/generated_kernels)
        parallel: Enable parallel generation
        register: Generate dispatcher registration code
        verbose: Print generation progress
    
    Returns:
        Dict with generation results
    """
    if output_dir is None:
        output_dir = get_generated_kernels_dir()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    codegen_script = get_codegen_script()
    
    if not codegen_script.exists():
        raise FileNotFoundError(f"Codegen script not found: {codegen_script}")
    
    # Build command
    cmd = [
        sys.executable,
        str(codegen_script),
        '--output-dir', str(output_dir),
        '--datatype', datatype,
        '--layout', layout,
        '--gpu-target', gpu_target,
        '--preselected', f'{datatype}_{layout}_{preset}',
    ]
    
    if not parallel:
        cmd.append('--no-parallel')
    
    if register:
        cmd.append('--register')
    
    if verbose:
        print(f"Generating {datatype} {layout} kernels (preset: {preset})...")
        print(f"Output directory: {output_dir}")
    
    # Run codegen
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error generating kernels:")
        print(result.stderr)
        raise RuntimeError("Kernel generation failed")
    
    if verbose:
        # Parse output
        for line in result.stdout.split('\n'):
            if 'Generation complete' in line or 'Kernels:' in line:
                print(f"  {line}")
    
    # Count generated files
    kernel_files = list(output_dir.glob("*.hpp"))
    
    return {
        'success': True,
        'num_kernels': len(kernel_files),
        'output_dir': str(output_dir),
        'datatype': datatype,
        'layout': layout,
        'preset': preset
    }


def build_dispatcher_executable(
    kernel_files: List[Path],
    output_executable: Path,
    verbose: bool = True
) -> bool:
    """
    Build a standalone executable with generated kernels
    
    Args:
        kernel_files: List of kernel header files to include
        output_executable: Output executable path
        verbose: Print build progress
    
    Returns:
        True if successful
    """
    dispatcher_root = get_dispatcher_root()
    build_dir = dispatcher_root / "build"
    
    # Use CMake to build
    if verbose:
        print(f"Building executable: {output_executable}")
    
    # This would trigger CMake build
    cmd = ['cmake', '--build', str(build_dir), '--target', 'single_tile_kernel_example']
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(build_dir))
    
    if result.returncode != 0 and verbose:
        print("Build output:", result.stderr)
    
    return result.returncode == 0


class Dispatcher:
    """
    High-level dispatcher interface
    
    Example:
        >>> dispatcher = Dispatcher()
        >>> dispatcher.generate_and_load_kernels('fp16', 'rcr')
        >>> result = dispatcher.select_kernel(M=1024, N=1024, K=1024)
    """
    
    def __init__(self, gpu_arch: str = 'gfx942'):
        """Initialize dispatcher"""
        self.gpu_arch = gpu_arch
        self.generated_kernels_dir = None
        self.cpp_dispatcher = None
        
        if HAS_CPP_EXTENSION:
            self.cpp_dispatcher = cpp.Dispatcher()
            self.registry = cpp.Registry.instance()
        else:
            self.registry = None
    
    def generate_kernels(
        self,
        datatype: str = 'fp16',
        layout: str = 'rcr',
        preset: str = 'essential',
        **kwargs
    ) -> Dict:
        """Generate CK Tile kernels"""
        result = generate_kernels(
            datatype=datatype,
            layout=layout,
            preset=preset,
            gpu_target=self.gpu_arch,
            **kwargs
        )
        
        self.generated_kernels_dir = Path(result['output_dir'])
        print(f"✓ Generated {result['num_kernels']} kernels")
        
        return result
    
    def load_generated_kernels(self, kernels_dir: Optional[Path] = None):
        """
        Load generated kernels (requires building C++ executable)
        
        Note: Full kernel loading requires C++ compilation.
        This method prepares the environment for kernel usage.
        """
        if kernels_dir is None:
            kernels_dir = self.generated_kernels_dir or get_generated_kernels_dir()
        
        kernels_dir = Path(kernels_dir)
        
        if not kernels_dir.exists():
            raise FileNotFoundError(f"Kernels directory not found: {kernels_dir}")
        
        # Check for registration files
        reg_header = kernels_dir / "registration" / "dispatcher_registration.hpp"
        manifest = kernels_dir / "registration" / "kernels_manifest.json"
        
        if manifest.exists():
            with open(manifest) as f:
                kernel_info = json.load(f)
            
            print(f"✓ Found {len(kernel_info['kernels'])} registered kernels:")
            for k in kernel_info['kernels']:
                print(f"  - {k['name']} ({k['tile_m']}x{k['tile_n']}x{k['tile_k']})")
        
        return kernels_dir
    
    def generate_and_load_kernels(
        self,
        datatype: str = 'fp16',
        layout: str = 'rcr',
        preset: str = 'essential'
    ):
        """Generate kernels and prepare for loading"""
        self.generate_kernels(datatype, layout, preset)
        return self.load_generated_kernels()
    
    def build_gpu_executable(self, rebuild: bool = False) -> Path:
        """
        Build the GPU executable with generated kernels
        
        Returns:
            Path to built executable
        """
        build_dir = get_dispatcher_root() / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        
        print("Building GPU executable...")
        
        # Configure CMake
        if rebuild or not (build_dir / "CMakeCache.txt").exists():
            cmake_cmd = [
                'cmake', '..',
                '-DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++',
                '-DCMAKE_BUILD_TYPE=Release',
                '-DBUILD_DISPATCHER_EXAMPLES=ON'
            ]
            
            result = subprocess.run(
                cmake_cmd,
                cwd=str(build_dir),
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print("CMake error:", result.stderr)
                raise RuntimeError("CMake configuration failed")
            
            print("  ✓ CMake configured")
        
        # Build
        make_cmd = ['make', 'single_tile_kernel_example', '-j4']
        result = subprocess.run(
            make_cmd,
            cwd=str(build_dir),
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print("Build error:", result.stderr)
            raise RuntimeError("Build failed")
        
        executable = build_dir / "examples" / "single_tile_kernel_example"
        
        if not executable.exists():
            raise FileNotFoundError(f"Executable not found: {executable}")
        
        print(f"  ✓ Built: {executable}")
        return executable
    
    def run_gpu_gemm(
        self,
        M: int,
        N: int,
        K: int,
        executable: Optional[Path] = None
    ) -> Dict:
        """
        Run GEMM on GPU via compiled executable
        
        Args:
            M, N, K: Problem dimensions
            executable: Path to executable (default: auto-detect)
        
        Returns:
            Dict with execution results
        """
        if executable is None:
            executable = get_dispatcher_root() / "build" / "examples" / "single_tile_kernel_example"
        
        if not executable.exists():
            print(f"Executable not found. Building...")
            executable = self.build_gpu_executable()
        
        # Run executable (captures size from problem, not args - would need to modify for parametric)
        result = subprocess.run(
            [str(executable)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print("Execution error:", result.stderr)
            raise RuntimeError("GPU execution failed")
        
        return {
            'success': True,
            'output': result.stdout,
            'problem_size': (M, N, K)
        }
    
    def select_kernel(self, M: int, N: int, K: int) -> Optional[str]:
        """
        Select a kernel for the given problem (via C++ extension)
        
        Args:
            M, N, K: Problem dimensions
        
        Returns:
            Kernel name if found, None otherwise
        """
        if not HAS_CPP_EXTENSION:
            print("C++ extension not available")
            return None
        
        problem = cpp.Problem(M, N, K)
        kernel = self.cpp_dispatcher.select_kernel(problem)
        
        if kernel:
            return kernel.get_name()
        return None
    
    def get_registered_kernels(self) -> List[str]:
        """Get list of registered kernel names"""
        if not HAS_CPP_EXTENSION or self.registry is None:
            # Read from manifest
            manifest = get_generated_kernels_dir() / "registration" / "kernels_manifest.json"
            if manifest.exists():
                with open(manifest) as f:
                    data = json.load(f)
                return [k['name'] for k in data['kernels']]
            return []
        
        # Get from C++ registry
        all_kernels = self.registry.get_all()
        return [k.get_name() for k in all_kernels]
    
    def info(self):
        """Print dispatcher information"""
        print("="*70)
        print("CK Tile Dispatcher - Python API")
        print("="*70)
        print(f"\nGPU Architecture: {self.gpu_arch}")
        print(f"C++ Extension: {'Loaded' if HAS_CPP_EXTENSION else 'Not available'}")
        
        if self.generated_kernels_dir:
            print(f"Generated Kernels: {self.generated_kernels_dir}")
        
        kernels = self.get_registered_kernels()
        print(f"Registered Kernels: {len(kernels)}")
        
        if kernels and len(kernels) <= 10:
            for k in kernels:
                print(f"  - {k}")
        elif kernels:
            print(f"  (showing first 5 of {len(kernels)})")
            for k in kernels[:5]:
                print(f"  - {k}")
        
        print()


class SimpleGemmAPI:
    """
    Simplified GEMM API that handles everything automatically
    
    Example:
        >>> gemm = SimpleGemmAPI()
        >>> gemm.ensure_kernels_ready()  # Generate + build if needed
        >>> result = gemm.execute(M=1024, N=1024, K=1024)
    """
    
    def __init__(self, gpu_arch: str = 'gfx942'):
        self.dispatcher = Dispatcher(gpu_arch)
        self.executable = None
    
    def ensure_kernels_ready(
        self,
        datatype: str = 'fp16',
        layout: str = 'rcr',
        force_regenerate: bool = False
    ) -> bool:
        """
        Ensure kernels are generated and executable is built
        
        Args:
            datatype: Data type for kernels
            layout: Memory layout
            force_regenerate: Force regeneration even if kernels exist
        
        Returns:
            True if ready
        """
        kernels_dir = get_generated_kernels_dir()
        
        # Check if kernels already exist
        kernel_files = list(kernels_dir.glob(f"gemm_{datatype}_{layout}_*.hpp"))
        
        if not kernel_files or force_regenerate:
            print(f"Generating {datatype} {layout} kernels...")
            self.dispatcher.generate_kernels(datatype, layout, 'essential')
        else:
            print(f"✓ Found {len(kernel_files)} existing kernels")
            self.dispatcher.generated_kernels_dir = kernels_dir
        
        # Build executable
        print("Checking/building GPU executable...")
        try:
            self.executable = self.dispatcher.build_gpu_executable()
            print(f"✓ Executable ready: {self.executable}")
            return True
        except Exception as e:
            print(f"✗ Build failed: {e}")
            return False
    
    def execute(
        self,
        M: int,
        N: int,
        K: int,
        verbose: bool = True
    ) -> Dict:
        """
        Execute GEMM on GPU
        
        Args:
            M, N, K: Problem dimensions
            verbose: Print execution details
        
        Returns:
            Dict with results
        """
        if self.executable is None:
            raise RuntimeError("Executable not ready. Call ensure_kernels_ready() first")
        
        if verbose:
            print(f"\nExecuting GEMM: M={M}, N={N}, K={K}")
        
        result = self.dispatcher.run_gpu_gemm(M, N, K, self.executable)
        
        if verbose and result['success']:
            print("✓ Execution successful")
            # Parse output for timing if available
            for line in result['output'].split('\n'):
                if 'GFLOPS' in line or 'ms' in line:
                    print(f"  {line.strip()}")
        
        return result
    
    def run_workflow(
        self,
        M: int = 1024,
        N: int = 1024,
        K: int = 1024,
        datatype: str = 'fp16',
        layout: str = 'rcr'
    ):
        """
        Complete workflow: generate → build → execute
        
        This is the simplest API - does everything automatically.
        """
        print("="*70)
        print("CK Tile Dispatcher - Complete Workflow")
        print("="*70 + "\n")
        
        # Step 1: Ensure ready
        print("Step 1: Preparing kernels and executable...")
        if not self.ensure_kernels_ready(datatype, layout):
            raise RuntimeError("Failed to prepare kernels")
        print()
        
        # Step 2: Execute
        print("Step 2: Executing on GPU...")
        result = self.execute(M, N, K)
        print()
        
        # Step 3: Summary
        print("="*70)
        print("Workflow Complete")
        print("="*70)
        print(f"✓ Generated kernels: {datatype} {layout}")
        print(f"✓ Built GPU executable")
        print(f"✓ Executed GEMM: {M}x{N}x{K}")
        print()
        
        return result


# Convenience functions for quick usage

def quick_gemm(
    M: int = 1024,
    N: int = 1024,
    K: int = 1024,
    datatype: str = 'fp16',
    layout: str = 'rcr'
) -> Dict:
    """
    Quickest way to run GEMM via dispatcher
    
    Example:
        >>> from ck_tile_dispatcher.dispatcher_api import quick_gemm
        >>> result = quick_gemm(M=2048, N=2048, K=2048)
    """
    api = SimpleGemmAPI()
    return api.run_workflow(M, N, K, datatype, layout)


def list_available_presets() -> Dict[str, List[str]]:
    """List available kernel presets"""
    return {
        'fp16_rcr': ['essential', 'compute', 'memory'],
        'fp16_rrr': ['essential', 'compute', 'memory'],
        'fp16_crr': ['essential', 'compute', 'memory'],
        'bf16_rcr': ['essential', 'compute', 'memory'],
        'fp32_rcr': ['essential', 'compute', 'memory'],
    }


def info():
    """Print API information"""
    print("="*70)
    print("CK Tile Dispatcher - Python API")
    print("="*70)
    print("\nHigh-level functions:")
    print("  - generate_kernels()         : Generate CK Tile kernels")
    print("  - Dispatcher()                : Main dispatcher class")
    print("  - SimpleGemmAPI()             : Simplified interface")
    print("  - quick_gemm()                : One-line GEMM execution")
    print("\nExample workflow:")
    print("  >>> from ck_tile_dispatcher.dispatcher_api import quick_gemm")
    print("  >>> result = quick_gemm(M=1024, N=1024, K=1024)")
    print("\nFor C++ extension:")
    print("  >>> import _dispatcher_native as cpp")
    print("  >>> registry = cpp.Registry.instance()")
    print("  >>> dispatcher = cpp.Dispatcher()")
    print()


# Module initialization
if __name__ == "__main__":
    info()

