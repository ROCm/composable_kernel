#!/usr/bin/env python3
"""
Systematic testing of grouped conv forward instances for given set of tuning parameters.
Tests compilation of various parameter combinations in parallel.
"""

import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any
import itertools


def calculate_mxdl_per_wave(block_size: int, m_per_block: int, n_per_block: int,
                           m_per_xdl: int, n_per_xdl: int, n_xdl_per_wave: int) -> int:
    """Calculate MXdlPerWave from other parameters."""
    waves = block_size // 64
    n_waves = n_per_block // (n_xdl_per_wave * n_per_xdl)
    if n_waves == 0:
        return 0
    m_waves = waves // n_waves
    if m_waves == 0:
        return 0
    m_xdl_per_wave = m_per_block // (m_waves * m_per_xdl)
    return m_xdl_per_wave


def calculate_thread_cluster_dim(block_size: int, m_per_block: int, k_per_block: int,
                                 m_per_xdl: int, ak1: int, m_xdl_per_wave: int) -> int:
    """Calculate middle dimension of thread cluster for given parameters."""
    # TODO: Fix this calculation to be more general
    # Thread cluster product must equal BlockSize (total number of threads)
    # For AK0_M_AK1, typical pattern is S<4, M/N, 1>
    # So k0=4, k1=1, we need to find M/N such that 4 * M/N * 1 = BlockSize
    return block_size // 4


def validate_blocksize(block_size: int, m_per_block: int, n_per_block: int,
                      m_per_xdl: int, n_per_xdl: int,
                      m_xdl_per_wave: int, n_xdl_per_wave: int) -> bool:
    """Validate BlockSize = MWaves × NWaves × 64."""
    m_waves = m_per_block // (m_xdl_per_wave * m_per_xdl)
    n_waves = n_per_block // (n_xdl_per_wave * n_per_xdl)
    
    if m_waves == 0 or n_waves == 0:
        return False
        
    expected_blocksize = m_waves * n_waves * 64
    return block_size == expected_blocksize


def generate_candidates() -> List[Dict[str, Any]]:
    """Generate all candidate parameter combinations with validation."""
    candidates = []
    candidate_id = 0
    
    # Parameter space from user requirements
    m_per_blocks = [128, 256, 512]
    n_per_blocks = [4, 8, 16, 32]
    k_per_blocks = [16, 32, 64]
    block_sizes = [128, 256, 512]
    
    # MFMA configurations: 16x16x4 and 4x4x4
    mfma_configs = [
        {"m_per_xdl": 16, "n_per_xdl": 16, "ak1": 4, "bk1": 4},
        {"m_per_xdl": 16, "n_per_xdl": 16, "ak1": 8, "bk1": 8},
        {"m_per_xdl": 4, "n_per_xdl": 4, "ak1": 4, "bk1": 4},
        {"m_per_xdl": 4, "n_per_xdl": 4, "ak1": 8, "bk1": 8},
    ]
    
    # Try different NXdlPerWave values
    n_xdl_per_wave_options = [1, 2, 4]
    
    for m_per_block, n_per_block, k_per_block, block_size in itertools.product(
        m_per_blocks, n_per_blocks, k_per_blocks, block_sizes
    ):
        # Only interested in M >> N for now.
        if m_per_block <= n_per_block:
            continue
            
        for mfma in mfma_configs:
            m_per_xdl = mfma["m_per_xdl"]
            n_per_xdl = mfma["n_per_xdl"]
            ak1 = mfma["ak1"]
            bk1 = mfma["bk1"]
            
            for n_xdl_per_wave in n_xdl_per_wave_options:
                # Calculate MXdlPerWave
                m_xdl_per_wave = calculate_mxdl_per_wave(
                    block_size, m_per_block, n_per_block,
                    m_per_xdl, n_per_xdl, n_xdl_per_wave
                )
                
                if m_xdl_per_wave == 0:
                    continue
                
                # Validate BlockSize constraint
                if not validate_blocksize(block_size, m_per_block, n_per_block,
                                        m_per_xdl, n_per_xdl,
                                        m_xdl_per_wave, n_xdl_per_wave):
                    continue
                
                # Calculate thread cluster dimensions
                thread_cluster_m = calculate_thread_cluster_dim(
                    block_size, m_per_block, k_per_block,
                    m_per_xdl, ak1, m_xdl_per_wave
                )

                thread_cluster_n = calculate_thread_cluster_dim(
                    block_size, n_per_block, k_per_block,
                    n_per_xdl, bk1, n_xdl_per_wave
                )

                # TODO: Generalize this calculation
                thread_cluster_k = block_size // 4
                
                # Build candidate configuration
                candidate = {
                    "id": candidate_id,
                    "block_size": block_size,
                    "m_per_block": m_per_block,
                    "n_per_block": n_per_block,
                    "k_per_block": k_per_block,
                    "ak1": ak1,
                    "bk1": bk1,
                    "m_per_xdl": m_per_xdl,
                    "n_per_xdl": n_per_xdl,
                    "m_xdl_per_wave": m_xdl_per_wave,
                    "n_xdl_per_wave": n_xdl_per_wave,
                    # Thread transfer parameters (conservative defaults based on existing instances)
                    "a_block_transfer_thread_cluster": f"4, {thread_cluster_m}, 1",
                    "a_block_transfer_arrange": "1, 0, 2",
                    "a_block_transfer_src_access": "1, 0, 2",
                    "a_block_transfer_src_vector_dim": 2,
                    "a_block_transfer_src_scalar_per_vector": 8,
                    "a_block_transfer_dst_scalar_per_vector": 8,
                    "b_block_transfer_thread_cluster": f"4, {thread_cluster_n}, 1",
                    "b_block_transfer_arrange": "1, 0, 2",
                    "b_block_transfer_src_access": "1, 0, 2",
                    "b_block_transfer_src_vector_dim": 2,
                    "b_block_transfer_src_scalar_per_vector": 8,
                    "b_block_transfer_dst_scalar_per_vector": 8,
                    "c_shuffle_m_xdl_per_wave_per_shuffle": 1,
                    "c_shuffle_n_xdl_per_wave_per_shuffle": 1,
                    "cde_block_transfer_cluster": f"1, {thread_cluster_k}, 1, 4",
                    "cde_block_transfer_scalar_per_vector": 4,
                }
                
                candidates.append(candidate)
                candidate_id += 1
    
    return candidates


def compile_candidate(candidate: Dict[str, Any], ck_root: str, template_dir: str) -> Dict[str, Any]:
    """Compile a single candidate instance in an isolated build directory."""
    candidate_id = candidate["id"]
    
    # Base path for temporary build
    base_dir = Path(__file__).parent / "results"
    base_dir.mkdir(exist_ok=True)

    # Create persistent build directory (won't be cleaned up)
    build_dir = base_dir / f"build_{candidate_id}"
    build_dir.mkdir(exist_ok=True)

    # Create temporary build directory with ignore_cleanup_errors for parallel safety
    # with tempfile.TemporaryDirectory(prefix=f"build_{candidate_id}_", 
    #                                 dir=base_dir,
    #                                 ignore_cleanup_errors=True) as build_dir:
    try:
        # Read templates
        cmake_template = (Path(template_dir) / "CMakeLists.txt").read_text()
        cpp_template = (Path(template_dir) / "test_instance.cpp").read_text()
        
        # Substitute parameters
        cmake_content = cmake_template.replace("{{CK_ROOT_DIR_VALUE}}", ck_root)
        
        cpp_content = cpp_template
        for key, value in candidate.items():
            if key != "id":
                placeholder = "{{" + key.upper() + "}}"
                cpp_content = cpp_content.replace(placeholder, str(value))
        
        # Write files to build directory
        Path(build_dir, "CMakeLists.txt").write_text(cmake_content)
        Path(build_dir, "test_instance.cpp").write_text(cpp_content)
        
        # Run CMake
        cmake_result = subprocess.run(
            ["cmake", "-D CMAKE_PREFIX_PATH=/opt/rocm", "-D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc", 
                "-D CMAKE_BUILD_TYPE=Release ", '-D GPU_TARGETS="gfx950"', "."],
            cwd=build_dir,
            capture_output=True,
            text=True,
            timeout=600 # Timeout - 10min
        )
        
        if cmake_result.returncode != 0:
            return {
                "id": candidate_id,
                "status": "cmake_failed",
                "params": candidate,
                "error": cmake_result.stderr[:500]
            }
        
        # Run make
        make_result = subprocess.run(
            ["make", "-j1"],
            cwd=build_dir,
            capture_output=True,
            text=True,
            timeout=600 # Timeout -10min
        )
        
        if make_result.returncode != 0:
            # Extract first error message
            error_lines = make_result.stderr.split('\n')
            error_msg = '\n'.join([l for l in error_lines if 'error:' in l][:3])
            
            return {
                "id": candidate_id,
                "status": "compile_failed",
                "params": candidate,
                "error": error_msg[:500]
            }
        else:
            # Run the test executable to get the type string
            exec_result = subprocess.run(
                ["./instance_test"],
                cwd=build_dir,
                capture_output=True,
                text=True,
                timeout=300 # Timeout - 5min
            )

            # The instance test outputs the instance type string on success
            return {
                "id": candidate_id,
                "status": "success",
                "params": candidate,
                "type_string": exec_result.stdout.strip()
            }
        
    except subprocess.TimeoutExpired:
        return {
            "id": candidate_id,
            "status": "timeout",
            "params": candidate
        }
    except Exception as e:
        return {
            "id": candidate_id,
            "status": "exception",
            "params": candidate,
            "error": str(e)
        }


def main():
    """Main testing function."""
    script_dir = Path(__file__).parent
    ck_root = str(script_dir.parent.parent.absolute())
    template_dir = str(script_dir / "template")
    
    print("Generating candidate configurations...")
    candidates = generate_candidates()
    
    print(f"Generated {len(candidates)} candidates after initial filtering")
    
    # Save candidates to JSON
    candidates_file = script_dir / "candidates.json"
    with open(candidates_file, 'w') as f:
        json.dump(candidates, f, indent=2)
    print(f"Saved candidates to {candidates_file}")
    
    # Test compilation in parallel
    print(f"\nTesting compilation with {os.cpu_count()} parallel jobs...")
    results = []
    results_file = script_dir / "compilation_results.json"
    
    def save_results_incremental():
        """Save current results to file."""
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] != "success"]
        
        with open(results_file, 'w') as f:
            json.dump({
                "summary": {
                    "total": len(candidates),
                    "tested": len(results),
                    "successful": len(successful),
                    "failed": len(failed)
                },
                "successful_configs": successful,
                "failed_configs": failed
            }, f, indent=2)
    
    with ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 8)) as executor:
        futures = {
            executor.submit(compile_candidate, candidate, ck_root, template_dir): candidate
            for candidate in candidates
        }
        
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            
            status_symbol = "✓" if result["status"] == "success" else "✗"
            print(f"[{i}/{len(candidates)}] {status_symbol} Candidate {result['id']}: {result['status']}")
            # Print also the type_string if successful
            if result["status"] == "success" and "type_string" in result:
                print(f"    Type String: {result['type_string']}")
            
            # Update results file after each completion
            save_results_incremental()
    
    # Print final summary
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]
    
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total candidates: {len(candidates)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    print(f"\nDetailed results saved to {results_file}")
    
    # Print some successful configs
    if successful:
        print(f"\nExample successful configurations:")
        for config in successful[:5]:
            params = config["params"]
            print(f"  ID {config['id']}: M={params['m_per_block']}, N={params['n_per_block']}, "
                  f"K={params['k_per_block']}, MFMA={params['m_per_xdl']}x{params['n_per_xdl']}x{params['ak1']}, "
                  f"Block={params['block_size']}")


if __name__ == "__main__":
    main()
