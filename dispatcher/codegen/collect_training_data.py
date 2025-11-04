#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Collect Training Data from Tile Engine

Run tile_engine benchmarks and collect performance data for ML training.
Supports:
- Automatic problem size generation
- Systematic configuration sweeps
- Parallel benchmark execution
- Data validation and cleaning
- Export to JSON/CSV for ML training
"""

import json
import subprocess
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

log = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark data collection"""
    # Problem sizes to benchmark
    problem_sizes: List[Tuple[int, int, int]]
    
    # Tile configurations to test
    tile_configs: List[Dict[str, int]]
    
    # Kernel traits to test
    pipelines: List[str] = None
    epilogues: List[str] = None
    schedulers: List[str] = None
    
    # Benchmark parameters
    num_warmup: int = 5
    num_iterations: int = 20
    timeout_seconds: int = 60
    
    # Parallel execution
    max_workers: int = 4
    
    # Output
    output_dir: Path = Path("./training_data")
    
    def __post_init__(self):
        if self.pipelines is None:
            self.pipelines = ["compv3", "compv4", "mem"]
        if self.epilogues is None:
            self.epilogues = ["cshuffle", "default"]
        if self.schedulers is None:
            self.schedulers = ["intrawave"]


# ============================================================================
# Problem Size Generator
# ============================================================================

class ProblemSizeGenerator:
    """Generate diverse problem sizes for training"""
    
    @staticmethod
    def generate_power_of_2_sizes(
        min_size: int = 64,
        max_size: int = 4096,
        square_only: bool = False
    ) -> List[Tuple[int, int, int]]:
        """Generate power-of-2 problem sizes"""
        sizes = []
        size = min_size
        
        while size <= max_size:
            if square_only:
                sizes.append((size, size, size))
            else:
                # Square
                sizes.append((size, size, size))
                # Rectangular
                if size * 2 <= max_size:
                    sizes.append((size, size * 2, size))
                    sizes.append((size * 2, size, size))
            
            size *= 2
        
        return sizes
    
    @staticmethod
    def generate_common_ml_sizes() -> List[Tuple[int, int, int]]:
        """Generate common ML workload sizes"""
        return [
            # Small (mobile/edge)
            (64, 64, 64),
            (128, 128, 128),
            (256, 256, 256),
            
            # Medium (inference)
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            
            # Large (training)
            (4096, 4096, 4096),
            (8192, 8192, 8192),
            
            # Rectangular (common in transformers)
            (1024, 4096, 1024),
            (4096, 1024, 1024),
            (2048, 8192, 2048),
            (8192, 2048, 2048),
            
            # Batch sizes
            (128, 768, 768),   # BERT-base
            (128, 1024, 1024), # BERT-large
            (256, 2048, 2048), # GPT-2
            (512, 4096, 4096), # GPT-3
        ]
    
    @staticmethod
    def generate_random_sizes(
        count: int = 100,
        min_dim: int = 64,
        max_dim: int = 4096
    ) -> List[Tuple[int, int, int]]:
        """Generate random problem sizes"""
        import random
        sizes = []
        
        for _ in range(count):
            # Bias towards multiples of 64 for better performance
            M = random.randrange(min_dim, max_dim + 1, 64)
            N = random.randrange(min_dim, max_dim + 1, 64)
            K = random.randrange(min_dim, max_dim + 1, 64)
            sizes.append((M, N, K))
        
        return sizes


# ============================================================================
# Tile Configuration Generator
# ============================================================================

class TileConfigGenerator:
    """Generate tile configurations to test"""
    
    @staticmethod
    def generate_standard_configs() -> List[Dict[str, int]]:
        """Generate standard tile configurations"""
        configs = []
        
        # Common tile sizes
        tile_sizes = [
            (128, 128, 32),
            (256, 256, 32),
            (128, 256, 32),
            (256, 128, 32),
            (64, 64, 32),
            (256, 256, 64),
        ]
        
        # Common warp configurations
        warp_configs = [
            (2, 2, 1),
            (4, 4, 1),
            (2, 4, 1),
            (4, 2, 1),
        ]
        
        # Common warp tile sizes
        warp_tile_sizes = [
            (32, 32, 16),
            (16, 16, 16),
            (32, 16, 16),
            (16, 32, 16),
        ]
        
        for (tm, tn, tk), (wm, wn, wk), (wtm, wtn, wtk) in itertools.product(
            tile_sizes, warp_configs, warp_tile_sizes
        ):
            # Validate configuration
            if tm % (wm * wtm) == 0 and tn % (wn * wtn) == 0 and tk % (wk * wtk) == 0:
                configs.append({
                    'tile_m': tm,
                    'tile_n': tn,
                    'tile_k': tk,
                    'warp_m': wm,
                    'warp_n': wn,
                    'warp_k': wk,
                    'warp_tile_m': wtm,
                    'warp_tile_n': wtn,
                    'warp_tile_k': wtk,
                })
        
        return configs


# ============================================================================
# Benchmark Runner
# ============================================================================

class BenchmarkRunner:
    """Run tile_engine benchmarks and collect data"""
    
    def __init__(self, tile_engine_path: Path, config: BenchmarkConfig):
        self.tile_engine_path = Path(tile_engine_path)
        self.config = config
        self.results = []
    
    def run_single_benchmark(
        self,
        problem_size: Tuple[int, int, int],
        tile_config: Dict[str, int],
        pipeline: str,
        epilogue: str,
        scheduler: str
    ) -> Optional[Dict]:
        """
        Run a single benchmark
        
        Returns performance data or None if failed
        """
        M, N, K = problem_size
        
        log.info(f"Benchmarking: M={M}, N={N}, K={K}, "
                f"tile={tile_config['tile_m']}x{tile_config['tile_n']}x{tile_config['tile_k']}, "
                f"{pipeline}/{epilogue}/{scheduler}")
        
        # Build command (placeholder - adjust for actual tile_engine interface)
        cmd = [
            str(self.tile_engine_path / "benchmark_gemm"),
            "--M", str(M),
            "--N", str(N),
            "--K", str(K),
            "--tile-m", str(tile_config['tile_m']),
            "--tile-n", str(tile_config['tile_n']),
            "--tile-k", str(tile_config['tile_k']),
            "--warp-m", str(tile_config['warp_m']),
            "--warp-n", str(tile_config['warp_n']),
            "--warp-k", str(tile_config['warp_k']),
            "--warp-tile-m", str(tile_config['warp_tile_m']),
            "--warp-tile-n", str(tile_config['warp_tile_n']),
            "--warp-tile-k", str(tile_config['warp_tile_k']),
            "--pipeline", pipeline,
            "--epilogue", epilogue,
            "--scheduler", scheduler,
            "--warmup", str(self.config.num_warmup),
            "--iterations", str(self.config.num_iterations),
            "--json",  # Output JSON
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds
            )
            
            if result.returncode != 0:
                log.warning(f"Benchmark failed: {result.stderr}")
                return None
            
            # Parse JSON output
            perf_data = json.loads(result.stdout)
            
            # Combine with configuration
            benchmark_result = {
                'problem': {'M': M, 'N': N, 'K': K, 'batch_size': 1},
                'config': {
                    **tile_config,
                    'pipeline': pipeline,
                    'epilogue': epilogue,
                    'scheduler': scheduler,
                    'persistent': False,
                    'block_size': 256,
                    'dtype_a': 'fp16',
                    'dtype_b': 'fp16',
                    'dtype_c': 'fp16',
                    'gpu_arch': 'gfx942',
                    'num_cus': 304,
                },
                'performance': perf_data
            }
            
            return benchmark_result
            
        except subprocess.TimeoutExpired:
            log.warning(f"Benchmark timed out")
            return None
        except Exception as e:
            log.error(f"Benchmark error: {e}")
            return None
    
    def run_all_benchmarks(self) -> List[Dict]:
        """Run all benchmark combinations"""
        # Generate all combinations
        tasks = []
        for problem_size in self.config.problem_sizes:
            for tile_config in self.config.tile_configs:
                for pipeline, epilogue, scheduler in itertools.product(
                    self.config.pipelines,
                    self.config.epilogues,
                    self.config.schedulers
                ):
                    tasks.append((problem_size, tile_config, pipeline, epilogue, scheduler))
        
        log.info(f"Total benchmarks to run: {len(tasks)}")
        
        # Run benchmarks (parallel or sequential)
        if self.config.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = [
                    executor.submit(self.run_single_benchmark, *task)
                    for task in tasks
                ]
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        self.results.append(result)
        else:
            for task in tasks:
                result = self.run_single_benchmark(*task)
                if result:
                    self.results.append(result)
        
        log.info(f"Completed {len(self.results)} successful benchmarks")
        return self.results
    
    def export_results(self, output_path: Path):
        """Export results to JSON"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'metadata': {
                'num_benchmarks': len(self.results),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'config': {
                    'num_warmup': self.config.num_warmup,
                    'num_iterations': self.config.num_iterations,
                }
            },
            'benchmarks': self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        log.info(f"Results exported to {output_path}")
    
    def export_to_csv(self, output_path: Path):
        """Export results to CSV (requires pandas)"""
        try:
            import pandas as pd
        except ImportError:
            log.error("Pandas required for CSV export")
            return
        
        # Flatten results
        rows = []
        for result in self.results:
            row = {}
            row.update(result['problem'])
            row.update(result['config'])
            row.update(result['performance'])
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        log.info(f"Results exported to CSV: {output_path}")


# ============================================================================
# Data Validator
# ============================================================================

class DataValidator:
    """Validate and clean collected data"""
    
    @staticmethod
    def validate_benchmark_result(result: Dict) -> Tuple[bool, str]:
        """Validate a single benchmark result"""
        # Check required fields
        required_fields = ['problem', 'config', 'performance']
        for field in required_fields:
            if field not in result:
                return False, f"Missing field: {field}"
        
        # Check performance metrics
        perf = result['performance']
        if 'execution_time_ms' not in perf or perf['execution_time_ms'] <= 0:
            return False, "Invalid execution time"
        
        if 'gflops' in perf and perf['gflops'] < 0:
            return False, "Negative GFLOPS"
        
        # Check for outliers (execution time > 1 second is suspicious)
        if perf['execution_time_ms'] > 1000:
            return False, "Execution time too high (possible error)"
        
        return True, "Valid"
    
    @staticmethod
    def clean_data(results: List[Dict]) -> List[Dict]:
        """Clean and validate data"""
        cleaned = []
        
        for result in results:
            valid, msg = DataValidator.validate_benchmark_result(result)
            if valid:
                cleaned.append(result)
            else:
                log.warning(f"Removing invalid result: {msg}")
        
        log.info(f"Cleaned data: {len(cleaned)}/{len(results)} valid results")
        return cleaned


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect training data from tile_engine')
    parser.add_argument('--tile-engine-path', type=Path, required=True,
                       help='Path to tile_engine binaries')
    parser.add_argument('--output-dir', type=Path, default=Path('./training_data'),
                       help='Output directory')
    parser.add_argument('--problem-sizes', type=str, default='ml',
                       choices=['power2', 'ml', 'random'],
                       help='Problem size generation strategy')
    parser.add_argument('--num-configs', type=int, default=20,
                       help='Number of tile configurations to test')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum parallel workers')
    parser.add_argument('--warmup', type=int, default=5,
                       help='Number of warmup iterations')
    parser.add_argument('--iterations', type=int, default=20,
                       help='Number of benchmark iterations')
    parser.add_argument('--export-csv', action='store_true',
                       help='Also export to CSV')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Generate problem sizes
    if args.problem_sizes == 'power2':
        problem_sizes = ProblemSizeGenerator.generate_power_of_2_sizes()
    elif args.problem_sizes == 'ml':
        problem_sizes = ProblemSizeGenerator.generate_common_ml_sizes()
    else:  # random
        problem_sizes = ProblemSizeGenerator.generate_random_sizes(count=50)
    
    log.info(f"Generated {len(problem_sizes)} problem sizes")
    
    # Generate tile configurations
    all_configs = TileConfigGenerator.generate_standard_configs()
    # Sample if too many
    if len(all_configs) > args.num_configs:
        import random
        tile_configs = random.sample(all_configs, args.num_configs)
    else:
        tile_configs = all_configs
    
    log.info(f"Testing {len(tile_configs)} tile configurations")
    
    # Create benchmark config
    config = BenchmarkConfig(
        problem_sizes=problem_sizes,
        tile_configs=tile_configs,
        num_warmup=args.warmup,
        num_iterations=args.iterations,
        max_workers=args.max_workers,
        output_dir=args.output_dir
    )
    
    # Run benchmarks
    runner = BenchmarkRunner(args.tile_engine_path, config)
    results = runner.run_all_benchmarks()
    
    # Clean data
    cleaned_results = DataValidator.clean_data(results)
    runner.results = cleaned_results
    
    # Export
    output_json = args.output_dir / "training_data.json"
    runner.export_results(output_json)
    
    if args.export_csv:
        output_csv = args.output_dir / "training_data.csv"
        runner.export_to_csv(output_csv)
    
    print(f"\n✅ Data collection complete!")
    print(f"   Total benchmarks: {len(cleaned_results)}")
    print(f"   Output: {output_json}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

