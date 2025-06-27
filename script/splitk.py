#!/usr/bin/env python3

import math
import argparse
import sys
import numpy as np

def parse_cli_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Analyze best Split-K values.")
    parser.add_argument("--gemm-m", type=int, dest="gemm_m", required=True, help="GEMM problem M-dimension.")
    parser.add_argument("--gemm-n", type=int, dest="gemm_n", required=True, help="GEMM problem N-dimension.")
    parser.add_argument("--gemm-k", type=int, dest="gemm_k", required=True, help="GEMM problem K-dimension.")
    parser.add_argument("--max-occupancy", type=int, required=True, help="Maximum number of simultaneously active workgroups on a single CU.")
    parser.add_argument("--blk-m", type=int, default=64, help="Block size for M dimension (default: 64).")
    parser.add_argument("--blk-n", type=int, default=64, help="Block size for N dimension (default: 64).")
    parser.add_argument("--blk-k", type=int, default=32, help="Block size for K dimension (default: 32).")
    parser.add_argument("--ncus", type=int, default=304, help="Number of compute units (default: 304).")
    
    args, unknown_args = parser.parse_known_args()
    
    if unknown_args:
        print(f"Unknown arguments: {unknown_args}", file=sys.stderr)
        sys.exit(1)
    
    return args

def find_optimal_split_k(grid_k, grid_size, nproc, max_split_k, print_analysis=False):
    """
    Find the optimal split-K value using the periodicity insights from StreamK.
    
    The key insight is that k-start will be periodic when grid_k/iters_per_proc 
    is a whole number. For split-K, this translates to finding split_k values
    where grid_k/split_k divides evenly into the number of processes.
    
    Parameters:
        grid_k (int): Number of tiles in the K dimension (K/BLK_K)
        grid_size (int): Total number of output tiles in the grid (M/BLK_M * N/BLK_N)
        nproc (int): Number of processes (effective compute units)
        max_split_k (int, optional): Maximum split-K to consider. Defaults to grid_k
        print_analysis (bool): Whether to print detailed analysis
        
    Returns:
        dict: Analysis results with optimal split-K values and their properties
    """
    results = []
    
    for split_k in range(1, max_split_k + 1):           
        k_tiles_per_split = (grid_k + split_k - 1) // split_k
        
        # Tail loop cost - the split_k may not divide grid_k evenly, leading to a tail loop
        tail_loop_cost = 0 if grid_k % split_k == 0 else 1.0 / (grid_k % split_k)
        
        # Check load balance - how evenly can we distribute the splits
        total_blocks = split_k * grid_size
        load_balance_score = 1.0 if total_blocks % nproc == 0 else (nproc - (total_blocks % nproc)) / nproc
        
        # Compute cache locality score based on k-tile size per split
        # Smaller k_tiles_per_split generally means better cache reuse within each split
        cache_locality_score = 1.0 / k_tiles_per_split if k_tiles_per_split > 0 else 0
        
        # Synchronization overhead increases with more splits
        sync_overhead_score = 1.0 / (1.0 + split_k)
        
        # Combined score (weighted)
        combined_score = (
            0.5 * load_balance_score + 
            0.3 * cache_locality_score +
            0.1 * tail_loop_cost +
            0.1 * sync_overhead_score
        )
        
        results.append({
            'split_k': split_k,
            'k_tiles_per_split': k_tiles_per_split,
            'load_balance_score': load_balance_score,
            'cache_locality_score': cache_locality_score,
            'sync_overhead_score': sync_overhead_score,
            'tail_loop_cost': tail_loop_cost,
            'combined_score': combined_score,
            'is_perfect_division': grid_k % split_k == 0
        })
    
    # Sort by combined score
    results.sort(key=lambda x: x['combined_score'], reverse=True)
    
    if print_analysis:
        print(f"\nSplit-K Analysis for grid_k={grid_k}, nproc={nproc}")
        print("=" * 80)
        print(f"{'Split-K':<8} {'Balance':<8} {'Cache':<8} {'Sync':<8} {'Tail':<8} {'Score':<8}")
        print("-" * 80)
        
        for result in results[:10]:  # Show top 10
            print(f"{result['split_k']:<8} "
                  f"{result['load_balance_score']:<8.3f} {result['cache_locality_score']:<8.3f}"
                  f"{result['sync_overhead_score']:<8.3f} {result['tail_loop_cost']:<8.3f} {result['combined_score']:<8.3f}")
    
    return {
        'optimal_split_k': results[0]['split_k'] if results else 1,
        'all_results': results,
        'top_candidates': results[:5] if len(results) >= 5 else results
    }


# Add a convenience function to find optimal split-K with both theoretical and empirical analysis
def find_best_split_k_comprehensive(
    M, N, K, BLK_M, BLK_N, BLK_K, 
    nproc, 
    max_split_k=None, print_analysis=True
):
    """
    Comprehensive split-K optimization using both theoretical analysis and cache simulation.
    
    Returns:
        dict: Complete analysis with recommendations
    """
    grid_m = math.ceil(M / BLK_M)
    grid_n = math.ceil(N / BLK_N) 
    grid_k = math.ceil(K / BLK_K)
    
    if max_split_k is None:
        max_split_k = grid_k

    grid_size = grid_m * grid_n
    
    theoretical_results = find_optimal_split_k(
        grid_k=grid_k, 
        grid_size=grid_size,
        nproc=nproc, 
        max_split_k=max_split_k,
        print_analysis=print_analysis
    )
    
    return {
        'theoretical_optimal': theoretical_results['optimal_split_k'],
        'theoretical_analysis': theoretical_results,
        'recommendation': theoretical_results['optimal_split_k'],
        'grid_dimensions': (grid_m, grid_n, grid_k)
    }

def main():
    args = parse_cli_args()
    
    nproc = args.ncus * args.max_occupancy
    results = find_best_split_k_comprehensive(
        M=args.gemm_m, N=args.gemm_n, K=args.gemm_k,
        BLK_M=args.blk_m, BLK_N=args.blk_n, BLK_K=args.blk_k,
        nproc=nproc, print_analysis=True
    )
    
    print("\nComprehensive Split-K Analysis Results:")
    print("=" * 60)
    print(f"Theoretical Optimal Split-K value: {results['theoretical_optimal']}")
    print(f"Recommended Split-K value: {results['recommendation']}")
    print(f"Grid Dimensions (M, N, K): {results['grid_dimensions']}")

if __name__ == "__main__":
    main()
