"""
Advanced profiling for CK Tile Dispatcher
"""

import time
import json
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import numpy as np


# ============================================================================
# Profile Data Structures
# ============================================================================

@dataclass
class KernelProfile:
    """Profile data for a single kernel execution"""
    kernel_name: str
    problem_size: tuple  # (M, N, K)
    execution_time_ms: float
    gflops: float
    bandwidth_gb_s: float
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ProfileReport:
    """Aggregated profile report"""
    total_calls: int = 0
    total_time_ms: float = 0.0
    kernel_stats: Dict[str, Dict] = field(default_factory=dict)
    problem_size_stats: Dict[tuple, Dict] = field(default_factory=dict)
    timeline: List[KernelProfile] = field(default_factory=list)
    
    def add_profile(self, profile: KernelProfile):
        """Add a profile to the report"""
        self.total_calls += 1
        self.total_time_ms += profile.execution_time_ms
        self.timeline.append(profile)
        
        # Update kernel stats
        if profile.kernel_name not in self.kernel_stats:
            self.kernel_stats[profile.kernel_name] = {
                "count": 0,
                "total_time_ms": 0.0,
                "avg_time_ms": 0.0,
                "min_time_ms": float('inf'),
                "max_time_ms": 0.0,
                "avg_gflops": 0.0,
            }
        
        stats = self.kernel_stats[profile.kernel_name]
        stats["count"] += 1
        stats["total_time_ms"] += profile.execution_time_ms
        stats["avg_time_ms"] = stats["total_time_ms"] / stats["count"]
        stats["min_time_ms"] = min(stats["min_time_ms"], profile.execution_time_ms)
        stats["max_time_ms"] = max(stats["max_time_ms"], profile.execution_time_ms)
        stats["avg_gflops"] = (stats.get("avg_gflops", 0.0) * (stats["count"] - 1) + 
                               profile.gflops) / stats["count"]
        
        # Update problem size stats
        if profile.problem_size not in self.problem_size_stats:
            self.problem_size_stats[profile.problem_size] = {
                "count": 0,
                "avg_time_ms": 0.0,
                "avg_gflops": 0.0,
            }
        
        ps_stats = self.problem_size_stats[profile.problem_size]
        ps_stats["count"] += 1
        ps_stats["avg_time_ms"] = (ps_stats["avg_time_ms"] * (ps_stats["count"] - 1) + 
                                    profile.execution_time_ms) / ps_stats["count"]
        ps_stats["avg_gflops"] = (ps_stats["avg_gflops"] * (ps_stats["count"] - 1) + 
                                   profile.gflops) / ps_stats["count"]
    
    def get_summary(self) -> str:
        """Get text summary of profile"""
        lines = []
        lines.append("=" * 80)
        lines.append("CK Tile Dispatcher Profile Report")
        lines.append("=" * 80)
        lines.append(f"Total calls: {self.total_calls}")
        lines.append(f"Total time: {self.total_time_ms:.2f} ms")
        lines.append(f"Average time per call: {self.total_time_ms / max(1, self.total_calls):.2f} ms")
        lines.append("")
        
        # Kernel statistics
        lines.append("Kernel Statistics:")
        lines.append("-" * 80)
        lines.append(f"{'Kernel':<40} {'Calls':>8} {'Avg (ms)':>12} {'GFLOPS':>12}")
        lines.append("-" * 80)
        
        for kernel_name, stats in sorted(self.kernel_stats.items(), 
                                         key=lambda x: x[1]["total_time_ms"], 
                                         reverse=True):
            lines.append(f"{kernel_name:<40} {stats['count']:>8} "
                        f"{stats['avg_time_ms']:>12.3f} {stats['avg_gflops']:>12.2f}")
        
        lines.append("")
        
        # Problem size statistics
        lines.append("Problem Size Statistics:")
        lines.append("-" * 80)
        lines.append(f"{'Size (MxNxK)':<30} {'Calls':>8} {'Avg (ms)':>12} {'GFLOPS':>12}")
        lines.append("-" * 80)
        
        for size, stats in sorted(self.problem_size_stats.items(), 
                                  key=lambda x: x[1]["count"], 
                                  reverse=True):
            size_str = f"{size[0]}x{size[1]}x{size[2]}"
            lines.append(f"{size_str:<30} {stats['count']:>8} "
                        f"{stats['avg_time_ms']:>12.3f} {stats['avg_gflops']:>12.2f}")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "total_calls": self.total_calls,
            "total_time_ms": self.total_time_ms,
            "kernel_stats": self.kernel_stats,
            "problem_size_stats": {str(k): v for k, v in self.problem_size_stats.items()},
            "timeline": [p.to_dict() for p in self.timeline],
        }
    
    def save(self, filename: str):
        """Save report to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✓ Profile report saved to {filename}")


# ============================================================================
# Profiler Class
# ============================================================================

class Profiler:
    """
    Advanced profiler for CK Tile Dispatcher
    
    Example:
        >>> profiler = Profiler()
        >>> with profiler:
        ...     result = dispatcher.gemm(A, B)
        >>> print(profiler.report.get_summary())
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize profiler
        
        Args:
            enabled: Whether profiling is enabled
        """
        self.enabled = enabled
        self.report = ProfileReport()
        self._start_time = None
    
    def start(self):
        """Start profiling"""
        if self.enabled:
            self._start_time = time.perf_counter()
    
    def stop(self):
        """Stop profiling"""
        if self.enabled and self._start_time is not None:
            elapsed = (time.perf_counter() - self._start_time) * 1000
            self._start_time = None
            return elapsed
        return 0.0
    
    def record(self, kernel_name: str, problem_size: tuple, 
               execution_time_ms: float, gflops: float, bandwidth_gb_s: float):
        """
        Record a kernel execution
        
        Args:
            kernel_name: Name of kernel
            problem_size: (M, N, K)
            execution_time_ms: Execution time in ms
            gflops: Performance in GFLOPS
            bandwidth_gb_s: Bandwidth in GB/s
        """
        if self.enabled:
            profile = KernelProfile(
                kernel_name=kernel_name,
                problem_size=problem_size,
                execution_time_ms=execution_time_ms,
                gflops=gflops,
                bandwidth_gb_s=bandwidth_gb_s
            )
            self.report.add_profile(profile)
    
    def reset(self):
        """Reset profiler"""
        self.report = ProfileReport()
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
        return False
    
    def print_summary(self):
        """Print profile summary"""
        print(self.report.get_summary())
    
    def save(self, filename: str):
        """Save profile to file"""
        self.report.save(filename)


# ============================================================================
# Decorator for Profiling
# ============================================================================

def profile(func: Callable) -> Callable:
    """
    Decorator to profile a function
    
    Example:
        >>> @profile
        ... def my_gemm(A, B):
        ...     return dispatcher.gemm(A, B)
    """
    def wrapper(*args, **kwargs):
        profiler = Profiler()
        profiler.start()
        result = func(*args, **kwargs)
        elapsed = profiler.stop()
        print(f"{func.__name__} took {elapsed:.3f} ms")
        return result
    return wrapper


# ============================================================================
# Comparative Profiling
# ============================================================================

class ComparativeProfiler:
    """
    Compare performance of different implementations
    
    Example:
        >>> cp = ComparativeProfiler()
        >>> cp.add_implementation("ck_tile", lambda: ck_gemm(A, B))
        >>> cp.add_implementation("pytorch", lambda: torch.matmul(A, B))
        >>> results = cp.run(num_iterations=100)
        >>> cp.print_comparison()
    """
    
    def __init__(self):
        self.implementations = {}
        self.results = {}
    
    def add_implementation(self, name: str, func: Callable):
        """Add an implementation to compare"""
        self.implementations[name] = func
    
    def run(self, num_warmup: int = 10, num_iterations: int = 100) -> Dict:
        """
        Run all implementations and collect results
        
        Args:
            num_warmup: Number of warmup iterations
            num_iterations: Number of benchmark iterations
        
        Returns:
            Dictionary with results for each implementation
        """
        self.results = {}
        
        for name, func in self.implementations.items():
            print(f"Benchmarking {name}...", end=" ")
            
            # Warmup
            for _ in range(num_warmup):
                func()
            
            # Benchmark
            times = []
            for _ in range(num_iterations):
                start = time.perf_counter()
                func()
                end = time.perf_counter()
                times.append((end - start) * 1000)
            
            # Statistics
            self.results[name] = {
                "mean_ms": np.mean(times),
                "std_ms": np.std(times),
                "min_ms": np.min(times),
                "max_ms": np.max(times),
                "median_ms": np.median(times),
            }
            
            print(f"✓ {self.results[name]['mean_ms']:.3f} ms")
        
        return self.results
    
    def print_comparison(self):
        """Print comparison table"""
        if not self.results:
            print("No results available. Run benchmark first.")
            return
        
        print("\n" + "=" * 80)
        print("Performance Comparison")
        print("=" * 80)
        print(f"{'Implementation':<20} {'Mean (ms)':>12} {'Std (ms)':>12} {'Speedup':>12}")
        print("-" * 80)
        
        # Find baseline (slowest)
        baseline_time = max(r["mean_ms"] for r in self.results.values())
        
        for name, result in sorted(self.results.items(), 
                                   key=lambda x: x[1]["mean_ms"]):
            speedup = baseline_time / result["mean_ms"]
            print(f"{name:<20} {result['mean_ms']:>12.3f} {result['std_ms']:>12.3f} "
                  f"{speedup:>12.2f}x")
        
        print("=" * 80)
    
    def plot_comparison(self, output_file: Optional[str] = None):
        """Plot comparison"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available")
            return
        
        if not self.results:
            print("No results available")
            return
        
        names = list(self.results.keys())
        means = [self.results[n]["mean_ms"] for n in names]
        stds = [self.results[n]["std_ms"] for n in names]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(names, means, yerr=stds, capsize=5)
        ax.set_ylabel("Execution Time (ms)")
        ax.set_title("Performance Comparison")
        ax.grid(True, alpha=0.3)
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {output_file}")
        else:
            plt.show()


# ============================================================================
# Timeline Visualization
# ============================================================================

def visualize_timeline(report: ProfileReport, output_file: Optional[str] = None):
    """
    Visualize execution timeline
    
    Args:
        report: ProfileReport
        output_file: Optional file to save plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return
    
    if not report.timeline:
        print("No timeline data available")
        return
    
    # Extract data
    timestamps = [p.timestamp - report.timeline[0].timestamp for p in report.timeline]
    exec_times = [p.execution_time_ms for p in report.timeline]
    kernel_names = [p.kernel_name for p in report.timeline]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Timeline
    ax1.scatter(timestamps, exec_times, alpha=0.6)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Execution Time (ms)")
    ax1.set_title("Execution Timeline")
    ax1.grid(True, alpha=0.3)
    
    # Histogram
    ax2.hist(exec_times, bins=50, alpha=0.7)
    ax2.set_xlabel("Execution Time (ms)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Execution Time Distribution")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Timeline plot saved to {output_file}")
    else:
        plt.show()

