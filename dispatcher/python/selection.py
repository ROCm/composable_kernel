"""
Kernel Selection Engine for CK Tile Dispatcher

Provides heuristic-guided kernel selection strategies.
"""

from typing import List, Optional, Callable
from enum import Enum
from dataclasses import dataclass


class SelectionStrategy(Enum):
    """Kernel selection strategy"""
    FIRST_FIT = "first_fit"  # First kernel that supports the problem
    HEURISTIC = "heuristic"  # Use heuristic function
    EXPLICIT = "explicit"  # Explicit kernel ID provided


@dataclass
class SelectionResult:
    """Result of kernel selection"""
    kernel_instance: Optional['KernelInstance']
    strategy_used: SelectionStrategy
    candidates_checked: int
    selection_time_ms: float
    error_message: str = ""
    
    @property
    def success(self) -> bool:
        return self.kernel_instance is not None


class SelectionEngine:
    """
    Kernel selection engine with multiple strategies
    
    Strategies:
    1. First-Fit: Iterate through registered kernels, return first match
    2. Heuristic: Query heuristic function for ordered candidates
    3. Explicit: Use provided kernel ID
    
    Example:
        >>> engine = SelectionEngine(registry)
        >>> engine.set_heuristic(my_heuristic_fn)
        >>> result = engine.select(problem, strategy=SelectionStrategy.HEURISTIC)
    """
    
    def __init__(self, registry):
        """
        Initialize selection engine
        
        Args:
            registry: Kernel registry
        """
        self.registry = registry
        self.heuristic_fn: Optional[Callable] = None
        self.default_strategy = SelectionStrategy.FIRST_FIT
    
    def set_heuristic(self, heuristic_fn: Callable):
        """
        Set heuristic function
        
        Args:
            heuristic_fn: Function that takes a Problem and returns
                         list of kernel IDs ordered by expected performance
        
        Example:
            >>> def my_heuristic(problem):
            ...     if problem.M > 2048:
            ...         return ["large_tile_kernel", "medium_tile_kernel"]
            ...     return ["small_tile_kernel"]
            >>> 
            >>> engine.set_heuristic(my_heuristic)
        """
        self.heuristic_fn = heuristic_fn
        self.default_strategy = SelectionStrategy.HEURISTIC
    
    def clear_heuristic(self):
        """Clear heuristic function"""
        self.heuristic_fn = None
        self.default_strategy = SelectionStrategy.FIRST_FIT
    
    def select(self, problem, strategy: Optional[SelectionStrategy] = None,
               kernel_id: Optional[str] = None) -> SelectionResult:
        """
        Select kernel for problem
        
        Args:
            problem: Problem specification
            strategy: Selection strategy (uses default if None)
            kernel_id: Explicit kernel ID (for EXPLICIT strategy)
        
        Returns:
            SelectionResult
        """
        import time
        
        start = time.perf_counter()
        
        # Determine strategy
        if kernel_id is not None:
            strategy = SelectionStrategy.EXPLICIT
        elif strategy is None:
            strategy = self.default_strategy
        
        # Execute strategy
        if strategy == SelectionStrategy.EXPLICIT:
            result = self._select_explicit(problem, kernel_id)
        elif strategy == SelectionStrategy.HEURISTIC:
            result = self._select_heuristic(problem)
        else:  # FIRST_FIT
            result = self._select_first_fit(problem)
        
        # Update timing
        result.selection_time_ms = (time.perf_counter() - start) * 1000
        
        return result
    
    def _select_explicit(self, problem, kernel_id: str) -> SelectionResult:
        """Select explicit kernel by ID"""
        kernel = self.registry.lookup(kernel_id)
        
        if kernel is None:
            return SelectionResult(
                kernel_instance=None,
                strategy_used=SelectionStrategy.EXPLICIT,
                candidates_checked=1,
                selection_time_ms=0.0,
                error_message=f"Kernel not found: {kernel_id}"
            )
        
        if not kernel.supports(problem):
            return SelectionResult(
                kernel_instance=None,
                strategy_used=SelectionStrategy.EXPLICIT,
                candidates_checked=1,
                selection_time_ms=0.0,
                error_message=f"Kernel {kernel_id} does not support problem"
            )
        
        return SelectionResult(
            kernel_instance=kernel,
            strategy_used=SelectionStrategy.EXPLICIT,
            candidates_checked=1,
            selection_time_ms=0.0
        )
    
    def _select_heuristic(self, problem) -> SelectionResult:
        """Select using heuristic function"""
        if self.heuristic_fn is None:
            # Fallback to first-fit
            return self._select_first_fit(problem)
        
        # Query heuristic
        try:
            candidate_ids = self.heuristic_fn(problem)
        except Exception as e:
            return SelectionResult(
                kernel_instance=None,
                strategy_used=SelectionStrategy.HEURISTIC,
                candidates_checked=0,
                selection_time_ms=0.0,
                error_message=f"Heuristic function failed: {e}"
            )
        
        # Try candidates in order
        candidates_checked = 0
        for kernel_id in candidate_ids:
            candidates_checked += 1
            kernel = self.registry.lookup(kernel_id)
            
            if kernel is None:
                continue
            
            if kernel.supports(problem):
                return SelectionResult(
                    kernel_instance=kernel,
                    strategy_used=SelectionStrategy.HEURISTIC,
                    candidates_checked=candidates_checked,
                    selection_time_ms=0.0
                )
        
        # Heuristic failed, fallback to first-fit
        result = self._select_first_fit(problem)
        result.candidates_checked += candidates_checked
        return result
    
    def _select_first_fit(self, problem) -> SelectionResult:
        """Select first kernel that supports problem"""
        kernels = self.registry.enumerate_all()
        
        candidates_checked = 0
        for kernel in kernels:
            candidates_checked += 1
            
            if kernel.supports(problem):
                return SelectionResult(
                    kernel_instance=kernel,
                    strategy_used=SelectionStrategy.FIRST_FIT,
                    candidates_checked=candidates_checked,
                    selection_time_ms=0.0
                )
        
        return SelectionResult(
            kernel_instance=None,
            strategy_used=SelectionStrategy.FIRST_FIT,
            candidates_checked=candidates_checked,
            selection_time_ms=0.0,
            error_message=f"No kernel found for problem: {problem}"
        )
    
    def enumerate_candidates(self, problem) -> List['KernelInstance']:
        """
        Enumerate all candidate kernels for a problem
        
        Args:
            problem: Problem specification
        
        Returns:
            List of kernel instances that support the problem
        """
        return self.registry.filter_by_problem(problem)
    
    def rank_candidates(self, problem) -> List[tuple]:
        """
        Rank candidates using heuristic
        
        Args:
            problem: Problem specification
        
        Returns:
            List of (kernel_instance, rank) tuples ordered by rank
        """
        if self.heuristic_fn is None:
            # No heuristic, return all candidates with equal rank
            candidates = self.enumerate_candidates(problem)
            return [(k, 0) for k in candidates]
        
        # Get heuristic ranking
        candidate_ids = self.heuristic_fn(problem)
        
        # Build ranked list
        ranked = []
        for rank, kernel_id in enumerate(candidate_ids):
            kernel = self.registry.lookup(kernel_id)
            if kernel and kernel.supports(problem):
                ranked.append((kernel, rank))
        
        return ranked
    
    def get_stats(self) -> dict:
        """Get selection engine statistics"""
        return {
            'has_heuristic': self.heuristic_fn is not None,
            'default_strategy': self.default_strategy.value,
            'registry_size': self.registry.size(),
        }


# Heuristic function examples

def size_based_heuristic(problem) -> List[str]:
    """
    Simple size-based heuristic
    
    Recommends kernels based on problem size:
    - Small problems: small tile sizes
    - Medium problems: medium tile sizes
    - Large problems: large tile sizes
    """
    total_size = problem.M * problem.N * problem.K
    
    if total_size < 1024 ** 3:  # < 1B elements
        # Small problem - prefer small tiles
        return [
            "128x128x32_kernel",
            "256x128x32_kernel",
            "256x256x32_kernel",
        ]
    elif total_size < 8 * 1024 ** 3:  # < 8B elements
        # Medium problem - prefer medium tiles
        return [
            "256x256x32_kernel",
            "256x256x64_kernel",
            "512x256x32_kernel",
        ]
    else:
        # Large problem - prefer large tiles
        return [
            "512x512x32_kernel",
            "512x512x64_kernel",
            "1024x512x32_kernel",
        ]


def datatype_aware_heuristic(problem) -> List[str]:
    """
    Datatype-aware heuristic
    
    Recommends kernels based on data type and problem size.
    """
    # This would need access to problem data types
    # Simplified example
    if hasattr(problem, 'dtype') and problem.dtype == 'fp16':
        return [
            "fp16_256x256x32_kernel",
            "fp16_512x256x32_kernel",
        ]
    else:
        return [
            "fp32_256x256x16_kernel",
            "fp32_512x256x16_kernel",
        ]


def ml_based_heuristic(model_path: str) -> Callable:
    """
    Create ML-based heuristic from trained model
    
    Args:
        model_path: Path to trained model
    
    Returns:
        Heuristic function
    
    Example:
        >>> heuristic = ml_based_heuristic("models/gemm_selector.pkl")
        >>> engine.set_heuristic(heuristic)
    """
    # Load model
    try:
        import pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    
    def heuristic(problem):
        # Extract features
        features = [problem.M, problem.N, problem.K]
        
        # Predict
        predictions = model.predict([features])
        
        # Return ranked kernel IDs
        return predictions[0]
    
    return heuristic

