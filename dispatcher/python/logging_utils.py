"""
Logging utilities for CK Tile Dispatcher

Provides structured logging with performance tracking.
"""

import logging
import time
from typing import Optional, Dict
from contextlib import contextmanager
from functools import wraps


# Create logger
logger = logging.getLogger("ck_tile_dispatcher")
logger.setLevel(logging.WARNING)

# Create console handler
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.DEBUG)

# Create formatter
_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
_console_handler.setFormatter(_formatter)

# Add handler
logger.addHandler(_console_handler)


def set_log_level(level: str):
    """
    Set logging level

    Args:
        level: One of DEBUG, INFO, WARNING, ERROR, CRITICAL
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    if level.upper() not in level_map:
        raise ValueError(f"Invalid log level: {level}")

    logger.setLevel(level_map[level.upper()])
    logger.info(f"Log level set to {level.upper()}")


def enable_file_logging(filepath: str, level: str = "DEBUG"):
    """
    Enable logging to file

    Args:
        filepath: Path to log file
        level: Logging level for file
    """
    file_handler = logging.FileHandler(filepath)
    file_handler.setLevel(getattr(logging, level.upper()))
    file_handler.setFormatter(_formatter)
    logger.addHandler(file_handler)
    logger.info(f"File logging enabled: {filepath}")


def disable_logging():
    """Disable all logging"""
    logger.setLevel(logging.CRITICAL + 1)


# Performance logging
class PerformanceLogger:
    """Track and log performance metrics"""

    def __init__(self):
        self.metrics: Dict[str, list] = {}

    def log_execution(self, operation: str, time_ms: float, **kwargs):
        """Log an execution"""
        if operation not in self.metrics:
            self.metrics[operation] = []

        self.metrics[operation].append(
            {"time_ms": time_ms, "timestamp": time.time(), **kwargs}
        )

        logger.debug(f"{operation}: {time_ms:.3f} ms")

    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation"""
        if operation not in self.metrics:
            return {}

        times = [m["time_ms"] for m in self.metrics[operation]]

        import numpy as np

        return {
            "count": len(times),
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "total_ms": np.sum(times),
        }

    def print_summary(self):
        """Print performance summary"""
        print("\n" + "=" * 70)
        print("Performance Summary")
        print("=" * 70)
        print(f"{'Operation':<30} {'Count':>8} {'Mean (ms)':>12} {'Total (ms)':>12}")
        print("-" * 70)

        for operation in sorted(self.metrics.keys()):
            stats = self.get_stats(operation)
            print(
                f"{operation:<30} {stats['count']:>8} "
                f"{stats['mean_ms']:>12.3f} {stats['total_ms']:>12.3f}"
            )

        print("=" * 70)

    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()


# Global performance logger
_perf_logger: Optional[PerformanceLogger] = None


def get_perf_logger() -> PerformanceLogger:
    """Get global performance logger"""
    global _perf_logger
    if _perf_logger is None:
        _perf_logger = PerformanceLogger()
    return _perf_logger


# Decorators
def log_call(func):
    """Decorator to log function calls"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling {func.__name__}")
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000
            logger.debug(f"{func.__name__} completed in {elapsed:.3f} ms")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed: {e}")
            raise

    return wrapper


def log_performance(operation_name: Optional[str] = None):
    """Decorator to log performance"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000

            perf_logger = get_perf_logger()
            perf_logger.log_execution(op_name, elapsed)

            return result

        return wrapper

    return decorator


# Context managers
@contextmanager
def log_context(operation: str, level: str = "INFO"):
    """
    Context manager for logging operations

    Example:
        >>> with log_context("GEMM computation"):
        ...     C = gemm(A, B)
    """
    log_func = getattr(logger, level.lower())
    log_func(f"Starting {operation}")
    start = time.perf_counter()

    try:
        yield
        elapsed = (time.perf_counter() - start) * 1000
        log_func(f"Completed {operation} in {elapsed:.3f} ms")
    except Exception as e:
        logger.error(f"Failed {operation}: {e}")
        raise


@contextmanager
def timed_operation(operation: str):
    """
    Context manager for timing operations

    Example:
        >>> with timed_operation("GEMM") as timer:
        ...     C = gemm(A, B)
        >>> print(f"Time: {timer.elapsed_ms:.3f} ms")
    """

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.elapsed_ms = None

    timer = Timer()
    timer.start_time = time.perf_counter()

    try:
        yield timer
    finally:
        timer.end_time = time.perf_counter()
        timer.elapsed_ms = (timer.end_time - timer.start_time) * 1000

        perf_logger = get_perf_logger()
        perf_logger.log_execution(operation, timer.elapsed_ms)


# Dispatch logging
class DispatchLogger:
    """Log kernel dispatch decisions"""

    def __init__(self):
        self.dispatches = []

    def log_dispatch(
        self, problem_size: tuple, kernel_name: str, selection_time_ms: float, **kwargs
    ):
        """Log a dispatch decision"""
        self.dispatches.append(
            {
                "problem_size": problem_size,
                "kernel_name": kernel_name,
                "selection_time_ms": selection_time_ms,
                "timestamp": time.time(),
                **kwargs,
            }
        )

        M, N, K = problem_size
        logger.info(
            f"Dispatched {M}x{N}x{K} to {kernel_name} "
            f"(selection: {selection_time_ms:.3f} ms)"
        )

    def print_summary(self):
        """Print dispatch summary"""
        if not self.dispatches:
            print("No dispatches logged")
            return

        print("\n" + "=" * 80)
        print("Dispatch Summary")
        print("=" * 80)

        # Count by kernel
        kernel_counts = {}
        for d in self.dispatches:
            kernel = d["kernel_name"]
            kernel_counts[kernel] = kernel_counts.get(kernel, 0) + 1

        print(f"\nTotal dispatches: {len(self.dispatches)}")
        print("\nKernel usage:")
        for kernel, count in sorted(
            kernel_counts.items(), key=lambda x: x[1], reverse=True
        ):
            pct = 100 * count / len(self.dispatches)
            print(f"  {kernel:<50} {count:>6} ({pct:>5.1f}%)")

        print("=" * 80)

    def reset(self):
        """Reset dispatch log"""
        self.dispatches.clear()


# Global dispatch logger
_dispatch_logger: Optional[DispatchLogger] = None


def get_dispatch_logger() -> DispatchLogger:
    """Get global dispatch logger"""
    global _dispatch_logger
    if _dispatch_logger is None:
        _dispatch_logger = DispatchLogger()
    return _dispatch_logger


# Utility functions
def log_system_info():
    """Log system information"""
    import platform
    import sys

    logger.info("=" * 60)
    logger.info("System Information")
    logger.info("=" * 60)
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Python version: {platform.python_version()}")

    try:
        import numpy as np

        logger.info(f"NumPy: {np.__version__}")
    except ImportError:
        pass

    try:
        import torch

        logger.info(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"CUDA: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass

    logger.info("=" * 60)


def log_config(config):
    """Log configuration"""
    logger.info("=" * 60)
    logger.info("Configuration")
    logger.info("=" * 60)
    for key, value in config.to_dict().items():
        logger.info(f"{key:30s}: {value}")
    logger.info("=" * 60)
