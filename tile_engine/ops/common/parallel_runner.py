#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Generic multi-GPU parallel job runner for tile engine benchmarks.

Op-agnostic: takes opaque jobs, distributes them across GPUs with one
job per GPU at a time, and yields results in completion order. Used by
fmha_benchmark.py and reusable for gemm/reduce/pooling benchmarks.
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterator, List, Optional, Tuple


def run_parallel_on_gpus(
    jobs: List[Any],
    gpu_ids: List[int],
    run_one: Callable[[Any, int], Any],
    max_workers: Optional[int] = None,
) -> Iterator[Tuple[int, Any]]:
    """Dispatch jobs across GPUs, one job per GPU at a time.

    Args:
        jobs: Opaque job objects passed to run_one.
        gpu_ids: GPU IDs to use (e.g. [0,1,2,3]). At most one job per GPU runs concurrently.
        run_one: Callable run_one(job, gpu_id) -> result. Caller is responsible
            for any subprocess isolation, environment setup, etc.
        max_workers: Thread pool size. Defaults to len(gpu_ids).

    Yields:
        (job_index, result) tuples in completion order. Caller can sort by
        job_index to restore submission order if needed.
    """
    if not jobs:
        return
    if max_workers is None:
        max_workers = len(gpu_ids)

    # One job per GPU at a time
    gpu_semas = {gid: threading.Semaphore(1) for gid in gpu_ids}
    cycle = [0]
    cycle_lock = threading.Lock()

    def _pick_gpu() -> int:
        with cycle_lock:
            gid = gpu_ids[cycle[0] % len(gpu_ids)]
            cycle[0] += 1
            return gid

    def _wrapper(job_idx: int, job: Any) -> Tuple[int, Any]:
        gid = _pick_gpu()
        gpu_semas[gid].acquire()
        try:
            return job_idx, run_one(job, gid)
        finally:
            gpu_semas[gid].release()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_wrapper, i, j) for i, j in enumerate(jobs)]
        for fut in as_completed(futures):
            yield fut.result()
