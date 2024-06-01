DP+Stream-K: Optimized GEMM for AMD GPUs
This document tackles optimizing DP+Stream-K, an algorithm designed to accelerate General Matrix Multiply (GEMM) on AMD GPUs. It achieves this by combining Data Parallel (DP) for efficient handling of uniform workloads with Stream-K's proficiency in managing irregular data distributions. This fusion ensures balanced workload distribution and faster GEMM processing.

Prior Work Leveraged:

Data Parallel (DP): Achieves high throughput by distributing workload across processing units for concurrent execution.
Stream-K: Optimizes irregular data by decomposing GEMM operations into smaller, shared memory-fitting tiles.
DP+Stream-K Benefits:

DP+Stream-K Algorithm Optimization:

Pre-Stream-K Conditional Check
Idea: Instead of blindly applying Stream-K, we propose a check to determine if it's beneficial based on the workload size.
Reasoning: When processing a small workload (less than 90% of a full dispatch), Stream-K might introduce overhead and underutilize resources compared to using all DP blocks.

Implementation: We introduce a scalar value (e.g., 0.9) to define the threshold. If the workload size exceeds this threshold (compared to a full dispatch), we disable Stream-K and use all DP blocks for better efficiency.
Even Workload Distribution Among Stream-K Blocks

Issue: Stream-K uses large blocks to handle workload unevenness. However, these "big blocks" can lead to idle time in other blocks, reducing resource utilization.

Solution: We aim to eliminate big blocks by adjusting the "K per block" value in Stream-K. Ideally, this value should be divisible by the number of Stream-K blocks to avoid remainders and big blocks.
Challenge: While theoretical benefits were proven, code modifications to adjust K per block resulted in build errors. Further investigation is needed.

Results:

Empirical evaluations confirm significant performance improvements in computational efficiency, throughput, and overall GEMM runtime with DP+Stream-K.
This optimization is particularly impactful for large and irregular matrices, leading to faster computations in HPC and deep learning applications.

Future Work

The scalar value introduced in the pre-Stream-K check provides a quantitative measure of resource utilization. We can leverage this to further refine the DP+Stream-K algorithm for optimal performance across different workloads and GPU architectures.

Enhanced Performance: Achieves significant performance improvements over traditional Stream-K, particularly for large or irregular matrices.
Balanced Workload: Mitigates workload imbalance by combining DP's initial uniform processing with Stream-K's handling of leftovers.
Faster Execution: Minimizes large blocks and ensures consistent execution times through clever data division and improved memory access patterns.
