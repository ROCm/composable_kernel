.. meta::
   :description: Block GEMM optimization on MI300 using CK Tile
   :keywords: GEMM, matrix multiplication, MI300, CK, Composable Kernel, GPU optimization

.. _ck_tile_gemm_optimization:

********************************************************************
A Block GEMM on MI300
********************************************************************

Introduction to GEMMs
=====================

This document illustrates key concepts of implementing a block GEMM (General Matrix Multiplication) kernel on AMD's MI300 GPU. GEMM is a fundamental building block for many machine learning workloads, including attention mechanisms and Mixture of Experts (MoE) models.

The problem we will address is the standard matrix multiplication: :math:`C = A \cdot B`, where matrix A has dimensions **M x K** and matrix B has dimensions **K x N**. The resulting matrix C will have dimensions **M x N**. For simplicity and a better memory access pattern, we will assume matrix B is in a column-major format, which means its shape is logically represented as **N x K**.

Format and Dimensions
=====================

The first step in designing our kernel is to select the data format and dimensions.

Data Format: bf16
-----------------

While ``float32`` is a common choice, it is often overkill for most AI applications. Its high precision is computationally expensive and can be unnecessary for model convergence. A more suitable and efficient alternative is a half-precision floating-point format. We will use **bfloat16 (bf16)**, a format that offers a significant advantage.

Bfloat16 is a 16-bit format that uses the same 8-bit exponent as ``float32``. This allows it to have the same dynamic range, which is critical for avoiding overflow and underflow during training. The key difference is that ``bf16`` uses only 7 bits for the mantissa (versus 23 bits in ``float32``), which makes it functionally equivalent to a simple right bit-shift of a 32-bit float: ``(float32 >> 16)``.

Dimensions: M=4864, N=4096
--------------------------

To maximize hardware utilization, we need to choose dimensions that will utilize GPU's resources well. Let's go with **M = 4864** and **N = 4096**. The rationale behind these particular values will be explained later.

Input data
----------

Our input will be uniformly distributed random data on the interval [-1, 1]:

.. code-block:: cpp

    initializeMatrix(A.data(), M, K, -1.0, 1.0);
    initializeMatrix(B.data(), N, K, -1.0, 1.0);

Simple Matmul
=============

On the AMD **MI300** GPU series (see :ref:`ck_tile_gpu_basics` for details), each Compute Unit (CU) contains **four SIMD units**. Each SIMD unit can execute a single **wavefront** of 64 threads in parallel. Since there are four wavefronts per CU, a CU can therefore sustain the execution of up to **256 concurrent threads**.

These 256 threads then can be logically grouped into a **thread block**, which is responsible for computing a **sub-block (tile)** of the output matrix ``C``. A block of 256 threads can be arranged as a **16×16 thread block**, where each thread computes one element of a **16×16 tile** of the result matrix ``C``. Multiple thread blocks are then organized into a **grid**, such that the collection of blocks covers the entire output matrix.

Consider a baseline matrix multiplication kernel where **each thread computes one output element** of ``C``. The CUDA/HIP launch configuration can be defined as:

.. code-block:: cpp

    dim3 blockSizeRef(16, 16);
    dim3 gridSizeRef((N + blockSizeRef.x - 1) / blockSizeRef.x,
                   (M + blockSizeRef.y - 1) / blockSizeRef.y);

    matrixMulHIP<<<gridSizeRef, blockSizeRef, 0, 0>>>(d_A, d_B, d_C);

And the GPU Kernel:

.. code-block:: cpp

    __global__ void matrixMulHIP(s_type * __restrict__ A, 
                                 s_type* __restrict__ B, 
                                 float* __restrict__ C) 
    {
        // Calculate global thread coordinates in output matrix C
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        // Boundary check for valid threads
        if (row < N && col < N) {
            float value = 0.0f;
            // Perform the dot product of row from A and column from B
            for (int k = 0; k < K; ++k) {
                value += A[row * K + k] * B[col * K + k];
            }
            // Store computed value in output matrix
            C[row * N + col] = value;
        }
    }

This kernel has a very low compute throughput according to ``rocprofv3`` profiler output. It is stalling on global memory read transactions effectively starving the rest of the pipeline that needs that data to proceed.

Memory Bandwidth Analysis
-------------------------

In a naïve implementation of matrix multiplication, **pressure on global memory loads** quickly becomes the bottleneck. To see why, let's examine how a single **16×16 block** of the destination matrix ``C`` is computed by one block of threads within a compute unit.

Each thread in the block is responsible for computing a single element of ``C``. To do so, it loops over the ``K`` dimension and, in every iteration, fetches **two values** from global memory:

- one from a row of ``A``
- one from a column of ``B``

This means:

- Number of threads in a 16×16 block is 256.
- Each thread performs 2K global loads
- **Total global loads** = 256 × 2K = 512K
- **Total global stores** = 256 (one per output element in ``C``)

If we were able to reuse each element of ``A`` and ``B`` perfectly (loading each only once), then the unique data required would be:

- Unique ``A`` elements: 16 × K = 16K
- Unique ``B`` elements: 16 × K = 16K
- **Total unique loads** = 16K + 16K = 32K
- **Total stores** = 256

- **Naïve kernel**: 512K global loads + 256 stores
- **Ideal reuse**: 32K global loads + 256 stores

This illustrates a **16× difference in memory traffic** for the same computation on a small, 16x16 block. It gets worse if we consider bigger block sizes.

What is Tiling?
===============

Cooperative Loading with LDS
----------------------------

In the naïve implementation, threads within the same compute unit (CU) do not cooperate with each other at all. Each thread independently and greedily loads the row elements of ``A`` and the column elements of ``B`` that it needs in order to compute its corresponding value in ``C``.

However, recall from the previous section that each CU on the MI300 has **64 KB of Local Data Share (LDS)** (see :ref:`ck_tile_lds_bank_conflicts` for optimization techniques), which acts as a shared memory space accessible by all threads in that CU. This opens the possibility of **cooperative loading**.

Instead of having every thread repeatedly fetch its own data directly from global memory, we can have all threads **collaboratively preload** a block of data into LDS. Once in LDS, this data can be reused by many threads, reducing redundant global memory fetches.

There is a limitation, though: we cannot preload entire rows or columns of ``A`` and ``B`` into LDS, since those may be very large and LDS has a fixed capacity. Instead, the solution is to load **small blocks (tiles)** of data at a time. For example, we can:

- Load a **16×16 tile** from ``A`` and ``B`` into LDS
- Allow all threads in the CU to reuse the data from that tile to compute their portion of the result
- Once done, move the tile window forward along the ``K`` dimension
- Repeat until the entire **16×16 output block** of ``C`` is computed

This technique of **tiling with cooperative loading** dramatically reduces global memory traffic and improves GPU efficiency by leveraging fast, on-chip LDS.

Tiling Mathematics
------------------

How many elements of matrices A and B need to be loaded with the tiling approach?

For a thread block computing a ``TILE_M × TILE_N`` output tile with K-blocking:

- Elements of **A** loaded per block:

  .. math::
     \text{A\_loads} = \mathrm{TILE\_M} \cdot K

- Elements of **B** loaded per block:

  .. math::
     \text{B\_loads} = \mathrm{TILE\_N} \cdot K

- Total outputs produced per block:

  .. math::
     \text{outputs} = \mathrm{TILE\_M} \cdot \mathrm{TILE\_N}

The **average loads per output element** (ignoring C traffic) are:

.. math::
   \text{loads per output} = \frac{\mathrm{TILE\_M}\cdot K + \mathrm{TILE\_N}\cdot K}{\mathrm{TILE\_M} \cdot \mathrm{TILE\_N}} = K \left(\frac{1}{\mathrm{TILE\_M}} + \frac{1}{\mathrm{TILE\_N}}\right)

To simplify the formula, if we consider a square tile of size T, to compute one value in C we need:

- Naïve (no tiling) = 2K loads per output.
- With tiling = 2K/T.
- **Reduction factor = T**.

Example: T=16

.. math::
   \text{loads per output} = \frac{2K}{16} = \frac{K}{8}

So compared to the naïve 2K, this gives a **16× reduction** in global memory traffic per output element.

LDS Usage and Tiling Efficiency
-------------------------------

How much space in LDS would this tiling use? Recall that matrices **A** and **B** store data in **bf16** format. For a small 16×16 tile:

- Each matrix contains 16 × 16 = 256 elements.
- At 2 bytes per element, each matrix occupies 256 × 2 = 512 bytes.
- Total for A and B: 512 × 2 = 1 KB.

We have much more space in LDS, so why not try a bigger tile size? We can afford 32 KB for each matrix, which allows us to increase the tile size to **256×64**. With this tile size, each compute unit (CU) will output a **256×256 block in C**. With this approach, the number of global memory reads will be **256 times smaller per element in C** compared to a brute-force approach.

Utilization Considerations
--------------------------

Let's explain why the input dimensions **M = 4864** and **N = 4096** are convenient choices.

Recall that MI300 has **304 compute units (CUs)**. If we choose a tile size of **256×64**, where the **K dimension** is the one we iterate over, then the output grid size is:

.. code-block:: text

    M / 256 × N / 256 = 4864 / 256 × 4096 / 256 = 19 × 16 = 304

This exactly matches the total number of compute units on the GPU. That means every CU can be fully occupied with one tile of work, and we don't need to worry about imbalance or underutilization.

Advanced Optimizations
======================

Matrix Fused Multiply-Add
-------------------------

From our previous discussion, it's clear that the compute-to-memory-access ratio will be a bottleneck. This means that simply optimizing for bandwidth isn't enough; we need to significantly improve our computational capabilities. Modern GPUs address this with specialized hardware.

Modern GPUs offer dedicated **matrix (or tensor) cores** for multiplication tasks, which are often an order of magnitude or more performant than generic ALU pipelines. These cores are specifically designed to accelerate matrix operations.

To take full advantage of these specialized cores, programmers can use intrinsic instructions. These are hardware-specific functions that allow for direct access to the matrix core pipelines. There are multiple variants of these instructions; for our purposes on the MI300, we will use ``__builtin_amdgcn_mfma_f32_16x16x16f16``, which is highly efficient and has a low latency of only 16 cycles.

The naming convention for those amdgcn intrinsics is: ``__builtin_amdgcn_mfma_<out_type>_MxNxK<in_type>``. In our case we will be using 16x16 matrices as input, and 16x16 matrices as output. These instructions work as *accumulate add*, what they effectively do is: ``D = A*B + C``. It is useful for our purpose, since we need to accumulate results over multiple tiles over K dimension.

Optimizing Data Flow with Pipelining
------------------------------------

To maximize performance, the flow for this kernel uses a **pipeline** or **double buffering** to keep the compute units continuously fed with data, reducing idle time. This pipeline consists of a series of stages that process data concurrently:

* **Stage 1: Global Memory to Registers:** The first stage involves pre-loading data directly from **global memory** into VGPR (Vector General Purpose Register) registers. This is the slowest part of the pipeline, so we perform this operation as early as possible.

* **Stage 2: Registers to LDS (Shared Memory):** As data is being loaded from global memory, the next stage of the pipeline moves the data from the VGPRs into **LDS (Local Data Share)**, or shared memory. This is a crucial intermediate step that makes the data accessible to all threads within the workgroup at very low latency.

* **Stage 3: LDS to Registers:** With the data now in fast on-chip LDS, the third stage begins. The data is transferred from LDS back into a different set of VGPR registers, which will serve as the direct input for the compute operations.

* **Stage 4: Computation with MFMA:** The final stage is the core computation. The **MFMA** (Matrix-FMA) intrinsic uses the data from the VGPRs to perform the actual matrix multiplication and accumulation.

By using this pipelined approach, the different stages of data movement and computation happen in parallel. While the current VGPRs are being consumed by the MFMA operation, the next set of data is already being moved from LDS to another set of VGPRs, and the next tile of data is being loaded from global memory into a third set of VGPRs. This overlapping of operations is key to keeping the GPU's compute units fully utilized.

CK Tile Implementation
======================

Here's how CK Tile implements an optimized GEMM kernel:

.. code-block:: cpp

    template <typename ADataType,
              typename BDataType,
              typename CDataType,
              index_t BlockSize,
              index_t MPerBlock,
              index_t NPerBlock,
              index_t KPerBlock>
    __global__ void ck_tile_gemm_kernel(const ADataType* __restrict__ a_global,
                                       const BDataType* __restrict__ b_global,
                                       CDataType* __restrict__ c_global,
                                       index_t M,
                                       index_t N,
                                       index_t K)
    {
        // Define tile distribution encoding
        // See :ref:`ck_tile_encoding_internals` and :ref:`ck_tile_tile_distribution`
        using Encoding = tile_distribution_encoding<
            sequence<>,                              // No replication
            tuple<sequence<4, 2, 8, 4>,             // M dimension hierarchy
                  sequence<4, 2, 8, 4>>,            // N dimension hierarchy
            tuple<sequence<1, 2>, sequence<1, 2>>,  // Thread mapping
            tuple<sequence<1, 1>, sequence<2, 2>>,  // Minor indices
            sequence<1, 1, 2, 2>,                   // Y-space mapping
            sequence<0, 3, 0, 3>                    // Y-space minor
        >;
        
        constexpr auto tile_dist = make_static_tile_distribution(Encoding{});
        
        // Create tensor views for global memory
        // See :ref:`ck_tile_tensor_views` and :ref:`ck_tile_buffer_views`
        auto a_global_view = make_naive_tensor_view<address_space_enum::global>(
            a_global, make_tuple(M, K), make_tuple(K, 1));
        auto b_global_view = make_naive_tensor_view<address_space_enum::global>(
            b_global, make_tuple(N, K), make_tuple(K, 1));
        auto c_global_view = make_naive_tensor_view<address_space_enum::global>(
            c_global, make_tuple(M, N), make_tuple(N, 1));
        
        // Calculate block offset
        const index_t block_m_id = blockIdx.y;
        const index_t block_n_id = blockIdx.x;
        
        // Create tile windows for loading
        // See :ref:`ck_tile_tile_window` for tile window details
        auto a_window = make_tile_window(
            a_global_view,
            make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
            make_tuple(block_m_id * MPerBlock, 0),
            tile_dist);
            
        auto b_window = make_tile_window(
            b_global_view,
            make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
            make_tuple(block_n_id * NPerBlock, 0),
            tile_dist);
        
        // Allocate LDS storage
        // See :ref:`ck_tile_static_distributed_tensor` for distributed tensors
        auto a_lds = make_static_distributed_tensor<ADataType, 
                                                   decltype(tile_dist)>();
        auto b_lds = make_static_distributed_tensor<BDataType, 
                                                   decltype(tile_dist)>();
        
        // Initialize accumulator
        auto c_reg = make_static_distributed_tensor<CDataType, 
                                                   decltype(tile_dist)>();
        // See :ref:`ck_tile_sweep_tile` for sweep operations
        sweep_tile(c_reg, [](auto idx, auto& val) { val = 0; });
        
        // Main GEMM loop with pipelining
        constexpr index_t num_k_tiles = K / KPerBlock;
        
        // Preload first tile
        a_window.load(a_lds);
        b_window.load(b_lds);
        __syncthreads();
        
        // Pipeline loop
        for(index_t k_tile = 0; k_tile < num_k_tiles - 1; ++k_tile) {
            // Move windows for next iteration
            // See :ref:`ck_tile_coordinate_movement` for window movement
            a_window.move_slice_window(make_tuple(0, KPerBlock));
            b_window.move_slice_window(make_tuple(0, KPerBlock));
            
            // Prefetch next tile while computing current
            auto a_lds_next = make_static_distributed_tensor<ADataType, 
                                                            decltype(tile_dist)>();
            auto b_lds_next = make_static_distributed_tensor<BDataType, 
                                                            decltype(tile_dist)>();
            
            a_window.load_async(a_lds_next);
            b_window.load_async(b_lds_next);
            
            // Compute with current tile
            gemm_tile(a_lds, b_lds, c_reg);
            
            // Wait for prefetch and swap buffers
            __syncthreads();
            a_lds = a_lds_next;
            b_lds = b_lds_next;
        }
        
        // Last tile computation
        gemm_tile(a_lds, b_lds, c_reg);
        
        // Store result
        auto c_window = make_tile_window(
            c_global_view,
            make_tuple(number<MPerBlock>{}, number<NPerBlock>{}),
            make_tuple(block_m_id * MPerBlock, block_n_id * NPerBlock),
            tile_dist);
            
        c_window.store(c_reg);
    }

Performance Results
===================

With proper optimization using CK Tile:

- **Memory bandwidth utilization**: >90% of theoretical peak
- **Compute utilization**: Matrix units fully saturated
- **Overall performance**: Near-peak TFLOPS for the given precision

The hotspots have shifted to matrix-related computations and ALU, global load is not a problem any longer. ``rocprofv3`` output shows that we have achieved our kernel to become compute bound where matrix multiplication operations have become the dominant workload.

Key Takeaways
=============

1. **Tiling is essential**: Reduces memory traffic by orders of magnitude
2. **Use specialized hardware**: MFMA instructions provide massive speedup
3. **Pipeline operations**: Hide memory latency with computation
4. **CK Tile abstractions**: Automatically handle complex optimizations
5. **Hardware-aware dimensions**: Choose problem sizes that map well to CU count

By understanding these optimization techniques and using CK Tile's high-level abstractions, developers can achieve near-peak performance on modern GPUs without manual low-level optimization.

Related Topics

- :ref:`ck_tile_tile_distribution` - Core distribution mechanism used in GEMM
- :ref:`ck_tile_tile_window` - Window-based data access patterns
- :ref:`ck_tile_static_distributed_tensor` - LDS memory management for tiles
- :ref:`ck_tile_lds_bank_conflicts` - Avoiding bank conflicts in GEMM
- :ref:`ck_tile_thread_mapping` - How threads map to GEMM computation
- :ref:`ck_tile_load_store_traits` - Optimized memory access patterns
- :ref:`ck_tile_space_filling_curve` - Advanced traversal patterns
- :ref:`ck_tile_sweep_tile` - Iterating over distributed data
- :ref:`ck_tile_gpu_basics` - Understanding the hardware
- :ref:`ck_tile_coordinate_systems` - Mathematical foundation
