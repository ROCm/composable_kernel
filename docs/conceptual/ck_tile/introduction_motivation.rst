.. _ck_tile_introduction:

Introduction and Motivation - Why Tile Distribution Matters
===========================================================

[TO DO: Most of this can be removed but kept somewhere in case we want to use it somewhere else.]

Overview [TO DO: can be removed]
--------

The evolution of GPU computing has brought increasing computational power to modern applications, yet harnessing this power efficiently remains one of the most challenging aspects of high-performance computing. This challenge arises from a mismatch between how developers conceptualize algorithms and how GPU hardware processes them. While developers think in terms of mathematical operations on multi-dimensional data structures, GPUs operate through thousands of threads accessing memory in complex patterns that must satisfy stringent hardware constraints.

This conceptual gap manifests most acutely in memory access patterns. GPUs achieve their high performance through massive parallelism, with thousands of threads processing simultaneously. However, this parallelism comes with a constraint: memory bandwidth. Despite continuous improvements in computational throughput, memory bandwidth has not scaled proportionally, creating what is often called the "memory wall." The efficiency with which threads access memory determines whether a GPU kernel achieves a few percent or near 100% of the hardware's theoretical performance.

The Composable Kernel (CK) framework addresses this challenge through its tile distribution system, a compile-time abstraction that automatically generates optimal memory access patterns while preserving the natural expression of algorithms. This documentation explores the mathematical foundations and practical implementation of tile distribution, demonstrating how it bridges the gap between algorithmic intent and hardware reality.

This introduction establishes the problems that tile distribution solves, explores why these problems matter for GPU performance, and provides the conceptual framework necessary to understand the compile-time coordinate transformation system that powers CK's approach to efficient GPU computation.

The GPU Memory Problem
----------------------

.. 
   Original mermaid diagram (edit here, then run update_diagrams.py)
   
   .. mermaid::
   
      graph TB
      subgraph "Random Access Pattern (Inefficient)"
          subgraph "Threads"
              T0_R["Thread 0"]
              T1_R["Thread 1"] 
              T2_R["Thread 2"]
              T3_R["Thread 3"]
          end

          subgraph "Memory"
              M0["Mem[0]"]
              M7["Mem[7]"]
              M15["Mem[15]"]
              M23["Mem[23]"]
              M31["Mem[31]"]
              M39["Mem[39]"]
              M47["Mem[47]"]
              M55["Mem[55]"]
          end

          T0_R -.-> M23
          T1_R -.-> M7
          T2_R -.-> M47
          T3_R -.-> M15
      end

      subgraph "Tile Distribution Pattern (Efficient)"
          subgraph "Threads_TD"
              T0_TD["Thread 0"]
              T1_TD["Thread 1"]
              T2_TD["Thread 2"]
              T3_TD["Thread 3"]
          end

          subgraph "Memory_TD"
              M0_TD["Mem[0]"]
              M1_TD["Mem[1]"]
              M2_TD["Mem[2]"]
              M3_TD["Mem[3]"]
              M4_TD["Mem[4]"]
              M5_TD["Mem[5]"]
              M6_TD["Mem[6]"]
              M7_TD["Mem[7]"]
          end

          T0_TD --> M0_TD
          T0_TD --> M1_TD
          T1_TD --> M2_TD
          T1_TD --> M3_TD
          T2_TD --> M4_TD
          T2_TD --> M5_TD
          T3_TD --> M6_TD
          T3_TD --> M7_TD
      end

      style T0_R fill:#fee2e2,stroke:#ef4444,stroke-width:2px
      style T1_R fill:#fee2e2,stroke:#ef4444,stroke-width:2px
      style T2_R fill:#fee2e2,stroke:#ef4444,stroke-width:2px
      style T3_R fill:#fee2e2,stroke:#ef4444,stroke-width:2px

      style T0_TD fill:#d1fae5,stroke:#10b981,stroke-width:2px
      style T1_TD fill:#d1fae5,stroke:#10b981,stroke-width:2px
      style T2_TD fill:#d1fae5,stroke:#10b981,stroke-width:2px
      style T3_TD fill:#d1fae5,stroke:#10b981,stroke-width:2px
   

[TO DO: the images need to be looked at because they look off in svg format]

.. image:: diagrams/introduction_motivation_1.svg
   :alt: Diagram
   :align: center

Why Random Memory Access is Slow [TO DO: keep -- and rewrite]

[TO DO: can have something where we say "memory access is slow but CK Tile can help with that" with examples]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The architecture of GPUs represents a study in trade-offs. While these devices can execute thousands of threads simultaneously and perform trillions of floating-point operations per second, they remain constrained by the physics of memory access. Understanding this constraint helps explain why tile distribution is not merely an optimization technique but an important component of high-performance GPU computing.

GPU memory systems are designed around the assumption of regular, predictable access patterns. The memory controller can service requests from 32 threads (a warp on AMD GPUs) in a single transaction when these threads access consecutive memory locations. This optimization, known as memory coalescing, can improve effective memory bandwidth by up to 32x compared to random access patterns. However, when threads within a warp access memory locations that are scattered throughout the address space, each access requires a separate memory transaction, reducing the effective bandwidth to a fraction of the theoretical maximum.

The impact extends beyond raw bandwidth. GPUs employ cache hierarchies to reduce memory latency, but these caches are effective only when access patterns exhibit spatial or temporal locality. Random access patterns defeat these optimizations, causing frequent cache misses that expose the full latency of global memory access, which can be hundreds of cycles. During these stalls, the computational units sit idle, unable to hide the latency even with the GPU's massive thread count.

Furthermore, the GPU's SIMT (Single Instruction, Multiple Thread) model means that all threads in a warp must process the same instruction at the same time. When threads access memory in unpredictable patterns, the memory controller cannot optimize the requests, leading to serialization of what should be parallel operations. This serialization effect compounds with each level of the memory hierarchy, from L1 cache through L2 cache to global memory, multiplying the performance impact.

The Thread Cooperation Challenge 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

[TO DO: keep also -- rewrite]



The challenge of efficient thread cooperation becomes particularly evident when examining an operation like matrix multiplication. Consider a scenario where 256 threads must cooperate to multiply two matrices. The naive approach, where each thread computes one element of the output matrix, illustrates precisely why GPU programming requires compile-time abstractions. [can be removed]

.. code-block:: cpp

   // Inefficient: Random access pattern
   __device__ void naive_matrix_multiply()
   {
       int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
       
       // Get this thread's output position
       int row = thread_id / MATRIX_WIDTH;
       int col = thread_id % MATRIX_WIDTH;
       
       // Each thread computes one element of C = A * B
       float result = 0.0f;
       for (int k = 0; k < MATRIX_WIDTH; k++)
       {
           // Random access pattern - threads in a warp access non-contiguous memory
           // Thread 0: A[0,0], A[0,1], A[0,2]...
           // Thread 1: A[1,0], A[1,1], A[1,2]...
           // These are far apart in memory!
           float a_element = global_memory_A[row * MATRIX_WIDTH + k];
           
           // Even worse for B - accessing column-wise causes strided access
           // Thread 0: B[0,0], B[1,0], B[2,0]...
           // Thread 1: B[0,1], B[1,1], B[2,1]...
           // Massive stride between accesses!
           float b_element = global_memory_B[k * MATRIX_WIDTH + col];
           
           result += a_element * b_element;
       }
       
       // Write result - adjacent threads write to adjacent locations (at least this is good)
       global_memory_C[row * MATRIX_WIDTH + col] = result;
   }

[TO DO: add more information about the code block in the text in addition to inline comments]

This seemingly straightforward implementation suffers from inefficiencies that stem from the mismatch between the algorithm's logical structure and the hardware's physical constraints. The memory access pattern is essentially random from the hardware's perspective, as adjacent threads access memory locations separated by large strides. This pattern prevents the memory controller from coalescing accesses, forcing it to issue separate transactions for each thread.

The lack of coordination between threads exacerbates the problem. While all threads in a warp execute the same instructions, they operate on completely different data with no sharing or reuse. This independence, which might seem desirable in traditional parallel programming, actually works against GPU architecture. The hardware cannot exploit any commonality in the access patterns, leading to severe underutilization of memory bandwidth.

Cache utilization suffers dramatically under this access pattern. Each thread traces a unique path through memory, with no overlap between threads' working sets. The L1 and L2 caches, designed to capture and exploit locality, instead thrash continuously as each thread's accesses evict data needed by others. The effective cache capacity approaches zero, exposing every memory access to the full latency of global memory.

This approach also fails to utilize the available memory bandwidth efficiently. GPUs can achieve memory bandwidths exceeding 1 TB/s, but only when accesses are properly structured. The random access pattern of the naive implementation might achieve less than 10% of this theoretical maximum, effectively reducing a high-performance GPU to the performance level of a much simpler processor.

[TO DO: Explain that the goal is to show why you wouldn't want to do what's in that code] 

The Tile Distribution Solution
------------------------------

Structured Mapping from Logical to Physical Coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tile distribution addresses the memory access problem through its approach. Rather than attempting to optimize the naive access patterns after the fact, tile distribution provides a mathematical framework that generates optimal patterns from the outset. This framework establishes a structured mapping between logical coordinates—the natural way developers think about their data—and physical coordinates that respect hardware constraints.

The essence of tile distribution is the recognition that efficient GPU computation requires a careful choreography of thread cooperation. Instead of each thread operating independently, threads are organized into hierarchical groups that work together on tiles of data. This organization ensures that when threads access memory, they do so in patterns that the hardware can optimize.

.. code-block:: cpp

   // Efficient: Tile-based distribution using CK Tile
   template<typename AType, typename BType, typename CType>
   __device__ void tile_distributed_matrix_multiply()
   {
       // 1. Define tile distribution encoding at compile time
       using Encoding = tile_distribution_encoding<
           sequence<>,                                    // No replication
           tuple<sequence<4, 2, 8, 4>,                   // M dimension hierarchy
                 sequence<4, 2, 8, 4>>,                  // N dimension hierarchy
           tuple<sequence<1, 2>, sequence<1, 2>>,        // P to RH major
           tuple<sequence<1, 1>, sequence<2, 2>>,        // P to RH minor
           sequence<1, 1, 2, 2>,                         // Y to RH major
           sequence<0, 3, 0, 3>                          // Y to RH minor
       >;
       
       // 2. Create the distribution
       constexpr auto distribution = make_static_tile_distribution(Encoding{});
       
       // 3. Create tile window for efficient memory access
       auto tile_window = make_tile_window(
           tensor_view, 
           window_lengths, 
           origin, 
           distribution
       );
       
       // 4. Load data with coalesced access pattern
       auto loaded_tensor = tile_window.load();
       
       // 5. Process tile data efficiently
       sweep_tile(loaded_tensor, [](auto y_indices) {
           auto value = loaded_tensor(y_indices);
           // ... efficient computation
       });
   }

[TO DO: add additional explanation to this since it introduces a lot of concepts; also link to any example that is relevant.]

The transformation from inefficient to efficient memory access is notable. Where the naive implementation scattered memory requests across the address space, tile distribution ensures that adjacent threads access adjacent memory locations. This transformation happens through an encoding system that captures the hierarchical nature of both the computation and the hardware.

The encoding shown above demonstrates the multi-level hierarchy that tile distribution employs. The sequence<4, 2, 8, 4> represents a four-level decomposition: four repetitions per thread, two warps per block, eight threads per warp, and four elements per vector operation. This hierarchical structure maps directly to the GPU's hardware organization, ensuring that each level of the hierarchy operates at maximum efficiency.

Memory access patterns become predictable and regular under tile distribution. The hardware's memory coalescing logic can now combine the requests from all threads in a warp into a single transaction, achieving the full memory bandwidth. The predictability extends beyond individual accesses to entire access sequences, enabling the hardware's prefetching mechanisms to anticipate and prepare data before it's needed.

Thread cooperation emerges naturally from the tile distribution structure. Threads within a warp work on adjacent data, enabling efficient data sharing through register shuffle operations. Warps within a block coordinate through shared memory, with access patterns that avoid bank conflicts. This cooperation transforms what was a collection of independent computations into a unified, efficient operation.

Cache utilization improves dramatically as well. The structured access patterns ensure that data loaded into cache by one thread is likely to be used by neighboring threads. Temporal locality emerges from the tile-based processing, where all operations on a tile complete before moving to the next tile. This locality transforms the cache from a liability into a high performance accelerator.

The scalability of tile distribution across different GPU architectures represents one of its important features. The same high-level code can achieve near-optimal performance on GPUs with different numbers of compute units, different cache sizes, and different memory bandwidths. The compile-time nature of the encoding allows the compiler to generate architecture-specific optimizations while maintaining portable source code.

The Coordinate Mapping Insight [TO DO: to be removed; more advanced]
------------------------------

Tile distribution uses a mathematical insight: efficient GPU computation requires a systematic framework for mapping between different coordinate spaces. This framework transforms the complex problem of thread-to-data assignment into a series of well-defined mathematical transformations, each serving a specific purpose in the journey from abstract algorithm to concrete hardware implementation.

.. 
   Original mermaid diagram (edit here, then run update_diagrams.py)
   
   .. mermaid::
   
      graph LR
          subgraph "Coordinate Spaces"
              P["P-space<br/>Thread Position<br/>(thread_x, thread_y,<br/>warp_id, block_id)"]
              Y["Y-space<br/>Local Data<br/>(y0, y1, y2, y3)"]
              X["X-space<br/>Global Position<br/>(x0, x1)"]
              D["D-space<br/>Memory Address<br/>(linearized)"]
          end

          subgraph "Transformations"
              T1["P + Y → X<br/>Thread data mapping"]
              T2["X → D<br/>Memory linearization"]
          end

          P --> T1
          Y --> T1
          T1 --> X
          X --> T2
          T2 --> D

          style P fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
          style Y fill:#fff3e0,stroke:#f57c00,stroke-width:2px
          style X fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
          style D fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
          style T1 fill:#fef3c7,stroke:#f59e0b,stroke-width:2px
          style T2 fill:#fef3c7,stroke:#f59e0b,stroke-width:2px
   
   

.. image:: diagrams/introduction_motivation_2.svg
   :alt: Diagram
   :align: center
The elegance of this approach emerges from its separation of concerns. Each coordinate space represents a distinct aspect of the computation, and the transformations between them encapsulate specific optimization strategies. This separation allows developers to reason about their algorithms in natural terms while the framework handles the complex mapping to efficient hardware processing patterns.

P-space (Thread Position Space)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

P-space represents the physical organization of threads on the GPU. This space captures the hierarchical nature of GPU thread organization, from individual threads identified by their x and y coordinates within a block, to warps that operate in lockstep, to thread blocks that share resources. The coordinates in P-space—``thread_x``, ``thread_y``, ``warp_id``, and ``block_id``—directly correspond to the hardware's thread model. P-space determines which threads can cooperate efficiently through shared memory and which threads will process their memory accesses simultaneously.

Y-space (Local Data Space)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Y-space embodies the algorithm's perspective on data organization. In this space, each thread reasons about its local portion of work using coordinates like ``y0``, ``y1``, ``y2``, and ``y3``. These coordinates are algorithm-specific and represent the natural way to index the data being processed. For matrix multiplication, Y-space might represent the local tile coordinates within a larger matrix. For convolution, it might represent the spatial dimensions and channels of a local receptive field. Y-space allows algorithms to be expressed in their most natural form, without concern for hardware-specific optimizations.

X-space (Global Position Space)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

X-space serves as the bridge between algorithmic intent and physical reality. This space represents the actual global coordinates of data in the problem domain—for instance, the row and column indices in a matrix or the spatial coordinates in an image. X-space is where the distributed nature of the computation becomes explicit, as each thread's local Y-space coordinates combine with its position in P-space to determine which global data elements it accesses.

D-space (Memory Address Space)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

D-space represents the final destination: linearized memory addresses that the hardware actually uses. This space accounts for the fact that multi-dimensional data structures must ultimately be stored in linear memory. The transformation to D-space incorporates layout optimizations such as padding for alignment, interleaving for better cache utilization, and address space considerations for different memory types (global, shared, or constant memory).

The transformative power of tile distribution emerges from the composition of these mappings. The **P + Y → X** transformation combines a thread's position with its local data coordinates to determine global data positions. This transformation encodes the distribution strategy, determining how work is partitioned across threads. The subsequent **X → D** transformation converts these logical positions into physical memory addresses, incorporating layout optimizations that ensure efficient memory access patterns.

The mathematical rigor of this framework enables optimizations. Because each transformation is well-defined and composable, the compiler can analyze the complete transformation chain and generate optimal code. The framework can automatically ensure memory coalescing by structuring the P + Y → X transformation appropriately. It can minimize bank conflicts in shared memory by carefully designing the X → D mapping. Most importantly, it can adapt these optimizations to different hardware architectures by adjusting the transformation parameters while keeping the high-level algorithm description unchanged.

What's Coming Next [TO DO: remove]
------------------

Having established the motivation for tile distribution and its coordinate mapping framework, the following sections provide a systematic overview of the complete CK Tile system. The documentation is carefully structured to build understanding layer by layer, starting from the most basic abstractions and progressing to advanced optimization techniques.

The foundation begins with raw memory access through :ref:`ck_tile_buffer_views`, which provides type-safe, address-space-aware access to GPU memory. Understanding BufferView establishes the patterns and principles that permeate the entire CK Tile system. From there, the documentation progresses to :ref:`ck_tile_tensor_views`, which adds multi-dimensional structure to raw memory, enabling natural expression of algorithms while maintaining the efficiency of the underlying buffer operations.

With these concepts established, the :ref:`ck_tile_coordinate_systems` documentation covers the tile distribution framework. This system implements the mathematical framework introduced above, providing compile-time transformations between P-space, Y-space, X-space, and D-space. Understanding these transformations at a deep level enables developers to reason about performance implications and design custom distribution strategies for novel algorithms. The :ref:`ck_tile_transforms` and :ref:`ck_tile_adaptors` provide the building blocks for these transformations.

The high-level :ref:`ck_tile_distribution` APIs represent the culmination of these lower-level abstractions. These APIs provide an accessible interface for common patterns while exposing enough flexibility for optimizations. Through concrete examples and detailed explanations, the documentation demonstrates how to leverage these APIs to achieve near-optimal performance across a variety of computational patterns. The :ref:`ck_tile_window` abstraction provides the gateway for efficient data access.

The exploration of coordinate systems goes beyond the basic P, Y, X, D framework to encompass topics such as multi-level tiling, replication strategies, and specialized coordinate systems for specific algorithm classes. The :ref:`ck_tile_encoding_internals` reveals the mathematical foundations, while :ref:`ck_tile_thread_mapping` shows how these abstractions map to hardware. This comprehensive treatment ensures that developers can handle not just common cases but also novel algorithms that require custom distribution strategies.

The implementation details reveal the template metaprogramming techniques that enable CK Tile's zero-overhead abstractions. Topics like :ref:`ck_tile_descriptors`, :ref:`ck_tile_load_store_traits`, and :ref:`ck_tile_static_distributed_tensor` show how these abstractions achieve zero overhead. By understanding these implementation strategies, developers can extend the framework, contribute optimizations, and debug performance issues at the deepest level.

The connection between abstract coordinate transformations and concrete hardware thread mapping represents an important piece of understanding. The documentation examines how logical thread organizations map to physical GPU resources, how to avoid common pitfalls like bank conflicts (see :ref:`ck_tile_lds_bank_conflicts` and :ref:`ck_tile_lds_index_swapping`) and divergent processing, and how to structure computations for maximum hardware utilization. The :ref:`ck_tile_hardware` section provides deep dives into architecture-specific optimizations.

Finally, additional topics include optimization techniques like :ref:`ck_tile_space_filling_curve` for optimal memory traversal, :ref:`ck_tile_sweep_tile` for clean iteration patterns, and practical examples like :ref:`ck_tile_convolution_example` and :ref:`ck_tile_gemm_optimization`. These topics prepare developers to push the boundaries of GPU performance and contribute to the ongoing evolution of high-performance computing.


Next Steps
----------

Continue to :ref:`ck_tile_buffer_views` for the foundational concepts.
