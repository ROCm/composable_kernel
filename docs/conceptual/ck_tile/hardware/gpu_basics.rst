.. meta::
   :description: Introduction to AMD CDNA Architecture for CK developers
   :keywords: CDNA, RDNA, ROCm, CK, Composable Kernel, GPU architecture, compute units

.. _ck_tile_gpu_basics:

********************************************************************
Intro to AMD CDNA Architecture
********************************************************************

The AMD CDNA architecture is a specialized GPU design for high-performance computing (HPC) and AI workloads. Unlike the RDNA architecture used in gaming GPUs, CDNA is optimized for data center tasks, prioritizing compute density, memory bandwidth, and scalability. This is achieved through several key architectural features.

XCD (eXtreme Chiplet Design)
=============================

A fundamental element of CDNA is the **eXtreme Chiplet Design (XCD)**. This design breaks the GPU into smaller, modular chiplets. Each XCD contains a portion of the GPU's compute resources and a dedicated slice of the L2 cache. This modular approach allows for greater manufacturing flexibility, higher yields, and improved scalability, as multiple XCDs can be connected together to form a single, high-performance GPU.

MI300 incorporates 8 accelerator complex dies (XCD) with 40 compute units (CUs) per XCD, however 2 of them stay disabled which brings total to 304 CUs.

.. figure:: https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/_images/gcn_compute_unit.png
   :alt: Conceptual block diagram of an accelerator complex die (XCD)
   :align: center

   Conceptual block diagram of an accelerator complex die (XCD)

L2 Cache
========

L2 cache is shared across CUs on XCD. On MI300, the L2 is a 4MB and 16-way set associative cache that is massively parallel with 16 channels that are each 256KB.

On the read side, each channel can read out a 128-byte cache line and the L2 cache can sustain four requests from different CUs per cycle for a combined throughput of 2KBytes/clock for each XCD. The 16 channels only support a half-line 64-byte write each with one fill request from the Infinity Fabric per clock. AMD CDNA 3 architecture has collectively up to eight instances across its 16 channels and up to 34.4 TB/s aggregate read bandwidth. The L2 itself is coherent within an XCD but XCD has a snoop filter to L2 of all the XCD, so that the vast majority of coherent requests from other XCDs will be resolved at the Infinity Cache without disturbing.

Infinity Cache
==============

Infinity cache is an LLC (Last Level Cache). It has a size of 256MB and boosts generational performance and efficiency by increasing cache bandwidth and reducing the number of off-chip memory accesses.

On MI300, just like the L2 cache, the AMD Infinity Cache is 16-way set-associative, and it is built around the concept of channels. Each stack of HBM memory is associated with 16 parallel channels. A channel is 64-bytes wide and connects to 2 MB of data arrays that are banked to sustain simultaneous reads and writes. In total, there are eight stacks of HBM across the four IODs, for 128 channels or 256MB of data. The peak bandwidth from the Infinity Cache is an astounding 17.2 TB/s, which is nearly as much as the total from the previous generation L2 caches and a welcome addition to the overall memory hierarchy.

CUs (Compute Units)
===================

Each XCD is composed of multiple **Compute Units (CUs)**. The CU is the fundamental building block of the GPU, responsible for executing a group of threads. Each CU contains a set of scalar and vector units, a local data share (LDS) for shared memory, and its own instruction and data caches. The number of CUs in a CDNA GPU directly correlates with its raw processing power, as it determines how many parallel operations can be performed.

A Compute Unit (CU) contains several specialized execution pipelines and functional blocks, such as the VALU (Vector ALU), SALU (Scalar ALU), Local Data Share (LDS), and a scheduler. Here are some key CU components:

SIMD
----

SIMD (Single Instruction, Multiple Data) units allow a single instruction to operate on multiple data points simultaneously. In a CDNA Compute Unit, each SIMD typically consists of 16 lanes, and four SIMDs work together to process a wavefront of 64 threads in parallel. Modern architectures, such as MI300, support 4 concurrent wavefronts per CU and provide 512 vector registers (VGPRs) shared among them.

VGPR
----

VGPRs (Vector General Purpose Registers) are one dword wide registers assigned to each thread in a wavefront. The number of VGPRs available per wavefront limits how many threads can be active at once. If a kernel requires more than 512 VGPRs per thread, the compiler will spill excess data to scratch memory in LDS (if available) or DRAM. The latter is much slower due to higher latency. This is primary motivation behind highly templated CK design, the API is aiming to perform as many computations at compile time as possible without allocating GPU registers at runtime.

SGPR
----

SGPRs (Scalar General Purpose Registers) are one dword wide registers shared by all threads in a wavefront. There are fewer SGPRs than VGPRs; for example, MI300 provides 128 SGPRs per CU.

LDS
---

LDS (Local Data Share) is a fast, shared memory block accessible by all threads within a CU, enabling efficient data exchange and synchronization. More details on LDS will be discussed in :ref:`ck_tile_lds_bank_conflicts`.

vL1D
----

The Vector L1 Data Cache (vL1D) is a high-speed, local cache within each Compute Unit (CU) of the AMD CDNA architecture. It is designed to handle memory requests from the vector units, which are responsible for parallel operations on a wavefront. Also known as the Texture Cache per Pipe (TCP), the vL1D cache works with a series of specialized units to quickly access data from memory, including an address translation unit and a tag RAM. The performance of this cache is critical for reducing memory latency and minimizing stalls in compute-intensive workloads.

Implications for CK Tile
========================

Understanding the CDNA architecture is crucial for effective use of CK Tile:

1. **Thread Organization**: CK Tile's hierarchical :ref:`ck_tile_thread_mapping` (blocks → warps → threads) directly maps to CDNA's hardware organization.

2. **Memory Hierarchy**: CK Tile's :ref:`ck_tile_buffer_views` and :ref:`ck_tile_tile_window` are designed to efficiently utilize the L2, Infinity Cache, and LDS hierarchy.

3. **Register Pressure**: CK Tile's compile-time optimizations help minimize VGPR usage, preventing spills to slower memory.

4. **Warp Execution**: CK Tile's :ref:`ck_tile_tile_distribution` ensures that threads within a warp access contiguous memory for optimal SIMD execution.

5. **LDS Utilization**: CK Tile's :ref:`ck_tile_static_distributed_tensor` and :ref:`ck_tile_tile_window` make effective use of the 64KB LDS per CU.

By understanding these architectural features, developers can better appreciate how CK Tile's abstractions map to hardware capabilities and why certain design decisions were made in the framework.

Further Reading

For comprehensive documentation on AMD GPU architecture and programming:

**AMD GPU Architecture Programming Documentation**
  https://gpuopen.com/amd-gpu-architecture-programming-documentation/

This resource provides in-depth guides covering:

- **CDNA Architecture Whitepapers**: Detailed technical specifications for MI200 and MI300 series
- **ISA (Instruction Set Architecture) Documentation**: Complete instruction reference for AMD GPUs
- **Optimization Guides**: Best practices for achieving peak performance
- **Programming Guides**: ROCm and HIP programming models
- **Performance Tools**: Profiling and analysis tool documentation

These guides complement the CK Tile documentation by providing the low-level hardware details that inform the design of CK's high-level abstractions. Understanding the underlying architecture enables developers to make informed decisions about tile sizes, distribution patterns, and optimization strategies.

Related Topics

- :ref:`ck_tile_thread_mapping` - How threads are organized and mapped to hardware
- :ref:`ck_tile_coordinate_systems` - Mathematical foundation for data distribution
- :ref:`ck_tile_lds_bank_conflicts` - Optimizing shared memory access patterns
- :ref:`ck_tile_load_store_traits` - Memory access optimization strategies
- :ref:`ck_tile_gemm_optimization` - Practical application of architecture knowledge
