.. meta::
   :description: CK Tile Hardware-Specific Documentation
   :keywords: CDNA, GPU architecture, LDS, GEMM, CK, Composable Kernel

.. _ck_tile_hardware:

********************************************************************
CK Tile Hardware Documentation - Radeon / Navi
********************************************************************

This section provides in-depth coverage of hardware-specific concepts and optimizations for CK Tile on AMD GPUs.

Overview
========

Understanding the underlying hardware architecture is crucial for achieving optimal performance with CK Tile. This documentation covers differences between AMD CDNA and AMD RDNA across the following areas:

- Instruction Set Architecture (ISA)
- Memory hierarchy and optimization techniques
- Practical examples of high-performance kernels

Key Hardware Considerations
===========================

When using CK Tile, keep these hardware aspects in mind:

Memory Hierarchy
----------------

The memory hierarchy on Radeon / Navi is similar to the memory hiararchy on CDNA / Instinct.  It is organised into Global Memory, L2/Infinity Cache, LDS, Registers.

[TODO: table with sizes at all levels for MI200, MI300, Navi31, Navi48 ]

Compute Resources
------------------

1. **Wavefront Execution**: multiple threads in lockstep as on CNDA / Instinct

   - 32 threads (in contrast to 64 on CDNA / Instinct)

2. **Matrix Units**: specialized Matrix Multiple Units as on CDNA / Instinct
   
   - [TODO - gfx11 / Navi 3 WMMA specs]
   - [TODO - gfx12 / Navi 4 WMMA specs]