.. meta::
  :description: Composable Kernel CK Tile API reference
  :keywords: composable kernel, CK, CK Tile, ROCm, API

***************************************************
Composable Kernel CK Tile API Reference
***************************************************

The following are some important CK Tile APIs.

Tensor descriptor transform 
=======================================
 
``make_merge_transform`` combines multiple tensor dimensions into one.

``make_unmerge_transform`` splits a dimension into multiple dimensions.

``make_pass_through_transform`` is an identity transform that preserves dimensions.

``make_xor_transform`` applies XOR-based swizzling to improve memory access patterns.

``transform_tensor_descriptor`` applies transforms to modify tensor representations.


Tile maker 
======================

``MakeADramTileDistribution`` creates a descriptor for matrix A in global memory.

``MakeALdsBlockDescriptor`` creates descriptor for matrix A in shared memory.

``MakeXBlockTileDistribution`` defines the threadblock distribution strategy.

``MakeTensorView`` creates views into tensors with specific access patterns.

 
GEMM kernel launch 
=============================

``launch_kernel<TileGemmKernel>`` launches the configured kernel with tile distributions.

