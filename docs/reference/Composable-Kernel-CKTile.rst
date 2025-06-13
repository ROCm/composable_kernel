.. meta::
  :description: Composable Kernel CK Tile
  :keywords: composable kernel, CK, CK TileROCm

***************************************************
Composable Kernel CK Tile
***************************************************

CK Tile is a high-level abstraction layer in the Composable Kernel (CK) library. CK Tile simplifies and optimizes the definition of tiled GPU workloads for matrix operations such as GEMM and convolution by providing a modular and declarative interface for tiling strategies, thread distribution strategies, and data movement strategies for GPU compute kernels. 

CK Tile is designed for performance portability and code reusability. It provides reusable policies and utilities for block tiling, warp tiling, thread-level tiling, memory layout transforms, tensor view generation, tensor descriptor transformations, and scheduling and pipeline customization. 
