 .. meta::
  :description: Composable Kernel CK Tile concepts
  :keywords: composable kernel, CK, CK Tile, concepts ROCm

***************************************************
Composable Kernel CK Tile concepts
***************************************************

CK Tile is a high-level abstraction layer in the Composable Kernel (CK) library.

It's important to understand the following concepts before using CK Tile. For information on how these concepts are used in CK Tile, see :doc:`How to use CK Tile <../how-to/Composable-Kernel-CKTile-workflow>`.
 
Problem
    The data shapes and types used to instantiate a kernel. For example, ``M``, ``N``, ``K``.

Tile distribution
    Describes how blocks, warps, and threads are mapped to a tiled workload.  

Tensor descriptor
    Abstract representation of a tensor's shape, layout, and strides.


Transforms
    Utilities such as merge, unmerge, pass-through, and XOR/swizzle that transform tensor descriptors. 

Policy
    Composable building blocks that control pipeline behavior. For example, memory loading and compute scheduling.
