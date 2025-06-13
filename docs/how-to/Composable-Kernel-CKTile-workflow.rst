.. meta::
  :description: Composable Kernel CK Tile workflow
  :keywords: composable kernel, CK, CK Tile, ROCm, how-to

***************************************************
Composable Kernel CK Tile workflow
***************************************************

When using CK Tile, a typical workflow begins with defining the problem shape and describing the distribution before launching the kernel.

For more information, see :doc:`CK Tile concepts <../conceptual/Composable-Kernel-CKTile-concept>`.
 

Define the problem shape:


.. code:: cpp

    struct GemmProblem {
        using ADataType = ck::half_t;
        using BDataType = ck::half_t;
        using CDataType = ck::half_t;
        static constexpr index_t M = 1024;
        static constexpr index_t N = 1024;
        static constexpr index_t K = 512;
    };

Describe tile distributions:

.. code:: cpp

    auto xblock = MakeXBlockTileDistribution<GemmProblem>();
    auto a_dram_desc = MakeADramTileDistribution<GemmProblem>();
    auto a_lds_desc  = MakeALdsBlockDescriptor<GemmProblem>();


Each of these ``Make*`` utilities is modular and interchangeable, enabling easy tuning.

Launch the kernel

.. code:: cpp

    launch_kernel<TileGemmKernel>(xblock, problem, ...);

 