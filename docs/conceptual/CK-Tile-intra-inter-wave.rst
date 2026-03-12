.. meta::
  :description: Intrawave and interwave scheduling with CK Tile
  :keywords: composable kernel, CK, CK Tile, ROCm, API, scheduling, intrawave, interwave

************************************************************
Intrawave and interwave scheduling with CK Tile
************************************************************

Two different scheduling pipelines are available to use with CK Tile's GEMM implementation. 

The interwave and intrawave scheduling pipelines coordinate waves in K dimension accumulation loops. Whether to use the interwave or intrawave pipeline depends on whether the workload is memory-bound or compute-bound.

In interwave scheduling, the K dimension is separated into chunks. The same chunk is loaded into each wave. When the chunk has been loaded into all the waves, the same operation is run on the chunk. 

Once all the waves have completed the operation, the next chunk is loaded into the waves. 

Because all the waves are synchronized, memory accesses are coordinated, and the cache hit rate is optimized, interwave scheduling is best for memory-bound workloads. 

In intrawave scheduling, the full K dimension is loaded into each wave. Each wave runs its own operation on the K dimension independently of the other waves, and without any synchronization with the other waves. The compute unit (CU) is responsible for interleaving the independent operations. 

Because the CU has flexibility in scheduling operations, intrawave scheduling is best for compute-bound workloads.

An example of both interwave and intrawave scheduling can be found in |gemm_utils.hpp|_, which is part of the `GEMM with CK Tile example <https://github.com/ROCm/rocm-libraries/tree/develop/projects/composablekernel/example/ck_tile/03_gemm/README.md>`_.

.. |gemm_utils.hpp| replace:: ``gemm_utils.hpp``
.. _gemm_utils.hpp: https://github.com/ROCm/rocm-libraries/tree/develop/projects/composablekernel/example/ck_tile/03_gemm/gemm_utils.hpp#L37