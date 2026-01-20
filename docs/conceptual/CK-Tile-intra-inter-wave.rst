.. meta::
  :description: Intrawave and interwave scheduling with CK Tile
  :keywords: composable kernel, CK, CK Tile, ROCm, API, scheduling, intrawave, interwave

************************************************************
Intrawave and interwave scheduling with CK Tile
************************************************************

Two different scheduling pipelines are available to use with CK Tile's GEMM implementation. 

The interwave and intrawave scheduling pipelines coordinate waves in K dimension accumulation loops. Whether to use the interwave or intrawave pipeline depends on whether the workload is memory-bound or compute-bound.

In interwave scheduling, the K dimension is separated into chunks. The same chunk is loaded into each wave and all the waves run the same operation on the chunk. The operation is only run once all the waves have loaded the chunk, and the next chunk is loaded only after all the waves have finished running their operation. 

Because all the waves are synchronized, memory accesses are coordinated and the cache hit rate is optimized, interwave scheduling is best for memory-bound workloads. 

In intrawave scheduling, the full k dimensions are loaded into each wave. The waves then all run their operations on the entire K dimension independently and without synchronization. The CU then interleaves the instructions from all the waves. 

Because the CU has flexibility in scheduling operations, intrawave scheduling is best for compute-bound workloads.
