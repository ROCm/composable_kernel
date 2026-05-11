// Minimal test to verify profiler accuracy
#include <hip/hip_runtime.h>
#include <iostream>

// Simple kernel: each thread in a warp reads the SAME LDS address
__global__ void test_max_conflict()
{
    __shared__ float lds[32];

    int tid = threadIdx.x;

    // All 64 threads in a wavefront read from lds[0] -> 64-way conflict!
    float val = lds[0];

    // Prevent optimization
    if(val > 1e10) {
        lds[tid % 32] = val;
    }
}

// Simple kernel: each thread reads different LDS address -> NO conflicts
__global__ void test_no_conflict()
{
    __shared__ float lds[64];

    int tid = threadIdx.x;

    // Each thread reads its own address -> no conflict
    float val = lds[tid];

    // Prevent optimization
    if(val > 1e10) {
        lds[tid] = val;
    }
}

int main()
{
    std::cout << "Test 1: Maximum conflict (all threads read same address)\n";
    test_max_conflict<<<1, 64>>>();
    (void)hipDeviceSynchronize();

    std::cout << "Test 2: No conflict (each thread reads different address)\n";
    test_no_conflict<<<1, 64>>>();
    (void)hipDeviceSynchronize();

    std::cout << "\nProfile these with:\n";
    std::cout << "rocprofv3 -f csv --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS -- ./bin/test_simple_conflict\n";

    return 0;
}
