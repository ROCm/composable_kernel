// Test if conflicts only count when MULTIPLE threads access SIMULTANEOUSLY
// Not when one thread makes sequential accesses
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <iostream>

#define HIP_CHECK(x) { \
    hipError_t err = x; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

// Test 1: ONE thread, sequential reads (our previous tests)
__global__ void one_thread_sequential(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;

    if (tid == 0) {
        float sum = 0;
        // Sequential reads hitting bank 0 multiple times
        // offsets: 0, 64, 128, 192, 256, 320, 384, 448
        // banks: 0, 0, 0, 0, 0, 0, 0, 0 (all bank 0, different slots!)
        for (int i = 0; i < 8; i++) {
            _Float16 val = lds[i * 64];  // Stride of 64
            sum += (float)val;
        }
        output[0] = sum;
    }
}

// Test 2: MULTIPLE threads, simultaneous reads of same bank
// Each thread reads one element, but all hit bank 0 at different slots
__global__ void multiple_threads_simultaneous_same_bank(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;

    // 8 threads each read from bank 0 (different slots) SIMULTANEOUSLY
    if (tid < 8) {
        // Thread 0: offset 0   -> slot 0  -> bank 0
        // Thread 1: offset 64  -> slot 32 -> bank 0
        // Thread 2: offset 128 -> slot 64 -> bank 0
        // Thread 3: offset 192 -> slot 96 -> bank 0
        // etc.
        // All access bank 0, different slots, AT THE SAME TIME!

        int offset = tid * 64;
        _Float16 val = lds[offset];
        output[tid] = (float)val;
    }
}

// Test 3: WARP of threads (32 threads), all read from different offsets
// But designed so they hit same banks
__global__ void warp_simultaneous_conflicts(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;

    // First 32 threads (one warp)
    if (tid < 32) {
        // Design: threads 0-15 hit banks 0-15
        //         threads 16-31 hit banks 0-15 again (conflicts!)
        int offset = (tid % 16) * 64 + (tid / 16) * 2;
        _Float16 val = lds[offset];
        output[tid] = (float)val;
    }
}

// Test 4: Real transpose pattern - 64 threads each reading their column
// This matches the real kernel!
__global__ void real_transpose_64_threads(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;

    // 64 threads, each reads a column (matching phase/lane structure)
    if (tid < 64) {
        int k = tid % 32;  // Which column (0-31)
        float sum = 0;

        // Each thread reads 8 M values - SEQUENTIAL for each thread
        // BUT all 64 threads do this AT THE SAME TIME
        for (int m = 0; m < 8; m++) {
            _Float16 val = lds[m * 32 + k];
            sum += (float)val;
        }
        output[tid] = sum;
    }
}

// Test 5: Explicit synchronization - all threads read together
__global__ void synchronized_simultaneous(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;

    if (tid < 64) {
        int k = tid % 32;

        // UNROLLED so all threads execute same instruction
        _Float16 v0 = lds[0 * 32 + k];  // All threads execute THIS together
        __syncthreads();
        _Float16 v1 = lds[1 * 32 + k];  // Then THIS together
        __syncthreads();
        _Float16 v2 = lds[2 * 32 + k];
        __syncthreads();
        _Float16 v3 = lds[3 * 32 + k];
        __syncthreads();
        _Float16 v4 = lds[4 * 32 + k];
        __syncthreads();
        _Float16 v5 = lds[5 * 32 + k];
        __syncthreads();
        _Float16 v6 = lds[6 * 32 + k];
        __syncthreads();
        _Float16 v7 = lds[7 * 32 + k];

        output[tid] = (float)(v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7);
    }
}

int main()
{
    const int N = 256;
    float *d_output;
    HIP_CHECK(hipMalloc(&d_output, N * sizeof(float)));

    std::cout << "=== SIMULTANEOUS ACCESS TESTS ===\n\n";

    std::cout << "Test 1: ONE thread, sequential reads (bank 0 × 8)\n";
    std::cout << "  Hypothesis: 0 conflicts (no simultaneous contention)\n";
    one_thread_sequential<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 2: MULTIPLE threads, simultaneous reads of bank 0\n";
    std::cout << "  8 threads hit bank 0 (different slots) AT THE SAME TIME\n";
    std::cout << "  Hypothesis: HIGH conflicts (true simultaneous contention!)\n";
    multiple_threads_simultaneous_same_bank<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 3: One warp (32 threads) with designed conflicts\n";
    std::cout << "  Hypothesis: Conflicts from simultaneous access\n";
    warp_simultaneous_conflicts<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 4: Real transpose - 64 threads each read column\n";
    std::cout << "  Each thread sequentially reads its column\n";
    std::cout << "  Hypothesis: Depends on whether sequential or simultaneous matters\n";
    real_transpose_64_threads<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 5: Synchronized/unrolled loads\n";
    std::cout << "  All threads execute same instruction together\n";
    std::cout << "  Hypothesis: Should show conflicts if simultaneity matters\n";
    synchronized_simultaneous<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    HIP_CHECK(hipFree(d_output));

    std::cout << "KEY HYPOTHESIS:\n";
    std::cout << "  LDS_BANK_CONFLICT only counts when MULTIPLE threads\n";
    std::cout << "  access the same bank AT THE SAME TIME (same instruction).\n";
    std::cout << "  NOT when one thread makes sequential accesses.\n\n";
    std::cout << "  If Test 2 shows conflicts but Test 1 shows 0,\n";
    std::cout << "  this hypothesis is CONFIRMED!\n";

    return 0;
}
