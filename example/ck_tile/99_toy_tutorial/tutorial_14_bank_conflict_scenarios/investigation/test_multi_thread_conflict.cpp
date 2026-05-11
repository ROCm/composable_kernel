// Test: Multiple threads accessing same bank to understand conflict counting
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

// Test 1: ALL 64 threads read from SAME bank 0, SAME slot (offset 0)
// FP16 same-slot optimization should mean 0 conflicts
__global__ void test_same_slot_all_threads(float* output)
{
    __shared__ _Float16 lds[64 * 32];
    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    // ALL threads read from offset 0 (bank 0, slot 0)
    _Float16 val = lds[0];
    output[threadIdx.x] = (float)val;
}

// Test 2: ALL 64 threads read from SAME bank 0 but DIFFERENT slots
// This should cause conflicts!
__global__ void test_same_bank_diff_slots_all_threads(float* output)
{
    __shared__ _Float16 lds[64 * 32];
    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;
    // Each thread reads a different slot in bank 0
    // tid=0: offset 0   -> slot 0   -> bank 0
    // tid=1: offset 64  -> slot 32  -> bank 0
    // tid=2: offset 128 -> slot 64  -> bank 0
    // ...
    _Float16 val = lds[tid * 64];  // All bank 0, but different slots
    output[tid] = (float)val;
}

// Test 3: 8 threads (one phase) read same bank with different slots
__global__ void test_one_phase_same_bank(float* output)
{
    __shared__ _Float16 lds[64 * 32];
    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;
    // Only first 8 threads participate
    if (tid < 8) {
        // Each of 8 threads accesses a different slot in bank 0
        _Float16 val = lds[tid * 64];
        output[tid] = (float)val;
    }
}

// Test 4: 64 threads where each lane reads sequentially (intra-lane pattern)
// Each lane reads 8 elements hitting same bank
__global__ void test_many_intra_lane_patterns(float* output)
{
    __shared__ _Float16 lds[64 * 32];
    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;
    float sum = 0;

    // Each thread reads 8 elements from different slots in same bank
    int base = tid % 32;  // Distribute across banks
    for (int i = 0; i < 8; i++) {
        // base + i*64 = base + i*64 (slot = base/2 + i*32, bank = (base/2)%32)
        _Float16 val = lds[base + i * 64];
        sum += (float)val;
    }
    output[tid] = sum;
}

// Test 5: Match the EXACT pattern from the real kernel
// 4 wavefronts × 8 dm values × 8 phases × 8 lanes
__global__ void test_exact_kernel_pattern(float* output)
{
    __shared__ _Float16 lds[64 * 32];

    // Initialize LDS
    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    // Match the production kernel pattern exactly
    // Each thread reads 8 elements with column-major access (transpose)
    int tid = threadIdx.x;  // 0-255 for 4 wavefronts
    int wf = tid / 64;
    int lane = tid % 64;

    // k distribution: wf gives k1 (0-3), lane%8 gives k2 (0-7)
    // m distribution: lane/8 gives m0 (0-7)
    int k1 = wf;
    int k2 = lane % 8;
    int m0 = lane / 8;

    int k = k1 * 8 + k2;

    float sum = 0;
    // Read 8 dm values (the intra-lane pattern)
    for (int dm = 0; dm < 8; dm++) {
        int m = m0 * 8 + dm;
        int offset = m * 32 + k;  // Row-major [M,K] layout
        _Float16 val = lds[offset];
        sum += (float)val;
    }
    output[tid] = sum;
}

// Test 6: Repeat the exact pattern multiple times (like K-loop iterations)
__global__ void test_repeated_exact_pattern(float* output, int iterations)
{
    __shared__ _Float16 lds[64 * 32];

    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;
    int wf = tid / 64;
    int lane = tid % 64;
    int k1 = wf;
    int k2 = lane % 8;
    int m0 = lane / 8;
    int k = k1 * 8 + k2;

    float sum = 0;

    for (int iter = 0; iter < iterations; iter++) {
        for (int dm = 0; dm < 8; dm++) {
            int m = m0 * 8 + dm;
            int offset = m * 32 + k;
            _Float16 val = lds[offset];
            sum += (float)val;
        }
        __syncthreads();  // Barrier between iterations
    }
    output[tid] = sum;
}

int main()
{
    const int N = 256;
    float *d_output;
    HIP_CHECK(hipMalloc(&d_output, N * sizeof(float)));

    std::cout << "=== MULTI-THREAD CONFLICT TESTS ===\n\n";

    std::cout << "Test 1: All 64 threads read SAME slot (bank 0, slot 0)\n";
    std::cout << "  Expected: 0 conflicts (FP16 same-slot optimization)\n";
    test_same_slot_all_threads<<<1, 64>>>(d_output);
    HIP_CHECK(hipDeviceSynchronize());

    std::cout << "Test 2: All 64 threads read SAME bank, DIFFERENT slots\n";
    std::cout << "  Pattern: tid*64 -> slots 0,32,64,96... (all bank 0)\n";
    std::cout << "  Expected: HIGH conflicts (64 slots in one bank)\n";
    test_same_bank_diff_slots_all_threads<<<1, 64>>>(d_output);
    HIP_CHECK(hipDeviceSynchronize());

    std::cout << "Test 3: 8 threads (one phase) read same bank, diff slots\n";
    std::cout << "  Expected: conflicts (8 slots in one bank)\n";
    test_one_phase_same_bank<<<1, 64>>>(d_output);
    HIP_CHECK(hipDeviceSynchronize());

    std::cout << "Test 4: 64 threads, each with 8-read intra-lane pattern\n";
    std::cout << "  Expected: Intra-lane conflicts × 64\n";
    test_many_intra_lane_patterns<<<1, 64>>>(d_output);
    HIP_CHECK(hipDeviceSynchronize());

    std::cout << "Test 5: Exact kernel pattern (256 threads = 4 WFs)\n";
    std::cout << "  Expected: Should match kernel conflict count\n";
    test_exact_kernel_pattern<<<1, 256>>>(d_output);
    HIP_CHECK(hipDeviceSynchronize());

    std::cout << "Test 6: Repeated pattern (4 K iterations like real kernel)\n";
    std::cout << "  Expected: 4× Test 5 conflicts\n";
    test_repeated_exact_pattern<<<1, 256>>>(d_output, 4);
    HIP_CHECK(hipDeviceSynchronize());

    // Also run with 4 blocks to match real kernel
    std::cout << "Test 7: 4 blocks × 256 threads (matches M=256 kernel)\n";
    std::cout << "  Expected: 4× Test 5 conflicts\n";
    test_exact_kernel_pattern<<<4, 256>>>(d_output);
    HIP_CHECK(hipDeviceSynchronize());

    std::cout << "Test 8: 4 blocks × repeated pattern (full match)\n";
    std::cout << "  Expected: Should match real kernel's 7,168 conflicts\n";
    test_repeated_exact_pattern<<<4, 256>>>(d_output, 4);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_output));

    std::cout << "\nAll tests completed.\n";
    std::cout << "Profile with: rocprofv3 --pmc SQ_LDS_BANK_CONFLICT -- ./test_multi_thread_conflict\n";

    return 0;
}
