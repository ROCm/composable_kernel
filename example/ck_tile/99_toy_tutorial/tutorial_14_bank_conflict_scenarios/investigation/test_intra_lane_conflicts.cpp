// Test INTRA-lane conflicts - one thread accessing same bank multiple times with different slots
// This is the pattern we see in transpose WITHOUT XOR
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

// Test 1: ONE thread reads column k=0 (hits banks {0,16,0,16,0,16,0,16})
// This is the INTRA-lane pattern WITHOUT XOR
__global__ void test_intra_lane_transpose_pattern(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;

    // Only thread 0 does the reads
    if (tid == 0) {
        float sum = 0;
        int k = 0;  // Read column 0

        // Read 8 consecutive M values (transpose pattern)
        for (int m = 0; m < 8; m++) {
            // m=0: offset 0   -> slot 0   -> bank 0
            // m=1: offset 32  -> slot 16  -> bank 16
            // m=2: offset 64  -> slot 32  -> bank 0   <- Same bank as m=0, different slot!
            // m=3: offset 96  -> slot 48  -> bank 16  <- Same bank as m=1, different slot!
            // m=4: offset 128 -> slot 64  -> bank 0   <- Same bank, different slot!
            // m=5: offset 160 -> slot 80  -> bank 16  <- Same bank, different slot!
            // m=6: offset 192 -> slot 96  -> bank 0   <- Same bank, different slot!
            // m=7: offset 224 -> slot 112 -> bank 16  <- Same bank, different slot!

            _Float16 val = lds[m * 32 + k];
            sum += (float)val;
        }
        output[0] = sum;
    }
}

// Test 2: ONE thread reads 8 elements that hit ALL DIFFERENT banks
// No conflicts expected
__global__ void test_intra_lane_no_conflicts(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;

    if (tid == 0) {
        float sum = 0;

        // Read elements that map to different banks
        // offset 0   -> slot 0  -> bank 0
        // offset 2   -> slot 1  -> bank 1
        // offset 4   -> slot 2  -> bank 2
        // offset 6   -> slot 3  -> bank 3
        // offset 8   -> slot 4  -> bank 4
        // offset 10  -> slot 5  -> bank 5
        // offset 12  -> slot 6  -> bank 6
        // offset 14  -> slot 7  -> bank 7
        for (int i = 0; i < 8; i++) {
            _Float16 val = lds[i * 2];
            sum += (float)val;
        }
        output[0] = sum;
    }
}

// Test 3: ONE thread reads 8 elements, SAME bank, DIFFERENT slots
// Should show maximum intra-lane conflicts
__global__ void test_intra_lane_same_bank_diff_slots(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;

    if (tid == 0) {
        float sum = 0;

        // All access bank 0, but different slots
        // offset 0   -> slot 0  -> bank 0
        // offset 64  -> slot 32 -> bank 0 (32 % 32 = 0)
        // offset 128 -> slot 64 -> bank 0 (64 % 32 = 0)
        // offset 192 -> slot 96 -> bank 0 (96 % 32 = 0)
        // offset 256 -> slot 128 -> bank 0 (128 % 32 = 0)
        // offset 320 -> slot 160 -> bank 0 (160 % 32 = 0)
        // offset 384 -> slot 192 -> bank 0 (192 % 32 = 0)
        // offset 448 -> slot 224 -> bank 0 (224 % 32 = 0)

        for (int i = 0; i < 8; i++) {
            _Float16 val = lds[i * 64];  // Stride of 64
            sum += (float)val;
        }
        output[0] = sum;
    }
}

// Test 4: ALL 8 threads (Phase 0 lanes) read their columns
// This is the FULL Phase 0 pattern
__global__ void test_full_phase0_pattern(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    const int phase0_lanes[8] = {0, 1, 2, 3, 20, 21, 22, 23};
    int tid = threadIdx.x;

    float sum = 0;
    for (int i = 0; i < 8; i++) {
        if (tid == phase0_lanes[i]) {
            int k = tid % 8;  // Each lane has its own k column

            // Each lane reads 8 M values (transpose pattern)
            for (int m = 0; m < 8; m++) {
                _Float16 val = lds[m * 32 + k];
                sum += (float)val;
            }
        }
    }

    if (tid < 8) {
        output[tid] = sum;
    }
}

// Test 5: SCALED - multiple threads each doing the transpose pattern
// This simulates multiple lanes all having intra-lane conflicts
__global__ void test_scaled_intra_conflicts(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;

    // First 32 threads each read a column (k = 0-31)
    if (tid < 32) {
        float sum = 0;
        int k = tid;

        // Each thread reads 8 M values (same intra-lane pattern)
        for (int m = 0; m < 8; m++) {
            _Float16 val = lds[m * 32 + k];
            sum += (float)val;
        }
        output[tid] = sum;
    }
}

int main()
{
    const int N = 256;
    float *d_output;
    HIP_CHECK(hipMalloc(&d_output, N * sizeof(float)));

    std::cout << "=== INTRA-LANE CONFLICT TESTS ===\n\n";

    std::cout << "Test 1: ONE thread, transpose pattern (column k=0)\n";
    std::cout << "  Pattern: Thread 0 reads m=[0-7], k=0\n";
    std::cout << "  Banks: {0, 16, 0, 16, 0, 16, 0, 16}\n";
    std::cout << "  Slots: {0, 16, 32, 48, 64, 80, 96, 112} (different slots!)\n";
    std::cout << "  Expected: INTRA-lane conflicts (bank 0: 4 slots, bank 16: 4 slots)\n";
    test_intra_lane_transpose_pattern<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 2: ONE thread, different banks (baseline)\n";
    std::cout << "  Pattern: Thread 0 reads offsets {0,2,4,6,8,10,12,14}\n";
    std::cout << "  Banks: {0,1,2,3,4,5,6,7} (all different)\n";
    std::cout << "  Expected: 0 conflicts\n";
    test_intra_lane_no_conflicts<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 3: ONE thread, same bank, different slots\n";
    std::cout << "  Pattern: Thread 0 reads offsets {0,64,128,192,256,320,384,448}\n";
    std::cout << "  Banks: All bank 0, slots {0,32,64,96,128,160,192,224}\n";
    std::cout << "  Expected: HIGH intra-lane conflicts (8 different slots)\n";
    test_intra_lane_same_bank_diff_slots<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 4: Full Phase 0 pattern (8 lanes)\n";
    std::cout << "  Pattern: Lanes {0,1,2,3,20,21,22,23} each read their column\n";
    std::cout << "  Each lane has intra-lane pattern like Test 1\n";
    std::cout << "  Expected: 8 lanes × intra-lane conflicts\n";
    test_full_phase0_pattern<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 5: Scaled - 32 threads each with transpose pattern\n";
    std::cout << "  Pattern: 32 threads (k=0-31) each read 8 M values\n";
    std::cout << "  Expected: 32 × intra-lane conflicts\n";
    test_scaled_intra_conflicts<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    HIP_CHECK(hipFree(d_output));

    std::cout << "All tests completed.\n\n";
    std::cout << "To profile:\n";
    std::cout << "  rocprofv3 -i lds_conflict.txt -d intra_results -f csv -- ./test_intra_lane_conflicts\n\n";
    std::cout << "KEY HYPOTHESIS:\n";
    std::cout << "  Test 1 should show conflicts (intra-lane, different slots)\n";
    std::cout << "  Test 4 should show 8× Test 1's conflicts\n";
    std::cout << "  Test 5 should show 32× Test 1's conflicts\n";
    std::cout << "  This is where the 7,168 conflicts come from!\n";

    return 0;
}
