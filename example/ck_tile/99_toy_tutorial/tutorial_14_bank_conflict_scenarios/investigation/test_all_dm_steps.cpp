// Test ALL 8 dm steps to find where conflicts actually occur
// This matches the real transpose pattern that reads 8 M values per lane
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

// Test dm=0 only (same as before)
__global__ void test_dm0_only(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    const int phase0_lanes[8] = {0, 1, 2, 3, 20, 21, 22, 23};
    int tid = threadIdx.x;

    for (int i = 0; i < 8; i++) {
        if (tid == phase0_lanes[i]) {
            int m = 0;           // Only dm=0
            int k = tid % 8;
            _Float16 val = lds[m * 32 + k];
            output[tid] = (float)val;
        }
    }
}

// Test dm=1 only
__global__ void test_dm1_only(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    const int phase0_lanes[8] = {0, 1, 2, 3, 20, 21, 22, 23};
    int tid = threadIdx.x;

    for (int i = 0; i < 8; i++) {
        if (tid == phase0_lanes[i]) {
            int m = 1;           // Only dm=1
            int k = tid % 8;
            _Float16 val = lds[m * 32 + k];
            output[tid] = (float)val;
        }
    }
}

// Test dm=2 only
__global__ void test_dm2_only(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    const int phase0_lanes[8] = {0, 1, 2, 3, 20, 21, 22, 23};
    int tid = threadIdx.x;

    for (int i = 0; i < 8; i++) {
        if (tid == phase0_lanes[i]) {
            int m = 2;           // Only dm=2
            int k = tid % 8;
            _Float16 val = lds[m * 32 + k];
            output[tid] = (float)val;
        }
    }
}

// Test ALL 8 dm steps (real pattern)
__global__ void test_all_dm_steps(const _Float16* __restrict__ lds_ptr, float* output)
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
            int k = tid % 8;

            // Read all 8 M values (dm=0 through dm=7)
            for (int dm = 0; dm < 8; dm++) {
                int m = dm;
                _Float16 val = lds[m * 32 + k];
                sum += (float)val;
            }
        }
    }

    if (tid < 8) {
        output[tid] = sum;
    }
}

// Test reading entire column (like real transpose)
// Each lane reads m=[m_start, m_start+7] for its assigned k
__global__ void test_full_column(const _Float16* __restrict__ lds_ptr, float* output)
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
            int k2_idx = tid % 8;
            int m0_idx = tid / 8;
            int k = k2_idx;
            int m_start = m0_idx * 8;

            // Read 8 consecutive M values
            for (int dm = 0; dm < 8; dm++) {
                int m = m_start + dm;
                // Lane 0: m=[0-7], k=0
                // Lane 1: m=[0-7], k=1
                // Lane 20: m=[16-23], k=4
                // etc.
                _Float16 val = lds[m * 32 + k];
                sum += (float)val;
            }
        }
    }

    if (tid < 8) {
        output[tid] = sum;
    }
}

int main()
{
    const int N = 256;
    float *d_output;
    HIP_CHECK(hipMalloc(&d_output, N * sizeof(float)));

    std::cout << "=== TESTING ALL DM STEPS ===\n\n";

    std::cout << "Test 1: dm=0 only\n";
    std::cout << "  Pattern: Lanes {0,1,2,3,20,21,22,23} read m=0, k={0-7}\n";
    std::cout << "  Expected: 0 conflicts (same-slot pairs)\n";
    test_dm0_only<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 2: dm=1 only\n";
    std::cout << "  Pattern: Lanes {0,1,2,3,20,21,22,23} read m=1, k={0-7}\n";
    std::cout << "  Check if still same-slot pairs\n";
    test_dm1_only<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 3: dm=2 only\n";
    std::cout << "  Pattern: Lanes {0,1,2,3,20,21,22,23} read m=2, k={0-7}\n";
    std::cout << "  Check if pattern changes\n";
    test_dm2_only<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 4: ALL dm steps (dm=0-7)\n";
    std::cout << "  Pattern: Each lane reads 8 M values\n";
    std::cout << "  This is the REAL transpose loop!\n";
    test_all_dm_steps<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 5: Full column read (with proper m_start)\n";
    std::cout << "  Pattern: Lane 0 reads m=[0-7], Lane 20 reads m=[16-23], etc.\n";
    std::cout << "  Exact match to real distribution\n";
    test_full_column<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    HIP_CHECK(hipFree(d_output));

    std::cout << "All tests completed.\n\n";
    std::cout << "To profile:\n";
    std::cout << "  rocprofv3 -i lds_conflict.txt -d dm_test_results -f csv -- ./test_all_dm_steps\n\n";
    std::cout << "This will show conflicts per dm value!\n";

    return 0;
}
