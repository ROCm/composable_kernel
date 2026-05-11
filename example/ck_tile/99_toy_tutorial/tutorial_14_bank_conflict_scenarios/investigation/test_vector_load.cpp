// Test with actual ds_read_b128 vector loads
// This matches what the real kernel likely uses
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_vector_types.h>
#include <iostream>

#define HIP_CHECK(x) { \
    hipError_t err = x; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

// Test 1: Scalar loads (our previous tests)
__global__ void test_scalar_loads(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;

    // Thread 0 reads column k=0 with scalar loads
    if (tid == 0) {
        float sum = 0;
        int k = 0;

        for (int m = 0; m < 8; m++) {
            _Float16 val = lds[m * 32 + k];  // Scalar load
            sum += (float)val;
        }
        output[0] = sum;
    }
}

// Test 2: Vector loads using float4 (16 bytes = 8 FP16 elements)
__global__ void test_vector_loads_float4(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;

    // Thread 0 reads column k=0 using vector loads
    if (tid == 0) {
        float sum = 0;
        int k = 0;

        // Read 8 FP16 elements as a single 128-bit vector load
        // This should compile to ds_read_b128
        float4* lds_vec = reinterpret_cast<float4*>(lds);

        for (int m = 0; m < 8; m++) {
            // Each row starts at offset m*32
            // We want to read 8 FP16 starting at offset m*32 + k
            // But float4 is 16 bytes, and we're at offset (m*32 + k)*2 bytes
            // This is complex, let's simplify...

            // Read the first 8 FP16 of row m (k=0-7)
            int vec_offset = (m * 32) / 8;  // Divide by 8 because float4 covers 8 FP16
            float4 vec = lds_vec[vec_offset];

            // Extract just the first FP16 (k=0)
            _Float16* ptr = reinterpret_cast<_Float16*>(&vec);
            sum += (float)ptr[k];
        }
        output[0] = sum;
    }
}

// Test 3: Transpose read using vector loads (read 8 M values at once)
// This is closer to what load_tile might do
__global__ void test_transpose_vector_load(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;

    // Thread 0 reads column k=0, ALL 8 M values in one vector load
    if (tid == 0) {
        int k = 0;

        // Try to read m=[0-7], k=0 as a vector
        // But they're NOT contiguous! (m=0,k=0 is offset 0, m=1,k=0 is offset 32)
        // Vector load requires contiguous memory
        // So we MUST do scalar loads for transpose!

        float sum = 0;
        for (int m = 0; m < 8; m++) {
            _Float16 val = lds[m * 32 + k];
            sum += (float)val;
        }
        output[0] = sum;
    }
}

// Test 4: Use inline assembly for ds_read_b128
// This is the actual hardware instruction
__global__ void test_ds_read_b128_inline_asm(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;

    if (tid == 0) {
        float sum = 0;

        // Read first 8 FP16 (16 bytes) from lds using builtin
        // This should compile to ds_read_b128
        uint4* lds_u4 = reinterpret_cast<uint4*>(lds);
        uint4 data = lds_u4[0];

        // Sum the loaded data
        _Float16* ptr = reinterpret_cast<_Float16*>(&data);
        for (int i = 0; i < 8; i++) {
            sum += (float)ptr[i];
        }

        output[0] = sum;
    }
}

// Test 5: Multiple threads using vector loads
// Each thread loads 16 bytes (8 FP16) from consecutive addresses
__global__ void test_multiple_threads_vector(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;

    // First 8 threads each load 16 bytes
    if (tid < 8) {
        // Thread 0: loads offsets 0-7
        // Thread 1: loads offsets 8-15
        // Thread 2: loads offsets 16-23
        // etc.

        int base_offset = tid * 8;
        float4* lds_vec = reinterpret_cast<float4*>(lds);
        float4 vec = lds_vec[tid];  // Each float4 is 8 FP16 elements

        _Float16* ptr = reinterpret_cast<_Float16*>(&vec);
        float sum = 0;
        for (int i = 0; i < 8; i++) {
            sum += (float)ptr[i];
        }

        output[tid] = sum;
    }
}

// Test 6: Phase 0 pattern but trying to use vector loads where possible
__global__ void test_phase0_with_vectors(const _Float16* __restrict__ lds_ptr, float* output)
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

            // Transpose pattern - MUST use scalar loads (not contiguous)
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

int main()
{
    const int N = 256;
    float *d_output;
    HIP_CHECK(hipMalloc(&d_output, N * sizeof(float)));

    std::cout << "=== VECTOR LOAD TESTS ===\n\n";

    std::cout << "Test 1: Scalar loads (baseline)\n";
    std::cout << "  Thread 0 reads column k=0 using 8 scalar loads\n";
    test_scalar_loads<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 2: Vector loads (float4)\n";
    std::cout << "  Thread 0 tries to use float4 vector loads\n";
    test_vector_loads_float4<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 3: Transpose with vector attempt\n";
    std::cout << "  Transpose forces scalar loads (non-contiguous)\n";
    test_transpose_vector_load<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 4: ds_read_b128 inline assembly\n";
    std::cout << "  Direct hardware instruction test\n";
    test_ds_read_b128_inline_asm<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 5: Multiple threads with vector loads\n";
    std::cout << "  8 threads each load 16 bytes (contiguous)\n";
    test_multiple_threads_vector<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 6: Phase 0 pattern (scalar loads required)\n";
    std::cout << "  Lanes {0,1,2,3,20,21,22,23} transpose pattern\n";
    test_phase0_with_vectors<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    HIP_CHECK(hipFree(d_output));

    std::cout << "All tests completed.\n\n";
    std::cout << "To profile:\n";
    std::cout << "  rocprofv3 -i lds_conflict.txt -d vector_results -f csv -- ./test_vector_load\n\n";
    std::cout << "KEY QUESTION:\n";
    std::cout << "  Does using vector loads (ds_read_b128) create different conflict patterns?\n";
    std::cout << "  Or is the real kernel doing something else entirely?\n";

    return 0;
}
