// Test multi-WF conflicts with REPEATED accesses (loop)
// Question: Does conflict behavior differ between single vs repeated access?
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

// Test 1: Two WFs, single access (baseline - should be 0 conflicts)
__global__ void two_wf_single_access(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;

    if (tid < 128) {
        int wf = tid / 64;
        int lane = tid % 64;

        if (lane < 8) {
            int k = lane * 2;  // Banks 0-7
            int m = 0;

            // SINGLE ACCESS
            _Float16 val = lds[m * 32 + k];
            output[tid] = (float)val;
        }
    }
}

// Test 2: Two WFs, 10 repeated accesses in loop
__global__ void two_wf_loop_10_iters(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;

    if (tid < 128) {
        int wf = tid / 64;
        int lane = tid % 64;

        if (lane < 8) {
            int k = lane * 2;  // Banks 0-7
            int m = 0;

            float sum = 0.0f;

            // LOOP: 10 ITERATIONS accessing same banks
            #pragma unroll
            for (int iter = 0; iter < 10; iter++) {
                _Float16 val = lds[m * 32 + k];
                sum += (float)val;
            }

            output[tid] = sum;
        }
    }
}

// Test 3: Two WFs, 100 repeated accesses
__global__ void two_wf_loop_100_iters(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;

    if (tid < 128) {
        int wf = tid / 64;
        int lane = tid % 64;

        if (lane < 8) {
            int k = lane * 2;  // Banks 0-7
            int m = 0;

            float sum = 0.0f;

            // LOOP: 100 ITERATIONS
            #pragma unroll 10
            for (int iter = 0; iter < 100; iter++) {
                _Float16 val = lds[m * 32 + k];
                sum += (float)val;
            }

            output[tid] = sum;
        }
    }
}

// Test 4: Two WFs, varying addresses in loop (simulating real K-loop)
__global__ void two_wf_loop_varying_addresses(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;

    if (tid < 128) {
        int wf = tid / 64;
        int lane = tid % 64;

        if (lane < 8) {
            float sum = 0.0f;

            // Simulate K-loop: access different rows (m) but same column (k)
            // This is like our actual GEMM transpose pattern
            #pragma unroll
            for (int m = 0; m < 16; m++) {
                int k = lane * 2;  // Same banks each iteration
                _Float16 val = lds[m * 32 + k];
                sum += (float)val;
            }

            output[tid] = sum;
        }
    }
}

// Test 5: Four WFs, single access
__global__ void four_wf_single_access(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;

    if (tid < 256) {
        int wf = tid / 64;
        int lane = tid % 64;

        if (lane < 8) {
            int k = lane * 2;  // Banks 0-7
            int m = 0;

            _Float16 val = lds[m * 32 + k];
            output[tid] = (float)val;
        }
    }
}

// Test 6: Four WFs, loop 10 iterations
__global__ void four_wf_loop_10_iters(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;

    if (tid < 256) {
        int wf = tid / 64;
        int lane = tid % 64;

        if (lane < 8) {
            int k = lane * 2;  // Banks 0-7
            int m = 0;

            float sum = 0.0f;

            #pragma unroll
            for (int iter = 0; iter < 10; iter++) {
                _Float16 val = lds[m * 32 + k];
                sum += (float)val;
            }

            output[tid] = sum;
        }
    }
}

int main()
{
    const int N = 256;
    float *d_output;
    HIP_CHECK(hipMalloc(&d_output, N * sizeof(float)));

    std::cout << "=== MULTI-WF CONFLICT TESTS WITH LOOPS ===\n\n";
    std::cout << "Testing if conflict behavior changes with repeated accesses...\n\n";

    std::cout << "Test 1: Two WFs, single access (baseline)\n";
    std::cout << "  Expected: 0 conflicts (each WF: 0 internal, no inter-WF)\n";
    two_wf_single_access<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 2: Two WFs, 10 iterations (repeated access to same banks)\n";
    std::cout << "  Expected: 0 conflicts if hardware handles loops well\n";
    std::cout << "           >0 conflicts if repeated access triggers different behavior\n";
    two_wf_loop_10_iters<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 3: Two WFs, 100 iterations (heavy repeated access)\n";
    std::cout << "  Expected: Same as Test 2, or conflicts might accumulate\n";
    two_wf_loop_100_iters<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 4: Two WFs, varying addresses (simulating K-loop)\n";
    std::cout << "  Pattern: Access different rows (m=0-15) but same banks\n";
    std::cout << "  Expected: 0 conflicts (different slots each iteration)\n";
    two_wf_loop_varying_addresses<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 5: Four WFs, single access (baseline)\n";
    std::cout << "  Expected: 0 conflicts\n";
    four_wf_single_access<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 6: Four WFs, 10 iterations\n";
    std::cout << "  Expected: Same as Test 2 but with 4 WFs\n";
    four_wf_loop_10_iters<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    HIP_CHECK(hipFree(d_output));

    std::cout << "All tests completed.\n\n";
    std::cout << "=== KEY QUESTIONS ===\n\n";
    std::cout << "1. Do conflicts change with repeated access?\n";
    std::cout << "   Compare Test 1 vs Test 2 vs Test 3\n\n";
    std::cout << "2. Does loop count matter?\n";
    std::cout << "   Compare Test 2 (10 iters) vs Test 3 (100 iters)\n\n";
    std::cout << "3. Does varying addresses change behavior?\n";
    std::cout << "   Compare Test 2 (same address) vs Test 4 (varying addresses)\n\n";
    std::cout << "4. Do multi-WF conflicts appear with loops?\n";
    std::cout << "   Compare Test 1/5 (single) vs Test 2/6 (loop)\n\n";
    std::cout << "=== PROFILING ===\n\n";
    std::cout << "rocprofv3 -i lds_metrics.txt -o loop_results -- ./test_inter_wf_with_loop\n";

    return 0;
}
