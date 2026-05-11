// Fixed transpose: UNROLLED so all threads execute same instruction together
// This matches how CK's tile operations work!
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <iostream>
#include <vector>

#define HIP_CHECK(x) { \
    hipError_t err = x; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

constexpr int kM = 64;
constexpr int kK = 32;
constexpr int kBlockSize = 256;

// CORRECT transpose: UNROLLED reads so all threads execute together
__global__ void transpose_unrolled_no_xor(
    const _Float16* __restrict__ input,
    _Float16* __restrict__ output,
    int M, int K)
{
    __shared__ _Float16 lds[kM * kK];

    const int tid = threadIdx.x;
    const int block_m = blockIdx.x * kM;

    if (block_m >= M) return;

    for (int k_base = 0; k_base < K; k_base += kK) {

        // Write to LDS
        for (int i = tid; i < kM * kK; i += kBlockSize) {
            int m_local = i / kK;
            int k_local = i % kK;
            int m_global = block_m + m_local;
            int k_global = k_base + k_local;

            if (m_global < M && k_global < K) {
                lds[m_local * kK + k_local] = input[m_global * K + k_global];
            }
        }
        __syncthreads();

        // TRANSPOSE READ - UNROLLED!
        // All threads execute each load instruction together
        if (tid < 64) {
            int k = tid % 32;
            int m_group = tid / 32;  // 0 or 1
            int m_base = m_group * 32;

            // UNROLL the loop - each line executes simultaneously across all threads!
            _Float16 v0 = lds[(m_base + 0) * kK + k];
            _Float16 v1 = lds[(m_base + 1) * kK + k];
            _Float16 v2 = lds[(m_base + 2) * kK + k];
            _Float16 v3 = lds[(m_base + 3) * kK + k];
            _Float16 v4 = lds[(m_base + 4) * kK + k];
            _Float16 v5 = lds[(m_base + 5) * kK + k];
            _Float16 v6 = lds[(m_base + 6) * kK + k];
            _Float16 v7 = lds[(m_base + 7) * kK + k];

            // Write to output
            int k_global = k_base + k;
            int m_global_base = block_m + m_base;

            if (k_global < K) {
                if (m_global_base + 0 < M) output[k_global * M + (m_global_base + 0)] = v0;
                if (m_global_base + 1 < M) output[k_global * M + (m_global_base + 1)] = v1;
                if (m_global_base + 2 < M) output[k_global * M + (m_global_base + 2)] = v2;
                if (m_global_base + 3 < M) output[k_global * M + (m_global_base + 3)] = v3;
                if (m_global_base + 4 < M) output[k_global * M + (m_global_base + 4)] = v4;
                if (m_global_base + 5 < M) output[k_global * M + (m_global_base + 5)] = v5;
                if (m_global_base + 6 < M) output[k_global * M + (m_global_base + 6)] = v6;
                if (m_global_base + 7 < M) output[k_global * M + (m_global_base + 7)] = v7;
            }
        }

        __syncthreads();
    }
}

// Simpler: just the LDS reads, UNROLLED
__global__ void transpose_lds_unrolled(
    _Float16* __restrict__ output)
{
    __shared__ _Float16 lds[kM * kK];

    const int tid = threadIdx.x;

    // Initialize
    for (int i = tid; i < kM * kK; i += kBlockSize) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    // 64 threads, transpose reads, UNROLLED
    if (tid < 64) {
        int k = tid % 32;
        int m_group = tid / 32;
        int m_base = m_group * 32;

        // Each load instruction executes across all 64 threads simultaneously
        _Float16 v0 = lds[(m_base + 0) * kK + k];
        _Float16 v1 = lds[(m_base + 1) * kK + k];
        _Float16 v2 = lds[(m_base + 2) * kK + k];
        _Float16 v3 = lds[(m_base + 3) * kK + k];
        _Float16 v4 = lds[(m_base + 4) * kK + k];
        _Float16 v5 = lds[(m_base + 5) * kK + k];
        _Float16 v6 = lds[(m_base + 6) * kK + k];
        _Float16 v7 = lds[(m_base + 7) * kK + k];

        float sum = (float)(v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7);
        output[tid] = (_Float16)sum;
    }
}

int main()
{
    constexpr int M = 256;
    constexpr int K = 128;

    std::vector<_Float16> h_input(M * K);
    std::vector<_Float16> h_output(K * M);

    for (int i = 0; i < M * K; i++) {
        h_input[i] = (_Float16)(i);
    }

    _Float16 *d_input, *d_output;
    HIP_CHECK(hipMalloc(&d_input, M * K * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&d_output, K * M * sizeof(_Float16)));

    HIP_CHECK(hipMemcpy(d_input, h_input.data(), M * K * sizeof(_Float16), hipMemcpyHostToDevice));

    const int grid_size = (M + kM - 1) / kM;

    std::cout << "=== UNROLLED TRANSPOSE TESTS ===\n\n";

    std::cout << "Test 1: Full transpose UNROLLED (4 blocks, 4 k_iterations)\n";
    std::cout << "  Key: Loads are UNROLLED so all threads execute together\n";
    std::cout << "  Expected: ~7,168 conflicts!\n";
    transpose_unrolled_no_xor<<<grid_size, kBlockSize>>>(d_input, d_output, M, K);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 2: LDS-only UNROLLED\n";
    std::cout << "  64 threads, 8 unrolled loads per thread\n";
    std::cout << "  Expected: Conflicts!\n";
    transpose_lds_unrolled<<<1, kBlockSize>>>(d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));

    std::cout << "CRITICAL DIFFERENCE:\n";
    std::cout << "  WRONG: for (m=0; m<8; m++) { load lds[m*32+k]; }\n";
    std::cout << "         → Sequential per thread, 0 conflicts\n\n";
    std::cout << "  RIGHT: v0=lds[0*32+k]; v1=lds[1*32+k]; ... v7=lds[7*32+k];\n";
    std::cout << "         → All threads execute each load together, CONFLICTS!\n\n";
    std::cout << "To profile:\n";
    std::cout << "  rocprofv3 -i lds_conflict.txt -d unrolled_results -f csv -- ./test_transpose_unrolled\n";

    return 0;
}
