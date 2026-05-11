// Full transpose kernel in pure HIP - mimics the real CK kernel WITHOUT XOR
// Goal: Recreate the 7,168 LDS bank conflicts we see in profiler
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

// Kernel parameters matching the real CK kernel
constexpr int kM = 64;
constexpr int kK = 32;
constexpr int kBlockSize = 256;

// Full transpose kernel WITHOUT XOR
// Reads [M, K] from global, writes to LDS, reads transposed [K, M], writes to global
__global__ void transpose_kernel_no_xor(
    const _Float16* __restrict__ input,   // [256, 128] row-major
    _Float16* __restrict__ output,        // [128, 256] row-major (transposed)
    int M, int K)
{
    __shared__ _Float16 lds[kM * kK];  // [64, 32] plain row-major layout

    const int tid = threadIdx.x;
    const int block_m = blockIdx.x * kM;  // Which M block (0, 64, 128, 192)

    if (block_m >= M) return;

    // Loop over K dimension (k_base = 0, 8, 16, 24 for K=128/32=4 iterations)
    for (int k_base = 0; k_base < K; k_base += kK) {

        // === STEP 1: WRITE to LDS (row-major [M, K]) ===
        // Each thread writes 8 FP16 elements (16 bytes)
        // This matches the "store_tile" pattern

        for (int i = tid; i < kM * kK; i += kBlockSize) {
            int m_local = i / kK;  // 0-63
            int k_local = i % kK;  // 0-31
            int m_global = block_m + m_local;
            int k_global = k_base + k_local;

            if (m_global < M && k_global < K) {
                // Read from global [M, K]
                _Float16 val = input[m_global * K + k_global];
                // Write to LDS [M, K] - plain row-major, NO XOR
                lds[m_local * kK + k_local] = val;
            }
        }

        __syncthreads();

        // === STEP 2: READ from LDS (transposed [K, M]) ===
        // This is where bank conflicts occur!
        // Each thread reads a COLUMN (transpose pattern)

        // Distribute threads: 256 threads reading 64x32 tile transposed
        // Phase-based execution (8 phases, 8 lanes per phase)
        const int phase = tid / 8;        // 0-31 (but we have 8 phases, so 0-7)
        const int lane_in_phase = tid % 8; // 0-7

        // Only process 64 threads (8 phases × 8 lanes)
        // Phases 0-7, lanes 0-7 in each phase
        if (tid < 64) {
            // Phase 0 lanes: {0, 1, 2, 3, 20, 21, 22, 23} (from our analysis)
            // But let's use simpler: tid 0-63 maps to phases

            // Each thread reads 8 consecutive M values for its assigned K
            int k_idx = lane_in_phase + (phase % 4) * 8;  // Which K column (0-31)
            int m_start = (phase / 4) * 32;                // Which M group

            if (k_idx < kK) {
                for (int dm = 0; dm < 8; dm++) {
                    int m_local = m_start + dm;
                    if (m_local < kM) {
                        // TRANSPOSE READ: read [K, M] from LDS stored as [M, K]
                        // This accesses: lds[m_local * kK + k_idx]
                        _Float16 val = lds[m_local * kK + k_idx];

                        // Write to global [K, M]
                        int k_global = k_base + k_idx;
                        int m_global = block_m + m_local;
                        if (k_global < K && m_global < M) {
                            output[k_global * M + m_global] = val;
                        }
                    }
                }
            }
        }

        __syncthreads();
    }
}

// Simpler version: just do the critical transpose read loop
// No global memory, just LDS reads
__global__ void transpose_lds_only_no_xor(
    _Float16* __restrict__ output,
    int M, int K)
{
    __shared__ _Float16 lds[kM * kK];

    const int tid = threadIdx.x;

    // Initialize LDS with a pattern
    for (int i = tid; i < kM * kK; i += kBlockSize) {
        lds[i] = (_Float16)(i);
    }
    __syncthreads();

    // === CRITICAL SECTION: Transpose reads ===
    // This is where conflicts should occur
    // Do this 4 times to match k_base iterations

    float sum = 0;
    for (int k_base_iter = 0; k_base_iter < 4; k_base_iter++) {

        if (tid < 64) {  // 64 threads active
            const int phase = tid / 8;
            const int lane = tid % 8;

            // Each thread reads its column (8 M values)
            int k = lane + (phase % 4) * 8;
            int m_start = (phase / 4) * 32;

            if (k < kK) {
                for (int dm = 0; dm < 8; dm++) {
                    int m = m_start + dm;
                    if (m < kM) {
                        // TRANSPOSE READ
                        _Float16 val = lds[m * kK + k];
                        sum += (float)val;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = (_Float16)sum;
    }
}

// Exact Phase 0 pattern from our analysis, but ALL 8 dm steps
__global__ void exact_phase0_all_dm(
    _Float16* __restrict__ output)
{
    __shared__ _Float16 lds[kM * kK];

    const int tid = threadIdx.x;

    // Initialize
    for (int i = tid; i < kM * kK; i += kBlockSize) {
        lds[i] = (_Float16)(i);
    }
    __syncthreads();

    // Phase 0 lanes
    const int phase0_lanes[8] = {0, 1, 2, 3, 20, 21, 22, 23};

    float sum = 0;
    for (int i = 0; i < 8; i++) {
        if (tid == phase0_lanes[i]) {
            int k = tid % 8;

            // Read ALL 8 M values (this is the full vector load pattern)
            for (int m = 0; m < 8; m++) {
                _Float16 val = lds[m * kK + k];
                sum += (float)val;
            }
        }
    }

    if (tid < 8) {
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

    const int grid_size = (M + kM - 1) / kM;  // 4 blocks

    std::cout << "=== FULL TRANSPOSE HIP TESTS ===\n\n";

    std::cout << "Test 1: Full transpose kernel (4 blocks, 4 k_iterations)\n";
    std::cout << "  Configuration: M=256, K=128, tile=64x32, blocks=4\n";
    std::cout << "  Pattern: Store [M,K], read transposed [K,M]\n";
    std::cout << "  Expected: ~7,168 conflicts (matching real kernel)\n";
    transpose_kernel_no_xor<<<grid_size, kBlockSize>>>(d_input, d_output, M, K);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 2: LDS-only transpose (simplified)\n";
    std::cout << "  Configuration: 4 k_base iterations, 64 active threads\n";
    std::cout << "  Pattern: Just the transpose read loop\n";
    transpose_lds_only_no_xor<<<1, kBlockSize>>>(d_output, M, K);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 3: Exact Phase 0 pattern (all dm)\n";
    std::cout << "  Configuration: Lanes {0,1,2,3,20,21,22,23}\n";
    std::cout << "  Pattern: Each reads 8 M values\n";
    exact_phase0_all_dm<<<1, kBlockSize>>>(d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));

    std::cout << "All tests completed.\n\n";
    std::cout << "To profile:\n";
    std::cout << "  rocprofv3 -i lds_conflict.txt -d full_transpose_results -f csv -- ./test_full_transpose_hip\n\n";
    std::cout << "CRITICAL TEST:\n";
    std::cout << "  If Test 1 shows 7,168 conflicts → We can recreate it!\n";
    std::cout << "  If Test 1 shows 0 conflicts → Something fundamental is different in real CK kernel\n";
    std::cout << "\n";
    std::cout << "  Real CK kernel: 7,168 conflicts (no XOR)\n";
    std::cout << "  Our hypothesis: Intra-lane conflicts during transpose reads\n";

    return 0;
}
