// CORRECT version: Multiple threads hit SAME bank simultaneously
// This is the actual transpose conflict pattern!
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

constexpr int kM = 64;
constexpr int kK = 32;

// The KEY: when reading m=0,1,2,3,4,5,6,7 for columns k=0 and k=2:
// Thread reads k=0: offsets {0, 32, 64, 96, 128, 160, 192, 224} → banks {0,16,0,16,0,16,0,16}
// Thread reads k=2: offsets {2, 34, 66, 98, 130, 162, 194, 226} → banks {1,17,1,17,1,17,1,17}
// Thread reads k=1: offsets {1, 33, 65, 97, 129, 161, 193, 225} → banks {0,16,0,16,0,16,0,16} <- SAME as k=0!

// So threads reading k=0 and k=1 both hit banks {0,16,0,16...}!
// When they execute simultaneously, CONFLICTS!

__global__ void transpose_real_pattern(
    _Float16* __restrict__ output)
{
    __shared__ _Float16 lds[kM * kK];

    const int tid = threadIdx.x;

    // Initialize
    for (int i = tid; i < kM * kK; i += 256) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    // 32 threads (k=0-31) each read 8 M values
    // When threads k=0,1 both read m=0, they hit SAME bank!
    if (tid < 32) {
        int k = tid;

        // UNROLLED - all threads execute each line together
        _Float16 v0 = lds[0 * kK + k];  // Thread k=0 hits bank 0, Thread k=1 hits bank 0 → CONFLICT!
        _Float16 v1 = lds[1 * kK + k];  // Thread k=0 hits bank 16, Thread k=1 hits bank 16 → CONFLICT!
        _Float16 v2 = lds[2 * kK + k];
        _Float16 v3 = lds[3 * kK + k];
        _Float16 v4 = lds[4 * kK + k];
        _Float16 v5 = lds[5 * kK + k];
        _Float16 v6 = lds[6 * kK + k];
        _Float16 v7 = lds[7 * kK + k];

        float sum = (float)(v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7);
        output[tid] = (_Float16)sum;
    }
}

// Full 64 threads (like CK pattern)
__global__ void transpose_full_64_threads(
    _Float16* __restrict__ output)
{
    __shared__ _Float16 lds[kM * kK];

    const int tid = threadIdx.x;

    // Initialize
    for (int i = tid; i < kM * kK; i += 256) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    // 64 threads total: tid 0-31 read m=[0-7], tid 32-63 read m=[8-15]
    // But wait, our analysis showed phases... let me use phase pattern

    // Phase 0: lanes {0,1,2,3,20,21,22,23}
    // Each reads k = lane_id % 8
    const int phase0_lanes[8] = {0, 1, 2, 3, 20, 21, 22, 23};

    for (int i = 0; i < 8; i++) {
        if (tid == phase0_lanes[i]) {
            int k = tid % 8;

            // UNROLLED
            _Float16 v0 = lds[0 * kK + k];
            _Float16 v1 = lds[1 * kK + k];
            _Float16 v2 = lds[2 * kK + k];
            _Float16 v3 = lds[3 * kK + k];
            _Float16 v4 = lds[4 * kK + k];
            _Float16 v5 = lds[5 * kK + k];
            _Float16 v6 = lds[6 * kK + k];
            _Float16 v7 = lds[7 * kK + k];

            output[tid] = (_Float16)(v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7);
        }
    }
}

int main()
{
    _Float16 *d_output;
    HIP_CHECK(hipMalloc(&d_output, 256 * sizeof(_Float16)));

    std::cout << "=== CORRECT TRANSPOSE CONFLICT PATTERN ===\n\n";

    std::cout << "Test 1: 32 threads (k=0-31), unrolled reads\n";
    std::cout << "  Thread k=0: reads m=[0-7] -> banks {0,16,0,16,0,16,0,16}\n";
    std::cout << "  Thread k=1: reads m=[0-7] -> banks {0,16,0,16,0,16,0,16}\n";
    std::cout << "  SAME BANKS! When executed together -> CONFLICTS!\n";
    std::cout << "  Expected: HIGH conflicts\n";
    transpose_real_pattern<<<1, 256>>>(d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 2: Phase 0 lanes {0,1,2,3,20,21,22,23}\n";
    std::cout << "  Lanes 0,1 both read banks {0,16,0,16...}\n";
    std::cout << "  Expected: Conflicts (or 0 if same-slot optimization applies)\n";
    transpose_full_64_threads<<<1, 256>>>(d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    HIP_CHECK(hipFree(d_output));

    std::cout << "KEY INSIGHT:\n";
    std::cout << "  Adjacent k values (k=0,1) map to SAME banks {0,16,0,16...}\n";
    std::cout << "  When threads read simultaneously, they hit same banks!\n";
    std::cout << "  But offsets 0 and 1 are in SAME SLOT -> FP16 optimization!\n";
    std::cout << "  So we might still see 0 conflicts...\n";

    return 0;
}
