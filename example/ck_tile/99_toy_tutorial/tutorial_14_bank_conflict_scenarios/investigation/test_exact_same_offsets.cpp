// ALL threads read THE EXACT SAME offsets simultaneously
// To verify if that creates the conflicts we see in CK
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

// ALL threads read column k=0 (offsets 0,32,64,96...)
// This creates maximum conflicts!
__global__ void all_threads_same_column(
    _Float16* __restrict__ output)
{
    __shared__ _Float16 lds[64 * 32];

    const int tid = threadIdx.x;

    // Initialize
    for (int i = tid; i < 64*32; i += 256) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    // ALL 64 threads read the SAME column (k=0)
    // offsets: 0, 32, 64, 96, 128, 160, 192, 224
    // banks: 0, 16, 0, 16, 0, 16, 0, 16
    if (tid < 64) {
        // UNROLLED - all 64 threads execute each line together
        _Float16 v0 = lds[0 * 32 + 0];   // All 64 threads read offset 0
        _Float16 v1 = lds[1 * 32 + 0];   // All 64 threads read offset 32
        _Float16 v2 = lds[2 * 32 + 0];   // All 64 threads read offset 64
        _Float16 v3 = lds[3 * 32 + 0];
        _Float16 v4 = lds[4 * 32 + 0];
        _Float16 v5 = lds[5 * 32 + 0];
        _Float16 v6 = lds[6 * 32 + 0];
        _Float16 v7 = lds[7 * 32 + 0];

        output[tid] = (_Float16)(v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7);
    }
}

int main()
{
    _Float16 *d_output;
    HIP_CHECK(hipMalloc(&d_output, 256 * sizeof(_Float16)));

    std::cout << "=== ALL THREADS READ SAME COLUMN ===\n\n";

    std::cout << "Test: 64 threads ALL read column k=0\n";
    std::cout << "  ALL threads read offset 0 together\n";
    std::cout << "  ALL threads read offset 32 together\n";
    std::cout << "  etc.\n";
    std::cout << "  Expected: MAXIMUM CONFLICTS (64-way broadcast)\n";
    all_threads_same_column<<<1, 256>>>(d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    HIP_CHECK(hipFree(d_output));

    std::cout << "If THIS shows conflicts, we know the pattern.\n";
    std::cout << "If THIS shows 0, then something else is going on.\n";

    return 0;
}
