// Test if hardware can efficiently read 2 FP16 elements from the same 4-byte bank slot
// Expected: Same-slot reads should have NO additional conflicts vs sequential reads
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

// Test 1: Read 2 FP16 from SAME 4-byte slot (adjacent elements in row-major)
// Row-major [64,32]: elements at (m,k) and (m,k+1) are adjacent -> same slot
__global__ void same_slot_reads(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    // Initialize LDS with pattern
    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    // Each thread reads 2 adjacent FP16 (same slot)
    // Thread 0 reads (0,0) and (0,1) -> offsets 0,1 -> both in slot 0
    int tid = threadIdx.x;
    int m = tid;
    int k0 = 0;
    int k1 = 1;

    _Float16 val0 = lds[m * 32 + k0];
    _Float16 val1 = lds[m * 32 + k1];

    output[tid] = (float)val0 + (float)val1;
}

// Test 2: Read 2 FP16 from DIFFERENT 4-byte slots (strided)
// Elements at (m,k) and (m+1,k) are 32 elements apart -> different slots
__global__ void different_slot_reads(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    // Initialize LDS with pattern
    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    // Each thread reads 2 strided FP16 (different slots)
    // Thread 0 reads (0,0) and (1,0) -> offsets 0,32 -> slots 0,16 (different banks)
    int tid = threadIdx.x;
    int m0 = tid;
    int m1 = tid + 1;
    int k = 0;

    if (m1 < 64) {
        _Float16 val0 = lds[m0 * 32 + k];
        _Float16 val1 = lds[m1 * 32 + k];

        output[tid] = (float)val0 + (float)val1;
    }
}

// Test 3: Read 2 FP16 from SAME bank but DIFFERENT slots
// This should cause actual conflict
__global__ void same_bank_different_slots(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    // Initialize LDS
    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    // Read elements that map to same bank but different slots
    // (0,0) -> offset 0 -> slot 0 -> bank 0
    // (2,0) -> offset 64 -> slot 32 -> bank 0 (32 % 32 = 0)
    int tid = threadIdx.x;
    int m0 = tid * 2;
    int m1 = tid * 2 + 2;
    int k = 0;

    if (m1 < 64) {
        _Float16 val0 = lds[m0 * 32 + k];
        _Float16 val1 = lds[m1 * 32 + k];

        output[tid] = (float)val0 + (float)val1;
    }
}

int main()
{
    const int N = 256;
    float *d_output;
    HIP_CHECK(hipMalloc(&d_output, N * sizeof(float)));

    std::cout << "Test 1: Same-slot reads (2 FP16 from same 4-byte slot)\n";
    std::cout << "  Expected: LOW conflicts (hardware services both in one cycle)\n";
    same_slot_reads<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 2: Different-slot reads (2 FP16 from different banks)\n";
    std::cout << "  Expected: NO conflicts (different banks)\n";
    different_slot_reads<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 3: Same-bank different-slot reads\n";
    std::cout << "  Expected: HIGH conflicts (2-way bank conflict)\n";
    same_bank_different_slots<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    HIP_CHECK(hipFree(d_output));

    std::cout << "All tests completed successfully.\n";
    std::cout << "\nTo profile, run:\n";
    std::cout << "  rocprofv3 --plugin file --plugin-version 2 \\\n";
    std::cout << "    -i input_same_slot.txt ./test_fp16_same_bank\n";
    std::cout << "\nwhere input_same_slot.txt contains:\n";
    std::cout << "  kernel: same_slot_reads\n";
    std::cout << "  metric: SQ_ACCUM_PREV_HIRES\n";
    std::cout << "  LDS_BANK_CONFLICT {}: $denom\n";

    return 0;
}
