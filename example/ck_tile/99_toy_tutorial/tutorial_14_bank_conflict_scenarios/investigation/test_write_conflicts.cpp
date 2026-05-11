// Test: Do ds_write operations cause bank conflicts?
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

// Test 1: Row-major writes (should be conflict-free)
__global__ void test_row_major_write(const _Float16* __restrict__ input, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    int tid = threadIdx.x;

    // Row-major write pattern (same as store_tile)
    // Each thread writes 8 elements in a row
    // Distribution: thread 0-3 in WF0 write different K values, same M row
    int wf = tid / 64;
    int lane = tid % 64;
    int k1 = wf;
    int k2 = lane % 8;
    int m0 = lane / 8;

    // Write 8 elements per thread
    for (int dm = 0; dm < 8; dm++) {
        int m = m0 * 8 + dm;
        int k = k1 * 8 + k2;
        lds[m * 32 + k] = (_Float16)(tid * 8 + dm);
    }
    __syncthreads();

    // Read back to prevent optimization
    float sum = 0;
    for (int dm = 0; dm < 8; dm++) {
        int m = m0 * 8 + dm;
        int k = k1 * 8 + k2;
        sum += (float)lds[m * 32 + k];
    }
    output[tid] = sum;
}

// Test 2: Column-major writes (should cause conflicts)
__global__ void test_column_major_write(const _Float16* __restrict__ input, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    int tid = threadIdx.x;
    int wf = tid / 64;
    int lane = tid % 64;
    int k1 = wf;
    int k2 = lane % 8;
    int m0 = lane / 8;

    // Column-major write pattern (transpose pattern)
    // This is the OPPOSITE of normal write - each thread writes different K values
    for (int dk = 0; dk < 8; dk++) {
        int k = k1 * 8 + dk;  // Iterate over K, not M
        int m = m0 * 8 + k2;  // Swap m/k pattern
        lds[m * 32 + k] = (_Float16)(tid * 8 + dk);
    }
    __syncthreads();

    // Read back
    float sum = 0;
    for (int dk = 0; dk < 8; dk++) {
        int k = k1 * 8 + dk;
        int m = m0 * 8 + k2;
        sum += (float)lds[m * 32 + k];
    }
    output[tid] = sum;
}

// Test 3: Write + Read sequence (exactly like real kernel)
__global__ void test_write_then_read(const _Float16* __restrict__ input, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    int tid = threadIdx.x;
    int wf = tid / 64;
    int lane = tid % 64;
    int k1 = wf;
    int k2 = lane % 8;
    int m0 = lane / 8;

    // Phase 1: Row-major WRITE (like store_tile from global)
    for (int dm = 0; dm < 8; dm++) {
        int m = m0 * 8 + dm;
        int k = k1 * 8 + k2;
        lds[m * 32 + k] = (_Float16)(tid * 8 + dm);
    }
    __syncthreads();

    // Phase 2: Column-major READ (transpose read)
    float sum = 0;
    int k = k1 * 8 + k2;
    for (int dm = 0; dm < 8; dm++) {
        int m = m0 * 8 + dm;
        sum += (float)lds[m * 32 + k];  // Row-major layout, column-major access
    }
    output[tid] = sum;
}

// Test 4: Only the write portion (no read)
__global__ void test_write_only(const _Float16* __restrict__ input, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    int tid = threadIdx.x;
    int wf = tid / 64;
    int lane = tid % 64;
    int k1 = wf;
    int k2 = lane % 8;
    int m0 = lane / 8;

    // Row-major WRITE only
    for (int dm = 0; dm < 8; dm++) {
        int m = m0 * 8 + dm;
        int k = k1 * 8 + k2;
        lds[m * 32 + k] = (_Float16)(tid * 8 + dm);
    }
    __syncthreads();

    // Prevent optimization - store some LDS value
    output[tid] = (float)lds[tid % (64*32)];
}

// Test 5: Only the read portion (LDS already initialized)
__global__ void test_read_only(float* output)
{
    __shared__ _Float16 lds[64 * 32];

    int tid = threadIdx.x;
    // Initialize uniformly to avoid write conflicts
    lds[tid] = (_Float16)tid;
    if (tid < 64*32 - 256) {
        lds[tid + 256] = (_Float16)(tid + 256);
    }
    __syncthreads();

    int wf = tid / 64;
    int lane = tid % 64;
    int k1 = wf;
    int k2 = lane % 8;
    int m0 = lane / 8;

    // Column-major READ only (transpose pattern)
    float sum = 0;
    int k = k1 * 8 + k2;
    for (int dm = 0; dm < 8; dm++) {
        int m = m0 * 8 + dm;
        sum += (float)lds[m * 32 + k];
    }
    output[tid] = sum;
}

// Test 6: Real kernel pattern with multiple K iterations
__global__ void test_full_k_loop(const _Float16* __restrict__ input, float* output, int k_iters)
{
    __shared__ _Float16 lds[64 * 32];

    int tid = threadIdx.x;
    int wf = tid / 64;
    int lane = tid % 64;
    int k1 = wf;
    int k2 = lane % 8;
    int m0 = lane / 8;

    float total = 0;

    for (int k_iter = 0; k_iter < k_iters; k_iter++) {
        // Write phase (row-major)
        for (int dm = 0; dm < 8; dm++) {
            int m = m0 * 8 + dm;
            int k = k1 * 8 + k2;
            lds[m * 32 + k] = (_Float16)(k_iter * 1000 + tid * 8 + dm);
        }
        __syncthreads();

        // Read phase (column-major - transpose)
        int k = k1 * 8 + k2;
        for (int dm = 0; dm < 8; dm++) {
            int m = m0 * 8 + dm;
            total += (float)lds[m * 32 + k];
        }
        __syncthreads();
    }
    output[tid] = total;
}

int main()
{
    const int N = 256;
    _Float16 *d_input;
    float *d_output;
    HIP_CHECK(hipMalloc(&d_input, 64*32*sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&d_output, N * sizeof(float)));

    std::cout << "=== WRITE CONFLICT TESTS ===\n\n";

    std::cout << "Test 1: Row-major writes (should be conflict-free)\n";
    test_row_major_write<<<1, 256>>>(d_input, d_output);
    HIP_CHECK(hipDeviceSynchronize());

    std::cout << "Test 2: Column-major writes (should cause conflicts)\n";
    test_column_major_write<<<1, 256>>>(d_input, d_output);
    HIP_CHECK(hipDeviceSynchronize());

    std::cout << "Test 3: Write (row) + Read (column) sequence\n";
    test_write_then_read<<<1, 256>>>(d_input, d_output);
    HIP_CHECK(hipDeviceSynchronize());

    std::cout << "Test 4: Write only (row-major)\n";
    test_write_only<<<1, 256>>>(d_input, d_output);
    HIP_CHECK(hipDeviceSynchronize());

    std::cout << "Test 5: Read only (column-major)\n";
    test_read_only<<<1, 256>>>(d_output);
    HIP_CHECK(hipDeviceSynchronize());

    std::cout << "Test 6: Full K loop (4 iterations, 4 blocks)\n";
    test_full_k_loop<<<4, 256>>>(d_input, d_output, 4);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));

    std::cout << "\nAll tests completed.\n";
    std::cout << "Profile with: rocprofv3 --pmc SQ_LDS_BANK_CONFLICT -- ./test_write_conflicts\n";

    return 0;
}
