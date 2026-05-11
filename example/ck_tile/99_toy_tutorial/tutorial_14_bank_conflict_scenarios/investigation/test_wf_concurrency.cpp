// Test if multiple wavefronts execute concurrently or serially
// Critical question: Are WFs actually active at the same time?
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <iostream>
#include <chrono>

#define HIP_CHECK(x) { \
    hipError_t err = x; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

// Helper: Busy-wait computation to create measurable work
__device__ float busy_work(int iterations) {
    float result = 0.0f;
    for (int i = 0; i < iterations; i++) {
        result += sqrtf((float)i + 1.0f);
    }
    return result;
}

// Test 1: Single wavefront with work
__global__ void one_wf_with_work(int work_amount, float* output, unsigned long long* start_time, unsigned long long* end_time)
{
    __shared__ _Float16 lds[64 * 32];

    int tid = threadIdx.x;

    // Record start time (first thread)
    if (tid == 0) {
        start_time[0] = clock64();
    }
    __syncthreads();

    // Initialize LDS
    for (int i = tid; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    if (tid < 64) {  // 1 wavefront
        int lane = tid % 64;
        if (lane < 8) {
            int k = lane * 2;
            int m = 0;

            // Do LDS access
            _Float16 val = lds[m * 32 + k];

            // Do busy work
            float work_result = busy_work(work_amount);

            output[tid] = (float)val + work_result;
        }
    }

    __syncthreads();
    if (tid == 0) {
        end_time[0] = clock64();
    }
}

// Test 2: Two wavefronts with work
__global__ void two_wf_with_work(int work_amount, float* output, unsigned long long* start_time, unsigned long long* end_time)
{
    __shared__ _Float16 lds[64 * 32];

    int tid = threadIdx.x;

    // Record start time
    if (tid == 0) {
        start_time[0] = clock64();
    }
    __syncthreads();

    // Initialize LDS
    for (int i = tid; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    if (tid < 128) {  // 2 wavefronts
        int wf = tid / 64;
        int lane = tid % 64;

        if (lane < 8) {
            int k = lane * 2;
            int m = 0;

            // Do LDS access (same pattern as Test 1)
            _Float16 val = lds[m * 32 + k];

            // Do busy work (same amount)
            float work_result = busy_work(work_amount);

            output[tid] = (float)val + work_result;
        }
    }

    __syncthreads();
    if (tid == 0) {
        end_time[0] = clock64();
    }
}

// Test 3: Four wavefronts with work
__global__ void four_wf_with_work(int work_amount, float* output, unsigned long long* start_time, unsigned long long* end_time)
{
    __shared__ _Float16 lds[64 * 32];

    int tid = threadIdx.x;

    if (tid == 0) {
        start_time[0] = clock64();
    }
    __syncthreads();

    for (int i = tid; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    if (tid < 256) {  // 4 wavefronts
        int wf = tid / 64;
        int lane = tid % 64;

        if (lane < 8) {
            int k = lane * 2;
            int m = 0;

            _Float16 val = lds[m * 32 + k];
            float work_result = busy_work(work_amount);

            output[tid] = (float)val + work_result;
        }
    }

    __syncthreads();
    if (tid == 0) {
        end_time[0] = clock64();
    }
}

// Test 4: Use atomic counter to track execution order
__global__ void two_wf_execution_order(float* output, unsigned int* exec_order, unsigned long long* timestamps)
{
    __shared__ _Float16 lds[64 * 32];

    int tid = threadIdx.x;

    for (int i = tid; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    if (tid < 128) {
        int wf = tid / 64;
        int lane = tid % 64;

        if (lane < 8) {
            // Record when this thread starts
            if (lane == 0) {  // First thread in each WF
                unsigned int order = atomicAdd(&exec_order[0], 1);
                timestamps[wf * 8 + order] = clock64();
            }

            int k = lane * 2;
            int m = 0;
            _Float16 val = lds[m * 32 + k];

            // Some work
            float work = busy_work(10000);

            output[tid] = (float)val + work;
        }
    }
}

int main()
{
    const int N = 256;
    const int WORK_AMOUNT = 100000;  // Enough to create measurable time difference

    float *d_output;
    unsigned long long *d_start_time, *d_end_time;
    unsigned int *d_exec_order;
    unsigned long long *d_timestamps;

    unsigned long long h_start_time, h_end_time;
    unsigned int h_exec_order;
    unsigned long long h_timestamps[16];

    HIP_CHECK(hipMalloc(&d_output, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_start_time, sizeof(unsigned long long)));
    HIP_CHECK(hipMalloc(&d_end_time, sizeof(unsigned long long)));
    HIP_CHECK(hipMalloc(&d_exec_order, sizeof(unsigned int)));
    HIP_CHECK(hipMalloc(&d_timestamps, 16 * sizeof(unsigned long long)));

    std::cout << "=== WAVEFRONT CONCURRENCY TEST ===\n\n";
    std::cout << "Question: Do multiple wavefronts execute concurrently or serially?\n";
    std::cout << "Method: Measure execution time with increasing WF count\n\n";
    std::cout << "Work amount per thread: " << WORK_AMOUNT << " iterations\n\n";

    // Test 1: Single WF
    std::cout << "Test 1: Single wavefront\n";
    one_wf_with_work<<<1, 256>>>(WORK_AMOUNT, d_output, d_start_time, d_end_time);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(&h_start_time, d_start_time, sizeof(unsigned long long), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&h_end_time, d_end_time, sizeof(unsigned long long), hipMemcpyDeviceToHost));
    unsigned long long time_1wf = h_end_time - h_start_time;
    std::cout << "  Execution time: " << time_1wf << " cycles\n\n";

    // Test 2: Two WFs
    std::cout << "Test 2: Two wavefronts (same work per WF)\n";
    two_wf_with_work<<<1, 256>>>(WORK_AMOUNT, d_output, d_start_time, d_end_time);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(&h_start_time, d_start_time, sizeof(unsigned long long), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&h_end_time, d_end_time, sizeof(unsigned long long), hipMemcpyDeviceToHost));
    unsigned long long time_2wf = h_end_time - h_start_time;
    std::cout << "  Execution time: " << time_2wf << " cycles\n";
    std::cout << "  Ratio to 1 WF: " << (double)time_2wf / time_1wf << "x\n";
    std::cout << "  (If serialized: ~2.0x, if concurrent: ~1.0x)\n\n";

    // Test 3: Four WFs
    std::cout << "Test 3: Four wavefronts (same work per WF)\n";
    four_wf_with_work<<<1, 256>>>(WORK_AMOUNT, d_output, d_start_time, d_end_time);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(&h_start_time, d_start_time, sizeof(unsigned long long), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&h_end_time, d_end_time, sizeof(unsigned long long), hipMemcpyDeviceToHost));
    unsigned long long time_4wf = h_end_time - h_start_time;
    std::cout << "  Execution time: " << time_4wf << " cycles\n";
    std::cout << "  Ratio to 1 WF: " << (double)time_4wf / time_1wf << "x\n";
    std::cout << "  (If serialized: ~4.0x, if concurrent: ~1.0x)\n\n";

    // Test 4: Execution order tracking
    std::cout << "Test 4: Execution order tracking\n";
    HIP_CHECK(hipMemset(d_exec_order, 0, sizeof(unsigned int)));
    HIP_CHECK(hipMemset(d_timestamps, 0, 16 * sizeof(unsigned long long)));
    two_wf_execution_order<<<1, 256>>>(d_output, d_exec_order, d_timestamps);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(&h_exec_order, d_exec_order, sizeof(unsigned int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_timestamps, d_timestamps, 16 * sizeof(unsigned long long), hipMemcpyDeviceToHost));

    std::cout << "  Execution order of WF0 lane 0 and WF1 lane 0:\n";
    std::cout << "  Order counter: " << h_exec_order << "\n";
    std::cout << "  Timestamps:\n";
    for (int i = 0; i < std::min(h_exec_order, 8u); i++) {
        std::cout << "    [" << i << "]: " << h_timestamps[i] << "\n";
    }
    std::cout << "  (If concurrent: timestamps should be close/overlapping)\n\n";

    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_start_time));
    HIP_CHECK(hipFree(d_end_time));
    HIP_CHECK(hipFree(d_exec_order));
    HIP_CHECK(hipFree(d_timestamps));

    std::cout << "=== INTERPRETATION ===\n\n";
    std::cout << "Time ratio analysis:\n";
    std::cout << "  1 WF: " << time_1wf << " cycles (baseline)\n";
    std::cout << "  2 WF: " << time_2wf << " cycles (" << (double)time_2wf / time_1wf << "x)\n";
    std::cout << "  4 WF: " << time_4wf << " cycles (" << (double)time_4wf / time_1wf << "x)\n\n";

    std::cout << "Conclusion:\n";
    if ((double)time_2wf / time_1wf > 1.5) {
        std::cout << "  ❌ SERIALIZED: Wavefronts execute sequentially!\n";
        std::cout << "  This means 0 conflicts is because WFs never overlap.\n";
        std::cout << "  Our inter-WF test is INVALID - need to force concurrency!\n";
    } else if ((double)time_2wf / time_1wf < 1.2) {
        std::cout << "  ✅ CONCURRENT: Wavefronts execute in parallel!\n";
        std::cout << "  This confirms 0 conflicts = genuine no inter-WF interference.\n";
        std::cout << "  Our inter-WF test is VALID!\n";
    } else {
        std::cout << "  ⚠️  PARTIAL: Some overlap but not fully concurrent.\n";
        std::cout << "  Need further investigation.\n";
    }

    return 0;
}
