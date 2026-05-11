// Test TRUE concurrency: Can multiple WFs issue LDS instructions on the same cycle?
// Method: Use a shared resource (atomic counter) that would show interference
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

// Test 1: Atomic counter to detect simultaneous execution
// If WFs are truly concurrent, they'll increment the counter simultaneously
__global__ void atomic_concurrency_test(unsigned int* shared_counter, unsigned int* wf_order, unsigned long long* timestamps)
{
    __shared__ _Float16 lds[64 * 32];
    __shared__ unsigned int local_counter;

    int tid = threadIdx.x;
    int wf_id = tid / 64;
    int lane = tid % 64;

    if (tid == 0) {
        local_counter = 0;
    }

    // Initialize LDS
    for (int i = tid; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    if (tid < 128) {  // 2 WFs
        if (lane < 8) {
            // Each thread tries to be the first to increment
            // If concurrent, both WFs will see the counter at nearly same value
            if (lane == 0) {
                unsigned long long before = clock64();
                unsigned int my_order = atomicAdd(shared_counter, 1);
                unsigned long long after = clock64();

                wf_order[wf_id * 3 + 0] = my_order;
                timestamps[wf_id * 2 + 0] = before;
                timestamps[wf_id * 2 + 1] = after;
            }

            // Do LDS access
            int k = lane * 2;
            int m = 0;
            _Float16 val = lds[m * 32 + k];

            // Prevent optimization
            if (val > (_Float16)1000000.0f) {
                wf_order[wf_id * 3 + 1] = 1;
            }
        }
    }
}

// Test 2: Barrier test - if concurrent, they should hit barrier at similar times
__global__ void barrier_timing_test(unsigned long long* barrier_times, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    int tid = threadIdx.x;
    int wf_id = tid / 64;
    int lane = tid % 64;

    // Initialize LDS
    for (int i = tid; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    // Record time just before LDS access
    if (tid < 128) {
        if (lane == 0) {
            barrier_times[wf_id * 4 + 0] = clock64();
        }

        // Do LDS access
        if (lane < 8) {
            int k = lane * 2;
            int m = 0;
            _Float16 val = lds[m * 32 + k];
            output[tid] = (float)val;
        }

        // Record time just after LDS access
        if (lane == 0) {
            barrier_times[wf_id * 4 + 1] = clock64();
        }
    }

    // If WFs are concurrent, the time windows should overlap:
    // WF0: [time0_before, time0_after]
    // WF1: [time1_before, time1_after]
    // Overlap = concurrent execution
}

// Test 3: Multiple iterations with timing on each LDS access
__global__ void iterative_lds_timing(unsigned long long* iteration_times, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    int tid = threadIdx.x;
    int wf_id = tid / 64;
    int lane = tid % 64;

    // Initialize LDS
    for (int i = tid; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    if (tid < 128) {  // 2 WFs
        if (lane < 8) {
            const int ITERS = 10;
            float sum = 0.0f;

            for (int iter = 0; iter < ITERS; iter++) {
                // Record time before LDS access
                unsigned long long before = clock64();

                // LDS access
                int k = lane * 2;
                int m = iter % 16;
                _Float16 val = lds[m * 32 + k];
                sum += (float)val;

                // Record time after
                unsigned long long after = clock64();

                // Lane 0 of each WF records the timing
                if (lane == 0) {
                    iteration_times[wf_id * ITERS * 2 + iter * 2 + 0] = before;
                    iteration_times[wf_id * ITERS * 2 + iter * 2 + 1] = after;
                }
            }

            output[tid] = sum;
        }
    }
}

// Test 4: Check hardware occupancy registers
__global__ void check_active_wavefronts(unsigned int* active_wf_count, unsigned int* hw_info)
{
    __shared__ _Float16 lds[64 * 32];

    int tid = threadIdx.x;
    int wf_id = tid / 64;
    int lane = tid % 64;

    // Get hardware ID
    unsigned int hw_id = __builtin_amdgcn_s_getreg(20);  // HW_ID register

    // Initialize LDS
    for (int i = tid; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    if (tid < 128) {
        if (lane == 0) {
            hw_info[wf_id * 2 + 0] = hw_id;
            // Try to get exec mask or other status
            hw_info[wf_id * 2 + 1] = __builtin_amdgcn_s_getreg(21); // STATUS register
        }

        if (lane < 8) {
            int k = lane * 2;
            int m = 0;
            _Float16 val = lds[m * 32 + k];

            // Count how many WFs appear to be active
            if (lane == 0) {
                atomicAdd(&active_wf_count[0], 1);
            }
        }
    }
}

int main()
{
    const int MAX_WF = 4;
    const int ITERS = 10;

    unsigned int *d_shared_counter, *d_wf_order, *d_active_count, *d_hw_info;
    unsigned long long *d_timestamps, *d_barrier_times, *d_iter_times;
    float *d_output;

    unsigned int h_shared_counter;
    unsigned int h_wf_order[MAX_WF * 3];
    unsigned int h_active_count;
    unsigned int h_hw_info[MAX_WF * 2];
    unsigned long long h_timestamps[MAX_WF * 2];
    unsigned long long h_barrier_times[MAX_WF * 4];
    unsigned long long h_iter_times[MAX_WF * ITERS * 2];

    HIP_CHECK(hipMalloc(&d_shared_counter, sizeof(unsigned int)));
    HIP_CHECK(hipMalloc(&d_wf_order, MAX_WF * 3 * sizeof(unsigned int)));
    HIP_CHECK(hipMalloc(&d_active_count, sizeof(unsigned int)));
    HIP_CHECK(hipMalloc(&d_hw_info, MAX_WF * 2 * sizeof(unsigned int)));
    HIP_CHECK(hipMalloc(&d_timestamps, MAX_WF * 2 * sizeof(unsigned long long)));
    HIP_CHECK(hipMalloc(&d_barrier_times, MAX_WF * 4 * sizeof(unsigned long long)));
    HIP_CHECK(hipMalloc(&d_iter_times, MAX_WF * ITERS * 2 * sizeof(unsigned long long)));
    HIP_CHECK(hipMalloc(&d_output, 256 * sizeof(float)));

    std::cout << "=== TRUE CONCURRENCY VERIFICATION ===\n\n";
    std::cout << "Testing if wavefronts execute instructions SIMULTANEOUSLY\n";
    std::cout << "(not just overlapped in time, but same-cycle instruction issue)\n\n";

    // Test 1: Atomic counter
    std::cout << "Test 1: Atomic counter race\n";
    std::cout << "  If WFs are concurrent, they'll both see counter ~0 simultaneously\n";
    HIP_CHECK(hipMemset(d_shared_counter, 0, sizeof(unsigned int)));
    HIP_CHECK(hipMemset(d_wf_order, 0, MAX_WF * 3 * sizeof(unsigned int)));
    HIP_CHECK(hipMemset(d_timestamps, 0, MAX_WF * 2 * sizeof(unsigned long long)));

    atomic_concurrency_test<<<1, 256>>>(d_shared_counter, d_wf_order, d_timestamps);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(&h_shared_counter, d_shared_counter, sizeof(unsigned int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_wf_order, d_wf_order, MAX_WF * 3 * sizeof(unsigned int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_timestamps, d_timestamps, MAX_WF * 2 * sizeof(unsigned long long), hipMemcpyDeviceToHost));

    std::cout << "  Results:\n";
    std::cout << "    WF0 saw counter at: " << h_wf_order[0] << "\n";
    std::cout << "    WF1 saw counter at: " << h_wf_order[3] << "\n";
    std::cout << "    Time WF0: " << h_timestamps[0] << " to " << h_timestamps[1] << "\n";
    std::cout << "    Time WF1: " << h_timestamps[2] << " to " << h_timestamps[3] << "\n";

    if (h_wf_order[0] == h_wf_order[3]) {
        std::cout << "    ✅ Both saw same value → TRULY CONCURRENT\n";
    } else if (abs((int)h_wf_order[0] - (int)h_wf_order[3]) <= 1) {
        std::cout << "    ⚠️  Values differ by 1 → MOSTLY concurrent (minor serialization)\n";
    } else {
        std::cout << "    ❌ Values differ significantly → SERIALIZED\n";
    }
    std::cout << "\n";

    // Test 2: Barrier timing
    std::cout << "Test 2: LDS access timing windows\n";
    std::cout << "  Check if WF0 and WF1 LDS access times overlap\n";
    HIP_CHECK(hipMemset(d_barrier_times, 0, MAX_WF * 4 * sizeof(unsigned long long)));

    barrier_timing_test<<<1, 256>>>(d_barrier_times, d_output);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_barrier_times, d_barrier_times, MAX_WF * 4 * sizeof(unsigned long long), hipMemcpyDeviceToHost));

    std::cout << "  WF0 LDS access window: [" << h_barrier_times[0] << " - " << h_barrier_times[1] << "]\n";
    std::cout << "  WF1 LDS access window: [" << h_barrier_times[4] << " - " << h_barrier_times[5] << "]\n";

    unsigned long long wf0_start = h_barrier_times[0];
    unsigned long long wf0_end = h_barrier_times[1];
    unsigned long long wf1_start = h_barrier_times[4];
    unsigned long long wf1_end = h_barrier_times[5];

    bool overlap = (wf0_start <= wf1_end) && (wf1_start <= wf0_end);
    std::cout << "  Time windows overlap: " << (overlap ? "✅ YES (concurrent)" : "❌ NO (serialized)") << "\n\n";

    // Test 3: Iterative timing
    std::cout << "Test 3: Iteration-by-iteration timing (10 LDS accesses)\n";
    HIP_CHECK(hipMemset(d_iter_times, 0, MAX_WF * ITERS * 2 * sizeof(unsigned long long)));

    iterative_lds_timing<<<1, 256>>>(d_iter_times, d_output);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_iter_times, d_iter_times, MAX_WF * ITERS * 2 * sizeof(unsigned long long), hipMemcpyDeviceToHost));

    std::cout << "  Checking if access times interleave:\n";
    int concurrent_iters = 0;
    for (int i = 0; i < ITERS; i++) {
        unsigned long long wf0_t0 = h_iter_times[0 * ITERS * 2 + i * 2 + 0];
        unsigned long long wf0_t1 = h_iter_times[0 * ITERS * 2 + i * 2 + 1];
        unsigned long long wf1_t0 = h_iter_times[1 * ITERS * 2 + i * 2 + 0];
        unsigned long long wf1_t1 = h_iter_times[1 * ITERS * 2 + i * 2 + 1];

        bool iter_overlap = (wf0_t0 <= wf1_t1) && (wf1_t0 <= wf0_t1);
        if (iter_overlap) concurrent_iters++;

        if (i < 3) {  // Show first 3 iterations
            std::cout << "    Iter " << i << ": WF0[" << wf0_t0 << "-" << wf0_t1 << "] vs WF1[" << wf1_t0 << "-" << wf1_t1 << "] ";
            std::cout << (iter_overlap ? "OVERLAP" : "NO OVERLAP") << "\n";
        }
    }
    std::cout << "  Concurrent iterations: " << concurrent_iters << "/" << ITERS << "\n\n";

    // Test 4: Hardware info
    std::cout << "Test 4: Hardware occupancy check\n";
    HIP_CHECK(hipMemset(d_active_count, 0, sizeof(unsigned int)));
    HIP_CHECK(hipMemset(d_hw_info, 0, MAX_WF * 2 * sizeof(unsigned int)));

    check_active_wavefronts<<<1, 256>>>(d_active_count, d_hw_info);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(&h_active_count, d_active_count, sizeof(unsigned int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_hw_info, d_hw_info, MAX_WF * 2 * sizeof(unsigned int), hipMemcpyDeviceToHost));

    std::cout << "  Active wavefront counter: " << h_active_count << "\n";
    std::cout << "  Hardware info:\n";
    for (int i = 0; i < 2; i++) {
        std::cout << "    WF" << i << ": HW_ID=0x" << std::hex << h_hw_info[i*2] << ", STATUS=0x" << h_hw_info[i*2+1] << std::dec << "\n";
    }
    std::cout << "\n";

    // Cleanup
    HIP_CHECK(hipFree(d_shared_counter));
    HIP_CHECK(hipFree(d_wf_order));
    HIP_CHECK(hipFree(d_active_count));
    HIP_CHECK(hipFree(d_hw_info));
    HIP_CHECK(hipFree(d_timestamps));
    HIP_CHECK(hipFree(d_barrier_times));
    HIP_CHECK(hipFree(d_iter_times));
    HIP_CHECK(hipFree(d_output));

    std::cout << "=== CONCLUSION ===\n\n";
    std::cout << "Look for:\n";
    std::cout << "  - Atomic counter: Both WFs see same/similar values = concurrent\n";
    std::cout << "  - Timing windows: Overlapping = concurrent execution\n";
    std::cout << "  - Iterations: Multiple overlaps = sustained concurrency\n\n";
    std::cout << "If all show concurrency → WFs truly execute simultaneously\n";
    std::cout << "If serialized → 0 conflicts is expected (no actual collision)\n";

    return 0;
}
