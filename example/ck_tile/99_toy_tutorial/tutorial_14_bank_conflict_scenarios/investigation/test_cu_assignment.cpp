// Test which CU (Compute Unit) each wavefront is assigned to
// Uses hardware IDs to determine if WFs are on same or different CUs
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

// Kernel to report hardware IDs for each wavefront
__global__ void report_hardware_ids(unsigned int* hw_ids, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    int tid = threadIdx.x;
    int wf_id = tid / 64;  // Wavefront ID within block
    int lane = tid % 64;

    // Get hardware IDs (AMD-specific)
    unsigned int cu_id = __builtin_amdgcn_s_getreg(20);  // HW_ID register
    // Bits 0-3: Wave ID in SIMD
    // Bits 4-7: SIMD ID (0-3)
    // Bits 8-11: CU ID

    // Initialize LDS
    for (int i = tid; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    // Do our standard test pattern
    if (tid < 256) {
        if (lane < 8) {
            int k = lane * 2;  // Banks 0-7
            int m = 0;

            _Float16 val = lds[m * 32 + k];
            output[tid] = (float)val;
        }

        // First thread of each WF reports its hardware ID
        if (lane == 0) {
            hw_ids[wf_id] = cu_id;
        }
    }
}

// Alternative: Use __smid() if available
__global__ void report_sm_ids(unsigned int* sm_ids, float* output)
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

    // Get SM ID (Compute Unit ID on AMD)
    unsigned int my_sm_id = 0;

    // AMD MI300X specific: use hardware register
    // HW_ID register (20 = 0x14) contains CU information
    unsigned int hw_id = __builtin_amdgcn_s_getreg(20);
    my_sm_id = (hw_id >> 8) & 0xF;  // Extract CU ID from bits 8-11

    // Do our standard test pattern
    if (tid < 256) {
        if (lane < 8) {
            int k = lane * 2;
            int m = 0;

            _Float16 val = lds[m * 32 + k];
            output[tid] = (float)val;
        }

        // First thread of each WF reports its CU ID
        if (lane == 0) {
            sm_ids[wf_id] = my_sm_id;
        }
    }
}

// Test with explicit block size to control WF allocation
__global__ void test_two_wf_cu_assignment(unsigned int* cu_ids, unsigned long long* timestamps, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    int tid = threadIdx.x;
    int wf_id = tid / 64;  // 0 or 1
    int lane = tid % 64;

    // Get hardware info
    unsigned int hw_id = __builtin_amdgcn_s_getreg(20);
    unsigned int cu_id = (hw_id >> 8) & 0xF;
    unsigned int simd_id = (hw_id >> 4) & 0xF;
    unsigned int wave_id = hw_id & 0xF;

    // Initialize LDS
    for (int i = tid; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    // Record start time for each WF
    if (lane == 0) {
        timestamps[wf_id] = clock64();
        cu_ids[wf_id * 4 + 0] = cu_id;
        cu_ids[wf_id * 4 + 1] = simd_id;
        cu_ids[wf_id * 4 + 2] = wave_id;
        cu_ids[wf_id * 4 + 3] = hw_id;  // Full HW_ID for reference
    }

    // Do LDS access (same pattern as our conflict tests)
    if (tid < 128) {
        if (lane < 8) {
            int k = lane * 2;  // Banks 0-7
            int m = 0;

            _Float16 val = lds[m * 32 + k];
            output[tid] = (float)val;
        }
    }
}

int main()
{
    const int MAX_WF = 4;
    const int N = 256;

    unsigned int *d_hw_ids, *d_sm_ids, *d_cu_ids;
    unsigned long long *d_timestamps;
    float *d_output;

    unsigned int h_hw_ids[MAX_WF];
    unsigned int h_sm_ids[MAX_WF];
    unsigned int h_cu_ids[MAX_WF * 4];
    unsigned long long h_timestamps[MAX_WF];

    HIP_CHECK(hipMalloc(&d_hw_ids, MAX_WF * sizeof(unsigned int)));
    HIP_CHECK(hipMalloc(&d_sm_ids, MAX_WF * sizeof(unsigned int)));
    HIP_CHECK(hipMalloc(&d_cu_ids, MAX_WF * 4 * sizeof(unsigned int)));
    HIP_CHECK(hipMalloc(&d_timestamps, MAX_WF * sizeof(unsigned long long)));
    HIP_CHECK(hipMalloc(&d_output, N * sizeof(float)));

    std::cout << "=== CU (Compute Unit) ASSIGNMENT TEST ===\n\n";
    std::cout << "Question: Are wavefronts in the same block assigned to the same CU?\n";
    std::cout << "Critical for understanding: Do our WFs share LDS banks (same CU)?\n\n";

    // Get device properties
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute Units: " << prop.multiProcessorCount << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Warp size: " << prop.warpSize << "\n\n";

    // Test 1: Report hardware IDs
    std::cout << "Test 1: Full hardware ID report\n";
    HIP_CHECK(hipMemset(d_hw_ids, 0, MAX_WF * sizeof(unsigned int)));
    report_hardware_ids<<<1, 256>>>(d_hw_ids, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_hw_ids, d_hw_ids, MAX_WF * sizeof(unsigned int), hipMemcpyDeviceToHost));

    std::cout << "  Hardware IDs for each WF:\n";
    for (int i = 0; i < MAX_WF; i++) {
        std::cout << "    WF" << i << ": 0x" << std::hex << h_hw_ids[i] << std::dec;
        std::cout << " (CU: " << ((h_hw_ids[i] >> 8) & 0xF) << ", SIMD: " << ((h_hw_ids[i] >> 4) & 0xF) << ", Wave: " << (h_hw_ids[i] & 0xF) << ")\n";
    }
    std::cout << "\n";

    // Test 2: Report SM/CU IDs
    std::cout << "Test 2: Compute Unit ID report\n";
    HIP_CHECK(hipMemset(d_sm_ids, 0, MAX_WF * sizeof(unsigned int)));
    report_sm_ids<<<1, 256>>>(d_sm_ids, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_sm_ids, d_sm_ids, MAX_WF * sizeof(unsigned int), hipMemcpyDeviceToHost));

    std::cout << "  CU IDs for each WF:\n";
    for (int i = 0; i < MAX_WF; i++) {
        std::cout << "    WF" << i << ": CU " << h_sm_ids[i] << "\n";
    }
    std::cout << "\n";

    // Test 3: Detailed 2-WF test
    std::cout << "Test 3: Two wavefront detailed assignment\n";
    HIP_CHECK(hipMemset(d_cu_ids, 0, MAX_WF * 4 * sizeof(unsigned int)));
    HIP_CHECK(hipMemset(d_timestamps, 0, MAX_WF * sizeof(unsigned long long)));
    test_two_wf_cu_assignment<<<1, 256>>>(d_cu_ids, d_timestamps, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_cu_ids, d_cu_ids, MAX_WF * 4 * sizeof(unsigned int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_timestamps, d_timestamps, MAX_WF * sizeof(unsigned long long), hipMemcpyDeviceToHost));

    std::cout << "  Detailed assignment:\n";
    for (int i = 0; i < 2; i++) {
        std::cout << "    WF" << i << ":\n";
        std::cout << "      CU ID:   " << h_cu_ids[i * 4 + 0] << "\n";
        std::cout << "      SIMD ID: " << h_cu_ids[i * 4 + 1] << "\n";
        std::cout << "      Wave ID: " << h_cu_ids[i * 4 + 2] << "\n";
        std::cout << "      HW_ID:   0x" << std::hex << h_cu_ids[i * 4 + 3] << std::dec << "\n";
        std::cout << "      Start time: " << h_timestamps[i] << " cycles\n";
    }
    std::cout << "\n";

    // Analysis
    std::cout << "=== ANALYSIS ===\n\n";

    bool same_cu = (h_cu_ids[0] == h_cu_ids[4]);
    bool same_simd = (h_cu_ids[1] == h_cu_ids[5]);

    std::cout << "WF0 vs WF1:\n";
    std::cout << "  Same CU?   " << (same_cu ? "✅ YES" : "❌ NO") << " (CU" << h_cu_ids[0] << " vs CU" << h_cu_ids[4] << ")\n";
    std::cout << "  Same SIMD? " << (same_simd ? "✅ YES" : "❌ NO") << " (SIMD" << h_cu_ids[1] << " vs SIMD" << h_cu_ids[5] << ")\n\n";

    unsigned long long time_diff = (h_timestamps[1] > h_timestamps[0]) ?
                                    (h_timestamps[1] - h_timestamps[0]) :
                                    (h_timestamps[0] - h_timestamps[1]);
    std::cout << "  Start time difference: " << time_diff << " cycles\n";
    std::cout << "  (Small difference = concurrent execution)\n\n";

    std::cout << "Interpretation:\n";
    if (same_cu) {
        std::cout << "  ✅ Both WFs on SAME CU → They SHARE LDS banks!\n";
        std::cout << "  This means 0 conflicts is VERY significant:\n";
        std::cout << "    - WFs are truly concurrent on the same hardware\n";
        std::cout << "    - They access the SAME physical LDS banks\n";
        std::cout << "    - Yet no conflicts detected!\n";
        std::cout << "  Conclusion: Hardware has robust WF isolation/pipelining\n";
    } else {
        std::cout << "  ⚠️  WFs on DIFFERENT CUs → They have SEPARATE LDS!\n";
        std::cout << "  This means 0 conflicts is EXPECTED:\n";
        std::cout << "    - Each CU has its own LDS memory\n";
        std::cout << "    - WFs can't conflict if using different physical memory\n";
        std::cout << "  Conclusion: Our test doesn't prove inter-WF isolation on same CU!\n";
        std::cout << "\n";
        std::cout << "  ⚠️  TO FIX: Need to force WFs onto same CU by:\n";
        std::cout << "    1. Launch more blocks (some will share CUs)\n";
        std::cout << "    2. Use smaller block sizes (might fit multiple in one CU)\n";
        std::cout << "    3. Query occupancy to understand scheduling\n";
    }

    HIP_CHECK(hipFree(d_hw_ids));
    HIP_CHECK(hipFree(d_sm_ids));
    HIP_CHECK(hipFree(d_cu_ids));
    HIP_CHECK(hipFree(d_timestamps));
    HIP_CHECK(hipFree(d_output));

    return 0;
}
