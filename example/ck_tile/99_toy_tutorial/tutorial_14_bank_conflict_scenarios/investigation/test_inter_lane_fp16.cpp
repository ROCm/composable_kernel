// Test INTER-lane bank conflicts with FP16 - matching our actual transpose pattern
// Key question: When 2 threads access the SAME bank/slot but DIFFERENT FP16 elements,
// can the hardware service both in one cycle?
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

// Test 1: Two threads access SAME slot, DIFFERENT FP16 elements
// This matches our actual case: lanes 0,1 both hit bank 0 at dm=0
// Lane 0: m=0, k=0 -> offset 0 -> slot 0, bank 0 (first FP16 in slot)
// Lane 1: m=0, k=1 -> offset 1 -> slot 0, bank 0 (second FP16 in slot)
__global__ void inter_lane_same_slot(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    // Initialize LDS
    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    // Simulate our transpose read pattern:
    // Threads 0-7 are in same wave, reading m=0 with k=0,1,2,3,4,5,6,7
    int tid = threadIdx.x;

    if (tid < 8) {
        int m = 0;      // All threads read same row
        int k = tid;    // Each thread reads different column

        // Thread 0: offset 0 (slot 0, first FP16)
        // Thread 1: offset 1 (slot 0, second FP16) <- SAME slot as thread 0!
        // Thread 2: offset 2 (slot 1, first FP16)
        // Thread 3: offset 3 (slot 1, second FP16) <- SAME slot as thread 2!

        _Float16 val = lds[m * 32 + k];
        output[tid] = (float)val;
    }
}

// Test 2: Two threads access SAME bank, DIFFERENT slots
// Lane 0: m=0, k=0 -> offset 0   -> slot 0, bank 0
// Lane 2: m=2, k=0 -> offset 64  -> slot 32, bank 0 (different slot, same bank!)
__global__ void inter_lane_different_slots(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    // Initialize LDS
    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;

    if (tid < 8) {
        // Simulate conflict: tid 0,1,2,3 all access bank 0, but different slots
        int m = tid * 2;  // m = 0, 2, 4, 6 for tid 0,1,2,3
        int k = 0;

        // tid 0: m=0, offset 0   -> slot 0  -> bank 0
        // tid 1: m=2, offset 64  -> slot 32 -> bank 0 (32 % 32 = 0)
        // tid 2: m=4, offset 128 -> slot 64 -> bank 0 (64 % 32 = 0)
        // tid 3: m=6, offset 192 -> slot 96 -> bank 0 (96 % 32 = 0)
        // All hit bank 0 but different slots!

        if (m < 64) {
            _Float16 val = lds[m * 32 + k];
            output[tid] = (float)val;
        }
    }
}

// Test 3: EXACT pattern from our analysis - Phase 0, dm=0
// Lanes 0,1 both hit bank 0 (but different FP16 elements in same slot)
// Lanes 2,3 both hit bank 1 (same slot)
// etc.
__global__ void exact_transpose_pattern(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    // Initialize LDS with row-major [64, 32]
    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    // Phase 0 lanes: {0, 1, 2, 3, 20, 21, 22, 23}
    // Each lane reads m=0 (same row) with k = lane_id % 8
    const int phase0_lanes[8] = {0, 1, 2, 3, 20, 21, 22, 23};

    int tid = threadIdx.x;

    for (int i = 0; i < 8; i++) {
        if (tid == phase0_lanes[i]) {
            int m = 0;           // All read m=0 at step dm=0
            int k = tid % 8;     // k = 0,1,2,3,4,5,6,7

            // Expected banks at dm=0:
            // Lane 0:  k=0 -> offset 0 -> slot 0 -> bank 0
            // Lane 1:  k=1 -> offset 1 -> slot 0 -> bank 0 (SAME slot!)
            // Lane 2:  k=2 -> offset 2 -> slot 1 -> bank 1
            // Lane 3:  k=3 -> offset 3 -> slot 1 -> bank 1 (SAME slot!)
            // Lane 20: k=4 -> offset 4 -> slot 2 -> bank 2
            // Lane 21: k=5 -> offset 5 -> slot 2 -> bank 2 (SAME slot!)
            // Lane 22: k=6 -> offset 6 -> slot 3 -> bank 3
            // Lane 23: k=7 -> offset 7 -> slot 3 -> bank 3 (SAME slot!)

            _Float16 val = lds[m * 32 + k];
            output[tid] = (float)val;
        }
    }
}

// Test 4: No conflicts baseline - each thread different bank
__global__ void no_conflicts_baseline(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    // Initialize LDS
    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;

    if (tid < 32) {
        // Each thread accesses different bank
        int offset = tid * 2;  // offsets 0,2,4,6,... -> slots 0,1,2,3,... -> banks 0-31

        _Float16 val = lds[offset];
        output[tid] = (float)val;
    }
}

// ============ MULTI-WAVEFRONT TESTS ============

// Test 5: Two Wavefronts, Pure Inter-WF Conflict Test (CORRECTED)
// CRITICAL: Each WF has 0 internal conflicts - this isolates PURE inter-WF conflicts
// Goal: Test if different wavefronts can conflict when accessing same banks
// WF0: 8 threads access banks {0,1,2,3,4,5,6,7} - 0 internal conflicts
// WF1: 8 threads access banks {0,1,2,3,4,5,6,7} - 0 internal conflicts
// Both WFs hit the SAME banks → isolates inter-WF conflict behavior
__global__ void two_wf_pure_inter_conflict(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    // Initialize LDS
    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;

    // WF0 (threads 0-63) and WF1 (threads 64-127)
    if (tid < 128) {
        int wf = tid / 64;  // 0 or 1
        int lane = tid % 64;

        if (lane < 8) {
            // CORRECTED: Each WF accesses 8 DIFFERENT banks (0 internal conflicts)
            // lane 0: k=0 → slot 0 → bank 0
            // lane 1: k=2 → slot 1 → bank 1
            // lane 2: k=4 → slot 2 → bank 2
            // ... lane 7: k=14 → slot 7 → bank 7
            // Both WFs use the SAME k values → same banks
            int k = lane * 2;  // 0,2,4,6,8,10,12,14 → banks 0,1,2,3,4,5,6,7
            int m = 0;

            // All threads read from same row (m=0)
            // WF0: 8 threads hit 8 different banks → 0 internal conflicts
            // WF1: 8 threads hit the SAME 8 banks → 0 internal conflicts
            // Result = 0 → WFs execute independently (no inter-WF conflicts)
            // Result > 0 → Inter-WF conflicts detected!
            _Float16 val = lds[m * 32 + k];
            output[tid] = (float)val;
        }
    }
}

// Test 6: Four Wavefronts, Pure Inter-WF Conflict Test (CORRECTED)
// CRITICAL: Each WF has 0 internal conflicts - isolates PURE inter-WF conflicts
// Goal: Test if inter-WF conflicts scale with number of wavefronts
// WF0-3: Each has 8 threads accessing banks {0,1,2,3,4,5,6,7} - 0 internal conflicts per WF
// All 4 WFs hit the SAME banks → tests scaling of inter-WF conflicts
__global__ void four_wf_pure_inter_conflict(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    // Initialize LDS
    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;

    // WF0-3 (threads 0-255)
    if (tid < 256) {
        int wf = tid / 64;  // 0, 1, 2, or 3
        int lane = tid % 64;

        if (lane < 8) {
            // CORRECTED: Each WF accesses 8 DIFFERENT banks (0 internal conflicts)
            // All 4 WFs use the SAME k values → same banks
            // lane 0: k=0 → bank 0
            // lane 1: k=2 → bank 1
            // ... lane 7: k=14 → bank 7
            int k = lane * 2;  // 0,2,4,6,8,10,12,14 → banks 0,1,2,3,4,5,6,7
            int m = 0;

            // All 32 threads (4 WFs × 8 threads) read from same row (m=0)
            // Each WF: 8 threads hit 8 different banks → 0 internal conflicts
            // All WFs hit the SAME 8 banks
            // Result = 0 → WFs execute independently
            // Result > 0 → Inter-WF conflicts scale with WF count
            _Float16 val = lds[m * 32 + k];
            output[tid] = (float)val;
        }
    }
}

// Test 7: Inter-WF with Same Slots (FP16 Optimization Test)
// Goal: Test if FP16 same-slot optimization works across wavefronts
// WF0 lane 0: k=0, m=0 -> offset 0 -> slot 0, bank 0
// WF1 lane 64: k=0, m=0 -> offset 0 -> slot 0, bank 0 (EXACT same slot!)
__global__ void inter_wf_same_slot(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    // Initialize LDS
    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;

    // Only specific threads from WF0 and WF1
    if (tid == 0 || tid == 1 || tid == 64 || tid == 65) {
        int wf = tid / 64;  // 0 or 1
        int lane = tid % 64;

        // Both WFs access the EXACT same location
        // WF0 lane 0: m=0, k=0 -> offset 0 -> slot 0 (first FP16)
        // WF0 lane 1: m=0, k=1 -> offset 1 -> slot 0 (second FP16)
        // WF1 lane 0: m=0, k=0 -> offset 0 -> slot 0 (first FP16) - SAME as WF0 lane 0!
        // WF1 lane 1: m=0, k=1 -> offset 1 -> slot 0 (second FP16) - SAME as WF0 lane 1!
        int m = 0;
        int k = lane % 2;  // 0 or 1

        _Float16 val = lds[m * 32 + k];
        output[tid] = (float)val;
    }
}

// Test 8: Actual Distribution Pattern (Multiple WFs)
// Goal: Simulate the real tile distribution with 4 wavefronts
// K1 distribution: WF0 handles k=0-7, WF1 k=8-15, WF2 k=16-23, WF3 k=24-31
// Each WF's phase 0 accesses their k range
__global__ void actual_distribution_pattern(const _Float16* __restrict__ lds_ptr, float* output)
{
    __shared__ _Float16 lds[64 * 32];

    // Initialize LDS with row-major [64, 32]
    for (int i = threadIdx.x; i < 64*32; i += blockDim.x) {
        lds[i] = (_Float16)i;
    }
    __syncthreads();

    int tid = threadIdx.x;

    // 256 threads = 4 wavefronts
    if (tid < 256) {
        int wf = tid / 64;      // Wavefront ID: 0, 1, 2, 3
        int lane = tid % 64;    // Lane within wavefront: 0-63

        // Phase 0 lanes for this wavefront: {0, 1, 2, 3, 20, 21, 22, 23}
        const int phase0_offsets[8] = {0, 1, 2, 3, 20, 21, 22, 23};

        bool is_phase0 = false;
        int k_local = 0;
        for (int i = 0; i < 8; i++) {
            if (lane == phase0_offsets[i]) {
                is_phase0 = true;
                k_local = i;  // 0-7
                break;
            }
        }

        if (is_phase0) {
            // Each WF handles different k range:
            // WF0: k = 0-7 -> banks 0-3
            // WF1: k = 8-15 -> banks 4-7
            // WF2: k = 16-23 -> banks 8-11
            // WF3: k = 24-31 -> banks 12-15
            int k = (wf * 8) + k_local;
            int m = 0;  // All read same row at dm=0

            if (k < 32) {
                _Float16 val = lds[m * 32 + k];
                output[tid] = (float)val;
            }
        }
    }
}

int main()
{
    const int N = 256;
    float *d_output;
    HIP_CHECK(hipMalloc(&d_output, N * sizeof(float)));

    std::cout << "=== INTER-LANE FP16 BANK CONFLICT TESTS ===\n\n";

    std::cout << "Test 1: Inter-lane same-slot (2 threads, same slot, different FP16)\n";
    std::cout << "  Pattern: Threads 0,1 both access bank 0 (slot 0, different FP16 elements)\n";
    std::cout << "  Question: Can hardware service both FP16 in one cycle?\n";
    inter_lane_same_slot<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 2: Inter-lane different-slots (multiple threads, same bank, different slots)\n";
    std::cout << "  Pattern: Threads 0,1,2,3 all access bank 0 (different slots)\n";
    std::cout << "  Expected: HIGH conflicts (4-way serialization)\n";
    inter_lane_different_slots<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 3: EXACT transpose pattern (Phase 0, dm=0)\n";
    std::cout << "  Pattern: Lanes {0,1,2,3,20,21,22,23} reading m=0, k={0-7}\n";
    std::cout << "  Banks: {0,0,1,1,2,2,3,3} - pairs hit same slot!\n";
    std::cout << "  This is our ACTUAL case - does FP16 optimization help?\n";
    exact_transpose_pattern<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 4: No conflicts baseline (all different banks)\n";
    std::cout << "  Expected: 0 conflicts\n";
    no_conflicts_baseline<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "=== MULTI-WAVEFRONT TESTS ===\n\n";

    std::cout << "Test 5: Two wavefronts, PURE inter-WF conflict test (CORRECTED)\n";
    std::cout << "  Pattern: Each WF has 8 threads accessing banks {0,1,2,3,4,5,6,7}\n";
    std::cout << "  WF0: 0 internal conflicts (each thread different bank)\n";
    std::cout << "  WF1: 0 internal conflicts (each thread different bank)\n";
    std::cout << "  Both WFs hit the SAME 8 banks → isolates pure inter-WF conflicts\n";
    std::cout << "  Expected: 0 conflicts → WFs independent, >0 conflicts → Inter-WF conflicts detected!\n";
    two_wf_pure_inter_conflict<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 6: Four wavefronts, PURE inter-WF conflict test (CORRECTED)\n";
    std::cout << "  Pattern: Each of 4 WFs has 8 threads accessing banks {0,1,2,3,4,5,6,7}\n";
    std::cout << "  Each WF: 0 internal conflicts (each thread different bank)\n";
    std::cout << "  All 4 WFs hit the SAME 8 banks → tests inter-WF conflict scaling\n";
    std::cout << "  Expected: 0 conflicts → WFs independent, >0 conflicts → Conflicts scale with WF count\n";
    four_wf_pure_inter_conflict<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 7: Inter-WF with same slots (FP16 optimization test)\n";
    std::cout << "  Pattern: WF0 lanes 0,1 and WF1 lanes 0,1 access EXACT same slots\n";
    std::cout << "  Question: Does FP16 same-slot optimization work across WF boundaries?\n";
    std::cout << "  Expected: 0 conflicts if FP16 optimization is inter-WF, else HIGH conflicts\n";
    inter_wf_same_slot<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    std::cout << "Test 8: Actual distribution pattern (4 WFs with real K1 distribution)\n";
    std::cout << "  Pattern: WF0 (k=0-7), WF1 (k=8-15), WF2 (k=16-23), WF3 (k=24-31)\n";
    std::cout << "  Expected: Each WF uses exclusive banks (no conflicts)\n";
    std::cout << "  This verifies our assumption from test_inter_wf_conflicts.cpp\n";
    actual_distribution_pattern<<<1, 256>>>(nullptr, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Completed.\n\n";

    HIP_CHECK(hipFree(d_output));

    std::cout << "All tests completed successfully.\n\n";
    std::cout << "=== KEY QUESTIONS TO ANSWER ===\n\n";
    std::cout << "Single-WF Tests:\n";
    std::cout << "  Test 1/3: LOW conflicts → FP16 optimization works intra-WF\n";
    std::cout << "           HIGH conflicts → FP16 optimization doesn't help\n\n";
    std::cout << "Multi-WF Tests (CORRECTED - Isolates Pure Inter-WF Conflicts):\n";
    std::cout << "  Test 5: 0 conflicts → WFs execute independently (serialized/pipelined, no inter-WF conflicts)\n";
    std::cout << "         >0 conflicts → Inter-WF conflicts detected! (different WFs can conflict)\n\n";
    std::cout << "  Test 6 vs Test 5:\n";
    std::cout << "         Both 0 conflicts → WFs execute independently regardless of count\n";
    std::cout << "         Test 6 > Test 5 → Inter-WF conflicts scale with number of wavefronts\n";
    std::cout << "         Test 6 = Test 5 > 0 → Fixed inter-WF conflict overhead (doesn't scale)\n\n";
    std::cout << "  CRITICAL: Each WF has 0 internal conflicts in Tests 5 and 6.\n";
    std::cout << "           Any conflicts measured are PURE inter-WF conflicts!\n\n";
    std::cout << "  Test 7: 0 conflicts → FP16 optimization works inter-WF\n";
    std::cout << "         HIGH conflicts → FP16 optimization only intra-WF\n\n";
    std::cout << "  Test 8: Low/0 conflicts → Each WF uses exclusive banks (as expected)\n";
    std::cout << "         HIGH conflicts → Our assumption was wrong\n\n";
    std::cout << "=== PROFILING INSTRUCTIONS ===\n\n";
    std::cout << "To profile all tests:\n";
    std::cout << "  rocprofv3 -i lds_metrics.txt -o inter_lane_results -- ./test_inter_lane_fp16\n\n";
    std::cout << "Where lds_metrics.txt contains:\n";
    std::cout << "  pmc: LDS_BANK_CONFLICT\n\n";
    std::cout << "Then analyze the conflict counts for each test to answer the questions above.\n";

    return 0;
}
