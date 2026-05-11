/*
 * FP16 LDS Bank Conflict Calculator
 *
 * Goal: Match profiler counts exactly
 * - WITHOUT XOR: 7,168 conflicts
 * - WITH XOR: 3,072 conflicts
 *
 * Key insights from investigation:
 * 1. SQ_LDS_BANK_CONFLICT counts "cycles stalled due to bank conflicts"
 * 2. Counter increments once per PHASE (8 lanes) that has conflicts
 * 3. FP16 same-slot optimization: 2 FP16 in same 4-byte slot = 0 conflict
 * 4. Same bank, different slots = conflicts (serialization)
 * 5. 32 LDS banks, 4 bytes per bank per cycle
 *
 * Hardware model:
 * - 64 lanes per wavefront
 * - 8 phases per wavefront (8 lanes each)
 * - Each phase executes in parallel (SIMD)
 * - Bank conflicts counted per-phase for cross-lane conflicts
 */

#include <iostream>
#include <set>
#include <map>
#include <vector>
#include <tuple>
#include <algorithm>

// Constants matching the kernel
constexpr int kM = 64;
constexpr int kK = 32;
constexpr int kKPack = 8;
constexpr int kBlockSize = 256;
constexpr int kWavefronts = kBlockSize / 64;  // 4

// XOR swizzling parameters
constexpr int DataTypeSize = 2;  // FP16 = 2 bytes
constexpr int MLdsLayer = (32 * 4 / kK / DataTypeSize);  // = 2

// Calculate element offset with XOR swizzling
int calculate_xor_offset(int m, int k) {
    // Step 0: Calculate components
    int m_div_layer = m / MLdsLayer;      // 0-31
    int layer = m % MLdsLayer;            // 0-1
    int k_div_pack = k / kKPack;          // 0-3
    int k_pack = k % kKPack;              // 0-7

    // Step 1: Base descriptor dims [K/Pack*Layer, M/Layer, Pack]
    // Index: [k_div_pack * MLdsLayer + layer, m_div_layer, k_pack]
    int dim0_idx = k_div_pack * MLdsLayer + layer;  // 0-7
    int dim1_idx = m_div_layer;                      // 0-31
    int dim2_idx = k_pack;                           // 0-7

    // Step 2: XOR transform on dims [1, 0]
    // Output indices after XOR:
    int xor_result = dim1_idx ^ dim0_idx;
    // Output layout: [dim1_idx_out, dim0_idx_out, dim2_idx]
    int dim1_out = xor_result;           // M/Layer dimension
    int dim0_out = dim0_idx;             // K/Pack*Layer dimension

    // Step 3: Unmerge dim0_out back to [layer_out, k_div_pack_out]
    int layer_out = dim0_out / (kK / kKPack);
    int k_div_pack_out = dim0_out % (kK / kKPack);

    // Step 4: Merge to [K, M] for read descriptor
    // K = k_div_pack_out * kKPack + dim2_idx
    // M = dim1_out * MLdsLayer + layer_out
    // But we need physical offset, not logical coordinates

    // Physical offset calculation using original strides:
    // desc_0 strides: [kKPack, kK * MLdsLayer, 1] = [8, 64, 1]
    int physical_offset = dim0_out * kKPack + dim1_out * (kK * MLdsLayer) + dim2_idx * 1;

    return physical_offset;
}

// Calculate element offset without XOR (plain row-major)
int calculate_plain_offset(int m, int k) {
    // Row-major [M, K] layout with K as stride
    return m * kK + k;
}

// Model the transpose read pattern
struct ConflictCalculator {
    bool use_xor;
    int total_conflicts;
    int total_reads;

    ConflictCalculator(bool xor_enabled) : use_xor(xor_enabled), total_conflicts(0), total_reads(0) {}

    int calculate_offset(int k, int m) const {
        if (use_xor) {
            return calculate_xor_offset(m, k);
        } else {
            return calculate_plain_offset(m, k);
        }
    }

    // Calculate conflicts for one ds_read_u16 instruction across all lanes
    // Each thread reads one element, 8 phases of 8 lanes each
    int calculate_read_conflicts(int k_iter, bool verbose = false) {
        int conflicts = 0;

        // For each wavefront
        for (int wf = 0; wf < kWavefronts; wf++) {
            // For each dm value (8 reads per thread)
            for (int dm = 0; dm < 8; dm++) {
                // For each phase (8 lanes execute in parallel)
                for (int phase = 0; phase < 8; phase++) {
                    // bank -> set of slots accessed
                    std::map<int, std::set<int>> bank_slots;

                    // Get addresses for 8 lanes in this phase
                    for (int lane_in_phase = 0; lane_in_phase < 8; lane_in_phase++) {
                        int lane = phase * 8 + lane_in_phase;

                        // Thread distribution:
                        // k1 = wf (0-3)
                        // k2 = lane % 8 (0-7)
                        // m0 = lane / 8 (0-7)
                        int k1 = wf;
                        int k2 = lane % 8;
                        int m0 = lane / 8;

                        int k = k1 * 8 + k2;
                        int m = m0 * 8 + dm;

                        // Calculate element offset
                        int elem_offset = calculate_offset(k, m);

                        // Convert to bytes
                        int byte_offset = elem_offset * DataTypeSize;

                        // Calculate slot (4-byte unit) and bank
                        int slot = byte_offset / 4;
                        int bank = slot % 32;

                        bank_slots[bank].insert(slot);
                        total_reads++;
                    }

                    // Calculate conflicts for this phase
                    // Debug: print access pattern
                    if (verbose) {
                        std::cout << "WF" << wf << " dm=" << dm << " phase=" << phase << ": ";
                        for (const auto& entry : bank_slots) {
                            int bank = entry.first;
                            std::cout << "B" << bank << "(" << entry.second.size() << ") ";
                        }
                        std::cout << "\n";
                    }

                    for (const auto& entry : bank_slots) {
                        int bank = entry.first;
                        const auto& slots = entry.second;

                        // Conflict = unique_slots - 1 for each bank
                        // (First slot is free, each additional slot costs 1 cycle)
                        if (slots.size() > 1) {
                            int phase_conflicts = slots.size() - 1;
                            conflicts += phase_conflicts;
                        }
                    }
                }
            }
        }

        return conflicts;
    }

    void run_analysis(int num_m_blocks, int num_k_iters, bool verbose = false) {
        total_conflicts = 0;
        total_reads = 0;

        for (int m_block = 0; m_block < num_m_blocks; m_block++) {
            for (int k_iter = 0; k_iter < num_k_iters; k_iter++) {
                int iter_conflicts = calculate_read_conflicts(k_iter, verbose);
                total_conflicts += iter_conflicts;

                if (verbose) {
                    std::cout << "Block " << m_block << " K-iter " << k_iter
                             << ": " << iter_conflicts << " conflicts\n";
                }
            }
        }
    }
};

int main() {
    std::cout << "=== FP16 LDS Bank Conflict Calculator ===\n\n";

    // Target: match profiler for M=256, K=128 kernel
    // Profiler results:
    // - WITHOUT XOR: 7,168 conflicts
    // - WITH XOR: 3,072 conflicts

    int num_m_blocks = 4;  // M=256 / kM=64 = 4 blocks
    int num_k_iters = 4;   // K=128 / kK=32 = 4 iterations

    std::cout << "Configuration:\n";
    std::cout << "  Tile size: [" << kM << ", " << kK << "]\n";
    std::cout << "  M blocks: " << num_m_blocks << "\n";
    std::cout << "  K iterations: " << num_k_iters << "\n";
    std::cout << "  Wavefronts per block: " << kWavefronts << "\n";
    std::cout << "  Total iterations: " << (num_m_blocks * num_k_iters) << "\n\n";

    // WITHOUT XOR
    std::cout << "=== WITHOUT XOR ===\n";
    ConflictCalculator calc_plain(false);
    calc_plain.run_analysis(num_m_blocks, num_k_iters, false);
    std::cout << "Total read conflicts: " << calc_plain.total_conflicts << "\n";
    std::cout << "Total reads: " << calc_plain.total_reads << "\n";
    std::cout << "Target: 7,168\n";
    std::cout << "Match: " << (calc_plain.total_conflicts == 7168 ? "YES" : "NO") << "\n\n";

    // WITH XOR
    std::cout << "=== WITH XOR ===\n";
    ConflictCalculator calc_xor(true);
    calc_xor.run_analysis(num_m_blocks, num_k_iters, false);
    std::cout << "Total read conflicts: " << calc_xor.total_conflicts << "\n";
    std::cout << "Total reads: " << calc_xor.total_reads << "\n";
    std::cout << "Target: 3,072\n";
    std::cout << "Match: " << (calc_xor.total_conflicts == 3072 ? "YES" : "NO") << "\n\n";

    // Per-iteration breakdown
    std::cout << "=== Per-Iteration Breakdown ===\n";
    std::cout << "Plain (per iter): " << (calc_plain.total_conflicts / (num_m_blocks * num_k_iters)) << "\n";
    std::cout << "XOR (per iter): " << (calc_xor.total_conflicts / (num_m_blocks * num_k_iters)) << "\n";

    // Detailed analysis for one iteration
    std::cout << "\n=== Single Iteration Detail (WITHOUT XOR) ===\n";
    ConflictCalculator detail_plain(false);
    detail_plain.run_analysis(1, 1, true);

    std::cout << "\n=== Single Iteration Detail (WITH XOR) ===\n";
    ConflictCalculator detail_xor(true);
    detail_xor.run_analysis(1, 1, true);

    return 0;
}
