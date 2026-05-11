/*
 * FP16 LDS Bank Conflict Calculator v2
 *
 * Key insight from profiler testing:
 * - test_same_bank_diff_slots_all_threads (64 threads, all bank 0, 64 slots) = 31 conflicts
 * - test_one_phase_same_bank (8 threads, bank 0, 8 slots) = 7 conflicts
 * - Pattern: conflicts = unique_slots - 1 when multiple threads hit same bank
 *
 * BUT: ds_read_u16 is per-thread, scalar operation
 * The profiler counts conflicts when LANES WITHIN A PHASE access same bank/different slot
 *
 * Real kernel does 8 ds_read_u16 per thread in sequence.
 * Each ds_read_u16 has 64 lanes accessing 64 different addresses simultaneously.
 *
 * Let me trace exactly what happens:
 * - Per-phase (8 lanes execute in parallel)
 * - For each dm read, 8 lanes in a phase access 8 different (k, m) pairs
 *
 * The problem: within a phase, different lanes access SAME dm but different (k, m0)
 * So for dm=0: lanes 0-7 access (k=0-7, m=0-7) -> 8 different K values
 * With row-major [M,K], these are all at offsets m*32 + k = 0*32+0, 0*32+1, ..., 0*32+7
 * = offsets 0,1,2,3,4,5,6,7 -> bytes 0,2,4,6,8,10,12,14 -> slots 0,1,2,3,4,5,6,7
 * = banks 0,1,2,3,4,5,6,7 (NO CONFLICT!)
 *
 * Wait - the transpose read is [K,M] not [M,K]!
 * When reading transposed, thread reads column k=const, varying m.
 * So for lane 0 (k=0), reads m=0,8,16,24,32,40,48,56 (8 dm values)
 * But these are 8 separate ds_read_u16 instructions, not one.
 *
 * Within ONE ds_read_u16:
 * Phase 0 lanes access: m0=0, dm varies by instruction
 * Actually wait - let me re-read the distribution...
 *
 * Distribution:
 * - k1 = wf (wavefront index)
 * - k2 = lane % 8
 * - m0 = lane / 8
 *
 * So within WF0, phase 0 (lanes 0-7):
 * lane 0: k=0, m0=0 -> reads dm=0..7 at offsets m*32+k = (0,8,16,24,...)*32+0
 * lane 1: k=1, m0=0 -> reads dm=0..7 at offsets (0,8,16,24,...)*32+1
 * lane 7: k=7, m0=0 -> reads dm=0..7 at offsets (0,8,16,24,...)*32+7
 *
 * For the SAME ds_read_u16 instruction (say, dm=0):
 * lane 0: offset = 0*32+0 = 0
 * lane 1: offset = 0*32+1 = 1
 * lane 2: offset = 0*32+2 = 2
 * ...
 * lane 7: offset = 0*32+7 = 7
 *
 * Bytes: 0,2,4,6,8,10,12,14 -> slots 0,1,2,3,4,5,6,7 -> banks 0,1,2,3,4,5,6,7
 * NO CONFLICTS in this phase for dm=0!
 *
 * BUT wait - what about m0?
 * Phase 0 has m0=0 for all 8 lanes
 * Phase 1 (lanes 8-15) has m0=1 for all 8 lanes
 * ...
 * Phase 7 (lanes 56-63) has m0=7 for all 8 lanes
 *
 * Hmm, so each phase has a different m0 value, which means different M rows.
 * For dm=0, m = m0*8 + 0 = m0*8
 * Phase 0: m=0, offset = 0*32+k
 * Phase 1: m=8, offset = 8*32+k = 256+k
 * Phase 7: m=56, offset = 56*32+k = 1792+k
 *
 * These phases execute in sequence, not parallel!
 *
 * OK I think I understand now:
 * The actual conflict happens because different threads within ONE phase
 * access addresses that COLLIDE on banks.
 *
 * Let me trace phase 0 for dm=0 again:
 * - 8 lanes (0-7) have k2=0-7, m0=0, dm=0
 * - m = 0*8 + 0 = 0
 * - k = wf*8 + k2 = 0*8 + 0-7 = 0-7
 * - offset = m*32 + k = 0*32 + 0-7 = 0-7
 *
 * Wait that's wrong for transpose read.
 * For [K,M] read, offset = k*M_stride + m
 * But LDS stores as [M,K] with stride K
 * So reading transposed means: offset = m*K + k
 *
 * OK so it's the same formula. Let me check the profiler test that DOES show conflicts:
 * test_same_bank_diff_slots: 64 threads read offsets 0,64,128,... (tid*64)
 * = slots 0,32,64,... = banks 0,0,0,... (all bank 0!)
 * = 64 accesses to bank 0, 64 different slots -> 63 conflicts expected, got 31
 *
 * Hmm, 31 is exactly half of 63-1=62... or 32-1=31.
 * Maybe gfx942 processes 32 lanes at a time?
 *
 * Let me check: gfx942 has 64-lane wavefronts but processes in groups of 32.
 * So 64 lanes / 32 = 2 "half-warps" or 32-lane groups.
 * Each 32-lane group: 32 accesses to bank 0, 32 slots -> 31 conflicts each.
 * Total = 31 * 2 = 62? But we got 31...
 *
 * Maybe the counter counts per-cycle, not per-access?
 * If 32 lanes accessing 32 different slots in same bank takes 32 cycles,
 * that's 31 EXTRA cycles = 31 conflicts.
 *
 * Let me rebuild the model with this understanding.
 */

#include <iostream>
#include <set>
#include <map>
#include <vector>

// Constants
constexpr int kM = 64;
constexpr int kK = 32;
constexpr int kBlockSize = 256;
constexpr int kWavefronts = kBlockSize / 64;  // 4
constexpr int kKPack = 8;
constexpr int DataTypeSize = 2;  // FP16
constexpr int MLdsLayer = 2;

// Calculate XOR offset
int calc_xor_offset(int m, int k) {
    int m_div = m / MLdsLayer;
    int layer = m % MLdsLayer;
    int k_div = k / kKPack;
    int k_pack = k % kKPack;

    // XOR transform
    int dim0 = k_div * MLdsLayer + layer;  // 0-7
    int dim1 = m_div;                       // 0-31
    int xor_dim1 = dim1 ^ dim0;

    // Physical offset with original strides [8, 64, 1]
    return dim0 * kKPack + xor_dim1 * (kK * MLdsLayer) + k_pack;
}

// Calculate plain offset
int calc_plain_offset(int m, int k) {
    return m * kK + k;
}

// Model: Each ds_read_u16 executes 64 lanes divided into 2 groups of 32
// Bank conflicts occur within each 32-lane group
void calculate_conflicts(bool use_xor, int num_blocks, int num_k_iters) {
    int total_conflicts = 0;

    auto calc_offset = use_xor ? calc_xor_offset : calc_plain_offset;

    for (int block = 0; block < num_blocks; block++) {
        for (int k_iter = 0; k_iter < num_k_iters; k_iter++) {
            // For each wavefront
            for (int wf = 0; wf < kWavefronts; wf++) {
                // For each dm value (8 reads per thread)
                for (int dm = 0; dm < 8; dm++) {
                    // Process 64 lanes in two 32-lane groups
                    for (int half = 0; half < 2; half++) {
                        std::map<int, std::set<int>> bank_slots;

                        for (int lane_in_half = 0; lane_in_half < 32; lane_in_half++) {
                            int lane = half * 32 + lane_in_half;

                            int k1 = wf;
                            int k2 = lane % 8;
                            int m0 = lane / 8;

                            int k = k1 * 8 + k2;
                            int m = m0 * 8 + dm;

                            int offset = calc_offset(m, k);
                            int byte_off = offset * DataTypeSize;
                            int slot = byte_off / 4;
                            int bank = slot % 32;

                            bank_slots[bank].insert(slot);
                        }

                        // Count conflicts: for each bank, extra slots cost 1 cycle each
                        for (const auto& [bank, slots] : bank_slots) {
                            if (slots.size() > 1) {
                                total_conflicts += (slots.size() - 1);
                            }
                        }
                    }
                }
            }
        }
    }

    std::cout << (use_xor ? "WITH XOR" : "WITHOUT XOR") << ":\n";
    std::cout << "  Calculated conflicts: " << total_conflicts << "\n";
    std::cout << "  Target: " << (use_xor ? "3,072" : "7,168") << "\n";
    std::cout << "  Match: " << (total_conflicts == (use_xor ? 3072 : 7168) ? "YES" : "NO") << "\n\n";
}

// Debug: show detailed bank/slot distribution for one ds_read_u16
void debug_bank_distribution(bool use_xor, int wf, int dm) {
    auto calc_offset = use_xor ? calc_xor_offset : calc_plain_offset;

    std::cout << "WF" << wf << " dm=" << dm << " distribution:\n";

    // Half 0 (lanes 0-31)
    std::map<int, std::set<int>> bank_slots_h0;  // bank -> set of slots
    for (int lane = 0; lane < 32; lane++) {
        int k1 = wf, k2 = lane % 8, m0 = lane / 8;
        int k = k1 * 8 + k2, m = m0 * 8 + dm;
        int offset = calc_offset(m, k);
        int slot = (offset * DataTypeSize) / 4;
        int bank = slot % 32;
        bank_slots_h0[bank].insert(slot);
    }

    std::cout << "  Half 0: ";
    int conflicts_h0 = 0;
    for (const auto& [bank, slots] : bank_slots_h0) {
        std::cout << "B" << bank << "(slots:" << slots.size() << ") ";
        if (slots.size() > 1) conflicts_h0 += slots.size() - 1;
    }
    std::cout << "=> " << conflicts_h0 << " conflicts\n";

    // Half 1 (lanes 32-63)
    std::map<int, std::set<int>> bank_slots_h1;
    for (int lane = 32; lane < 64; lane++) {
        int k1 = wf, k2 = lane % 8, m0 = lane / 8;
        int k = k1 * 8 + k2, m = m0 * 8 + dm;
        int offset = calc_offset(m, k);
        int slot = (offset * DataTypeSize) / 4;
        int bank = slot % 32;
        bank_slots_h1[bank].insert(slot);
    }

    std::cout << "  Half 1: ";
    int conflicts_h1 = 0;
    for (const auto& [bank, slots] : bank_slots_h1) {
        std::cout << "B" << bank << "(slots:" << slots.size() << ") ";
        if (slots.size() > 1) conflicts_h1 += slots.size() - 1;
    }
    std::cout << "=> " << conflicts_h1 << " conflicts\n";

    // Show actual slot values for first bank
    if (!bank_slots_h0.empty()) {
        int first_bank = bank_slots_h0.begin()->first;
        std::cout << "  Detail B" << first_bank << " slots: { ";
        for (int s : bank_slots_h0[first_bank]) {
            std::cout << s << " ";
        }
        std::cout << "}\n";
    }
}

int main() {
    std::cout << "=== FP16 Conflict Calculator v2 ===\n\n";

    int num_blocks = 4;
    int num_k_iters = 4;

    std::cout << "Config: " << num_blocks << " blocks × " << num_k_iters << " K-iters\n\n";

    calculate_conflicts(false, num_blocks, num_k_iters);
    calculate_conflicts(true, num_blocks, num_k_iters);

    std::cout << "=== Debug: Bank distribution for one ds_read ===\n";
    std::cout << "\nWITHOUT XOR:\n";
    debug_bank_distribution(false, 0, 0);
    debug_bank_distribution(false, 0, 1);

    std::cout << "\nWITH XOR:\n";
    debug_bank_distribution(true, 0, 0);
    debug_bank_distribution(true, 0, 1);

    return 0;
}
