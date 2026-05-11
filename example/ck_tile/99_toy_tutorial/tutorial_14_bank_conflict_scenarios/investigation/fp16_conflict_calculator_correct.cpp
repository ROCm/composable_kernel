/*
 * FP16 Conflict Calculator - Corrected Thread Distribution
 *
 * Based on GDB debug output showing actual values:
 * - Lane 0: reads k=0, m=0,1,2,3,4,5,6,7
 * - Lane 1: reads k=0, m=8,9,10,11,12,13,14,15
 * - Lane 8: reads k=1, m=0,1,2,3,4,5,6,7
 *
 * Corrected distribution:
 *   k = lane / 8
 *   m_base = (lane % 8) * 8
 *   For dm=0..7: m = m_base + dm
 *
 * Phase-based: 8 phases of 8 lanes each
 * Phase P contains lanes P*8 to P*8+7
 *
 * Within Phase 0 (lanes 0-7):
 *   All lanes have k = lane / 8 = 0 (same k!)
 *   m_base = 0, 8, 16, 24, 32, 40, 48, 56 (different m!)
 *
 * Targets:
 * - NO XOR: 7,168 conflicts
 * - WITH XOR: 3,072 conflicts
 */

#include <iostream>
#include <set>
#include <map>
#include <vector>

constexpr int kM = 64;           // Tile M dimension
constexpr int kK = 32;           // Tile K dimension
constexpr int kBlockSize = 256;  // Threads per block
constexpr int kWavefronts = 4;   // 256 / 64
constexpr int kMLdsLayer = 2;    // XOR parameter
constexpr int kKPack = 8;        // XOR parameter

// Plain row-major offset [M, K]
int plain_offset(int m, int k) {
    return m * kK + k;
}

// XOR swizzled offset
int xor_offset(int m, int k) {
    int m_div = m / kMLdsLayer;
    int layer = m % kMLdsLayer;
    int k_div = k / kKPack;
    int k_pack = k % kKPack;
    int dim0 = k_div * kMLdsLayer + layer;
    int dim1 = m_div;
    int xor_dim1 = dim1 ^ dim0;
    return dim0 * kKPack + xor_dim1 * (kK * kMLdsLayer) + k_pack;
}

// Analyze read conflicts for one wavefront, one dm value
// Returns conflicts for 8 phases of 8 lanes each
int analyze_read_one_dm(int wf, int dm, bool use_xor, bool verbose = false) {
    auto calc_offset = use_xor ? xor_offset : plain_offset;
    int total_conflicts = 0;

    // 8 phases of 8 lanes each
    for (int phase = 0; phase < 8; phase++) {
        std::map<int, std::set<int>> bank_slots;

        for (int lane_in_phase = 0; lane_in_phase < 8; lane_in_phase++) {
            int lane = phase * 8 + lane_in_phase;

            // CORRECTED distribution from GDB:
            // Each wavefront handles a different k range
            // Within wavefront: k = lane / 8, m_base = (lane % 8) * 8
            int k_within_wf = lane / 8;  // 0-7 within wavefront
            int k = wf * 8 + k_within_wf;  // Full k coordinate

            int m_base = (lane % 8) * 8;
            int m = m_base + dm;

            int offset = calc_offset(m, k);
            int byte_addr = offset * 2;  // FP16
            int slot = byte_addr / 4;
            int bank = slot % 32;

            bank_slots[bank].insert(slot);
        }

        // Count conflicts: same bank, different slots
        int phase_conflicts = 0;
        for (const auto& [bank, slots] : bank_slots) {
            if (slots.size() > 1) {
                phase_conflicts += slots.size() - 1;
            }
        }
        total_conflicts += phase_conflicts;

        if (verbose && phase_conflicts > 0) {
            std::cout << "    Phase " << phase << ": ";
            for (const auto& [bank, slots] : bank_slots) {
                if (slots.size() > 1) {
                    std::cout << "B" << bank << "(" << slots.size() << ") ";
                }
            }
            std::cout << "-> " << phase_conflicts << " conflicts\n";
        }
    }

    return total_conflicts;
}

// Calculate total read conflicts per iteration (1 block, 1 K-tile)
int calc_read_conflicts_per_iter(bool use_xor, bool verbose = false) {
    int total = 0;

    for (int wf = 0; wf < kWavefronts; wf++) {
        if (verbose) std::cout << "  WF" << wf << ":\n";
        for (int dm = 0; dm < 8; dm++) {
            int c = analyze_read_one_dm(wf, dm, use_xor, verbose);
            if (verbose) std::cout << "    dm=" << dm << ": " << c << " conflicts\n";
            total += c;
        }
    }

    return total;
}

// Analyze write conflicts
// ds_write_b128 = 16 bytes = 8 FP16 elements per instruction
int calc_write_conflicts_per_iter(bool use_xor, bool verbose = false) {
    auto calc_offset = use_xor ? xor_offset : plain_offset;
    int total = 0;

    // Write pattern: each thread writes 8 consecutive dm values
    // The ds_write_b128 writes 16 bytes to base address

    for (int wf = 0; wf < kWavefronts; wf++) {
        for (int phase = 0; phase < 8; phase++) {
            std::map<int, std::set<int>> bank_slots;

            for (int lane_in_phase = 0; lane_in_phase < 8; lane_in_phase++) {
                int lane = phase * 8 + lane_in_phase;

                int k_within_wf = lane / 8;
                int k = wf * 8 + k_within_wf;
                int m_base = (lane % 8) * 8;

                // Thread writes to m=m_base, k=k (base of 8 dm values)
                // But ds_write_b128 writes 16 bytes = 4 consecutive slots
                int offset = calc_offset(m_base, k);
                int byte_addr = offset * 2;

                // 4 slots per ds_write_b128
                for (int s = 0; s < 4; s++) {
                    int slot = (byte_addr + s * 4) / 4;
                    int bank = slot % 32;
                    bank_slots[bank].insert(slot);
                }
            }

            for (const auto& [bank, slots] : bank_slots) {
                if (slots.size() > 1) {
                    total += slots.size() - 1;
                }
            }
        }
    }

    return total;
}

void show_phase0_detail(bool use_xor) {
    auto calc_offset = use_xor ? xor_offset : plain_offset;
    const char* label = use_xor ? "XOR" : "PLAIN";

    std::cout << "\n=== Phase 0, WF0, dm=0 Detail (" << label << ") ===\n";
    std::cout << "Lane | k | m | offset | byte | slot | bank\n";
    std::cout << "-----|---|---|--------|------|------|-----\n";

    for (int lane = 0; lane < 8; lane++) {
        int k = lane / 8;  // All 0 for phase 0
        int m_base = (lane % 8) * 8;
        int m = m_base + 0;  // dm=0

        int offset = calc_offset(m, k);
        int byte_addr = offset * 2;
        int slot = byte_addr / 4;
        int bank = slot % 32;

        std::cout << "  " << lane << "  |  " << k << " | " << m << " | "
                  << offset << " | " << byte_addr << " | " << slot << " | " << bank << "\n";
    }
}

int main() {
    std::cout << "=== FP16 Conflict Calculator - Corrected Distribution ===\n\n";

    std::cout << "Thread distribution (from GDB):\n";
    std::cout << "  k = wf*8 + lane/8\n";
    std::cout << "  m = (lane%8)*8 + dm\n\n";

    // Show phase detail
    show_phase0_detail(false);
    show_phase0_detail(true);

    std::cout << "\n=== Per-Iteration Conflict Calculation ===\n\n";

    int read_plain = calc_read_conflicts_per_iter(false, false);
    int read_xor = calc_read_conflicts_per_iter(true, false);
    int write_plain = calc_write_conflicts_per_iter(false, false);
    int write_xor = calc_write_conflicts_per_iter(true, false);

    std::cout << "Per iteration (1 block × 1 K-tile):\n";
    std::cout << "  PLAIN: read=" << read_plain << ", write=" << write_plain
              << ", total=" << (read_plain + write_plain) << "\n";
    std::cout << "  XOR:   read=" << read_xor << ", write=" << write_xor
              << ", total=" << (read_xor + write_xor) << "\n\n";

    int num_blocks = 4;
    int num_k_iters = 4;
    int total_iters = num_blocks * num_k_iters;

    int total_plain = (read_plain + write_plain) * total_iters;
    int total_xor = (read_xor + write_xor) * total_iters;

    std::cout << "Total (" << total_iters << " iterations):\n";
    std::cout << "  PLAIN: " << total_plain << " (target: 7168)\n";
    std::cout << "  XOR:   " << total_xor << " (target: 3072)\n\n";

    std::cout << "Match status:\n";
    std::cout << "  PLAIN: " << (total_plain == 7168 ? "MATCH" : "NO MATCH")
              << " (diff: " << (total_plain - 7168) << ")\n";
    std::cout << "  XOR:   " << (total_xor == 3072 ? "MATCH" : "NO MATCH")
              << " (diff: " << (total_xor - 3072) << ")\n\n";

    // Show detailed breakdown
    std::cout << "=== Detailed Read Conflicts (PLAIN, WF0) ===\n";
    int wf0_plain = 0;
    for (int dm = 0; dm < 8; dm++) {
        int c = analyze_read_one_dm(0, dm, false, true);
        wf0_plain += c;
    }
    std::cout << "WF0 total: " << wf0_plain << "\n";

    return 0;
}
