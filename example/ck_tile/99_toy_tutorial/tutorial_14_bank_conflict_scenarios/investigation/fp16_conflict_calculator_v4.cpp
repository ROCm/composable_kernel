/*
 * FP16 Conflict Calculator v4 - Half-wavefront max model
 *
 * Hypothesis: Hardware processes wavefront in 2 halves of 32 lanes.
 * Each half has 4 phases of 8 lanes. Counter reports max conflicts
 * per half, then sums the two halves.
 *
 * For plain layout:
 * - Each phase has 8 lanes reading same k, different m
 * - All 8 hit same bank with 8 different slots = 7 conflicts
 * - Max per half = 7 conflicts
 * - Total per instruction = 7 × 2 = 14 conflicts
 * - Per iteration = 14 × 4 WFs × 8 dm = 448 ✓
 *
 * For XOR layout:
 * - XOR spreads m values across different banks
 * - Conflicts reduced to ~3 per phase
 * - Total per instruction = 3 × 2 = 6 conflicts
 * - Per iteration = 6 × 4 WFs × 8 dm = 192 ✓
 */

#include <iostream>
#include <set>
#include <map>
#include <algorithm>

constexpr int kM = 64;
constexpr int kK = 32;
constexpr int kWavefronts = 4;
constexpr int kMLdsLayer = 2;
constexpr int kKPack = 8;

int plain_offset(int m, int k) {
    return m * kK + k;
}

// Real XOR transform from 04_row_major_xor.cpp
int xor_offset(int m, int k) {
    // Based on: make_xor_lds_descriptor<MLdsLayer=2, LdsAlignment=8>
    // dim0 = k / KPack, dim1 = m / MLdsLayer (XOR'd)
    int m_div = m / kMLdsLayer;
    int m_layer = m % kMLdsLayer;
    int k_div = k / kKPack;
    int k_pack = k % kKPack;

    // dim0 = k_div * MLdsLayer + m_layer
    int dim0 = k_div * kMLdsLayer + m_layer;
    // dim1 = m_div XOR dim0
    int dim1 = m_div ^ dim0;

    // offset = dim0 * KPack + dim1 * (K * MLdsLayer) + k_pack
    return dim0 * kKPack + dim1 * (kK * kMLdsLayer) + k_pack;
}

// Model: Max conflicts per half-wavefront
// Half 0: phases 0-3 (lanes 0-31)
// Half 1: phases 4-7 (lanes 32-63)
// Report: max(phase0-3) + max(phase4-7)
int calc_conflicts_per_dm(int wf, int dm, bool use_xor, bool verbose = false) {
    auto calc_offset = use_xor ? xor_offset : plain_offset;

    // Process each half-wavefront
    int half0_max = 0, half1_max = 0;

    for (int phase = 0; phase < 8; phase++) {
        std::map<int, std::set<int>> bank_slots;

        for (int lane_in_phase = 0; lane_in_phase < 8; lane_in_phase++) {
            int lane = phase * 8 + lane_in_phase;
            int k = wf * 8 + lane / 8;
            int m = (lane % 8) * 8 + dm;

            int offset = calc_offset(m, k);
            int byte_addr = offset * 2;
            int slot = byte_addr / 4;
            int bank = slot % 32;
            bank_slots[bank].insert(slot);
        }

        int phase_conflicts = 0;
        for (const auto& [bank, slots] : bank_slots) {
            if (slots.size() > 1) {
                phase_conflicts += slots.size() - 1;
            }
        }

        if (phase < 4) {
            half0_max = std::max(half0_max, phase_conflicts);
        } else {
            half1_max = std::max(half1_max, phase_conflicts);
        }

        if (verbose) {
            std::cout << "  Phase " << phase << " (half " << (phase < 4 ? 0 : 1) << "): ";
            for (const auto& [bank, slots] : bank_slots) {
                std::cout << "B" << bank << "(" << slots.size() << ") ";
            }
            std::cout << "-> " << phase_conflicts << "\n";
        }
    }

    return half0_max + half1_max;
}

void analyze_xor_transform() {
    std::cout << "\n=== XOR Transform Analysis ===\n";
    std::cout << "Plain vs XOR offsets for k=0 (first column):\n";
    std::cout << "m  | plain_off | plain_bank | xor_off | xor_bank\n";
    std::cout << "---|-----------|------------|---------|--------\n";

    for (int m = 0; m < 64; m += 8) {
        int plain_off = plain_offset(m, 0);
        int plain_bank = (plain_off * 2 / 4) % 32;
        int xor_off = xor_offset(m, 0);
        int xor_bank = (xor_off * 2 / 4) % 32;
        std::cout << m << "  | " << plain_off << " | " << plain_bank
                  << " | " << xor_off << " | " << xor_bank << "\n";
    }

    std::cout << "\nXOR effect on Phase 0 (lanes 0-7, k=0, dm=0):\n";
    std::cout << "Lane | m | plain_bank | xor_bank\n";
    for (int lane = 0; lane < 8; lane++) {
        int k = 0;  // lane / 8 = 0
        int m = lane * 8;  // (lane % 8) * 8
        int plain_off = plain_offset(m, k);
        int plain_bank = (plain_off * 2 / 4) % 32;
        int xor_off = xor_offset(m, k);
        int xor_bank = (xor_off * 2 / 4) % 32;
        std::cout << "  " << lane << "  | " << m << " | " << plain_bank
                  << " | " << xor_bank << "\n";
    }
}

int main() {
    std::cout << "=== FP16 Conflict Calculator v4 ===\n";
    std::cout << "Model: Max conflicts per half-wavefront × 2\n\n";

    analyze_xor_transform();

    int target_plain = 7168;
    int target_xor = 3072;
    int total_iters = 16;

    std::cout << "\n=== Conflict Calculation ===\n\n";

    int per_iter_plain = 0, per_iter_xor = 0;

    for (int wf = 0; wf < kWavefronts; wf++) {
        for (int dm = 0; dm < 8; dm++) {
            per_iter_plain += calc_conflicts_per_dm(wf, dm, false);
            per_iter_xor += calc_conflicts_per_dm(wf, dm, true);
        }
    }

    int total_plain = per_iter_plain * total_iters;
    int total_xor = per_iter_xor * total_iters;

    std::cout << "Per iteration:\n";
    std::cout << "  PLAIN: " << per_iter_plain << " (target: 448)\n";
    std::cout << "  XOR:   " << per_iter_xor << " (target: 192)\n\n";

    std::cout << "Total (" << total_iters << " iterations):\n";
    std::cout << "  PLAIN: " << total_plain << " (target: " << target_plain << ")\n";
    std::cout << "  XOR:   " << total_xor << " (target: " << target_xor << ")\n\n";

    std::cout << "Match status:\n";
    std::cout << "  PLAIN: " << (total_plain == target_plain ? "MATCH" : "NO MATCH")
              << " (diff: " << (total_plain - target_plain) << ")\n";
    std::cout << "  XOR:   " << (total_xor == target_xor ? "MATCH" : "NO MATCH")
              << " (diff: " << (total_xor - target_xor) << ")\n\n";

    // Detailed view for WF0
    std::cout << "=== Detail: WF0, dm=0 (PLAIN) ===\n";
    calc_conflicts_per_dm(0, 0, false, true);
    std::cout << "Total: " << calc_conflicts_per_dm(0, 0, false) << "\n\n";

    std::cout << "=== Detail: WF0, dm=0 (XOR) ===\n";
    calc_conflicts_per_dm(0, 0, true, true);
    std::cout << "Total: " << calc_conflicts_per_dm(0, 0, true) << "\n";

    return 0;
}
