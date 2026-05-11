/*
 * FP16 Conflict Calculator v5 - Correct XOR Transform
 *
 * Based on 04_row_major_xor.cpp XOR descriptor:
 *
 * Step 0: MLdsLayer = 128 / (32 * 2) = 2 for FP16
 *
 * Step 1: Reshape [M, K] = [64, 32] to 3D:
 *   [K/Pack*Layer, M/Layer, Pack] = [4*2, 32, 8] = [8, 32, 8]
 *   Strides: [8, 64, 1]
 *
 * Step 2: XOR transform on dimensions [M/Layer, K/Pack*Layer]
 *   new_dim1 = old_dim1 XOR old_dim0
 *   This means: m_idx' = m_idx XOR (k_idx / Pack * Layer)
 *
 * The physical offset calculation:
 *   Given logical (m, k):
 *     m_layer = m % MLdsLayer = m % 2
 *     m_div = m / MLdsLayer = m / 2
 *     k_pack = k % Pack = k % 8
 *     k_div = k / Pack = k / 8
 *     dim0 = k_div * MLdsLayer + m_layer
 *     dim1 = m_div
 *     xor_dim1 = dim1 XOR dim0 = (m/2) XOR (k/8 * 2 + m%2)
 *     offset = dim0 * Pack + xor_dim1 * (K * MLdsLayer) + k_pack
 *            = (k_div * 2 + m_layer) * 8 + xor_dim1 * 64 + k_pack
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

// XOR offset - following the descriptor transform exactly
int xor_offset(int m, int k) {
    int m_layer = m % kMLdsLayer;
    int m_div = m / kMLdsLayer;
    int k_pack = k % kKPack;
    int k_div = k / kKPack;

    // dim0 corresponds to K dimension component
    int dim0 = k_div * kMLdsLayer + m_layer;

    // dim1 corresponds to M dimension component
    int dim1 = m_div;

    // XOR transform: xor_dim1 = dim1 XOR dim0
    int xor_dim1 = dim1 ^ dim0;

    // Physical offset
    int offset = dim0 * kKPack + xor_dim1 * (kK * kMLdsLayer) + k_pack;

    return offset;
}

// Model: Max conflicts per half-wavefront
int calc_conflicts_per_dm(int wf, int dm, bool use_xor, bool verbose = false) {
    auto calc_offset = use_xor ? xor_offset : plain_offset;

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
    std::cout << "For column k=0 read (transpose pattern):\n";
    std::cout << "m  | m_layer | m_div | dim0 | dim1 | xor_dim1 | plain_off | xor_off | plain_bank | xor_bank\n";
    std::cout << "---|---------|-------|------|------|----------|-----------|---------|------------|--------\n";

    for (int m = 0; m < 64; m += 8) {
        int k = 0;
        int m_layer = m % kMLdsLayer;
        int m_div = m / kMLdsLayer;
        int k_div = k / kKPack;
        int dim0 = k_div * kMLdsLayer + m_layer;
        int dim1 = m_div;
        int xor_dim1 = dim1 ^ dim0;

        int plain_off = plain_offset(m, k);
        int xor_off = xor_offset(m, k);
        int plain_bank = (plain_off * 2 / 4) % 32;
        int xor_bank = (xor_off * 2 / 4) % 32;

        std::cout << m << "  | " << m_layer << " | " << m_div << " | " << dim0
                  << " | " << dim1 << " | " << xor_dim1 << " | " << plain_off
                  << " | " << xor_off << " | " << plain_bank << " | " << xor_bank << "\n";
    }

    std::cout << "\nPhase 0 (lanes 0-7) accessing k=0:\n";
    std::cout << "Lane | m | plain_bank | xor_bank\n";
    for (int lane = 0; lane < 8; lane++) {
        int k = 0;
        int m = lane * 8;
        int plain_off = plain_offset(m, k);
        int xor_off = xor_offset(m, k);
        int plain_bank = (plain_off * 2 / 4) % 32;
        int xor_bank = (xor_off * 2 / 4) % 32;
        std::cout << "  " << lane << "  | " << m << " | " << plain_bank << " | " << xor_bank << "\n";
    }
}

int main() {
    std::cout << "=== FP16 Conflict Calculator v5 - Correct XOR ===\n";

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

    // Detailed view
    std::cout << "=== Detail: WF0, dm=0 (PLAIN) ===\n";
    calc_conflicts_per_dm(0, 0, false, true);
    std::cout << "Total: " << calc_conflicts_per_dm(0, 0, false) << "\n\n";

    std::cout << "=== Detail: WF0, dm=0 (XOR) ===\n";
    calc_conflicts_per_dm(0, 0, true, true);
    std::cout << "Total: " << calc_conflicts_per_dm(0, 0, true) << "\n";

    return 0;
}
