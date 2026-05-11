/*
 * FP16 Conflict Calculator - Per-Instruction Model
 *
 * Hypothesis: Profiler counts conflicts per ds_read instruction (64 lanes total),
 * not per phase (8 lanes). The 64 lanes are processed in 8 phases, but
 * the counter might aggregate to one count per instruction.
 *
 * If 8 phases each have 7 conflicts = 56 conflicts per instruction
 * But if profiler counts max(conflicts_per_phase) = 7 per instruction
 * Or if profiler counts (unique_slots - 1) across all 64 lanes = ?
 *
 * Let's try different models and see which matches:
 * - Target NO XOR: 7168 / 16 iterations / 4 WFs / 8 dm = 14 per (WF, dm)
 * - Target XOR: 3072 / 16 / 4 / 8 = 6 per (WF, dm)
 */

#include <iostream>
#include <set>
#include <map>
#include <vector>

constexpr int kM = 64;
constexpr int kK = 32;
constexpr int kBlockSize = 256;
constexpr int kWavefronts = 4;
constexpr int kMLdsLayer = 2;
constexpr int kKPack = 8;

int plain_offset(int m, int k) {
    return m * kK + k;
}

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

// Model 1: Sum of conflicts across all 8 phases
int model1_sum_phases(int wf, int dm, bool use_xor) {
    auto calc_offset = use_xor ? xor_offset : plain_offset;
    int total = 0;

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

        for (const auto& [bank, slots] : bank_slots) {
            if (slots.size() > 1) {
                total += slots.size() - 1;
            }
        }
    }
    return total;
}

// Model 2: Max conflicts across phases (if counter reports worst case)
int model2_max_phase(int wf, int dm, bool use_xor) {
    auto calc_offset = use_xor ? xor_offset : plain_offset;
    int max_conflicts = 0;

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
        if (phase_conflicts > max_conflicts) {
            max_conflicts = phase_conflicts;
        }
    }
    return max_conflicts;
}

// Model 3: 32-lane halves (if hardware processes 32 lanes together)
int model3_half_wavefront(int wf, int dm, bool use_xor) {
    auto calc_offset = use_xor ? xor_offset : plain_offset;
    int total = 0;

    for (int half = 0; half < 2; half++) {
        std::map<int, std::set<int>> bank_slots;

        for (int lane_in_half = 0; lane_in_half < 32; lane_in_half++) {
            int lane = half * 32 + lane_in_half;
            int k = wf * 8 + lane / 8;
            int m = (lane % 8) * 8 + dm;

            int offset = calc_offset(m, k);
            int byte_addr = offset * 2;
            int slot = byte_addr / 4;
            int bank = slot % 32;
            bank_slots[bank].insert(slot);
        }

        for (const auto& [bank, slots] : bank_slots) {
            if (slots.size() > 1) {
                total += slots.size() - 1;
            }
        }
    }
    return total;
}

// Model 4: Full 64-lane wavefront together
int model4_full_wavefront(int wf, int dm, bool use_xor) {
    auto calc_offset = use_xor ? xor_offset : plain_offset;
    std::map<int, std::set<int>> bank_slots;

    for (int lane = 0; lane < 64; lane++) {
        int k = wf * 8 + lane / 8;
        int m = (lane % 8) * 8 + dm;

        int offset = calc_offset(m, k);
        int byte_addr = offset * 2;
        int slot = byte_addr / 4;
        int bank = slot % 32;
        bank_slots[bank].insert(slot);
    }

    int total = 0;
    for (const auto& [bank, slots] : bank_slots) {
        if (slots.size() > 1) {
            total += slots.size() - 1;
        }
    }
    return total;
}

int main() {
    std::cout << "=== FP16 Conflict Calculator - Multiple Models ===\n\n";

    // Targets
    int target_plain = 7168;
    int target_xor = 3072;
    int total_iters = 16;  // 4 blocks × 4 K-iters

    std::cout << "Targets:\n";
    std::cout << "  NO XOR: " << target_plain << " total, "
              << (target_plain / total_iters) << " per iter, "
              << (target_plain / total_iters / kWavefronts / 8) << " per (WF, dm)\n";
    std::cout << "  XOR:    " << target_xor << " total, "
              << (target_xor / total_iters) << " per iter, "
              << (target_xor / total_iters / kWavefronts / 8) << " per (WF, dm)\n\n";

    // Test each model
    auto test_model = [&](const char* name, auto model_func) {
        int sum_plain = 0, sum_xor = 0;
        for (int wf = 0; wf < kWavefronts; wf++) {
            for (int dm = 0; dm < 8; dm++) {
                sum_plain += model_func(wf, dm, false);
                sum_xor += model_func(wf, dm, true);
            }
        }

        int total_plain = sum_plain * total_iters;
        int total_xor = sum_xor * total_iters;

        std::cout << name << ":\n";
        std::cout << "  Per iter: plain=" << sum_plain << ", xor=" << sum_xor << "\n";
        std::cout << "  Total:    plain=" << total_plain << " (target " << target_plain << ", diff " << (total_plain - target_plain) << ")\n";
        std::cout << "            xor=" << total_xor << " (target " << target_xor << ", diff " << (total_xor - target_xor) << ")\n\n";
    };

    test_model("Model 1: Sum of 8 phases", model1_sum_phases);
    test_model("Model 2: Max per phase", model2_max_phase);
    test_model("Model 3: 32-lane halves", model3_half_wavefront);
    test_model("Model 4: Full 64-lane WF", model4_full_wavefront);

    // Detailed analysis for WF0, dm=0
    std::cout << "=== Detail: WF0, dm=0, PLAIN ===\n";
    std::cout << "Phase | Lane Range | k values | Conflicts\n";
    std::cout << "------|------------|----------|----------\n";

    for (int phase = 0; phase < 8; phase++) {
        std::map<int, std::set<int>> bank_slots;
        std::cout << "  " << phase << "   | " << (phase*8) << "-" << (phase*8+7) << "      | ";

        for (int lane_in_phase = 0; lane_in_phase < 8; lane_in_phase++) {
            int lane = phase * 8 + lane_in_phase;
            int k = lane / 8;  // WF0, so no wf*8 offset
            int m = (lane % 8) * 8 + 0;  // dm=0

            if (lane_in_phase == 0) std::cout << k;
            else if (k != (lane - 1) / 8) std::cout << "," << k;

            int offset = plain_offset(m, k);
            int byte_addr = offset * 2;
            int slot = byte_addr / 4;
            int bank = slot % 32;
            bank_slots[bank].insert(slot);
        }
        std::cout << "        | ";

        int phase_conflicts = 0;
        for (const auto& [bank, slots] : bank_slots) {
            if (slots.size() > 1) {
                phase_conflicts += slots.size() - 1;
            }
        }
        std::cout << phase_conflicts << "\n";
    }

    // Phase 0: All 8 lanes access k=0!
    std::cout << "\nPhase 0 detail:\n";
    std::cout << "Lane | k | m | offset | slot | bank\n";
    for (int lane = 0; lane < 8; lane++) {
        int k = lane / 8;  // = 0 for all phase 0 lanes
        int m = (lane % 8) * 8;  // = 0, 8, 16, 24, 32, 40, 48, 56
        int offset = plain_offset(m, k);
        int slot = offset * 2 / 4;
        int bank = slot % 32;
        std::cout << "  " << lane << "  | " << k << " | " << m << " | " << offset << " | " << slot << " | " << bank << "\n";
    }

    return 0;
}
