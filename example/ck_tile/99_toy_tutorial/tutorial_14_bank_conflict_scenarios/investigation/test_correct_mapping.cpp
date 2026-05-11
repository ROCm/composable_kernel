// Find the correct lane-to-coordinate mapping
// The key is: each lane in a phase accesses DIFFERENT coordinates
// Phase 0 = {0,1,2,3,20,21,22,23} - 8 lanes that execute together

#include <iostream>
#include <set>
#include <map>
#include <vector>
#include "ck_tile/core.hpp"

using namespace ck_tile;

template<typename DataType, bool UseXor>
struct TestDescriptors
{
    static constexpr index_t kM = 64;
    static constexpr index_t kK = 32;
    static constexpr index_t kKPack = 8;

    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDescriptorKM()
    {
        if constexpr (UseXor)
        {
            constexpr auto DataTypeSize = sizeof(DataType);
            constexpr auto MLdsLayer =
                (32 * 4 / kK / DataTypeSize) < 1 ? 1 : (32 * 4 / kK / DataTypeSize);

            constexpr auto lds_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<kK / kKPack * MLdsLayer>{},
                           number<kM / MLdsLayer>{},
                           number<kKPack>{}),
                make_tuple(number<kKPack>{}, number<kK * MLdsLayer>{}, number<1>{}),
                number<kKPack>{},
                number<1>{});

            constexpr auto lds_desc_permuted = transform_tensor_descriptor(
                lds_desc_0,
                make_tuple(make_xor_transform(make_tuple(number<kM / MLdsLayer>{},
                                                         number<kK / kKPack * MLdsLayer>{})),
                           make_pass_through_transform(number<kKPack>{})),
                make_tuple(sequence<1, 0>{}, sequence<2>{}),
                make_tuple(sequence<1, 0>{}, sequence<2>{}));

            constexpr auto lds_desc_unmerged = transform_tensor_descriptor(
                lds_desc_permuted,
                make_tuple(make_unmerge_transform(
                               make_tuple(number<MLdsLayer>{}, number<kK / kKPack>{})),
                           make_pass_through_transform(number<kM / MLdsLayer>{}),
                           make_pass_through_transform(number<kKPack>{})),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

            constexpr auto lds_desc = transform_tensor_descriptor(
                lds_desc_unmerged,
                make_tuple(
                    make_merge_transform(make_tuple(number<kK / kKPack>{}, number<kKPack>{})),
                    make_merge_transform(make_tuple(number<kM / MLdsLayer>{}, number<MLdsLayer>{}))),
                make_tuple(sequence<2, 3>{}, sequence<1, 0>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return lds_desc;
        }
        else
        {
            return make_naive_tensor_descriptor(
                make_tuple(number<kK>{}, number<kM>{}),
                make_tuple(number<1>{}, number<kK>{}));
        }
    }
};

// From the distribution encoding:
// K0=1, K1=4, K2=8
// M0=8, M1=8
//
// The encoding pattern is:
//   tuple<sequence<K0, K1, K2>, sequence<M0, M1>>
//   tuple<sequence<1>, sequence<1, 2>>      <- which dims are thread-distributed
//   tuple<sequence<1>, sequence<2, 0>>      <- which indices map to threads
//
// This means:
//   - K1 is iterated (not thread-mapped), K2 is thread-mapped via index 2
//   - M0 is NOT iterated (thread-mapped via sequence<2>), M1 is iterated
//
// Thread mapping:
//   256 threads = 64 phases × 4 (K1 iterations)
//   Within 64 threads: M0=8 groups, each with K2=8 threads
//
// So for lane i (0-63):
//   M0_idx = i / 8  (which M group, 0-7)
//   K2_idx = i % 8  (which K within the group, 0-7)
//   M = M0_idx * M1 + m1_iter (m1_iter = 0..7)
//   K = k1_iter * K2 + K2_idx (k1_iter = 0..3)

// Phase structure comes from memory access coalescing
// Phases 0-7 correspond to different (M0_idx, K2_idx) combinations

// Let me trace through which lanes are in which phase
void analyze_phase_structure()
{
    std::cout << "=== Phase Structure Analysis ===\n\n";

    std::vector<std::vector<int>> phases = {
        {0, 1, 2, 3, 20, 21, 22, 23},
        {4, 5, 6, 7, 16, 17, 18, 19},
        {8, 9, 10, 11, 28, 29, 30, 31},
        {12, 13, 14, 15, 24, 25, 26, 27},
        {32, 33, 34, 35, 52, 53, 54, 55},
        {36, 37, 38, 39, 48, 49, 50, 51},
        {40, 41, 42, 43, 60, 61, 62, 63},
        {44, 45, 46, 47, 56, 57, 58, 59}
    };

    // Decode each lane's M0_idx and K2_idx
    std::cout << "Lane → (M0_idx, K2_idx) mapping:\n";
    for (size_t p = 0; p < phases.size(); p++) {
        std::cout << "Phase " << p << ": ";
        for (int lane : phases[p]) {
            int m0_idx = lane / 8;
            int k2_idx = lane % 8;
            std::cout << "L" << lane << "=(M" << m0_idx << ",K" << k2_idx << ") ";
        }
        std::cout << "\n";
    }

    std::cout << "\nPhase Pattern:\n";
    std::cout << "  Phase 0: M0={0,2}, K2={0,1,2,3}\n";
    std::cout << "  Phase 1: M0={0,2}, K2={4,5,6,7}\n";
    std::cout << "  Phase 2: M0={1,3}, K2={0,1,2,3}\n";
    std::cout << "  Phase 3: M0={1,3}, K2={4,5,6,7}\n";
    std::cout << "  Phase 4: M0={4,6}, K2={0,1,2,3}\n";
    std::cout << "  Phase 5: M0={4,6}, K2={4,5,6,7}\n";
    std::cout << "  Phase 6: M0={5,7}, K2={0,1,2,3}\n";
    std::cout << "  Phase 7: M0={5,7}, K2={4,5,6,7}\n";
}

// Now calculate conflicts properly
template<bool UseXor>
void analyze_conflicts_correct()
{
    using DataType = half_t;
    constexpr index_t DataTypeSize = sizeof(DataType);

    constexpr auto desc_km = TestDescriptors<DataType, UseXor>::MakeLdsDescriptorKM();

    std::cout << "\n=== " << (UseXor ? "WITH XOR" : "WITHOUT XOR")
              << " (Correct Mapping) ===\n\n";

    index_t total_conflicts = 0;

    std::vector<std::vector<int>> phases = {
        {0, 1, 2, 3, 20, 21, 22, 23},
        {4, 5, 6, 7, 16, 17, 18, 19},
        {8, 9, 10, 11, 28, 29, 30, 31},
        {12, 13, 14, 15, 24, 25, 26, 27},
        {32, 33, 34, 35, 52, 53, 54, 55},
        {36, 37, 38, 39, 48, 49, 50, 51},
        {40, 41, 42, 43, 60, 61, 62, 63},
        {44, 45, 46, 47, 56, 57, 58, 59}
    };

    // For each phase
    for (size_t p = 0; p < phases.size(); p++) {
        const auto& phase = phases[p];
        index_t phase_conflicts = 0;

        // K1 iterations (0-3)
        for (int k1 = 0; k1 < 4; k1++) {
            // M1 iterations (vector load positions 0-7)
            for (int m1 = 0; m1 < 8; m1++) {

                std::map<index_t, std::vector<int>> bank_to_lanes;
                std::map<index_t, std::set<index_t>> bank_to_slots;

                // Each lane accesses:
                //   k = k1 * 8 + K2_idx (where K2_idx = lane % 8)
                //   m = M0_idx * 8 + m1 (where M0_idx = lane / 8)
                for (int lane : phase) {
                    int k2_idx = lane % 8;
                    int m0_idx = lane / 8;

                    int k = k1 * 8 + k2_idx;
                    int m = m0_idx * 8 + m1;

                    if (k >= 32 || m >= 64) continue;

                    auto offset = desc_km.calculate_offset(make_tuple(k, m));
                    index_t byte_offset = offset * DataTypeSize;
                    index_t slot = byte_offset / 4;
                    index_t bank = slot % 32;

                    bank_to_lanes[bank].push_back(lane);
                    bank_to_slots[bank].insert(slot);
                }

                // Count conflicts
                for (const auto& [bank, lanes] : bank_to_lanes) {
                    if (lanes.size() > 1) {
                        const auto& slots = bank_to_slots[bank];
                        if (slots.size() > 1) {
                            // Different slots = real conflict
                            phase_conflicts += (lanes.size() - 1);
                        }
                        // Same slot = FP16 optimization, 0 conflicts
                    }
                }
            }
        }

        std::cout << "Phase " << p << ": " << phase_conflicts << " conflicts\n";
        total_conflicts += phase_conflicts;
    }

    std::cout << "\n=== TOTAL (64x32 tile) ===\n";
    std::cout << "  Conflicts: " << total_conflicts << "\n";
    std::cout << "  Scaled (4 blocks): " << total_conflicts * 4 << "\n\n";
}

// Debug: Show exactly what Phase 0 accesses for k1=0, m1=0
void debug_phase0()
{
    using DataType = half_t;
    constexpr index_t DataTypeSize = sizeof(DataType);

    constexpr auto desc_km_plain = TestDescriptors<DataType, false>::MakeLdsDescriptorKM();
    constexpr auto desc_km_xor = TestDescriptors<DataType, true>::MakeLdsDescriptorKM();

    std::vector<int> phase0 = {0, 1, 2, 3, 20, 21, 22, 23};

    std::cout << "\n=== DEBUG: Phase 0, k1=0, m1=0 ===\n\n";
    std::cout << "Lane | (M0,K2) | (k,m) | Plain offset | XOR offset | Plain bank | XOR bank\n";
    std::cout << "-----|---------|-------|--------------|------------|------------|----------\n";

    for (int lane : phase0) {
        int k2_idx = lane % 8;
        int m0_idx = lane / 8;
        int k = 0 * 8 + k2_idx;  // k1=0
        int m = m0_idx * 8 + 0;  // m1=0

        auto plain_offset = desc_km_plain.calculate_offset(make_tuple(k, m));
        auto xor_offset = desc_km_xor.calculate_offset(make_tuple(k, m));

        index_t plain_slot = (plain_offset * DataTypeSize) / 4;
        index_t xor_slot = (xor_offset * DataTypeSize) / 4;

        std::cout << " " << lane << "   | (" << m0_idx << "," << k2_idx << ")   | (" << k << "," << m << ")  | "
                  << plain_offset << "          | " << xor_offset << "          | "
                  << (plain_slot % 32) << "          | " << (xor_slot % 32) << "\n";
    }
}

int main()
{
    std::cout << "=============================================\n";
    std::cout << "Bank Conflict Analysis - Correct Mapping\n";
    std::cout << "=============================================\n\n";

    analyze_phase_structure();
    debug_phase0();
    analyze_conflicts_correct<false>();
    analyze_conflicts_correct<true>();

    std::cout << "=============================================\n";
    std::cout << "PROFILER COMPARISON:\n";
    std::cout << "  WITHOUT XOR: 7,168\n";
    std::cout << "  WITH XOR:    3,072\n";
    std::cout << "=============================================\n";

    return 0;
}
