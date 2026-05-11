// Test actual bank conflicts using the XOR descriptor
// Key: Check ALL accesses made by each phase, count conflicts
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

    // Same XOR descriptor from 04_row_major_xor.cpp
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

// Phase groupings from tile distribution (actual phase structure)
static std::vector<std::vector<int>> get_read_phases() {
    return {
        {0, 1, 2, 3, 20, 21, 22, 23},
        {4, 5, 6, 7, 16, 17, 18, 19},
        {8, 9, 10, 11, 28, 29, 30, 31},
        {12, 13, 14, 15, 24, 25, 26, 27},
        {32, 33, 34, 35, 52, 53, 54, 55},
        {36, 37, 38, 39, 48, 49, 50, 51},
        {40, 41, 42, 43, 60, 61, 62, 63},
        {44, 45, 46, 47, 56, 57, 58, 59}
    };
}

// Given the distribution encoding for [K, M]:
// K0=1, K1=4, K2=8, M0=8, M1=8
// This function returns what (k, m) coordinates a lane accesses
// across its K1 and M1 iterations
struct LaneAccess {
    int k;      // k coordinate
    int m;      // m coordinate
};

std::vector<LaneAccess> get_lane_accesses(int lane_id) {
    // From distribution encoding analysis:
    // K2 = 8, threads distributed across K2 dimension via lane_id % 8
    // M0 = 8, threads distributed across M0 dimension via lane_id / 8
    //
    // Each thread iterates over:
    //   K1 = 4 (4 iterations along K)
    //   M1 = 8 (vector load of 8 M elements)

    int k2_idx = lane_id % 8;  // 0-7
    int m0_idx = lane_id / 8;  // 0-7 (for 64 threads)

    std::vector<LaneAccess> accesses;

    // For K1 iterations (0-3), for each M1 elements (0-7)
    for (int k1 = 0; k1 < 4; k1++) {
        int k = k1 * 8 + k2_idx;  // k = K1_idx * K2 + K2_idx
        int m_base = m0_idx * 8;  // M0_idx * M1

        for (int m1 = 0; m1 < 8; m1++) {
            int m = m_base + m1;
            accesses.push_back({k, m});
        }
    }

    return accesses;
}

template<bool UseXor>
void analyze_conflicts()
{
    using DataType = half_t;
    constexpr index_t DataTypeSize = sizeof(DataType);

    constexpr auto desc_km = TestDescriptors<DataType, UseXor>::MakeLdsDescriptorKM();

    std::cout << "\n=== " << (UseXor ? "WITH XOR" : "WITHOUT XOR") << " ===\n\n";

    index_t total_conflicts = 0;
    auto phases = get_read_phases();

    // Process each phase (8 threads execute simultaneously)
    for (size_t phase_idx = 0; phase_idx < phases.size(); phase_idx++) {
        const auto& phase = phases[phase_idx];

        std::cout << "Phase " << phase_idx << " (lanes: ";
        for (int lane : phase) std::cout << lane << " ";
        std::cout << "):\n";

        index_t phase_conflicts = 0;

        // For each K1 iteration (4 iterations)
        for (int k1_iter = 0; k1_iter < 4; k1_iter++) {
            // For each M1 element (vector load position 0-7)
            for (int m1_pos = 0; m1_pos < 8; m1_pos++) {

                // Map bank -> {lanes, slots}
                std::map<index_t, std::vector<int>> bank_to_lanes;
                std::map<index_t, std::set<index_t>> bank_to_slots;

                // Check each lane in this phase
                for (int lane : phase) {
                    int k2_idx = lane % 8;
                    int m0_idx = lane / 8;

                    int k = k1_iter * 8 + k2_idx;
                    int m = m0_idx * 8 + m1_pos;

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
                            // Multiple slots = real conflict
                            phase_conflicts += (lanes.size() - 1);
                        }
                        // Same slot with FP16 = 0 conflicts
                    }
                }
            }
        }

        std::cout << "  Conflicts: " << phase_conflicts << "\n";
        total_conflicts += phase_conflicts;
    }

    std::cout << "\n=== TOTAL CONFLICTS PER TILE (64x32) ===\n";
    std::cout << "  Total: " << total_conflicts << "\n";
    std::cout << "  Scaled (4 blocks): " << total_conflicts * 4 << "\n\n";
}

int main()
{
    std::cout << "=============================================\n";
    std::cout << "Bank Conflict Analysis - Actual Descriptor\n";
    std::cout << "=============================================\n";
    std::cout << "\nDistribution parameters:\n";
    std::cout << "  K0=1, K1=4, K2=8\n";
    std::cout << "  M0=8, M1=8\n";
    std::cout << "  K2_idx = lane % 8\n";
    std::cout << "  M0_idx = lane / 8\n\n";

    analyze_conflicts<false>();  // Without XOR
    analyze_conflicts<true>();   // With XOR

    std::cout << "=============================================\n";
    std::cout << "PROFILER COMPARISON:\n";
    std::cout << "  WITHOUT XOR: 7,168\n";
    std::cout << "  WITH XOR:    3,072\n";
    std::cout << "=============================================\n";

    return 0;
}
