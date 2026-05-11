// Test if lanes within a phase access the SAME column (as suggested by assembly)
// Key insight from assembly: All 8 threads use SAME base address, different row offsets
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

// Phase groupings
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

// Assembly shows ALL threads in a phase read the SAME column
// But different m values (rows 0-7 for dm=0)
// So the pattern is: 8 threads × 8 dm steps = 64 M elements, all reading column k
template<bool UseXor>
void analyze_same_column_access()
{
    using DataType = half_t;
    constexpr index_t DataTypeSize = sizeof(DataType);

    constexpr auto desc_km = TestDescriptors<DataType, UseXor>::MakeLdsDescriptorKM();

    std::cout << "\n=== " << (UseXor ? "WITH XOR" : "WITHOUT XOR")
              << " (Same-Column Access Model) ===\n\n";

    index_t total_conflicts = 0;
    auto phases = get_read_phases();

    // ASSEMBLY MODEL:
    // Each phase processes ONE column at a time
    // 8 threads in the phase read 8 DIFFERENT rows of that column
    // Iterate through dm=0..7, so each thread reads 8 rows total

    for (size_t phase_idx = 0; phase_idx < phases.size(); phase_idx++) {
        const auto& phase = phases[phase_idx];
        index_t phase_conflicts = 0;

        std::cout << "Phase " << phase_idx << ":\n";

        // For each column k (0-31)
        for (int k = 0; k < 32; k++) {
            // For each dm step (0-7)
            for (int dm = 0; dm < 8; dm++) {
                // Map bank -> lanes and slots
                std::map<index_t, std::vector<int>> bank_to_lanes;
                std::map<index_t, std::set<index_t>> bank_to_slots;

                // ALL 8 threads read the SAME column k, but different rows
                // Thread layout:
                // - Lane 0: m = 0*8 + dm = dm
                // - Lane 1: m = 0*8 + dm = dm  (but lane 1 in phase has different M0_idx!)
                // Need to understand how lanes map to m values

                // From phase groupings:
                // Phase 0: lanes 0,1,2,3 have M0_idx = 0 (m = dm)
                //          lanes 20,21,22,23 have M0_idx = 2 (m = 16 + dm)
                // So each thread gets m = M0_idx * 8 + dm

                for (int lane : phase) {
                    int m0_idx = lane / 8;
                    int m = m0_idx * 8 + dm;

                    if (m >= 64) continue;

                    auto offset = desc_km.calculate_offset(make_tuple(k, m));
                    index_t byte_offset = offset * DataTypeSize;
                    index_t slot = byte_offset / 4;
                    index_t bank = slot % 32;

                    bank_to_lanes[bank].push_back(lane);
                    bank_to_slots[bank].insert(slot);
                }

                // Count conflicts for this (k, dm) access
                for (const auto& [bank, lanes] : bank_to_lanes) {
                    if (lanes.size() > 1) {
                        const auto& slots = bank_to_slots[bank];
                        if (slots.size() > 1) {
                            phase_conflicts += (lanes.size() - 1);
                        }
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

// Debug: Show what each lane in Phase 0 accesses
void debug_phase0_access()
{
    using DataType = half_t;
    constexpr index_t DataTypeSize = sizeof(DataType);

    constexpr auto desc_km = TestDescriptors<DataType, false>::MakeLdsDescriptorKM();

    std::cout << "\n=== DEBUG: Phase 0 Access Pattern (WITHOUT XOR) ===\n";
    std::cout << "\nFor column k=0, dm=0:\n";

    auto phases = get_read_phases();
    const auto& phase0 = phases[0];

    for (int lane : phase0) {
        int m0_idx = lane / 8;
        int m = m0_idx * 8 + 0;  // dm=0

        auto offset = desc_km.calculate_offset(make_tuple(0, m));
        index_t byte_offset = offset * DataTypeSize;
        index_t slot = byte_offset / 4;
        index_t bank = slot % 32;

        std::cout << "  Lane " << lane << ": M0_idx=" << m0_idx
                  << ", m=" << m << ", k=0 → offset=" << offset
                  << ", slot=" << slot << ", bank=" << bank << "\n";
    }

    std::cout << "\nNote: Phase 0 lanes are {0,1,2,3,20,21,22,23}\n";
    std::cout << "  Lanes 0-3: M0_idx = 0 → m = 0,1,2,3 (for dm=0,1,2,3)\n";
    std::cout << "  Lanes 20-23: M0_idx = 2 → m = 16,17,18,19 (for dm=0,1,2,3)\n";
}

int main()
{
    std::cout << "=============================================\n";
    std::cout << "Bank Conflict Analysis - Same Column Model\n";
    std::cout << "=============================================\n";
    std::cout << "\nKey insight from assembly:\n";
    std::cout << "  All 8 threads in a phase read the SAME column\n";
    std::cout << "  Different dm values give different rows\n\n";

    debug_phase0_access();

    analyze_same_column_access<false>();  // Without XOR
    analyze_same_column_access<true>();   // With XOR

    std::cout << "=============================================\n";
    std::cout << "PROFILER COMPARISON:\n";
    std::cout << "  WITHOUT XOR: 7,168\n";
    std::cout << "  WITH XOR:    3,072\n";
    std::cout << "=============================================\n";

    return 0;
}
