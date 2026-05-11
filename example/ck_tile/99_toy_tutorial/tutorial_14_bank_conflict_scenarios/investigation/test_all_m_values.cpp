// Test conflicts across ALL m values, not just m1=0
// Each phase accesses 8 different m values (m1=0..7)
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

std::vector<std::vector<int>> get_phases() {
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

template<bool UseXor>
void analyze_all_m_values()
{
    using DataType = half_t;
    constexpr index_t DataTypeSize = sizeof(DataType);

    constexpr auto desc_km = TestDescriptors<DataType, UseXor>::MakeLdsDescriptorKM();

    std::cout << "\n=== " << (UseXor ? "WITH XOR" : "WITHOUT XOR") << " ===\n\n";

    index_t total_conflicts = 0;
    auto phases = get_phases();

    // For each phase
    for (size_t p = 0; p < phases.size(); p++) {
        const auto& phase = phases[p];
        index_t phase_conflicts = 0;

        // K1 iterations (0-3)
        for (int k1 = 0; k1 < 4; k1++) {
            // M1 iterations - ALL 8 values (this is the vector load)
            for (int m1 = 0; m1 < 8; m1++) {

                std::map<index_t, std::vector<std::pair<int, index_t>>> bank_entries;

                // Each lane accesses its (k, m) coordinate
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

                    bank_entries[bank].push_back({lane, slot});
                }

                // Count conflicts
                for (const auto& [bank, entries] : bank_entries) {
                    if (entries.size() > 1) {
                        std::set<index_t> unique_slots;
                        for (const auto& [lane, slot] : entries) {
                            unique_slots.insert(slot);
                        }
                        if (unique_slots.size() > 1) {
                            phase_conflicts += (entries.size() - 1);
                        }
                    }
                }
            }
        }

        std::cout << "Phase " << p << ": " << phase_conflicts << " conflicts\n";
        total_conflicts += phase_conflicts;
    }

    std::cout << "\n=== TOTAL (64x32 tile) ===\n";
    std::cout << "  Conflicts: " << total_conflicts << "\n";
    std::cout << "  Scaled (4 blocks): " << total_conflicts * 4 << "\n";
}

// Debug: show one conflicting case in detail
template<bool UseXor>
void debug_conflict_case()
{
    using DataType = half_t;
    constexpr index_t DataTypeSize = sizeof(DataType);

    constexpr auto desc_km = TestDescriptors<DataType, UseXor>::MakeLdsDescriptorKM();

    std::cout << "\n=== DEBUG: Phase 0, k1=0, m1=1 (" << (UseXor ? "XOR" : "PLAIN") << ") ===\n";

    std::vector<int> phase0 = {0, 1, 2, 3, 20, 21, 22, 23};

    std::map<index_t, std::vector<std::tuple<int, int, int, index_t>>> bank_map;

    for (int lane : phase0) {
        int k2_idx = lane % 8;
        int m0_idx = lane / 8;
        int k = 0 * 8 + k2_idx;  // k1=0
        int m = m0_idx * 8 + 1;  // m1=1

        auto offset = desc_km.calculate_offset(make_tuple(k, m));
        index_t byte_offset = offset * DataTypeSize;
        index_t slot = byte_offset / 4;
        index_t bank = slot % 32;

        std::cout << "  Lane " << lane << ": (k=" << k << ",m=" << m << ") → offset=" << offset
                  << ", byte=" << byte_offset << ", slot=" << slot << ", bank=" << bank << "\n";

        bank_map[bank].push_back({lane, k, m, slot});
    }

    std::cout << "\n  Bank grouping:\n";
    for (const auto& [bank, entries] : bank_map) {
        std::cout << "    Bank " << bank << ": ";
        std::set<index_t> slots;
        for (const auto& [lane, k, m, slot] : entries) {
            std::cout << "L" << lane << "(slot=" << slot << ") ";
            slots.insert(slot);
        }
        if (entries.size() > 1) {
            std::cout << " → " << (slots.size() == 1 ? "SAME slot (0 conflicts)" : "DIFFERENT slots (" + std::to_string(entries.size()-1) + " conflicts)");
        }
        std::cout << "\n";
    }
}

int main()
{
    std::cout << "=============================================\n";
    std::cout << "Bank Conflict Analysis - ALL M Values\n";
    std::cout << "=============================================\n";

    analyze_all_m_values<false>();
    analyze_all_m_values<true>();

    debug_conflict_case<false>();
    debug_conflict_case<true>();

    std::cout << "\n=============================================\n";
    std::cout << "PROFILER COMPARISON:\n";
    std::cout << "  WITHOUT XOR: 7,168\n";
    std::cout << "  WITH XOR:    3,072\n";
    std::cout << "=============================================\n";

    return 0;
}
