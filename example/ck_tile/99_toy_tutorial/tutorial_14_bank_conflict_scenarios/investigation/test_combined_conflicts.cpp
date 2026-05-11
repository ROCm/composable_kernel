// Combine INTRA-LANE and INTER-LANE conflicts
// Now we understand:
// - INTRA-LANE: One thread's vector load hitting same bank multiple times
// - INTER-LANE: Different threads in same phase hitting same bank with different slots
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
void analyze_combined()
{
    using DataType = half_t;
    constexpr index_t DataTypeSize = sizeof(DataType);

    constexpr auto desc_km = TestDescriptors<DataType, UseXor>::MakeLdsDescriptorKM();

    std::cout << "\n=== " << (UseXor ? "WITH XOR" : "WITHOUT XOR") << " ===\n\n";

    index_t total_intra = 0;
    index_t total_inter = 0;

    auto phases = get_phases();

    // Process each phase
    for (size_t p = 0; p < phases.size(); p++) {
        const auto& phase = phases[p];
        index_t phase_intra = 0;
        index_t phase_inter = 0;

        // K1 iterations
        for (int k1 = 0; k1 < 4; k1++) {

            // For INTRA-LANE: each lane's vector load
            for (int lane : phase) {
                int k2_idx = lane % 8;
                int m0_idx = lane / 8;
                int k = k1 * 8 + k2_idx;
                int m_start = m0_idx * 8;

                std::map<index_t, int> lane_bank_hits;
                for (int m1 = 0; m1 < 8; m1++) {
                    int m = m_start + m1;
                    auto offset = desc_km.calculate_offset(make_tuple(k, m));
                    index_t byte_offset = offset * DataTypeSize;
                    index_t slot = byte_offset / 4;
                    index_t bank = slot % 32;
                    lane_bank_hits[bank]++;
                }

                for (const auto& [bank, count] : lane_bank_hits) {
                    if (count > 1) {
                        phase_intra += (count - 1);
                    }
                }
            }

            // For INTER-LANE: different lanes hitting same bank with different slots
            // This happens at each m1 position within the vector load
            for (int m1 = 0; m1 < 8; m1++) {
                std::map<index_t, std::vector<std::pair<int, index_t>>> bank_entries;

                for (int lane : phase) {
                    int k2_idx = lane % 8;
                    int m0_idx = lane / 8;
                    int k = k1 * 8 + k2_idx;
                    int m = m0_idx * 8 + m1;

                    auto offset = desc_km.calculate_offset(make_tuple(k, m));
                    index_t byte_offset = offset * DataTypeSize;
                    index_t slot = byte_offset / 4;
                    index_t bank = slot % 32;

                    bank_entries[bank].push_back({lane, slot});
                }

                for (const auto& [bank, entries] : bank_entries) {
                    if (entries.size() > 1) {
                        std::set<index_t> unique_slots;
                        for (const auto& [lane, slot] : entries) {
                            unique_slots.insert(slot);
                        }
                        if (unique_slots.size() > 1) {
                            phase_inter += (entries.size() - 1);
                        }
                    }
                }
            }
        }

        std::cout << "Phase " << p << ": intra=" << phase_intra << ", inter=" << phase_inter << "\n";
        total_intra += phase_intra;
        total_inter += phase_inter;
    }

    std::cout << "\n=== SUMMARY (64x32 tile) ===\n";
    std::cout << "  Intra-lane: " << total_intra << "\n";
    std::cout << "  Inter-lane: " << total_inter << "\n";
    std::cout << "  Total:      " << (total_intra + total_inter) << "\n";
    std::cout << "\n  Scaled (4 blocks):\n";
    std::cout << "    Intra-lane: " << total_intra * 4 << "\n";
    std::cout << "    Inter-lane: " << total_inter * 4 << "\n";
    std::cout << "    Total:      " << (total_intra + total_inter) * 4 << "\n";
}

int main()
{
    std::cout << "=============================================\n";
    std::cout << "Combined INTRA + INTER Lane Conflict Analysis\n";
    std::cout << "=============================================\n";

    analyze_combined<false>();
    analyze_combined<true>();

    std::cout << "\n=============================================\n";
    std::cout << "PROFILER COMPARISON:\n";
    std::cout << "  WITHOUT XOR: 7,168\n";
    std::cout << "  WITH XOR:    3,072\n";
    std::cout << "=============================================\n";

    return 0;
}
