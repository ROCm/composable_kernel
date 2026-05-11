// Correct conflict model:
// Within one bank, each UNIQUE SLOT requires a separate cycle
// Multiple threads accessing the SAME slot = 0 conflicts between them
// Conflicts = (number of unique slots per bank) - 1
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

template<bool UseXor>
void analyze_slot_based()
{
    using DataType = half_t;
    constexpr index_t DataTypeSize = sizeof(DataType);

    constexpr auto desc_km = TestDescriptors<DataType, UseXor>::MakeLdsDescriptorKM();

    std::cout << "\n=== SLOT-BASED MODEL (" << (UseXor ? "XOR" : "PLAIN") << ") ===\n\n";

    index_t total_conflicts = 0;

    // Each K1 iteration is a separate wavefront operation
    for (int k1 = 0; k1 < 4; k1++) {
        std::cout << "K1 iteration " << k1 << ":\n";
        index_t k1_conflicts = 0;

        // Each dm step is executed by ALL 64 threads SIMULTANEOUSLY
        for (int dm = 0; dm < 8; dm++) {
            // Group by bank, then by slot within bank
            // bank -> set of unique slots
            std::set<index_t> bank_slots[32];

            for (int lane = 0; lane < 64; lane++) {
                int k2_idx = lane % 8;
                int m0_idx = lane / 8;
                int k = k1 * 8 + k2_idx;
                int m = m0_idx * 8 + dm;

                auto offset = desc_km.calculate_offset(make_tuple(k, m));
                index_t byte_offset = offset * DataTypeSize;
                index_t slot = byte_offset / 4;
                index_t bank = slot % 32;

                bank_slots[bank].insert(slot);
            }

            // Conflicts = (unique slots per bank - 1) for each bank
            index_t dm_conflicts = 0;
            for (int bank = 0; bank < 32; bank++) {
                int unique_slots = bank_slots[bank].size();
                if (unique_slots > 1) {
                    dm_conflicts += (unique_slots - 1);
                }
            }

            k1_conflicts += dm_conflicts;
        }

        std::cout << "  Conflicts: " << k1_conflicts << "\n";
        total_conflicts += k1_conflicts;
    }

    std::cout << "\n=== SUMMARY ===\n";
    std::cout << "  Total per tile (64x32): " << total_conflicts << "\n";
    std::cout << "  Scaled (4 blocks): " << total_conflicts * 4 << "\n";
}

// Debug: show slots per bank for one dm step
template<bool UseXor>
void debug_dm0_slots()
{
    using DataType = half_t;
    constexpr index_t DataTypeSize = sizeof(DataType);

    constexpr auto desc_km = TestDescriptors<DataType, UseXor>::MakeLdsDescriptorKM();

    std::cout << "\n=== DEBUG: k1=0, dm=0 (" << (UseXor ? "XOR" : "PLAIN") << ") ===\n";
    std::cout << "Slot grouping within each bank:\n\n";

    // bank -> slot -> list of lanes
    std::map<index_t, std::map<index_t, std::vector<int>>> bank_slot_lanes;

    for (int lane = 0; lane < 64; lane++) {
        int k2_idx = lane % 8;
        int m0_idx = lane / 8;
        int k = k2_idx;  // k1=0
        int m = m0_idx * 8;  // dm=0

        auto offset = desc_km.calculate_offset(make_tuple(k, m));
        index_t byte_offset = offset * DataTypeSize;
        index_t slot = byte_offset / 4;
        index_t bank = slot % 32;

        bank_slot_lanes[bank][slot].push_back(lane);
    }

    for (const auto& bank_entry : bank_slot_lanes) {
        index_t bank = bank_entry.first;
        const auto& slot_map = bank_entry.second;

        std::cout << "Bank " << bank << " (" << slot_map.size() << " unique slots):\n";
        for (const auto& slot_entry : slot_map) {
            index_t slot = slot_entry.first;
            const auto& lanes = slot_entry.second;

            std::cout << "  Slot " << slot << ": lanes [";
            for (size_t i = 0; i < lanes.size(); i++) {
                if (i > 0) std::cout << ", ";
                std::cout << lanes[i];
            }
            std::cout << "]\n";
        }
        std::cout << "  → Conflicts: " << (slot_map.size() > 1 ? slot_map.size() - 1 : 0) << "\n\n";
    }
}

int main()
{
    std::cout << "=============================================\n";
    std::cout << "Slot-Based Conflict Model\n";
    std::cout << "=============================================\n";
    std::cout << "Key insight: Multiple threads accessing the SAME\n";
    std::cout << "4-byte slot get serviced together (0 conflicts).\n";
    std::cout << "Only UNIQUE slots within a bank cause conflicts.\n";
    std::cout << "Conflicts per bank per dm = (unique_slots - 1)\n";
    std::cout << "=============================================\n";

    debug_dm0_slots<false>();
    debug_dm0_slots<true>();

    analyze_slot_based<false>();
    analyze_slot_based<true>();

    std::cout << "\n=============================================\n";
    std::cout << "PROFILER COMPARISON:\n";
    std::cout << "  WITHOUT XOR: 7,168\n";
    std::cout << "  WITH XOR:    3,072\n";
    std::cout << "=============================================\n";

    return 0;
}
