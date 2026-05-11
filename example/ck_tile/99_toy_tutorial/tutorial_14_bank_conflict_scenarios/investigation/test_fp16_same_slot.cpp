// Check FP16 same-slot optimization in detail
// When 2 FP16 values share the same 4-byte slot, accessing them together = 0 conflicts
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

// FP16 same-slot rule:
// - 2 FP16 values at adjacent k positions share a 4-byte slot
// - If multiple threads access the SAME slot, they can be serviced together = 0 conflicts
// - If threads access the same BANK but different SLOTS, that's a conflict

template<bool UseXor>
void analyze_with_fp16_slot_grouping()
{
    using DataType = half_t;
    constexpr index_t DataTypeSize = sizeof(DataType);

    constexpr auto desc_km = TestDescriptors<DataType, UseXor>::MakeLdsDescriptorKM();

    std::cout << "\n=== " << (UseXor ? "XOR" : "PLAIN") << " with FP16 Same-Slot Grouping ===\n\n";

    index_t total_conflicts = 0;

    // K1 iterations
    for (int k1 = 0; k1 < 4; k1++) {
        index_t k1_conflicts = 0;

        // Each dm step
        for (int dm = 0; dm < 8; dm++) {
            // Group by (bank, slot) - accesses to same slot don't conflict
            std::map<index_t, std::map<index_t, int>> bank_slot_counts;  // bank -> slot -> count

            for (int lane = 0; lane < 64; lane++) {
                int k2_idx = lane % 8;
                int m0_idx = lane / 8;
                int k = k1 * 8 + k2_idx;
                int m = m0_idx * 8 + dm;

                auto offset = desc_km.calculate_offset(make_tuple(k, m));
                index_t byte_offset = offset * DataTypeSize;
                index_t slot = byte_offset / 4;
                index_t bank = slot % 32;

                bank_slot_counts[bank][slot]++;
            }

            // Count conflicts per bank
            // For each bank: each unique slot adds 0 conflicts (FP16 optimization)
            // BUT: N unique slots in same bank = (total_accesses - N) conflicts?
            // Actually the rule is: within one cycle, ONE slot per bank can be serviced
            // So N unique slots = (N-1) cycles = need serialization

            // WAIT - that's not right either.
            // The FP16 optimization is: if 2 threads access SAME 4-byte slot,
            // they can be serviced in one cycle (no conflict).
            // If N threads access same bank but M different slots,
            // we need M cycles, so (M-1) conflicts per bank per cycle.

            // Actually let me think again:
            // - 64 threads access 64 locations
            // - Group by bank: bank 0 has some threads, bank 1 has some, etc.
            // - Within each bank:
            //   - Group by slot: threads accessing same slot = 0 conflicts between them
            //   - Different slots = serialized
            //   - Number of unique slots = number of cycles needed
            //   - Conflicts = total_accesses - num_unique_slots (since first slot is free)

            for (const auto& [bank, slot_counts] : bank_slot_counts) {
                int total_accesses = 0;
                int num_unique_slots = slot_counts.size();
                for (const auto& [slot, count] : slot_counts) {
                    total_accesses += count;
                }

                // Actually the profiler counts conflicts differently!
                // Conflicts = number of additional cycles beyond the first
                // For N unique slots: (N-1) cycles of delay = (N-1) conflicts per access?

                // Let me try: conflicts = (num_unique_slots - 1) per bank per dm step
                if (num_unique_slots > 1) {
                    k1_conflicts += (num_unique_slots - 1);
                }
            }
        }

        std::cout << "K1 " << k1 << ": " << k1_conflicts << " conflicts\n";
        total_conflicts += k1_conflicts;
    }

    std::cout << "\nTotal per tile: " << total_conflicts << "\n";
    std::cout << "Scaled (4 blocks): " << total_conflicts * 4 << "\n";
}

// Debug detail for one dm step
template<bool UseXor>
void debug_dm0_slot_detail()
{
    using DataType = half_t;
    constexpr index_t DataTypeSize = sizeof(DataType);

    constexpr auto desc_km = TestDescriptors<DataType, UseXor>::MakeLdsDescriptorKM();

    std::cout << "\n=== DEBUG dm=0, k1=0 (" << (UseXor ? "XOR" : "PLAIN") << ") ===\n";

    std::map<index_t, std::map<index_t, std::vector<int>>> bank_slot_lanes;

    for (int lane = 0; lane < 64; lane++) {
        int k2_idx = lane % 8;
        int m0_idx = lane / 8;
        int k = k2_idx;
        int m = m0_idx * 8;

        auto offset = desc_km.calculate_offset(make_tuple(k, m));
        index_t byte_offset = offset * DataTypeSize;
        index_t slot = byte_offset / 4;
        index_t bank = slot % 32;

        bank_slot_lanes[bank][slot].push_back(lane);
    }

    for (const auto& [bank, slot_map] : bank_slot_lanes) {
        std::cout << "Bank " << bank << ": " << slot_map.size() << " unique slots\n";
        for (const auto& [slot, lanes] : slot_map) {
            std::cout << "  Slot " << slot << " (" << lanes.size() << " lanes): ";
            for (int l : lanes) std::cout << l << " ";
            std::cout << "\n";
        }
    }
}

int main()
{
    std::cout << "=============================================\n";
    std::cout << "FP16 Same-Slot Optimization Analysis\n";
    std::cout << "=============================================\n";

    debug_dm0_slot_detail<false>();
    debug_dm0_slot_detail<true>();

    analyze_with_fp16_slot_grouping<false>();
    analyze_with_fp16_slot_grouping<true>();

    std::cout << "\n=============================================\n";
    std::cout << "PROFILER:\n";
    std::cout << "  WITHOUT XOR: 7,168\n";
    std::cout << "  WITH XOR:    3,072\n";
    std::cout << "=============================================\n";

    return 0;
}
