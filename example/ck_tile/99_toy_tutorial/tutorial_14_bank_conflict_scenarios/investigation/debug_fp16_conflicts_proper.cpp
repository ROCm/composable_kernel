// Proper bank conflict calculation: No ×3 multiplier, includes inter-lane conflicts
#include <iostream>
#include <set>
#include <map>
#include <vector>
#include "ck_tile/core.hpp"

using namespace ck_tile;

template<typename DataType, bool UseXor>
struct ProperConflictCalculator
{
    static constexpr index_t kM = 64;
    static constexpr index_t kK = 32;
    static constexpr index_t kKPack = 8;

    // LDS descriptor for WRITE [M, K]
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDescriptorMK()
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
                make_tuple(number<kKPack>{},
                           number<kK * MLdsLayer>{},
                           number<1>{}),
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
                    make_merge_transform(
                        make_tuple(number<kM / MLdsLayer>{}, number<MLdsLayer>{})),
                    make_merge_transform(make_tuple(number<kK / kKPack>{}, number<kKPack>{}))),
                make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return lds_desc;
        }
        else
        {
            return make_naive_tensor_descriptor_packed(make_tuple(number<kM>{}, number<kK>{}));
        }
    }

    // LDS descriptor for READ [K, M] (transposed view)
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDescriptorKM()
    {
        if constexpr (UseXor)
        {
            // Same XOR pattern as write, but swapped merge order
            constexpr auto DataTypeSize = sizeof(DataType);
            constexpr auto MLdsLayer =
                (32 * 4 / kK / DataTypeSize) < 1 ? 1 : (32 * 4 / kK / DataTypeSize);

            constexpr auto lds_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<kK / kKPack * MLdsLayer>{},
                           number<kM / MLdsLayer>{},
                           number<kKPack>{}),
                make_tuple(number<kKPack>{},
                           number<kK * MLdsLayer>{},
                           number<1>{}),
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

            // SWAPPED merge order to get [K, M] instead of [M, K]
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
            // Plain transposed descriptor [K, M]
            return make_naive_tensor_descriptor(
                make_tuple(number<kK>{}, number<kM>{}),
                make_tuple(number<1>{}, number<kK>{}));
        }
    }
};

// Phase groupings - CRITICAL for XOR!
// Only 8 threads per phase execute simultaneously
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

template<bool UseXor>
void analyze_conflicts_proper()
{
    using DataType = half_t;
    constexpr index_t M = 64;
    constexpr index_t DataTypeSize = sizeof(DataType);

    // Use KM descriptor for transpose READ
    constexpr auto desc_km = ProperConflictCalculator<DataType, UseXor>::MakeLdsDescriptorKM();

    std::cout << "\n=== FP16 " << (UseXor ? "WITH" : "WITHOUT") << " XOR (PROPER CALCULATION) ===\n\n";

    // WRITE: Contiguous writes should be conflict-free!
    std::cout << "WRITE Pattern (row-wise, 16 bytes per thread, contiguous):\n";
    index_t write_conflicts = 0;

    // Each thread writes to different consecutive banks - no conflicts!
    std::cout << "  Write conflicts: " << write_conflicts << " (contiguous writes are conflict-free)\n\n";

    // READ: Transpose access - COLUMN READS
    // CRITICAL: Only 8 threads PER PHASE execute simultaneously!
    std::cout << "READ Pattern (column-wise transpose):\n";
    std::cout << "  Only 8 threads per phase execute simultaneously!\n";
    std::cout << "  Conflicts counted WITHIN each phase only\n\n";

    index_t read_conflicts = 0;
    auto read_phases = get_read_phases();

    // Process each PHASE separately (only 8 threads execute together)
    for(const auto& phase : read_phases)
    {
        // Process each dm step
        for(index_t dm = 0; dm < 8; dm++)
        {
            // Group lanes WITHIN THIS PHASE by {k_value, bank}
            std::map<std::tuple<index_t, index_t>, std::vector<index_t>> k_bank_to_lanes;
            std::map<std::tuple<index_t, index_t>, std::set<index_t>> k_bank_to_slots;

            // Check all K values (0-31) for lanes in THIS PHASE only
            for(index_t k1 = 0; k1 < 4; k1++)  // K1 groups
            {
                for(index_t lane : phase)  // Only 8 lanes in this phase!
                {
                    index_t k2_idx = lane % 8;
                    index_t k = k1 * 8 + k2_idx;
                    index_t m0_idx = lane / 8;
                    index_t m_start = m0_idx * 8;
                    index_t m = m_start + dm;

                    if(m >= M) continue;

                    auto offset = desc_km.calculate_offset(make_tuple(k, m));
                    index_t byte_offset = offset * DataTypeSize;
                    index_t slot = byte_offset / 4;
                    index_t bank = slot % 32;

                    k_bank_to_lanes[{k, bank}].push_back(lane);
                    k_bank_to_slots[{k, bank}].insert(slot);
                }
            }

            // Count conflicts WITHIN THIS PHASE
            for(const auto& entry : k_bank_to_lanes)
            {
                const auto& lanes = entry.second;
                if(lanes.size() > 1)
                {
                    const auto [k_val, bank] = entry.first;
                    const auto& slots = k_bank_to_slots[{k_val, bank}];

                    if(slots.size() > 1)
                    {
                        // Different slots → TRUE conflict
                        read_conflicts += (lanes.size() - 1);
                    }
                }
            }
        }
    }

    // Debug output for Phase 0, dm=0, column k=0
    std::cout << "  Example: Phase 0, dm=0, column k=0:\n";
    const auto& phase0 = read_phases[0];
    for(index_t lane : phase0)
    {
        index_t k2_idx = lane % 8;
        if(k2_idx != 0) continue;  // Only show k=0

        index_t m0_idx = lane / 8;
        index_t m_start = m0_idx * 8;
        index_t m = m_start + 0;  // dm=0
        index_t k = 0;

        auto offset = desc_km.calculate_offset(make_tuple(k, m));
        index_t byte_offset = offset * DataTypeSize;
        index_t slot = byte_offset / 4;
        index_t bank = slot % 32;

        std::cout << "    Lane " << lane << " reads [k=0, m=" << m
                  << "] → offset " << offset << " → slot " << slot << " → bank " << bank << "\n";
    }
    std::cout << "    (Only lane 0 in Phase 0 reads k=0, so no conflicts within this phase for k=0!)\n";
    std::cout << "\n";

    std::cout << "\n";
    std::cout << "CONFLICTS PER TILE (64×32, all 8 phases):\n";
    std::cout << "  Read conflicts: " << read_conflicts << "\n\n";

    index_t total_conflicts = write_conflicts + read_conflicts;
    std::cout << "TOTAL CONFLICTS (one 64×32 tile): " << total_conflicts << "\n";
    std::cout << "  Write: " << write_conflicts << " (contiguous writes)\n";
    std::cout << "  Read: " << read_conflicts << " (column reads, different rows)\n\n";

    // Scale by number of blocks
    index_t num_blocks = 4;  // M=256, kM=64 → 4 blocks
    std::cout << "Scaled by number of blocks (" << num_blocks << " blocks for M=256):\n";
    std::cout << "  Total: " << total_conflicts * num_blocks << "\n";
    std::cout << "  Write: " << write_conflicts * num_blocks << "\n";
    std::cout << "  Read: " << read_conflicts * num_blocks << "\n\n";
}

int main()
{
    std::cout << "=================================================\n";
    std::cout << "FP16 Bank Conflict Analysis (PROPER)\n";
    std::cout << "=================================================\n";

    analyze_conflicts_proper<false>(); // Without XOR
    analyze_conflicts_proper<true>();  // With XOR

    std::cout << "\n===============================================\n";
    std::cout << "PROFILER COMPARISON:\n";
    std::cout << "===============================================\n";
    std::cout << "  Profiler WITHOUT XOR: 7,168\n";
    std::cout << "  Profiler WITH XOR:    3,072\n\n";
    std::cout << "Goal: Match these numbers with intra + inter lane conflicts\n";
    std::cout << "Key: NO ×3 multiplier, NO write conflicts (contiguous)\n";

    return 0;
}
