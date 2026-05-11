// Bank conflict calculator based on assembly analysis
// Key insight: All 8 threads in a phase read SAME column (base address) with row-stride offsets
#include <iostream>
#include <set>
#include <map>
#include <vector>
#include "ck_tile/core.hpp"

using namespace ck_tile;

template<typename DataType, bool UseXor>
struct AssemblyBasedCalculator
{
    static constexpr index_t kM = 64;
    static constexpr index_t kK = 32;
    static constexpr index_t kKPack = 8;

    // LDS descriptor for READ [K, M] (transposed view)
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

// Phase groupings from actual tile distribution
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
void analyze_conflicts_assembly_based()
{
    using DataType = half_t;
    constexpr index_t M = 64;
    constexpr index_t K = 32;
    constexpr index_t DataTypeSize = sizeof(DataType);

    constexpr auto desc_km = AssemblyBasedCalculator<DataType, UseXor>::MakeLdsDescriptorKM();

    std::cout << "\n=== FP16 " << (UseXor ? "WITH" : "WITHOUT") << " XOR (Assembly-Based) ===\n\n";

    index_t total_conflicts = 0;
    auto read_phases = get_read_phases();

    // ASSEMBLY INSIGHT: Each phase reads one column at a time
    // All 8 threads read SAME column (k value), different rows (m values)
    // Pattern: base_address + {0, 64, 128, 192, 256, 320, 384, 448} byte offsets

    std::cout << "Transpose READ pattern (column-wise):\n";
    std::cout << "  From assembly: All threads in phase use SAME base address\n";
    std::cout << "  Different row offsets: {0, 64, 128, 192, 256, 320, 384, 448} bytes\n";
    std::cout << "  Row stride: 64 bytes (32 elements × 2 bytes/element)\n\n";

    // For each column (k=0 to k=31)
    for(index_t k = 0; k < K; k++)
    {
        // For each dm step (8 dm values per thread)
        for(index_t dm = 0; dm < 8; dm++)
        {
            // Count conflicts for this (k, dm) access across all phases
            // In reality, phases execute sequentially, so we count per phase
            for(const auto& phase : read_phases)
            {
                // Map bank -> {lanes, slots}
                std::map<index_t, std::vector<index_t>> bank_to_lanes;
                std::map<index_t, std::set<index_t>> bank_to_slots;

                // Each thread in this phase reads (k, m) where m = m_start + dm
                for(index_t lane : phase)
                {
                    index_t m0_idx = lane / 8;  // Which M group (0-7)
                    index_t m_start = m0_idx * 8;
                    index_t m = m_start + dm;

                    if(m >= M) continue;

                    // Calculate LDS address for [k, m]
                    auto offset = desc_km.calculate_offset(make_tuple(k, m));
                    index_t byte_offset = offset * DataTypeSize;
                    index_t slot = byte_offset / 4;
                    index_t bank = slot % 32;

                    bank_to_lanes[bank].push_back(lane);
                    bank_to_slots[bank].insert(slot);
                }

                // Count conflicts per bank
                for(const auto& [bank, lanes] : bank_to_lanes)
                {
                    if(lanes.size() > 1)
                    {
                        const auto& slots = bank_to_slots[bank];
                        if(slots.size() > 1)
                        {
                            // Multiple threads, different slots → conflicts
                            total_conflicts += (lanes.size() - 1);
                        }
                        // else: same slot → 0 conflicts (FP16 optimization)
                    }
                }
            }
        }
    }

    std::cout << "Conflict calculation:\n";
    std::cout << "  Columns (k): " << K << "\n";
    std::cout << "  DM steps: 8\n";
    std::cout << "  Phases: " << read_phases.size() << "\n";
    std::cout << "  Total access points: " << K << " × 8 × " << read_phases.size() << " = " << K * 8 * read_phases.size() << "\n\n";

    std::cout << "CONFLICTS PER TILE (64×32):\n";
    std::cout << "  Total: " << total_conflicts << "\n\n";

    // Example: Show bank pattern for column k=0, dm=0, Phase 0
    std::cout << "Example: Column k=0, dm=0, Phase 0:\n";
    const auto& phase0 = read_phases[0];
    std::map<index_t, std::vector<std::tuple<index_t, index_t>>> bank_pattern;

    for(index_t lane : phase0)
    {
        index_t m0_idx = lane / 8;
        index_t m = m0_idx * 8;  // dm=0

        auto offset = desc_km.calculate_offset(make_tuple(0, m));
        index_t byte_offset = offset * DataTypeSize;
        index_t slot = byte_offset / 4;
        index_t bank = slot % 32;

        bank_pattern[bank].push_back({lane, slot});
    }

    for(const auto& [bank, lane_slots] : bank_pattern)
    {
        std::cout << "  Bank " << bank << ": ";
        for(const auto& [lane, slot] : lane_slots)
        {
            std::cout << "(lane " << lane << ", slot " << slot << ") ";
        }
        if(lane_slots.size() > 1)
        {
            std::set<index_t> unique_slots;
            for(const auto& [_, slot] : lane_slots)
                unique_slots.insert(slot);
            if(unique_slots.size() > 1)
                std::cout << " → " << (lane_slots.size() - 1) << "-way conflict";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // Scale by number of blocks
    index_t num_blocks = 4;  // M=256, kM=64 → 4 blocks
    std::cout << "Scaled by number of blocks (" << num_blocks << " blocks for M=256):\n";
    std::cout << "  Total conflicts: " << total_conflicts * num_blocks << "\n\n";
}

int main()
{
    std::cout << "=================================================\n";
    std::cout << "Bank Conflict Analysis (Assembly-Based)\n";
    std::cout << "=================================================\n";
    std::cout << "\nKey insight from assembly analysis:\n";
    std::cout << "  ds_read_u16 v8,  v6          // offset 0\n";
    std::cout << "  ds_read_u16 v9,  v6 offset:64\n";
    std::cout << "  ds_read_u16 v10, v6 offset:128\n";
    std::cout << "  ...\n";
    std::cout << "All 8 reads use SAME base (v6) = same column!\n";
    std::cout << "Different offsets = different rows (stride 64 bytes)\n\n";

    analyze_conflicts_assembly_based<false>(); // Without XOR
    analyze_conflicts_assembly_based<true>();  // With XOR

    std::cout << "\n===============================================\n";
    std::cout << "PROFILER COMPARISON:\n";
    std::cout << "===============================================\n";
    std::cout << "  Profiler WITHOUT XOR: 7,168\n";
    std::cout << "  Profiler WITH XOR:    3,072\n\n";

    return 0;
}
