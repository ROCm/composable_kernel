// Model based on wavefront execution
// ALL 64 threads execute SIMULTANEOUSLY
// Each thread does 8 scalar reads (the dm loop)
// Conflicts occur when multiple threads access same bank at same time
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
void analyze_wavefront_model()
{
    using DataType = half_t;
    constexpr index_t DataTypeSize = sizeof(DataType);

    constexpr auto desc_km = TestDescriptors<DataType, UseXor>::MakeLdsDescriptorKM();

    std::cout << "\n=== WAVEFRONT MODEL (" << (UseXor ? "XOR" : "PLAIN") << ") ===\n\n";

    index_t total_conflicts = 0;

    // Each K1 iteration is a separate wavefront operation
    for (int k1 = 0; k1 < 4; k1++) {
        std::cout << "K1 iteration " << k1 << ":\n";
        index_t k1_conflicts = 0;

        // Each dm step is executed by ALL 64 threads SIMULTANEOUSLY
        for (int dm = 0; dm < 8; dm++) {
            // All 64 threads access their (k, m) location
            // bank -> {(lane, slot)}
            std::map<index_t, std::vector<std::pair<int, index_t>>> bank_accesses;

            for (int lane = 0; lane < 64; lane++) {
                int k2_idx = lane % 8;
                int m0_idx = lane / 8;
                int k = k1 * 8 + k2_idx;
                int m = m0_idx * 8 + dm;

                auto offset = desc_km.calculate_offset(make_tuple(k, m));
                index_t byte_offset = offset * DataTypeSize;
                index_t slot = byte_offset / 4;
                index_t bank = slot % 32;

                bank_accesses[bank].push_back({lane, slot});
            }

            // Count conflicts: N threads accessing same bank = N-1 conflicts
            // (unless they all access the SAME slot for FP16)
            index_t dm_conflicts = 0;
            for (const auto& [bank, accesses] : bank_accesses) {
                if (accesses.size() > 1) {
                    std::set<index_t> unique_slots;
                    for (const auto& [lane, slot] : accesses) {
                        unique_slots.insert(slot);
                    }
                    if (unique_slots.size() == 1) {
                        // All same slot = 0 conflicts (FP16 optimization)
                    } else {
                        // Different slots = conflicts
                        dm_conflicts += (accesses.size() - 1);
                    }
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

// Debug: show one dm step in detail
template<bool UseXor>
void debug_dm0()
{
    using DataType = half_t;
    constexpr index_t DataTypeSize = sizeof(DataType);

    constexpr auto desc_km = TestDescriptors<DataType, UseXor>::MakeLdsDescriptorKM();

    std::cout << "\n=== DEBUG: k1=0, dm=0 (" << (UseXor ? "XOR" : "PLAIN") << ") ===\n";
    std::cout << "All 64 threads accessing simultaneously:\n\n";

    std::map<index_t, std::vector<std::tuple<int, int, int, index_t>>> bank_groups;

    for (int lane = 0; lane < 64; lane++) {
        int k2_idx = lane % 8;
        int m0_idx = lane / 8;
        int k = k2_idx;  // k1=0
        int m = m0_idx * 8;  // dm=0

        auto offset = desc_km.calculate_offset(make_tuple(k, m));
        index_t byte_offset = offset * DataTypeSize;
        index_t slot = byte_offset / 4;
        index_t bank = slot % 32;

        bank_groups[bank].push_back({lane, k, m, slot});
    }

    for (const auto& [bank, entries] : bank_groups) {
        std::cout << "Bank " << bank << " (" << entries.size() << " accesses): ";
        std::set<index_t> unique_slots;
        for (const auto& [lane, k, m, slot] : entries) {
            std::cout << "L" << lane << "(k=" << k << ",m=" << m << ",s=" << slot << ") ";
            unique_slots.insert(slot);
        }
        if (entries.size() > 1) {
            std::cout << " → " << (unique_slots.size() == 1 ? "0 conflicts (same slot)" :
                std::to_string(entries.size()-1) + " conflicts");
        }
        std::cout << "\n";
    }
}

int main()
{
    std::cout << "=============================================\n";
    std::cout << "Wavefront Execution Model\n";
    std::cout << "=============================================\n";
    std::cout << "All 64 threads execute simultaneously.\n";
    std::cout << "Each dm step is one wavefront operation.\n";

    debug_dm0<false>();
    debug_dm0<true>();

    analyze_wavefront_model<false>();
    analyze_wavefront_model<true>();

    std::cout << "\n=============================================\n";
    std::cout << "PROFILER COMPARISON:\n";
    std::cout << "  WITHOUT XOR: 7,168\n";
    std::cout << "  WITH XOR:    3,072\n";
    std::cout << "=============================================\n";

    return 0;
}
