// Test: Do different wavefronts hit the same banks with XOR?
#include <iostream>
#include <set>
#include <map>
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
void analyze_inter_wf_conflicts()
{
    using DataType = half_t;
    constexpr index_t DataTypeSize = sizeof(DataType);

    constexpr auto desc_km = TestDescriptors<DataType, UseXor>::MakeLdsDescriptorKM();

    std::cout << "\n=== " << (UseXor ? "WITH XOR" : "WITHOUT XOR") << " ===\n\n";

    // For dm=0, check which banks each wavefront hits
    std::cout << "Banks accessed by each wavefront at dm=0:\n\n";

    std::set<int> all_wf_banks[4];  // Banks per wavefront

    for (int wf = 0; wf < 4; wf++) {
        int k1 = wf;
        std::set<int> banks_hit;

        for (int lane = 0; lane < 64; lane++) {
            int k2_idx = lane % 8;
            int m0_idx = lane / 8;
            int k = k1 * 8 + k2_idx;
            int m = m0_idx * 8 + 0;  // dm=0

            auto offset = desc_km.calculate_offset(make_tuple(k, m));
            int byte_offset = offset * DataTypeSize;
            int slot = byte_offset / 4;
            int bank = slot % 32;

            banks_hit.insert(bank);
        }

        all_wf_banks[wf] = banks_hit;

        std::cout << "WF" << wf << " (k=" << k1*8 << "-" << (k1*8+7) << "): banks { ";
        for (int b : banks_hit) {
            std::cout << b << " ";
        }
        std::cout << "}\n";
    }

    // Check for overlaps
    std::cout << "\nInter-wavefront bank overlaps:\n";

    int total_overlaps = 0;
    for (int wf1 = 0; wf1 < 4; wf1++) {
        for (int wf2 = wf1 + 1; wf2 < 4; wf2++) {
            std::set<int> overlap;
            for (int b : all_wf_banks[wf1]) {
                if (all_wf_banks[wf2].count(b)) {
                    overlap.insert(b);
                }
            }
            if (!overlap.empty()) {
                std::cout << "  WF" << wf1 << " ∩ WF" << wf2 << ": { ";
                for (int b : overlap) {
                    std::cout << b << " ";
                }
                std::cout << "} (" << overlap.size() << " shared banks)\n";
                total_overlaps += overlap.size();
            }
        }
    }

    if (total_overlaps == 0) {
        std::cout << "  None! Each wavefront uses exclusive banks.\n";
    }

    // Now count actual conflicts if all 4 wavefronts execute simultaneously
    std::cout << "\n--- If all 4 wavefronts execute simultaneously at dm=0 ---\n";

    // bank -> list of (wf, lane, slot)
    std::map<int, std::vector<std::tuple<int, int, int>>> bank_accesses;

    for (int wf = 0; wf < 4; wf++) {
        int k1 = wf;
        for (int lane = 0; lane < 64; lane++) {
            int k2_idx = lane % 8;
            int m0_idx = lane / 8;
            int k = k1 * 8 + k2_idx;
            int m = m0_idx * 8 + 0;

            auto offset = desc_km.calculate_offset(make_tuple(k, m));
            int byte_offset = offset * DataTypeSize;
            int slot = byte_offset / 4;
            int bank = slot % 32;

            bank_accesses[bank].push_back({wf, lane, slot});
        }
    }

    int intra_wf_conflicts = 0;
    int inter_wf_conflicts = 0;

    for (auto& entry : bank_accesses) {
        int bank = entry.first;
        auto& accesses = entry.second;

        if (accesses.size() <= 1) continue;

        // Count unique slots
        std::set<int> unique_slots;
        for (auto& acc : accesses) {
            unique_slots.insert(std::get<2>(acc));
        }

        // Check if accesses are from same or different wavefronts
        std::set<int> wfs_involved;
        for (auto& acc : accesses) {
            wfs_involved.insert(std::get<0>(acc));
        }

        int conflicts = unique_slots.size() - 1;

        if (wfs_involved.size() == 1) {
            intra_wf_conflicts += conflicts;
        } else {
            inter_wf_conflicts += conflicts;
        }

        if (accesses.size() > 8) {  // More than one wavefront hitting this bank
            std::cout << "Bank " << bank << ": " << accesses.size() << " accesses from "
                      << wfs_involved.size() << " WFs, " << unique_slots.size() << " unique slots\n";
        }
    }

    std::cout << "\nConflict breakdown (dm=0 only):\n";
    std::cout << "  Intra-wavefront: " << intra_wf_conflicts << "\n";
    std::cout << "  Inter-wavefront: " << inter_wf_conflicts << "\n";
    std::cout << "  Total: " << (intra_wf_conflicts + inter_wf_conflicts) << "\n";
}

int main()
{
    std::cout << "=============================================\n";
    std::cout << "Inter-Wavefront Conflict Analysis\n";
    std::cout << "=============================================\n";
    std::cout << "Question: Does XOR introduce inter-WF conflicts?\n";

    analyze_inter_wf_conflicts<false>();
    analyze_inter_wf_conflicts<true>();

    return 0;
}
