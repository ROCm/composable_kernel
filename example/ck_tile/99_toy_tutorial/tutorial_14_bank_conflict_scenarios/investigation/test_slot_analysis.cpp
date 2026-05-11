// Analyze slots more carefully
// Check if lanes hitting same bank are at same or different slots
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

void analyze_slots()
{
    using DataType = half_t;
    constexpr index_t DataTypeSize = sizeof(DataType);

    constexpr auto desc_plain = TestDescriptors<DataType, false>::MakeLdsDescriptorKM();
    constexpr auto desc_xor = TestDescriptors<DataType, true>::MakeLdsDescriptorKM();

    std::vector<int> phase0 = {0, 1, 2, 3, 20, 21, 22, 23};

    std::cout << "=== SLOT ANALYSIS for Phase 0, k1=0, m1=0 ===\n\n";
    std::cout << "Lane | (k,m) | Plain: offset→byte→slot→bank | XOR: offset→byte→slot→bank\n";
    std::cout << "-----|-------|------------------------------|-----------------------------\n";

    for (int lane : phase0) {
        int k2_idx = lane % 8;
        int m0_idx = lane / 8;
        int k = 0 * 8 + k2_idx;  // k1=0
        int m = m0_idx * 8 + 0;  // m1=0

        auto plain_off = desc_plain.calculate_offset(make_tuple(k, m));
        auto xor_off = desc_xor.calculate_offset(make_tuple(k, m));

        index_t plain_byte = plain_off * DataTypeSize;
        index_t xor_byte = xor_off * DataTypeSize;
        index_t plain_slot = plain_byte / 4;
        index_t xor_slot = xor_byte / 4;
        index_t plain_bank = plain_slot % 32;
        index_t xor_bank = xor_slot % 32;

        std::cout << " " << lane << "   | (" << k << "," << m << ")  | "
                  << plain_off << "→" << plain_byte << "→" << plain_slot << "→bank" << plain_bank << "       | "
                  << xor_off << "→" << xor_byte << "→" << xor_slot << "→bank" << xor_bank << "\n";
    }

    std::cout << "\n\n=== GROUPING BY BANK ===\n";

    // Group by bank for plain
    std::map<index_t, std::vector<std::tuple<int, int, int, index_t>>> plain_banks; // bank -> (lane, k, m, slot)
    std::map<index_t, std::vector<std::tuple<int, int, int, index_t>>> xor_banks;

    for (int lane : phase0) {
        int k2_idx = lane % 8;
        int m0_idx = lane / 8;
        int k = k2_idx;  // k1=0
        int m = m0_idx * 8;  // m1=0

        auto plain_off = desc_plain.calculate_offset(make_tuple(k, m));
        auto xor_off = desc_xor.calculate_offset(make_tuple(k, m));

        index_t plain_slot = (plain_off * DataTypeSize) / 4;
        index_t xor_slot = (xor_off * DataTypeSize) / 4;

        plain_banks[plain_slot % 32].push_back({lane, k, m, plain_slot});
        xor_banks[xor_slot % 32].push_back({lane, k, m, xor_slot});
    }

    std::cout << "\nPLAIN:\n";
    for (const auto& [bank, entries] : plain_banks) {
        std::cout << "  Bank " << bank << ": ";
        std::set<index_t> unique_slots;
        for (const auto& [lane, k, m, slot] : entries) {
            std::cout << "L" << lane << "(k=" << k << ",m=" << m << ",slot=" << slot << ") ";
            unique_slots.insert(slot);
        }
        if (entries.size() > 1) {
            if (unique_slots.size() == 1) {
                std::cout << " → SAME SLOT (0 conflicts)";
            } else {
                std::cout << " → DIFFERENT SLOTS (" << (entries.size() - 1) << " conflicts)";
            }
        }
        std::cout << "\n";
    }

    std::cout << "\nXOR:\n";
    for (const auto& [bank, entries] : xor_banks) {
        std::cout << "  Bank " << bank << ": ";
        std::set<index_t> unique_slots;
        for (const auto& [lane, k, m, slot] : entries) {
            std::cout << "L" << lane << "(k=" << k << ",m=" << m << ",slot=" << slot << ") ";
            unique_slots.insert(slot);
        }
        if (entries.size() > 1) {
            if (unique_slots.size() == 1) {
                std::cout << " → SAME SLOT (0 conflicts)";
            } else {
                std::cout << " → DIFFERENT SLOTS (" << (entries.size() - 1) << " conflicts)";
            }
        }
        std::cout << "\n";
    }
}

int main()
{
    std::cout << "=============================================\n";
    std::cout << "Slot Analysis for Bank Conflicts\n";
    std::cout << "=============================================\n\n";

    analyze_slots();

    std::cout << "\n=============================================\n";
    std::cout << "KEY INSIGHT:\n";
    std::cout << "  If lanes hitting same bank have SAME slot → 0 conflicts (FP16 opt)\n";
    std::cout << "  If lanes hitting same bank have DIFFERENT slots → N-1 conflicts\n";
    std::cout << "=============================================\n";

    return 0;
}
