#include <iostream>
#include <map>
#include <set>
#include <vector>
#include "ck_tile/core.hpp"

using namespace ck_tile;

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

int main() {
    constexpr index_t kM = 64;
    constexpr index_t kK = 32;
    constexpr index_t kKPack = 8;
    constexpr index_t DataTypeSize = 2;
    
    constexpr auto MLdsLayer = (32 * 4 / kK / DataTypeSize) < 1 ? 1 : (32 * 4 / kK / DataTypeSize);

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
    
    auto phases = get_phases();
    const auto& phase0 = phases[0];
    
    std::cout << "Phase 0 lanes: ";
    for(int l : phase0) std::cout << l << " ";
    std::cout << "\n\nDetailed view - Phase 0, column k=0, dm values 0-7:\n";
    
    for(int dm = 0; dm < 8; dm++) {
        std::cout << "\ndm=" << dm << ":\n";
        std::map<int, std::vector<std::tuple<int,int>>> bank_to_lane_slot;
        
        for(int lane : phase0) {
            int m0_idx = lane / 8;
            int m = m0_idx * 8 + dm;
            
            auto offset = lds_desc.calculate_offset(make_tuple(0, m));
            int byte_offset = offset * DataTypeSize;
            int slot = byte_offset / 4;
            int bank = slot % 32;
            
            bank_to_lane_slot[bank].push_back({lane, slot});
        }
        
        int conflicts = 0;
        for(const auto& [bank, lane_slots] : bank_to_lane_slot) {
            std::set<int> unique_slots;
            for(const auto& [_, slot] : lane_slots) unique_slots.insert(slot);
            
            std::cout << "  Bank " << bank << ": " << lane_slots.size() << " threads, " 
                      << unique_slots.size() << " slots";
            
            if(lane_slots.size() > 1 && unique_slots.size() > 1) {
                int conf = lane_slots.size() - 1;
                conflicts += conf;
                std::cout << " → " << conf << "-way conflict";
            }
            std::cout << "\n";
        }
        std::cout << "  Total dm=" << dm << ": " << conflicts << " conflicts\n";
    }
    
    return 0;
}
