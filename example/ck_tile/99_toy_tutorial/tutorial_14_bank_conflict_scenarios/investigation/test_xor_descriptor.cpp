#include <iostream>
#include "ck_tile/core.hpp"

using namespace ck_tile;

int main() {
    constexpr index_t kM = 64;
    constexpr index_t kK = 32;
    constexpr index_t kKPack = 8;
    constexpr index_t DataTypeSize = 2;  // FP16
    
    // XOR descriptor for [K, M]
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
    
    std::cout << "WITH XOR - Column k=0 bank pattern:\n";
    for(int m = 0; m < 8; m++) {
        auto offset = lds_desc.calculate_offset(make_tuple(0, m));
        int byte_offset = offset * DataTypeSize;
        int slot = byte_offset / 4;
        int bank = slot % 32;
        std::cout << "  m=" << m << " → offset=" << offset << " → byte=" << byte_offset 
                  << " → slot=" << slot << " → bank=" << bank << "\n";
    }
    
    std::cout << "\nPhase 0 lanes accessing column k=0, dm=0 with XOR:\n";
    std::vector<int> phase0 = {0, 1, 2, 3, 20, 21, 22, 23};
    for(int lane : phase0) {
        int m0_idx = lane / 8;
        int m = m0_idx * 8;  // dm=0
        
        auto offset = lds_desc.calculate_offset(make_tuple(0, m));
        int byte_offset = offset * DataTypeSize;
        int slot = byte_offset / 4;
        int bank = slot % 32;
        
        std::cout << "  Lane " << lane << " → m=" << m << " → slot=" << slot << " → bank=" << bank << "\n";
    }
    
    return 0;
}
