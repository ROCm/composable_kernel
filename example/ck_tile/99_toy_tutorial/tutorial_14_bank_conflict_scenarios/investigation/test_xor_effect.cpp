// Test if XOR transform is actually changing offsets
#include <iostream>
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

            // MLdsLayer = 128 / 64 = 2 for FP16

            constexpr auto lds_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<kK / kKPack * MLdsLayer>{},  // 32/8*2 = 8
                           number<kM / MLdsLayer>{},            // 64/2 = 32
                           number<kKPack>{}),                   // 8
                make_tuple(number<kKPack>{},                   // stride 8
                           number<kK * MLdsLayer>{},            // stride 64
                           number<1>{}),                        // stride 1
                number<kKPack>{},
                number<1>{});

            constexpr auto lds_desc_permuted = transform_tensor_descriptor(
                lds_desc_0,
                make_tuple(make_xor_transform(make_tuple(number<kM / MLdsLayer>{},      // 32
                                                         number<kK / kKPack * MLdsLayer>{})),  // 8
                           make_pass_through_transform(number<kKPack>{})),
                make_tuple(sequence<1, 0>{}, sequence<2>{}),
                make_tuple(sequence<1, 0>{}, sequence<2>{}));

            constexpr auto lds_desc_unmerged = transform_tensor_descriptor(
                lds_desc_permuted,
                make_tuple(make_unmerge_transform(
                               make_tuple(number<MLdsLayer>{}, number<kK / kKPack>{})),  // 2, 4
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

int main()
{
    using DataType = half_t;

    constexpr auto desc_plain = TestDescriptors<DataType, false>::MakeLdsDescriptorKM();
    constexpr auto desc_xor = TestDescriptors<DataType, true>::MakeLdsDescriptorKM();

    std::cout << "=== Testing XOR Effect ===\n\n";
    std::cout << "MLdsLayer = " << (32 * 4 / 32 / 2) << " = 2\n\n";

    std::cout << "Comparing Plain vs XOR offsets:\n";
    std::cout << "(k, m) | Plain offset | XOR offset | Different?\n";
    std::cout << "-------|--------------|------------|------------\n";

    int differences = 0;
    for (int k = 0; k < 32; k += 4) {
        for (int m = 0; m < 64; m += 8) {
            auto plain = desc_plain.calculate_offset(make_tuple(k, m));
            auto xor_off = desc_xor.calculate_offset(make_tuple(k, m));
            bool diff = (plain != xor_off);
            if (diff) differences++;

            std::cout << "(" << k << "," << m << ")  | " << plain << "           | " << xor_off << "          | " << (diff ? "YES" : "no") << "\n";
        }
    }

    std::cout << "\nTotal different coordinates: " << differences << "\n";

    if (differences == 0) {
        std::cout << "\n*** WARNING: XOR transform appears to have NO EFFECT! ***\n";
        std::cout << "This could be a bug in the descriptor construction.\n";
    }

    return 0;
}
