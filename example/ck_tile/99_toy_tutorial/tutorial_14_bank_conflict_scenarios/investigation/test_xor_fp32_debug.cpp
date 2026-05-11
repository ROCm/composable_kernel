// Debug test to verify XOR transform for FP32
#include <iostream>
#include "ck_tile/core.hpp"

using namespace ck_tile;

template<typename DataType, bool UseXor>
struct DebugXorTest
{
    static constexpr index_t kM = 64;
    static constexpr index_t kK = 32;
    static constexpr index_t kKPack = 8;

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
};

int main()
{
    using DataType = float;

    std::cout << "=== FP32 XOR Debug Test ===\n\n";

    constexpr auto desc_xor = DebugXorTest<DataType, true>::MakeLdsDescriptorMK();
    constexpr auto desc_plain = DebugXorTest<DataType, false>::MakeLdsDescriptorMK();

    constexpr index_t DataTypeSize = sizeof(DataType);
    constexpr index_t MLdsLayer = (32 * 4 / 32 / DataTypeSize) < 1 ? 1 : (32 * 4 / 32 / DataTypeSize);

    std::cout << "MLdsLayer = " << MLdsLayer << "\n";
    std::cout << "XOR dimensions: (64/" << MLdsLayer << ", 32/8*" << MLdsLayer << ") = ("
              << 64/MLdsLayer << ", " << 32/8*MLdsLayer << ")\n\n";

    std::cout << "First column (k=0) addresses - WITHOUT XOR:\n";
    std::cout << "Element [m,k] -> Offset -> Bank\n";
    for(index_t m = 0; m < 16; m++)
    {
        auto offset = desc_plain.calculate_offset(make_tuple(m, 0));
        index_t byte_offset = offset * DataTypeSize;
        index_t bank = byte_offset / 4 % 32;
        std::cout << "[" << m << ",0] -> " << offset << " -> Bank " << bank << "\n";
    }

    std::cout << "\nFirst column (k=0) addresses - WITH XOR:\n";
    std::cout << "Element [m,k] -> Offset -> Bank\n";
    for(index_t m = 0; m < 16; m++)
    {
        auto offset = desc_xor.calculate_offset(make_tuple(m, 0));
        index_t byte_offset = offset * DataTypeSize;
        index_t bank = byte_offset / 4 % 32;
        std::cout << "[" << m << ",0] -> " << offset << " -> Bank " << bank << "\n";
    }

    std::cout << "\n=== WRITE Pattern (row-wise) ===\n";
    std::cout << "\nFirst row (m=0) addresses - WITH XOR:\n";
    std::cout << "Element [m,k] -> Offset -> Bank\n";
    for(index_t k = 0; k < 8; k++)
    {
        auto offset = desc_xor.calculate_offset(make_tuple(0, k));
        index_t byte_offset = offset * DataTypeSize;
        index_t bank = byte_offset / 4 % 32;
        std::cout << "[0," << k << "] -> " << offset << " -> Bank " << bank << "\n";
    }

    std::cout << "\nSecond row (m=1) addresses - WITH XOR:\n";
    std::cout << "Element [m,k] -> Offset -> Bank\n";
    for(index_t k = 0; k < 8; k++)
    {
        auto offset = desc_xor.calculate_offset(make_tuple(1, k));
        index_t byte_offset = offset * DataTypeSize;
        index_t bank = byte_offset / 4 % 32;
        std::cout << "[1," << k << "] -> " << offset << " -> Bank " << bank << "\n";
    }

    std::cout << "\n=== INTER-LANE Conflict Check (Multiple Columns Read Simultaneously) ===\n";
    std::cout << "\nWITH XOR - First 4 elements of each column:\n";
    for(index_t col = 0; col < 8; col++)
    {
        std::cout << "Column " << col << " [0-3]: Banks [";
        for(index_t m = 0; m < 4; m++)
        {
            auto offset = desc_xor.calculate_offset(make_tuple(m, col));
            index_t byte_offset = offset * DataTypeSize;
            index_t bank = byte_offset / 4 % 32;
            std::cout << bank;
            if(m < 3) std::cout << ", ";
        }
        std::cout << "]\n";
    }

    std::cout << "\nWITHOUT XOR - First 4 elements of each column:\n";
    for(index_t col = 0; col < 8; col++)
    {
        std::cout << "Column " << col << " [0-3]: Banks [";
        for(index_t m = 0; m < 4; m++)
        {
            auto offset = desc_plain.calculate_offset(make_tuple(m, col));
            index_t byte_offset = offset * DataTypeSize;
            index_t bank = byte_offset / 4 % 32;
            std::cout << bank;
            if(m < 3) std::cout << ", ";
        }
        std::cout << "]\n";
    }

    return 0;
}
