// Debug: Use CK-Tile descriptor directly to get bank mapping
// Prints bank for each (m, k) coordinate for both plain and XOR descriptors

#include <iostream>
#include <vector>
#include <set>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

constexpr index_t kM = 64;
constexpr index_t kK = 32;
constexpr index_t kKPack = 8;
constexpr index_t kBlockSize = 256;

// Plain [M,K] descriptor
CK_TILE_HOST_DEVICE static constexpr auto MakePlainDescriptorMK()
{
    return make_naive_tensor_descriptor_packed(make_tuple(number<kM>{}, number<kK>{}));
}

// XOR [M,K] descriptor - from 04_row_major_xor.cpp
CK_TILE_HOST_DEVICE static constexpr auto MakeXorDescriptorMK()
{
    using DataType = half_t;
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
            make_merge_transform(make_tuple(number<kM / MLdsLayer>{}, number<MLdsLayer>{})),
            make_merge_transform(make_tuple(number<kK / kKPack>{}, number<kKPack>{}))),
        make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
        make_tuple(sequence<0>{}, sequence<1>{}));

    return lds_desc;
}

template<typename Desc>
__global__ void print_banks_kernel(int* output, Desc desc)
{
    int idx = threadIdx.x;
    if(idx == 0)
    {
        // Print offset and bank for each (m, k)
        for(int m = 0; m < kM; m += 8)
        {
            for(int k = 0; k < 8; k++)
            {
                auto offset = desc.calculate_offset(make_multi_index(m, k));
                int byte_off = offset * sizeof(half_t);
                int bank = (byte_off / 4) % 32;
                output[m/8 * 8 + k] = bank;
            }
        }
    }
}

int main()
{
    std::cout << "=== Direct Descriptor Bank Analysis ===\n\n";

    constexpr auto plain_desc = MakePlainDescriptorMK();
    constexpr auto xor_desc = MakeXorDescriptorMK();

    std::vector<int> h_plain(64, 0), h_xor(64, 0);
    DeviceMem d_plain(64 * sizeof(int)), d_xor(64 * sizeof(int));

    stream_config stream;

    // Launch kernels to get bank mappings
    print_banks_kernel<<<1, 1>>>(static_cast<int*>(d_plain.GetDeviceBuffer()), plain_desc);
    hip_check_error(hipDeviceSynchronize());
    d_plain.FromDevice(h_plain.data(), 64 * sizeof(int));

    print_banks_kernel<<<1, 1>>>(static_cast<int*>(d_xor.GetDeviceBuffer()), xor_desc);
    hip_check_error(hipDeviceSynchronize());
    d_xor.FromDevice(h_xor.data(), 64 * sizeof(int));

    std::cout << "Banks for column k=0 (every 8 rows):\n";
    std::cout << "m  | plain_bank | xor_bank\n";
    std::cout << "---|------------|--------\n";
    for(int i = 0; i < 8; i++)
    {
        std::cout << (i * 8) << "  | " << h_plain[i * 8] << " | " << h_xor[i * 8] << "\n";
    }

    std::cout << "\nFull bank mapping for k=0..7:\n";
    std::cout << "m\\k |";
    for(int k = 0; k < 8; k++) std::cout << " " << k << " |";
    std::cout << "\n----|";
    for(int k = 0; k < 8; k++) std::cout << "---|";
    std::cout << "\n";

    std::cout << "PLAIN:\n";
    for(int m = 0; m < kM; m += 8)
    {
        std::cout << m << "  |";
        for(int k = 0; k < 8; k++)
        {
            auto offset = plain_desc.calculate_offset(make_multi_index(m, k));
            int byte_off = offset * sizeof(half_t);
            int bank = (byte_off / 4) % 32;
            std::cout << " " << bank << " |";
        }
        std::cout << "\n";
    }

    std::cout << "\nXOR:\n";
    for(int m = 0; m < kM; m += 8)
    {
        std::cout << m << "  |";
        for(int k = 0; k < 8; k++)
        {
            auto offset = xor_desc.calculate_offset(make_multi_index(m, k));
            int byte_off = offset * sizeof(half_t);
            int bank = (byte_off / 4) % 32;
            std::cout << " " << bank << " |";
        }
        std::cout << "\n";
    }

    // Phase 0 analysis
    std::cout << "\n=== Phase 0 Bank Analysis (dm=0, WF0) ===\n";
    std::cout << "Lanes 0-7 all access k=0 with m=0,8,16,24,32,40,48,56\n\n";

    std::cout << "Lane | m | plain_bank | xor_bank\n";
    for(int lane = 0; lane < 8; lane++)
    {
        int m = lane * 8;
        int k = 0;
        auto plain_off = plain_desc.calculate_offset(make_multi_index(m, k));
        auto xor_off = xor_desc.calculate_offset(make_multi_index(m, k));
        int plain_bank = (plain_off * 2 / 4) % 32;
        int xor_bank = (xor_off * 2 / 4) % 32;
        std::cout << "  " << lane << "  | " << m << " | " << plain_bank << " | " << xor_bank << "\n";
    }

    std::cout << "\nCounting unique banks:\n";
    std::set<int> plain_banks, xor_banks;
    for(int lane = 0; lane < 8; lane++)
    {
        int m = lane * 8;
        int k = 0;
        auto plain_off = plain_desc.calculate_offset(make_multi_index(m, k));
        auto xor_off = xor_desc.calculate_offset(make_multi_index(m, k));
        plain_banks.insert((plain_off * 2 / 4) % 32);
        xor_banks.insert((xor_off * 2 / 4) % 32);
    }
    std::cout << "PLAIN: " << plain_banks.size() << " unique banks -> "
              << (8 - plain_banks.size()) << " conflicts\n";
    std::cout << "XOR:   " << xor_banks.size() << " unique banks -> "
              << (8 - xor_banks.size()) << " conflicts\n";

    return 0;
}
