// PURE Read-only LDS test - pre-initialize LDS with NO bank conflicts
// Expected: 4,096 read conflicts (transpose pattern)
#include <iostream>
#include <vector>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

template<typename DataType>
struct PureReadKernel
{
    static constexpr index_t kBlockSize = 256;
    static constexpr index_t kM = 64;
    static constexpr index_t kK = 32;
    static constexpr index_t kKPack = 8;

    CK_TILE_HOST_DEVICE static constexpr index_t GetStaticLdsSize()
    {
        return kM * kK * sizeof(DataType);
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDescriptorMK()
    {
        return make_naive_tensor_descriptor_packed(make_tuple(kM, kK));
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDescriptorKM()
    {
        // Plain transposed descriptor - will have bank conflicts
        return make_naive_tensor_descriptor(
            make_tuple(kK, kM),
            make_tuple(number<1>{}, number<kK>{}));
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeDistributionMK()
    {
        constexpr index_t K1 = 16 / sizeof(DataType);
        constexpr index_t K0 = kK / K1;
        constexpr index_t M2 = 64 / K0;
        constexpr index_t M1 = kBlockSize / 64;
        constexpr index_t M0 = kM / (M2 * M1);

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 0>>,
                sequence<1, 2>,
                sequence<0, 1>
            >{});
    }

    CK_TILE_DEVICE void operator()(const DataType* __restrict__ input,
                                    DataType* __restrict__ output,
                                    index_t M,
                                    index_t K) const
    {
        __shared__ DataType lds[kM * kK];

        const index_t block_m = blockIdx.x * kM;
        if(block_m >= M) return;

        // Use store_tile to initialize LDS - this is CONFLICT-FREE
        constexpr auto lds_desc_mk = MakeLdsDescriptorMK();
        auto lds_view_mk = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<DataType*>(lds), lds_desc_mk);
        constexpr auto dist_mk = MakeDistributionMK();
        auto lds_window_mk = make_tile_window(
            lds_view_mk, make_tuple(kM, kK), {0, 0}, dist_mk);

        // Load from global
        const auto gmem_desc_in = make_naive_tensor_descriptor(
            make_tuple(number<kM>{}, number<kK>{}),
            make_tuple(K, number<1>{}));
        auto gmem_view_in = make_tensor_view<address_space_enum::global>(
            input + block_m * K, gmem_desc_in);
        auto gmem_window_in = make_tile_window(
            gmem_view_in, make_tuple(kM, kK), {0, 0}, dist_mk);

        auto reg_tile = load_tile(gmem_window_in);
        store_tile(lds_window_mk, reg_tile);  // Conflict-free write!
        block_sync_lds();

        // NOW DO TRANSPOSE READS - this is what we're measuring
        for(index_t k_block = 0; k_block < K; k_block += kK)
        {
            constexpr auto lds_desc_km = MakeLdsDescriptorKM();

            auto lds_view_km = make_tensor_view<address_space_enum::lds>(
                reinterpret_cast<DataType*>(lds), lds_desc_km);

            constexpr index_t M1 = 16 / sizeof(DataType);
            constexpr index_t M0 = kM / M1;
            constexpr index_t K2 = 64 / M0;
            constexpr index_t K1 = kBlockSize / 64;
            constexpr index_t K0 = kK / (K2 * K1);

            constexpr auto dist_km = make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<1>,
                    tuple<sequence<K0, K1, K2>, sequence<M0, M1>>,
                    tuple<sequence<1>, sequence<1, 2>>,
                    tuple<sequence<1>, sequence<2, 0>>,
                    sequence<1, 2>,
                    sequence<0, 1>
                >{});

            auto lds_window_km = make_tile_window(
                lds_view_km, make_tuple(kK, kM), {0, 0}, dist_km);

            auto reg_final = load_tile(lds_window_km);
            block_sync_lds();

            const auto gmem_desc_out = make_naive_tensor_descriptor(
                make_tuple(number<kK>{}, number<kM>{}),
                make_tuple(M, number<1>{}));

            auto gmem_view_out = make_tensor_view<address_space_enum::global>(
                output + k_block * M + block_m, gmem_desc_out);

            auto gmem_window_out = make_tile_window(
                gmem_view_out, make_tuple(kK, kM), {0, 0}, dist_km);

            store_tile(gmem_window_out, reg_final);
            block_sync_lds();
        }
    }
};

int main()
{
    constexpr index_t M = 256;
    constexpr index_t K = 128;
    using DataType = half_t;

    std::vector<DataType> h_input(M * K);
    std::vector<DataType> h_output(K * M);
    for(index_t i = 0; i < M * K; ++i) h_input[i] = static_cast<DataType>(i);

    DeviceMem d_input(M * K * sizeof(DataType));
    DeviceMem d_output(K * M * sizeof(DataType));
    d_input.ToDevice(h_input.data(), M * K * sizeof(DataType));

    constexpr index_t kM = 64;
    constexpr index_t block_size = 256;
    const index_t grid_size = (M + kM - 1) / kM;

    stream_config stream;
    constexpr index_t lds_size = PureReadKernel<DataType>::GetStaticLdsSize();

    launch_kernel(stream,
                 make_kernel<block_size>(
                     PureReadKernel<DataType>{},
                     dim3(grid_size),
                     dim3(block_size),
                     lds_size,
                     static_cast<const DataType*>(d_input.GetDeviceBuffer()),
                     static_cast<DataType*>(d_output.GetDeviceBuffer()),
                     M, K));

    hip_check_error(hipDeviceSynchronize());
    std::cout << "Pure Read NO XOR completed\n";
    return 0;
}
