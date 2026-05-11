// Write-only LDS test - NO XOR
#include <iostream>
#include <vector>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

template<typename DataType>
struct WriteOnlyKernel
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

        constexpr auto lds_desc_mk = MakeLdsDescriptorMK();
        auto lds_view_mk = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<DataType*>(lds), lds_desc_mk);

        constexpr auto dist_mk = MakeDistributionMK();
        auto lds_window_mk = make_tile_window(
            lds_view_mk, make_tuple(kM, kK), {0, 0}, dist_mk);

        for(index_t k_block = 0; k_block < K; k_block += kK)
        {
            const auto gmem_desc_in = make_naive_tensor_descriptor(
                make_tuple(number<kM>{}, number<kK>{}),
                make_tuple(K, number<1>{}));

            auto gmem_view_in = make_tensor_view<address_space_enum::global>(
                input + block_m * K + k_block, gmem_desc_in);

            auto gmem_window_in = make_tile_window(
                gmem_view_in, make_tuple(kM, kK), {0, 0}, dist_mk);

            auto reg_tile = load_tile(gmem_window_in);
            store_tile(lds_window_mk, reg_tile);
            block_sync_lds();

            if(threadIdx.x == 0 && k_block == 0)
                output[block_m] = lds[0];

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
    std::vector<DataType> h_output(M);
    for(index_t i = 0; i < M * K; ++i) h_input[i] = static_cast<DataType>(i);

    DeviceMem d_input(M * K * sizeof(DataType));
    DeviceMem d_output(M * sizeof(DataType));
    d_input.ToDevice(h_input.data(), M * K * sizeof(DataType));

    constexpr index_t kM = 64;
    constexpr index_t block_size = 256;
    const index_t grid_size = (M + kM - 1) / kM;

    stream_config stream;
    constexpr index_t lds_size = WriteOnlyKernel<DataType>::GetStaticLdsSize();

    launch_kernel(stream,
                 make_kernel<block_size>(
                     WriteOnlyKernel<DataType>{},
                     dim3(grid_size),
                     dim3(block_size),
                     lds_size,
                     static_cast<const DataType*>(d_input.GetDeviceBuffer()),
                     static_cast<DataType*>(d_output.GetDeviceBuffer()),
                     M, K));

    hip_check_error(hipDeviceSynchronize());
    std::cout << "Write-only NO XOR completed\n";
    return 0;
}
