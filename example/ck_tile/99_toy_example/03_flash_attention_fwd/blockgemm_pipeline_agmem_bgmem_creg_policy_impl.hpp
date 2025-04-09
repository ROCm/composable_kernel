// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"

namespace ck_tile {
namespace policy_impl {

// 3d + padding
template <typename Problem>
__host__ __device__ static constexpr auto make_a_lds_block_descriptor_3d_pad()
{
    constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
    constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

    constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
        make_tuple(number<kKPerBlock / 8>{}, number<kMPerBlock>{}, number<8>{}),
        make_tuple(number<(kMPerBlock + 1) * 8>{}, number<8>{}, number<1>{}),
        number<8>{},
        number<1>{});

    constexpr auto a_lds_block_desc =
        transform_tensor_descriptor(a_lds_block_desc_0,
                                    make_tuple(make_pass_through_transform(kMPerBlock),
                                               make_merge_transform(make_tuple(kKPerBlock / 8, 8))),
                                    make_tuple(sequence<1>{}, sequence<0, 2>{}),
                                    make_tuple(sequence<0>{}, sequence<1>{}));

    return a_lds_block_desc;
}

// 3d + padding
template <typename Problem>
__host__ __device__ static constexpr auto make_b_lds_block_descriptor_3d_pad()
{
    constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
    constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

    constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
        make_tuple(number<kKPerBlock / 8>{}, number<kNPerBlock>{}, number<8>{}),
        make_tuple(number<(kNPerBlock + 1) * 8>{}, number<8>{}, number<1>{}),
        number<8>{},
        number<1>{});

    constexpr auto b_lds_block_desc =
        transform_tensor_descriptor(b_lds_block_desc_0,
                                    make_tuple(make_pass_through_transform(kNPerBlock),
                                               make_merge_transform(make_tuple(kKPerBlock / 8, 8))),
                                    make_tuple(sequence<1>{}, sequence<0, 2>{}),
                                    make_tuple(sequence<0>{}, sequence<1>{}));

    return b_lds_block_desc;
}

template <typename Problem, typename BlockGemm>
__host__ __device__ static constexpr auto make_a_reg_block_descriptor()
{
    constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
    constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

    constexpr auto config = BlockGemm::BlockGemmPolicy::template GetWarpGemmMWarpNWarp<Problem>();

    using WG = remove_cvref_t<decltype(config.template get<0>())>;

    constexpr index_t MWarp = config.template get<1>();
    constexpr index_t NWarp = config.template get<2>();

    constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WG::kM);
    constexpr index_t KIterPerWarp = kKPerBlock / WG::kK;

    constexpr auto a_block_outer_dstr_encoding =
        tile_distribution_encoding<sequence<NWarp>,
                                   tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                                   tuple<sequence<1, 0>>,
                                   tuple<sequence<1, 0>>,
                                   sequence<1, 2>,
                                   sequence<0, 0>>{};

    constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
        a_block_outer_dstr_encoding, typename WG::AWarpDstrEncoding{});

    constexpr auto a_block_dstr = make_static_tile_distribution(a_block_dstr_encode);

    return a_block_dstr;
}

template <typename Problem>
__host__ __device__ static constexpr auto make_a_dram_tile_distribution()
{
    using ADataType = remove_cvref_t<typename Problem::ADataType>;

    constexpr index_t kBlockSize = Problem::kBlockSize;

    constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
    constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

    constexpr index_t K1 = 16 / sizeof(ADataType);
    constexpr index_t K0 = kKPerBlock / K1;
    constexpr index_t M2 = get_warp_size() / K0;

    constexpr index_t M1 = kBlockSize / get_warp_size();
    constexpr index_t M0 = kMPerBlock / (M2 * M1);

    return make_static_tile_distribution(
        tile_distribution_encoding<sequence<1>,
                                   tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                                   tuple<sequence<1>, sequence<1, 2>>,
                                   tuple<sequence<1>, sequence<2, 0>>,
                                   sequence<1, 2>,
                                   sequence<0, 1>>{});
}

template <typename Problem, typename BlockGemm>
__host__ __device__ static constexpr auto make_a_dram_tile_distribution_skip_lds()
{
    constexpr auto config = BlockGemm::BlockGemmPolicy::template GetWarpGemmMWarpNWarp<Problem>();

    using WG = remove_cvref_t<decltype(config.template get<0>())>;

    constexpr index_t MWarp = config.template get<1>();

    constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
    constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

    constexpr index_t K2 =
        WG::kK / WG::WarpGemmAttribute::Impl::kABKLane; // WG::WarpGemmAttribute::Impl::kABKPerLane;
                                                        // // 16 / sizeof(ADataType);
    constexpr index_t K1 = WG::WarpGemmAttribute::Impl::kABKLane;
    constexpr index_t K0 = kKPerBlock / (K1 * K2);

    constexpr index_t M2 = WG::WarpGemmAttribute::Impl::kAMLane;
    constexpr index_t M1 = MWarp;
    constexpr index_t M0 = kMPerBlock / (M2 * M1);

    return make_static_tile_distribution(
        tile_distribution_encoding<sequence<1>,
                                   tuple<sequence<M0, M1, M2>, sequence<K0, K1, K2>>,
                                   tuple<sequence<1>, sequence<2, 1>>,
                                   tuple<sequence<1>, sequence<1, 2>>,
                                   sequence<2, 1, 2>,
                                   sequence<0, 0, 2>>{});
}

template <typename Problem>
__host__ __device__ static constexpr auto make_b_dram_tile_distribution()
{
    using BDataType = remove_cvref_t<typename Problem::BDataType>;

    constexpr index_t kBlockSize = Problem::kBlockSize;

    constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
    constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

    constexpr index_t K1 = 16 / sizeof(BDataType);
    constexpr index_t K0 = kKPerBlock / K1;
    constexpr index_t N2 = get_warp_size() / K0;

    constexpr index_t N1 = kBlockSize / get_warp_size();
    constexpr index_t N0 = kNPerBlock / (N2 * N1);

    return make_static_tile_distribution(
        tile_distribution_encoding<sequence<1>,
                                   tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                   tuple<sequence<1>, sequence<1, 2>>,
                                   tuple<sequence<1>, sequence<2, 0>>,
                                   sequence<1, 2>,
                                   sequence<0, 1>>{});
}

template <typename Problem>
__host__ __device__ static constexpr auto get_block_gemm()
{
    using BlockGemmPolicy = BlockGemmASmemBSmemCRegDefaultPolicy;

    return BlockGemmASmemBSmemCReg<Problem, BlockGemmPolicy>{};
}

} // namespace policy_impl
} // namespace ck_tile
