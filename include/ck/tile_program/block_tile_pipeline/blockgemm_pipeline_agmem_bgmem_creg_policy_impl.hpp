#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/tile/tile_gemm_shape.hpp"
#include "ck/tile_program/warp_tile/warp_gemm.hpp"

namespace ck {
namespace tile_program {
namespace block {
namespace policy_impl {
// 3d + padding
template <typename Problem>
__host__ __device__ static constexpr auto make_a_lds_block_descriptor_3d_pad()
{
    using namespace ck;

    constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
    constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

    constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
        make_tuple(Number<kKPerBlock / 8>{}, Number<kMPerBlock>{}, Number<8>{}),
        make_tuple(Number<(kMPerBlock + 1) * 8>{}, Number<8>{}, Number<1>{}),
        Number<8>{},
        Number<1>{});

    constexpr auto a_lds_block_desc =
        transform_tensor_descriptor(a_lds_block_desc_0,
                                    make_tuple(make_pass_through_transform(kMPerBlock),
                                               make_merge_transform(make_tuple(kKPerBlock / 8, 8))),
                                    make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
                                    make_tuple(Sequence<0>{}, Sequence<1>{}));

    return a_lds_block_desc;
}

// 3d + padding
template <typename Problem>
__host__ __device__ static constexpr auto make_b_lds_block_descriptor_3d_pad()
{
    using namespace ck;

    constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
    constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

    constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
        make_tuple(Number<kKPerBlock / 8>{}, Number<kNPerBlock>{}, Number<8>{}),
        make_tuple(Number<(kNPerBlock + 1) * 8>{}, Number<8>{}, Number<1>{}),
        Number<8>{},
        Number<1>{});

    constexpr auto b_lds_block_desc =
        transform_tensor_descriptor(b_lds_block_desc_0,
                                    make_tuple(make_pass_through_transform(kNPerBlock),
                                               make_merge_transform(make_tuple(kKPerBlock / 8, 8))),
                                    make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
                                    make_tuple(Sequence<0>{}, Sequence<1>{}));

    return b_lds_block_desc;
}

template <typename Problem, typename BlockGemm>
__host__ __device__ static constexpr auto make_a_reg_block_descriptor()
{
    using namespace ck;

    constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
    constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

    constexpr auto config = BlockGemm::BlockGemmPolicy::template GetWarpGemmMWarpNWarp<Problem>();

    using WG = remove_cvref_t<decltype(config.template At<0>())>;

    constexpr index_t MWarp = config.template At<1>();
    constexpr index_t NWarp = config.template At<2>();

    constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WG::kM);
    constexpr index_t KIterPerWarp = kKPerBlock / WG::kK;

    constexpr auto a_block_outer_dstr_encoding =
        StaticTileDistributionEncoding<Sequence<NWarp>,
                                       Tuple<Sequence<MIterPerWarp, MWarp>, Sequence<KIterPerWarp>>,
                                       Tuple<Sequence<1, 0>>,
                                       Tuple<Sequence<1, 0>>,
                                       Sequence<1, 2>,
                                       Sequence<0, 0>>{};

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
        StaticTileDistributionEncoding<Sequence<1>,
                                       Tuple<Sequence<M0, M1, M2>, Sequence<K0, K1>>,
                                       Tuple<Sequence<1>, Sequence<1, 2>>,
                                       Tuple<Sequence<1>, Sequence<2, 0>>,
                                       Sequence<1, 2>,
                                       Sequence<0, 1>>{});
}

template <typename Problem, typename BlockGemm>
__host__ __device__ static constexpr auto make_a_dram_tile_distribution_skip_lds()
{
    constexpr auto config = BlockGemm::BlockGemmPolicy::template GetWarpGemmMWarpNWarp<Problem>();

    using WG = remove_cvref_t<decltype(config.template At<0>())>;

    constexpr index_t MWarp = config.template At<1>();

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
        StaticTileDistributionEncoding<Sequence<1>,
                                       Tuple<Sequence<M0, M1, M2>, Sequence<K0, K1, K2>>,
                                       Tuple<Sequence<1>, Sequence<2, 1>>,
                                       Tuple<Sequence<1>, Sequence<1, 2>>,
                                       Sequence<2, 1, 2>,
                                       Sequence<0, 0, 2>>{});
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
        StaticTileDistributionEncoding<Sequence<1>,
                                       Tuple<Sequence<N0, N1, N2>, Sequence<K0, K1>>,
                                       Tuple<Sequence<1>, Sequence<1, 2>>,
                                       Tuple<Sequence<1>, Sequence<2, 0>>,
                                       Sequence<1, 2>,
                                       Sequence<0, 1>>{});
}

template <typename Problem>
__host__ __device__ static constexpr auto get_block_gemm()
{
    using BlockGemmPolicy = BlockGemmASmemBSmemCRegV1DefaultPolicy;

    return BlockGemmASmemBSmemCRegV1<Problem, BlockGemmPolicy>{};
}

} // namespace policy_impl
} // namespace block
} // namespace tile_program
} // namespace ck
