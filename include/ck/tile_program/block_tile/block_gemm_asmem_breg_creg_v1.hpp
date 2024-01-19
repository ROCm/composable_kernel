// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/static_tile_distribution_helper.hpp"
#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/tile/tile_gemm_shape.hpp"
#include "ck/tile_program/warp_tile/warp_gemm.hpp"
#include "ck/tile_program/block_tile/block_gemm_asmem_breg_creg_problem.hpp"
#include "ck/tile_program/block_tile/block_gemm_asmem_breg_creg_v1_default_policy.hpp"

namespace ck {
namespace tile_program {
namespace block {

// A is block window on shared memory
// B is block distributed tensor
// C is block distributed tensor
template <typename Problem_, typename Policy_ = BlockGemmASmemBRegCRegV1DefaultPolicy>
struct BlockGemmASmemBRegCRegV1
{
    using Problem        = remove_cvref_t<Problem_>;
    using Policy         = remove_cvref_t<Policy_>;
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    // C += A * B
    template <typename CBlockTensor, typename ABlockWindowTmp, typename BBlockTensorTmp>
    __device__ void operator()(CBlockTensor& c_block_tensor,
                               const ABlockWindowTmp& a_block_window_tmp,
                               const BBlockTensorTmp& b_block_tensor_tmp) const
    {
        static_assert(is_same_v<ADataType, remove_cv_t<typename ABlockWindowTmp::DataType>> &&
                          is_same_v<BDataType, remove_cv_t<typename BBlockTensorTmp::DataType>> &&
                          is_same_v<CDataType, remove_cv_t<typename CBlockTensor::DataType>>,
                      "wrong!");

        // constexpr index_t MPerBlock = ABlockWindowTmp{}.GetWindowLengths()[Number<0>{}];
        // constexpr index_t NPerBlock = BBlockTensorTmp{}.GetLengths()[Number<0>{}];
        // constexpr index_t KPerBlock = ABlockWindowTmp{}.GetWindowLengths()[Number<1>{}];
        constexpr index_t MPerBlock = BlockGemmShape::kM;
        constexpr index_t NPerBlock = BlockGemmShape::kN;
        constexpr index_t KPerBlock = BlockGemmShape::kK;

        // static_assert(MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
        //                   KPerBlock == BlockGemmShape::kK,
        //               "wrong!");

        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template At<0>())>;

        constexpr index_t MWarp = config.template At<1>();
        constexpr index_t NWarp = config.template At<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

        constexpr index_t MPerBlockPerIter = MPerBlock / MIterPerWarp;
        constexpr index_t KPerBlockPerIter = KPerBlock / KIterPerWarp;

        const index_t iMWarp = get_warp_id() / NWarp;

        constexpr auto b_block_outer_dstr_encoding = StaticTileDistributionEncoding<
            Sequence<MWarp>,
            Tuple<Sequence<NIterPerWarp, NWarp>, Sequence<KIterPerWarp>>,
            Tuple<Sequence<0, 1>>,
            Tuple<Sequence<0, 1>>,
            Sequence<1, 2>,
            Sequence<0, 0>>{};

        constexpr auto c_block_outer_dstr_encoding = StaticTileDistributionEncoding<
            Sequence<>,
            Tuple<Sequence<MIterPerWarp, MWarp>, Sequence<NIterPerWarp, NWarp>>,
            Tuple<Sequence<1, 2>>,
            Tuple<Sequence<1, 1>>,
            Sequence<1, 2>,
            Sequence<0, 0>>{};

        constexpr auto b_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            b_block_outer_dstr_encoding, typename WG::BWarpDstrEncoding{});

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});

        constexpr auto b_block_dstr = make_static_tile_distribution(b_block_dstr_encode);

        // constrcut from B-block-tensor from B-Block-tensor-tmp
        // FIXME: need method to check b_block_tensor and b_block_tensor_tmp have equivalent
        // distribution
        auto b_block_tensor =
            make_static_distributed_tensor<typename BBlockTensorTmp::DataType>(b_block_dstr);

        b_block_tensor.GetThreadBuffer() = b_block_tensor_tmp.GetThreadBuffer();

        // construct A-warp-window
        auto a_warp_window_tmp = make_tile_window(
            a_block_window_tmp.GetBottomTensorView(),
            make_tuple(Number<WG::kM>{}, Number<WG::kK>{}),
            a_block_window_tmp.GetWindowOrigin() + MultiIndex<2>{iMWarp * WG::kM, 0},
            make_static_tile_distribution(typename WG::AWarpDstrEncoding{}));

#if 0 // FIXME: using Array will cause register spill
        Array<Array<decltype(b_warp_window_tmp), KIterPerWarp>, NIterPerWarp> b_warp_windows{
            {b_warp_window_tmp}};

        for(index_t nIter = 0; nIter < NIterPerWarp; nIter++)
        {
            for(index_t kIter = 0; kIter < KIterPerWarp; kIter++)
            {
                move_tile_window(b_warp_windows(nIter)(kIter),
                                 {nIter * NPerBlockPerIter, kIter * KPerBlockPerIter});
            }
        }
#else
        StaticallyIndexedArray<StaticallyIndexedArray<decltype(a_warp_window_tmp), KIterPerWarp>,
                               MIterPerWarp>
            a_warp_windows;

        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                a_warp_windows(mIter)(kIter) = a_warp_window_tmp;

                move_tile_window(a_warp_windows(mIter)(kIter),
                                 {mIter * MPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });
#endif

        // check C-block-distribution
        static_assert(is_same_v<remove_cvref_t<decltype(c_block_dstr_encode)>,
                                remove_cvref_t<decltype(CBlockTensor::GetTileDistribution()
                                                            .GetStaticTileDistributionEncoding())>>,
                      "wrong!");

        using BWarpDstr = typename WG::BWarpDstr;
        using CWarpDstr = typename WG::CWarpDstr;

        using BWarpTensor = typename WG::BWarpTensor;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto b_warp_y_lengths = to_sequence(BWarpDstr{}.GetYs2DDescriptor().GetLengths());
        constexpr auto c_warp_y_lengths = to_sequence(CWarpDstr{}.GetYs2DDescriptor().GetLengths());

        constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        // hot loop:
        static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                // read A warp tensor from A Block window
                const auto a_warp_tensor = load_tile(a_warp_windows(mIter)(kIter));
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    // read B warp tensor from B block tensor
                    BWarpTensor b_warp_tensor;

                    b_warp_tensor.GetThreadBuffer() = b_block_tensor.GetYSlicedThreadData(
                        merge_sequences(Sequence<nIter, kIter>{}, b_warp_y_index_zeros),
                        merge_sequences(Sequence<1, 1>{}, b_warp_y_lengths));

                    // read C warp tensor from C block tensor
                    CWarpTensor c_warp_tensor;

                    c_warp_tensor.GetThreadBuffer() = c_block_tensor.GetYSlicedThreadData(
                        merge_sequences(Sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(Sequence<1, 1>{}, c_warp_y_lengths));

                    // warp GEMM
                    WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                    // write C warp tensor into C block tensor
                    c_block_tensor.SetYSlicedThreadData(
                        merge_sequences(Sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(Sequence<1, 1>{}, c_warp_y_lengths),
                        c_warp_tensor.GetThreadBuffer());
                });
            });
        });
    }

    __device__ constexpr auto MakeCBlockTile() const
    {
        constexpr index_t MPerBlock = BlockGemmShape::kM;
        constexpr index_t NPerBlock = BlockGemmShape::kN;

        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template At<0>())>;

        constexpr index_t MWarp = config.template At<1>();
        constexpr index_t NWarp = config.template At<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
        // constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

        constexpr auto c_block_outer_dstr_encoding = StaticTileDistributionEncoding<
            Sequence<>,
            Tuple<Sequence<MIterPerWarp, MWarp>, Sequence<NIterPerWarp, NWarp>>,
            Tuple<Sequence<1, 2>>,
            Tuple<Sequence<1, 1>>,
            Sequence<1, 2>,
            Sequence<0, 0>>{};

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});
        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);
        auto c_block_tensor         = make_static_distributed_tensor<CDataType>(c_block_dstr);
        return c_block_tensor;
    }

    // C = A * B
    template <typename ABlockWindowTmp, typename BBlockTensorTmp>
    __device__ auto operator()(const ABlockWindowTmp& a_block_window_tmp,
                               const BBlockTensorTmp& b_block_tensor_tmp) const
    {
        static_assert(is_same_v<ADataType, remove_cv_t<typename ABlockWindowTmp::DataType>> &&
                          is_same_v<BDataType, remove_cv_t<typename BBlockTensorTmp::DataType>>,
                      "wrong!");

        // constexpr index_t MPerBlock = ABlockWindowTmp{}.GetWindowLengths()[Number<0>{}];
        // constexpr index_t NPerBlock = BBlockTensorTmp{}.GetLengths()[Number<0>{}];
        // constexpr index_t KPerBlock = ABlockWindowTmp{}.GetWindowLengths()[Number<1>{}];
        constexpr index_t MPerBlock = BlockGemmShape::kM;
        constexpr index_t NPerBlock = BlockGemmShape::kN;
        constexpr index_t KPerBlock = BlockGemmShape::kK;

        // static_assert(MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
        //                   KPerBlock == BlockGemmShape::kK,
        //               "wrong!");

        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template At<0>())>;

        constexpr index_t MWarp = config.template At<1>();
        constexpr index_t NWarp = config.template At<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

        constexpr index_t MPerBlockPerIter = MPerBlock / MIterPerWarp;
        constexpr index_t KPerBlockPerIter = KPerBlock / KIterPerWarp;

        const index_t iMWarp = get_warp_id() / NWarp;

        constexpr auto b_block_outer_dstr_encoding = StaticTileDistributionEncoding<
            Sequence<MWarp>,
            Tuple<Sequence<NIterPerWarp, NWarp>, Sequence<KIterPerWarp>>,
            Tuple<Sequence<0, 1>>,
            Tuple<Sequence<0, 1>>,
            Sequence<1, 2>,
            Sequence<0, 0>>{};

        constexpr auto c_block_outer_dstr_encoding = StaticTileDistributionEncoding<
            Sequence<>,
            Tuple<Sequence<MIterPerWarp, MWarp>, Sequence<NIterPerWarp, NWarp>>,
            Tuple<Sequence<1, 2>>,
            Tuple<Sequence<1, 1>>,
            Sequence<1, 2>,
            Sequence<0, 0>>{};

        constexpr auto b_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            b_block_outer_dstr_encoding, typename WG::BWarpDstrEncoding{});

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});

        constexpr auto b_block_dstr = make_static_tile_distribution(b_block_dstr_encode);
        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);

        // constrcut from B-block-tensor from B-Block-tensor-tmp
        // FIXME: need method to check b_block_tensor and b_block_tensor_tmp have equivalent
        // distribution
        auto b_block_tensor =
            make_static_distributed_tensor<typename BBlockTensorTmp::DataType>(b_block_dstr);

        b_block_tensor.GetThreadBuffer() = b_block_tensor_tmp.GetThreadBuffer();

        // construct A-warp-window
        auto a_warp_window_tmp = make_tile_window(
            a_block_window_tmp.GetBottomTensorView(),
            make_tuple(Number<WG::kM>{}, Number<WG::kK>{}),
            a_block_window_tmp.GetWindowOrigin() + MultiIndex<2>{iMWarp * WG::kM, 0},
            make_static_tile_distribution(typename WG::AWarpDstrEncoding{}));

#if 0 // FIXME: using Array will cause register spill
        Array<Array<decltype(b_warp_window_tmp), KIterPerWarp>, NIterPerWarp> b_warp_windows{
            {b_warp_window_tmp}};

        for(index_t nIter = 0; nIter < NIterPerWarp; nIter++)
        {
            for(index_t kIter = 0; kIter < KIterPerWarp; kIter++)
            {
                move_tile_window(b_warp_windows(nIter)(kIter),
                                 {nIter * NPerBlockPerIter, kIter * KPerBlockPerIter});
            }
        }
#else
        StaticallyIndexedArray<StaticallyIndexedArray<decltype(a_warp_window_tmp), KIterPerWarp>,
                               MIterPerWarp>
            a_warp_windows;

        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                a_warp_windows(mIter)(kIter) = a_warp_window_tmp;

                move_tile_window(a_warp_windows(mIter)(kIter),
                                 {mIter * MPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });
#endif

        // Construct C-Block-Tensor
        auto c_block_tensor = make_static_distributed_tensor<CDataType>(c_block_dstr);

        using BWarpDstr = typename WG::BWarpDstr;
        using CWarpDstr = typename WG::CWarpDstr;

        using BWarpTensor = typename WG::BWarpTensor;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto b_warp_y_lengths = to_sequence(BWarpDstr{}.GetYs2DDescriptor().GetLengths());
        constexpr auto c_warp_y_lengths = to_sequence(CWarpDstr{}.GetYs2DDescriptor().GetLengths());

        constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        // hot loop:
        static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                // read A warp tensor from A Block window
                const auto a_warp_tensor = load_tile(a_warp_windows(mIter)(kIter));
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    // read B warp tensor from B block tensor
                    BWarpTensor b_warp_tensor;

                    b_warp_tensor.GetThreadBuffer() = b_block_tensor.GetYSlicedThreadData(
                        merge_sequences(Sequence<nIter, kIter>{}, b_warp_y_index_zeros),
                        merge_sequences(Sequence<1, 1>{}, b_warp_y_lengths));

                    // read C warp tensor from C block tensor
                    CWarpTensor c_warp_tensor;

                    c_warp_tensor.GetThreadBuffer() = c_block_tensor.GetYSlicedThreadData(
                        merge_sequences(Sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(Sequence<1, 1>{}, c_warp_y_lengths));

                    // warp GEMM
                    WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                    // write C warp tensor into C block tensor
                    c_block_tensor.SetYSlicedThreadData(
                        merge_sequences(Sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(Sequence<1, 1>{}, c_warp_y_lengths),
                        c_warp_tensor.GetThreadBuffer());
                });
            });
        });

        return c_block_tensor;
    }
};

} // namespace block
} // namespace tile_program
} // namespace ck
