// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1_default_policy.hpp"

namespace ck_tile {

// A is block distributed tensor
// B is block distributed tensor
// C is block distributed tensor
template <typename Problem_, typename Policy_ = BlockGemmARegBRegCRegV1DefaultPolicy>
struct BlockGemmARegBRegCRegV1
{
    private:
    template <typename PipelineProblem_, typename GemmPolicy_>
    struct GemmTraits_
    {
        using Problem        = remove_cvref_t<PipelineProblem_>;
        using Policy         = remove_cvref_t<GemmPolicy_>;
        using ADataType      = remove_cvref_t<typename Problem::ADataType>;
        using BDataType      = remove_cvref_t<typename Problem::BDataType>;
        using CDataType      = remove_cvref_t<typename Problem::CDataType>;
        using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

        static constexpr index_t kBlockSize = Problem::kBlockSize;

        static constexpr index_t MPerBlock = BlockGemmShape::kM;
        static constexpr index_t NPerBlock = BlockGemmShape::kN;
        static constexpr index_t KPerBlock = BlockGemmShape::kK;

        static constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm               = remove_cvref_t<decltype(config.template at<0>())>;

        static constexpr index_t MWarp = config.template at<1>() / (Problem::kNumWaveGroups);
        static constexpr index_t NWarp = config.template at<2>();
        static constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WarpGemm::kM);
        static constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WarpGemm::kN);
        static constexpr index_t KIterPerWarp = KPerBlock / WarpGemm::kK;

        static constexpr index_t OrigMWarp = config.template at<1>();
        static constexpr index_t OrigNWarp = config.template at<2>();

        static constexpr index_t KPack = WarpGemm::kKPerThread;
    };

    public:
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    using Traits = GemmTraits_<Problem, Policy>;

    using WarpGemm       = typename Traits::WarpGemm;
    using BlockGemmShape = typename Traits::BlockGemmShape;

    using ADataType = remove_cvref_t<typename Traits::ADataType>;
    using BDataType = remove_cvref_t<typename Traits::BDataType>;
    using CDataType = remove_cvref_t<typename Traits::CDataType>;

    static constexpr index_t KIterPerWarp = Traits::KIterPerWarp;
    static constexpr index_t MIterPerWarp = Traits::MIterPerWarp;
    static constexpr index_t NIterPerWarp = Traits::NIterPerWarp;

    static constexpr index_t MWarp = Traits::MWarp;
    static constexpr index_t NWarp = Traits::NWarp;
    static constexpr index_t OrigMWarp = Traits::OrigMWarp;
    static constexpr index_t OrigNWarp = Traits::OrigNWarp;

    static constexpr index_t MPerBlock = Traits::MPerBlock;
    static constexpr index_t NPerBlock = Traits::NPerBlock;
    static constexpr index_t KPack = Traits::KPack;

    CK_TILE_DEVICE static constexpr auto MakeABlockDistributionEncode()
    {
        constexpr auto a_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<OrigNWarp>,
                                       tuple<sequence<MIterPerWarp, OrigMWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};
        /*
        constexpr auto a_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<OrigNWarp>,
                                       tuple<sequence<MIterPerWarp, OrigMWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};
        */
        /*
        constexpr auto a_block_outer_dstr_encoding = 
            tile_distribution_encoding<sequence<MWarp>, 
                                        tuple<sequence<MIterPerWarp>, sequence<KIterPerWarp>>, 
                                        tuple<>, 
                                        tuple<>, 
                                        sequence<1, 2>, 
                                        sequence<0, 0>>{};
        */
        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encoding, typename WarpGemm::AWarpDstrEncoding{});

        return a_block_dstr_encode;
    }

    CK_TILE_DEVICE static constexpr auto MakeBBlockDistributionEncode()
    {
        constexpr auto b_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<OrigMWarp>,
                                       tuple<sequence<NIterPerWarp, OrigNWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};
        
        /*
        constexpr auto b_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<MWarp>,                                                 
                                       tuple<sequence<NIterPerWarp>, sequence<KIterPerWarp>>,    
                                       tuple<>,                                           
                                       tuple<>,
                                       sequence<1, 2>,                                                  
                                       sequence<0, 0>>{};
        */
        constexpr auto b_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            b_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});

        return b_block_dstr_encode;
    }

    CK_TILE_DEVICE static constexpr auto MakeCBlockDistributionEncode()
    {
        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<OrigMWarp>,
            tuple<sequence<MIterPerWarp>, sequence<NIterPerWarp, OrigNWarp>>,
            tuple<sequence<0, 2>>,
            tuple<sequence<0, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};  
        
        /*
        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<MWarp>,
            tuple<sequence<MIterPerWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<>,
            tuple<>,
            sequence<1, 2>,
            sequence<0, 0>>{};        
        */
        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WarpGemm::CWarpDstrEncoding{});

        return c_block_dstr_encode;
    }

    CK_TILE_DEVICE static constexpr auto MakeCBlockLdsDescriptor()
    {
        constexpr auto c_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<NPerBlock / KPack>{}, number<MPerBlock>{}, number<KPack>{}),
            make_tuple(number<MPerBlock * KPack>{}, number<KPack>{}, number<1>{}),
            number<KPack>{},
            number<1>{});

        constexpr auto c_lds_block_desc = transform_tensor_descriptor(
                c_lds_block_desc_0,
                make_tuple(
                    make_pass_through_transform(number<MPerBlock>{}),
                    make_merge_transform(make_tuple(number<NPerBlock>{} / KPack, number<KPack>{}))),
                make_tuple(sequence<1>{}, sequence<0, 2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

        return c_lds_block_desc;
    }

    // C += A * B
    template <typename CBlockTensor, typename ABlockTensor, typename BBlockTensor>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ABlockTensor& a_block_tensor,
                                   const BBlockTensor& b_block_tensor) const
    {

        static_assert(std::is_same_v<ADataType, remove_cv_t<typename ABlockTensor::DataType>> &&
                          std::is_same_v<BDataType, remove_cv_t<typename BBlockTensor::DataType>> &&
                          std::is_same_v<CDataType, remove_cv_t<typename CBlockTensor::DataType>>,
                      "wrong!");

        constexpr auto a_block_dstr_encode = MakeABlockDistributionEncode();

        constexpr auto b_block_dstr_encode = MakeBBlockDistributionEncode();

        constexpr auto c_block_dstr_encode = MakeCBlockDistributionEncode();

        // check ABC-block-distribution
        static_assert(
            std::is_same_v<remove_cvref_t<decltype(a_block_dstr_encode)>,
                           remove_cvref_t<decltype(ABlockTensor::get_tile_distribution()
                                                       .get_static_tile_distribution_encoding())>>,
            "A distribution is wrong!");
        static_assert(
            std::is_same_v<remove_cvref_t<decltype(b_block_dstr_encode)>,
                           remove_cvref_t<decltype(BBlockTensor::get_tile_distribution()
                                                       .get_static_tile_distribution_encoding())>>,
            "B distribution is wrong!");
        static_assert(
            std::is_same_v<remove_cvref_t<decltype(c_block_dstr_encode)>,
                           remove_cvref_t<decltype(CBlockTensor::get_tile_distribution()
                                                       .get_static_tile_distribution_encoding())>>,
            "C distribution is wrong!");

        using AWarpDstr = typename WarpGemm::AWarpDstr;
        using BWarpDstr = typename WarpGemm::BWarpDstr;
        using CWarpDstr = typename WarpGemm::CWarpDstr;

        using AWarpTensor = typename WarpGemm::AWarpTensor;
        using BWarpTensor = typename WarpGemm::BWarpTensor;
        using CWarpTensor = typename WarpGemm::CWarpTensor;

        constexpr auto a_warp_y_lengths =
            to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto b_warp_y_lengths =
            to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        // hot loop:
        static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                // read A warp tensor from A Block window
                AWarpTensor a_warp_tensor;

                a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    // read B warp tensor from B block tensor
                    BWarpTensor b_warp_tensor;

                    b_warp_tensor.get_thread_buffer() = b_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<nIter, kIter>{}, b_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

                    // read C warp tensor from C block tensor
                    CWarpTensor c_warp_tensor;

                    c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                    // warp GEMM
                    WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                    // write C warp tensor into C block tensor
                    c_block_tensor.set_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                        c_warp_tensor.get_thread_buffer());
                });
            });
        });
    }

    CK_TILE_DEVICE static constexpr auto MakeCBlockTile()
    {
        /* Gold standard */
        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<OrigMWarp>,
            tuple<sequence<MIterPerWarp>, sequence<NIterPerWarp, OrigNWarp>>,
            tuple<sequence<0, 2>>,
            tuple<sequence<0, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};  
        
        /*
        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<MWarp>,
            tuple<sequence<MIterPerWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<>,
            tuple<>,
            sequence<1, 2>,
            sequence<0, 0>>{};        
        */
        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WarpGemm::CWarpDstrEncoding{});
        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);
        auto c_block_tensor         = make_static_distributed_tensor<CDataType>(c_block_dstr);
        /*
        auto c_block_tensor_view    = make_naive_tensor_view<address_space_enum::vgpr, 
                                                              memory_operation_enum::set,
                                                              amd_buffer_coherence_enum::slc>(
                                                                c_block_tensor.mData, 
                                                                make_tuple(MPerBlock, NPerBlock), 
                                                                make_tuple(NPerBlock, 1), 
                                                                number<1>{}, number<1>{});
        */                                               


        return c_block_tensor;
    }

    // C = A * B
    template <typename ABlockTensor, typename BBlockTensor>
    CK_TILE_DEVICE auto operator()(const ABlockTensor& a_block_tensor,
                                   const BBlockTensor& b_block_tensor) const
    {
        auto c_block_tensor = MakeCBlockTile();
        operator()(c_block_tensor, a_block_tensor, b_block_tensor);
        return c_block_tensor;
    }
};

} // namespace ck_tile
