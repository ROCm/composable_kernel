// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"

#include "block_gemm_areg_bsmem_creg_problem.hpp"
#include "block_gemm_areg_bsmem_creg_v1_default_policy.hpp"
#include "block_gemm_areg_bsmem_creg_v1_iteratek_policy.hpp"

namespace ck_tile {

// A is block distributed tensor
// B is block window on shared memory
// C is block distributed tensor
template <typename Problem, typename Policy = BlockGemmARegBSmemCRegV1DefaultPolicy>
struct BlockGemmARegBSmemCRegV1
{
    using ADataType       = remove_cvref_t<typename Problem::ADataType>;
    using BDataType       = remove_cvref_t<typename Problem::BDataType>;
    using CDataType       = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape  = remove_cvref_t<typename Problem::BlockGemmShape>;
    using BlockGemmPolicy = Policy;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<ADataType>>::PackedSize;

    // B block tile distribution for load from lds
    CK_TILE_DEVICE static constexpr auto MakeBBlockDistributionEncode()
    {
        constexpr auto config =
            Policy::template GetWarpGemmMWarpNWarp<Problem, Problem::BlockGemmShape::kM>();
        using WG = remove_cvref_t<decltype(config.template get<0>())>;

        constexpr index_t MWarp = config.template get<1>();
        constexpr index_t NWarp = config.template get<2>();

        constexpr index_t NIterPerWarp = Problem::BlockGemmShape::kN / (NWarp * WG::kN);
        constexpr index_t KPerBlock    = Problem::BlockGemmShape::kK;
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

        constexpr auto b_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<MWarp>,
                                       tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};
        constexpr auto b_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            b_block_outer_dstr_encoding, typename WG::BWarpDstrEncoding{});

        return b_block_dstr_encode;
    }

    static constexpr auto BLdsTileDistr =
        decltype(make_static_tile_distribution(MakeBBlockDistributionEncode())){};
    using BLdsTile = decltype(make_static_distributed_tensor<BDataType>(BLdsTileDistr));

    template <index_t VectorSizeB = 8, index_t SmemPack = 8>
    CK_TILE_DEVICE static constexpr auto HotLoopScheduler()
    {
        constexpr index_t MPerBlock = BlockGemmShape::kM;
        constexpr index_t NPerBlock = BlockGemmShape::kN;
        constexpr index_t KPerBlock = BlockGemmShape::kK;

        constexpr auto config     = Policy::template GetWarpGemmMWarpNWarp<Problem, MPerBlock>();
        using WG                  = remove_cvref_t<decltype(config.template get<0>())>;
        constexpr index_t MPerXDL = WG::kM;
        constexpr index_t NPerXDL = WG::kN;
        constexpr index_t KPerXDL = WG::WarpGemmAttribute::Impl::kK;

        constexpr index_t WaveSize = get_warp_size();
        constexpr index_t WaveNumM = config.template get<1>();

        constexpr index_t B_LDS_RW_Width = SmemPack;

        constexpr index_t B_Buffer_Load_Inst_Num =
            NPerBlock * KPerBlock / (kBlockSize * VectorSizeB);

        constexpr index_t B_LDS_Write_Inst_Num =
            NPerBlock * KPerBlock / (kBlockSize * B_LDS_RW_Width);

        constexpr index_t B_LDS_Read_Inst_Num =
            WaveNumM * NPerBlock * KPerBlock / (kBlockSize * B_LDS_RW_Width);

        constexpr index_t C_MFMA_Inst_Num = MPerBlock * NPerBlock * KPerBlock /
                                            (kBlockSize / WaveSize) / (MPerXDL * NPerXDL * KPerXDL);

        // B split schedule
        constexpr auto num_ds_read_inst_b = B_LDS_RW_Width * sizeof(BDataType) / kPackedSize == 16
                                                ? B_LDS_Read_Inst_Num
                                                : B_LDS_Read_Inst_Num / 2;

        constexpr auto num_ds_write_inst_b = B_LDS_Write_Inst_Num;

        constexpr auto num_buffer_load_inst_b = B_Buffer_Load_Inst_Num;

        constexpr auto num_mfma_inst = C_MFMA_Inst_Num;

        constexpr auto mfma_cycle = NPerXDL == 16 ? 16 : 32;
        constexpr auto ds_read_b_issue_cycle =
            B_LDS_RW_Width * sizeof(BDataType) / kPackedSize == 16 ? 8 : 4;
        constexpr auto ds_read_b_mfma_rate =
            (mfma_cycle - 4 + 2 * ds_read_b_issue_cycle - 1) / (2 * ds_read_b_issue_cycle);

        constexpr auto num_dsread_b_mfma =
            (num_ds_read_inst_b + ds_read_b_mfma_rate - 1) / ds_read_b_mfma_rate;

        // stage 1
        constexpr auto num_mfma_stage1         = num_mfma_inst - (num_dsread_b_mfma);
        constexpr auto num_mfma_per_issue      = num_mfma_stage1 / (num_buffer_load_inst_b);
        constexpr auto num_dswrite_per_issue_b = num_ds_write_inst_b / num_buffer_load_inst_b;
        constexpr auto num_mfma_per_dswrite_b =
            (num_mfma_per_issue - num_dswrite_per_issue_b * 2 >= 1) ? 2 : 1;

        static_for<0, num_buffer_load_inst_b, 1>{}([&](auto i) {
            ignore = i;
            static_for<0, num_dswrite_per_issue_b, 1>{}([&](auto idswrite) {
                ignore = idswrite;
                __builtin_amdgcn_sched_group_barrier(0x200, 1, 0);                      // DS write
                __builtin_amdgcn_sched_group_barrier(0x008, num_mfma_per_dswrite_b, 0); // MFMA
            });
            __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
            __builtin_amdgcn_sched_group_barrier(0x008,
                                                 num_mfma_per_issue - num_mfma_per_dswrite_b *
                                                                          num_dswrite_per_issue_b,
                                                 0); // MFMA
        });

        // stage 2
        static_for<0, num_dsread_b_mfma, 1>{}([&](auto i) {
            if constexpr((num_ds_read_inst_b - (i + 1) * ds_read_b_mfma_rate) >=
                         ds_read_b_mfma_rate)
            {
                __builtin_amdgcn_sched_group_barrier(0x100, ds_read_b_mfma_rate, 0); // DS read
            }
            else
            {
                __builtin_amdgcn_sched_group_barrier(0x100,
                                                     num_ds_read_inst_b - (num_dsread_b_mfma - 1) *
                                                                              ds_read_b_mfma_rate,
                                                     0); // DS read
            }
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
        });
    }

    // C += A * B
    template <typename CBlockTensor, typename ABlockTensorTmp>
    __device__ void operator()(CBlockTensor& c_block_tensor,
                               const ABlockTensorTmp& a_block_tensor_tmp,
                               const BLdsTile& b_block_tensor_tmp) const
    {
        static_assert(std::is_same_v<ADataType, remove_cv_t<typename ABlockTensorTmp::DataType>> &&
                          std::is_same_v<BDataType, remove_cv_t<typename BLdsTile::DataType>> &&
                          std::is_same_v<CDataType, remove_cv_t<typename CBlockTensor::DataType>>,
                      "wrong!");

        constexpr index_t MPerBlock = ABlockTensorTmp{}.get_lengths()[number<0>{}];
        constexpr index_t NPerBlock = CBlockTensor{}.get_lengths()[number<1>{}];
        constexpr index_t KPerBlock = ABlockTensorTmp{}.get_lengths()[number<1>{}];

        static_assert(MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
                          KPerBlock == BlockGemmShape::kK,
                      "wrong!");

        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem, MPerBlock>();

        using WG = remove_cvref_t<decltype(config.template get<0>())>;

        constexpr index_t MWarp = config.template get<1>();
        constexpr index_t NWarp = config.template get<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

        constexpr auto a_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encoding, typename WG::AWarpDstrEncoding{});

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});

        constexpr auto a_block_dstr = make_static_tile_distribution(a_block_dstr_encode);

        // constrcut from A-block-tensor from A-Block-tensor-tmp
        // FIXME: need method to check a_block_tensor and a_block_tensor_tmp have equivalent
        // distribution
        auto a_block_tensor =
            make_static_distributed_tensor<typename ABlockTensorTmp::DataType>(a_block_dstr);

        a_block_tensor.get_thread_buffer() = a_block_tensor_tmp.get_thread_buffer();
        // check C-block-distribution
        static_assert(
            std::is_same_v<remove_cvref_t<decltype(c_block_dstr_encode)>,
                           remove_cvref_t<decltype(CBlockTensor::get_tile_distribution()
                                                       .get_static_tile_distribution_encoding())>>,
            "wrong!");

        using AWarpDstr = typename WG::AWarpDstr;
        using BWarpDstr = typename WG::BWarpDstr;
        using CWarpDstr = typename WG::CWarpDstr;

        using AWarpTensor = typename WG::AWarpTensor;
        using BWarpTensor = typename WG::BWarpTensor;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto a_warp_y_lengths =
            to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        static constexpr auto b_warp_y_lengths =
            to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        // hot loop:
        static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                // read A warp tensor from A block tensor
                AWarpTensor a_warp_tensor;
                a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    // read B warp tensor from B Block window
                    BWarpTensor b_warp_tensor;
                    b_warp_tensor.get_thread_buffer() = b_block_tensor_tmp.get_y_sliced_thread_data(
                        merge_sequences(sequence<nIter, kIter>{}, b_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

                    // read C warp tensor from C block tensor
                    CWarpTensor c_warp_tensor;
                    c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                    // warp GEMM
                    WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                    // write C warp tensor into C block tensor
                    c_block_tensor.set_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                        c_warp_tensor.get_thread_buffer());
                });
            });
        });
    }

    // C += A * B
    template <typename CBlockTensor, typename ABlockTensorTmp, typename BBlockWindowTmp>
    __device__ void operator()(CBlockTensor& c_block_tensor,
                               const ABlockTensorTmp& a_block_tensor_tmp,
                               const BBlockWindowTmp& b_block_window_tmp) const
    {
        static_assert(
            std::is_same_v<ADataType, remove_cv_t<typename ABlockTensorTmp::DataType>> &&
                std::is_same_v<BDataType, remove_cv_t<typename BBlockWindowTmp::DataType>> &&
                std::is_same_v<CDataType, remove_cv_t<typename CBlockTensor::DataType>>,
            "wrong!");

        static_assert((BlockGemmShape::kM == BlockGemmShape::kN), "wrong!");

        constexpr index_t MPerBlock = ABlockTensorTmp{}.get_lengths()[number<0>{}];
        constexpr index_t NPerBlock = BBlockWindowTmp{}.get_window_lengths()[number<0>{}];
        constexpr index_t KPerBlock = ABlockTensorTmp{}.get_lengths()[number<1>{}];

        static_assert(MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
                          KPerBlock == BlockGemmShape::kK,
                      "wrong!");

        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem, MPerBlock>();

        using WG = remove_cvref_t<decltype(config.template get<0>())>;

        constexpr index_t MWarp = config.template get<1>();
        constexpr index_t NWarp = config.template get<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

        constexpr index_t NPerBlockPerIter = NPerBlock / NIterPerWarp;
        constexpr index_t KPerBlockPerIter = KPerBlock / KIterPerWarp;

        const index_t iNWarp = get_warp_id() % NWarp;

        constexpr auto a_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encoding, typename WG::AWarpDstrEncoding{});

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});

        constexpr auto a_block_dstr = make_static_tile_distribution(a_block_dstr_encode);

        // constrcut from A-block-tensor from A-Block-tensor-tmp
        // FIXME: need method to check a_block_tensor and a_block_tensor_tmp have equivalent
        // distribution
        auto a_block_tensor =
            make_static_distributed_tensor<typename ABlockTensorTmp::DataType>(a_block_dstr);

        a_block_tensor.get_thread_buffer() = a_block_tensor_tmp.get_thread_buffer();

        // construct B-warp-window
        auto b_warp_window_tmp = make_tile_window(
            b_block_window_tmp.get_bottom_tensor_view(),
            make_tuple(number<WG::kN>{}, number<WG::kK>{}),
            {b_block_window_tmp.get_window_origin().at(number<0>{}) + iNWarp * WG::kN,
             b_block_window_tmp.get_window_origin().at(number<1>{})},
            make_static_tile_distribution(typename WG::BWarpDstrEncoding{}));

        statically_indexed_array<
            statically_indexed_array<decltype(b_warp_window_tmp), KIterPerWarp>,
            NIterPerWarp>
            b_warp_windows;

        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                b_warp_windows(nIter)(kIter) = b_warp_window_tmp;

                move_tile_window(b_warp_windows(nIter)(kIter),
                                 {nIter * NPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });

        // check C-block-distribution
        static_assert(
            std::is_same_v<remove_cvref_t<decltype(c_block_dstr_encode)>,
                           remove_cvref_t<decltype(CBlockTensor::get_tile_distribution()
                                                       .get_static_tile_distribution_encoding())>>,
            "wrong!");

        using AWarpDstr = typename WG::AWarpDstr;
        using CWarpDstr = typename WG::CWarpDstr;

        using AWarpTensor = typename WG::AWarpTensor;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto a_warp_y_lengths =
            to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        // hot loop:
        static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                // read A warp tensor from A block tensor
                AWarpTensor a_warp_tensor;

                a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    // read B warp tensor from B Block window
                    const auto b_warp_tensor = load_tile(b_warp_windows(nIter)(kIter));
                    // read C warp tensor from C block tensor
                    CWarpTensor c_warp_tensor;

                    c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                    // warp GEMM
                    WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                    // write C warp tensor into C block tensor
                    c_block_tensor.set_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                        c_warp_tensor.get_thread_buffer());
                });
            });
        });
    }

    // C = A * B
    template <typename ABlockTensorTmp, typename BBlockWindowTmp>
    __device__ auto operator()(const ABlockTensorTmp& a_block_tensor_tmp,
                               const BBlockWindowTmp& b_block_window_tmp) const
    {
        static_assert(
            std::is_same_v<ADataType, remove_cv_t<typename ABlockTensorTmp::DataType>> &&
                std::is_same_v<BDataType, remove_cv_t<typename BBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert((BlockGemmShape::kM == BlockGemmShape::kN), "wrong!");

        constexpr index_t MPerBlock = ABlockTensorTmp{}.get_lengths()[number<0>{}];
        constexpr index_t NPerBlock = BBlockWindowTmp{}.get_window_lengths()[number<0>{}];
        constexpr index_t KPerBlock = ABlockTensorTmp{}.get_lengths()[number<1>{}];

        static_assert(MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
                          KPerBlock == BlockGemmShape::kK,
                      "wrong!");

        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem, MPerBlock>();

        using WG = remove_cvref_t<decltype(config.template get<0>())>;

        constexpr index_t MWarp = config.template get<1>();
        constexpr index_t NWarp = config.template get<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

        constexpr index_t NPerBlockPerIter = NPerBlock / NIterPerWarp;
        constexpr index_t KPerBlockPerIter = KPerBlock / KIterPerWarp;

        const index_t iNWarp = get_warp_id() % NWarp;

        constexpr auto a_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encoding, typename WG::AWarpDstrEncoding{});

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});

        constexpr auto a_block_dstr = make_static_tile_distribution(a_block_dstr_encode);
        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);

        // constrcut from A-block-tensor from A-Block-tensor-tmp
        // FIXME: need method to check a_block_tensor and a_block_tensor_tmp have equivalent
        // distribution
        auto a_block_tensor =
            make_static_distributed_tensor<typename ABlockTensorTmp::DataType>(a_block_dstr);

        a_block_tensor.get_thread_buffer() = a_block_tensor_tmp.get_thread_buffer();

        // construct B-warp-window
        auto b_warp_window_tmp = make_tile_window(
            b_block_window_tmp.get_bottom_tensor_view(),
            make_tuple(number<WG::kN>{}, number<WG::kK>{}),
            {b_block_window_tmp.get_window_origin().at(number<0>{}) + iNWarp * WG::kN,
             b_block_window_tmp.get_window_origin().at(number<1>{})},
            make_static_tile_distribution(typename WG::BWarpDstrEncoding{}));

        statically_indexed_array<
            statically_indexed_array<decltype(b_warp_window_tmp), KIterPerWarp>,
            NIterPerWarp>
            b_warp_windows;

        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                b_warp_windows(nIter)(kIter) = b_warp_window_tmp;

                move_tile_window(b_warp_windows(nIter)(kIter),
                                 {nIter * NPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });

        // Construct C-Block-Tensor
        auto c_block_tensor = make_static_distributed_tensor<CDataType>(c_block_dstr);

        using AWarpDstr = typename WG::AWarpDstr;
        using CWarpDstr = typename WG::CWarpDstr;

        using AWarpTensor = typename WG::AWarpTensor;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto a_warp_y_lengths =
            to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        // hot loop:
        static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                // read A warp tensor from A block tensor
                AWarpTensor a_warp_tensor;

                a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    // read B warp tensor from B Block window
                    const auto b_warp_tensor = load_tile(b_warp_windows(nIter)(kIter));

                    // read C warp tensor from C block tensor
                    CWarpTensor c_warp_tensor;

                    c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                    // warp GEMM
                    WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                    // write C warp tensor into C block tensor
                    c_block_tensor.set_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                        c_warp_tensor.get_thread_buffer());
                });
            });
        });

        return c_block_tensor;
    }
};

} // namespace ck_tile
