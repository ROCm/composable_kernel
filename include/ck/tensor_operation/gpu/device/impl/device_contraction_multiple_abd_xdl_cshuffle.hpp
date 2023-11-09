// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>
#include <vector>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_contraction_multiple_abd.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_abd_xdl_cshuffle.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {

template <typename GridwiseGemm,
          typename AsPointer,
          typename BsPointer,
          typename DsPointer,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename AsGridDesc_AK0_M_AK1,
          typename BsGridDesc_BK0_N_BK1,
          typename DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename Block2ETileMap,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_contraction_multiple_abd_xdl_cshuffle(
            AsPointer p_as_grid,
            BsPointer p_bs_grid,
            DsPointer p_ds_grid,
            EDataType* __restrict__ p_e_grid,
            const AElementwiseOperation a_element_op,
            const BElementwiseOperation b_element_op,
            const CDEElementwiseOperation cde_element_op,
            const AsGridDesc_AK0_M_AK1 as_grid_desc_ak0_m_ak1,
            const BsGridDesc_BK0_N_BK1 bs_grid_desc_bk0_n_bk1,
            const DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                ds_grid_desc_mblock_mperblock_nblock_nperblock,
            const EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                e_grid_desc_mblock_mperblock_nblock_nperblock,
            const Block2ETileMap block_2_etile_map)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__) || \
    defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    GridwiseGemm::template Run<HasMainKBlockLoop>(p_as_grid,
                                                  p_bs_grid,
                                                  p_ds_grid,
                                                  p_e_grid,
                                                  p_shared,
                                                  a_element_op,
                                                  b_element_op,
                                                  cde_element_op,
                                                  as_grid_desc_ak0_m_ak1,
                                                  bs_grid_desc_bk0_n_bk1,
                                                  ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  e_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  block_2_etile_map);
#else
    ignore = p_as_grid;
    ignore = p_bs_grid;
    ignore = p_ds_grid;
    ignore = p_e_grid;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = cde_element_op;
    ignore = as_grid_desc_ak0_m_ak1;
    ignore = bs_grid_desc_bk0_n_bk1;
    ignore = ds_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = e_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = block_2_etile_map;
#endif
}

} // namespace ck

namespace ck {
namespace tensor_operation {
namespace device {

// GEMM:
//   input : A[M, K]
//   input : B[N, K]
//   input : D0[M, N], D1[M, N], ...
//   output : E[M, N]
//   C = a_op(A) * b_op(B)
//   E = cde_op(C, D0, D1, ...)
// Assume:
//   D0, D1, ... and E have the same layout
template <index_t NumDimM,
          index_t NumDimN,
          index_t NumDimK,
          typename AsDataType,
          typename BsDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          GemmSpecialization GemmSpec,
          index_t NumGemmKPrefetchStage,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          index_t BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEBlockTransferScalarPerVector_NPerBlock,
          LoopScheduler LoopSched     = make_default_loop_scheduler(),
          PipelineVersion PipelineVer = PipelineVersion::v1>
struct DeviceContractionMultipleABD_Xdl_CShuffle
    : public DeviceContractionMultipleABD<NumDimM,
                                          NumDimN,
                                          NumDimK,
                                          AsDataType,
                                          BsDataType,
                                          DsDataType,
                                          EDataType,
                                          AElementwiseOperation,
                                          BElementwiseOperation,
                                          CDEElementwiseOperation>
{
    using DeviceOp = DeviceContractionMultipleABD_Xdl_CShuffle;

    static constexpr index_t NumATensor = AsDataType::Size();
    static constexpr index_t NumBTensor = BsDataType::Size();
    static constexpr index_t NumDTensor = DsDataType::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    using ComputeDataType = EDataType;

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemmMultipleABD_xdl_cshuffle<
        AsDataType,
        BsDataType,
        ComputeDataType,
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        InMemoryDataOperationEnum::Set,
        NumGemmKPrefetchStage,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        false,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEBlockTransferScalarPerVector_NPerBlock,
        LoopSched,
        PipelineVer>;

    static constexpr auto matrix_padder =
        ck::tensor_operation::device::MatrixPadder<GemmSpec, index_t, index_t, index_t>{
            MPerBlock, NPerBlock, KPerBlock};

    static auto MakeAGridDescriptor_M_K(const std::vector<index_t>& a_ms_ks_lengths_,
                                        const std::vector<index_t>& a_ms_ks_strides_)
    {
        assert(a_ms_ks_lengths_.size() == NumDimM + NumDimK &&
               a_ms_ks_strides_.size() == NumDimM + NumDimK);

        const auto to_tuple = [&](auto& vec, auto num) {
            return generate_tuple([&](auto i) { return vec[i]; }, num);
        };

        const auto a_ms_ks_lengths = to_tuple(a_ms_ks_lengths_, Number<NumDimM + NumDimK>{});
        const auto a_ms_ks_strides = to_tuple(a_ms_ks_strides_, Number<NumDimM + NumDimK>{});

        // dimension Ids for M0, M1, ...
        constexpr auto mDimIds = typename arithmetic_sequence_gen<0, NumDimM, 1>::type{};

        // dimension Ids for K0, K1, ...
        constexpr auto kDimIds =
            typename arithmetic_sequence_gen<NumDimM, NumDimM + NumDimK, 1>::type{};

        // lengths for M0, M1, ...
        const auto mLengths = get_container_subset(a_ms_ks_lengths, mDimIds);

        // lengths for K0, K1, ...
        const auto kLengths = get_container_subset(a_ms_ks_lengths, kDimIds);

        // naive tensor A[M0, M1, M2, ..., K0, K1, K2...]
        const auto a_grid_desc_ms_ks =
            make_naive_tensor_descriptor(a_ms_ks_lengths, a_ms_ks_strides);

        // transformed tensor A[MRaw = M0 * M1 * M2 * ... , KRaw = K0 * K1 * K2 * ...]
        const auto a_grid_desc_mraw_kraw = transform_tensor_descriptor(
            a_grid_desc_ms_ks,
            make_tuple(make_merge_transform(mLengths), make_merge_transform(kLengths)),
            make_tuple(mDimIds, kDimIds),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return matrix_padder.PadADescriptor_M_K(a_grid_desc_mraw_kraw);
    }

    __host__ __device__ static auto
    MakeAsGridDescriptor_M_K(const std::array<std::vector<index_t>, NumATensor>& as_ms_ks_lengths,
                             const std::array<std::vector<index_t>, NumATensor>& as_ms_ks_strides)
    {
        return generate_tuple(
            [&](auto i) {
                return MakeAGridDescriptor_M_K(as_ms_ks_lengths[i], as_ms_ks_strides[i]);
            },
            Number<NumATensor>{});
    }

    // Assume: B[N0, N1, N2, ..., K0, K1, K2, ...]
    static auto MakeBGridDescriptor_N_K(const std::vector<index_t>& b_ns_ks_lengths_,
                                        const std::vector<index_t>& b_ns_ks_strides_)
    {
        assert(b_ns_ks_lengths_.size() == NumDimN + NumDimK &&
               b_ns_ks_strides_.size() == NumDimN + NumDimK);

        const auto to_tuple = [&](auto& vec, auto num) {
            return generate_tuple([&](auto i) { return vec[i]; }, num);
        };

        const auto b_ns_ks_lengths = to_tuple(b_ns_ks_lengths_, Number<NumDimN + NumDimK>{});
        const auto b_ns_ks_strides = to_tuple(b_ns_ks_strides_, Number<NumDimN + NumDimK>{});

        // dimension Ids for N0, N1, ...
        constexpr auto nDimIds = typename arithmetic_sequence_gen<0, NumDimN, 1>::type{};

        // dimension Ids for K0, K1, ...
        constexpr auto kDimIds =
            typename arithmetic_sequence_gen<NumDimN, NumDimN + NumDimK, 1>::type{};

        // lengths for K0, K1, ...
        const auto kLengths = get_container_subset(b_ns_ks_lengths, kDimIds);

        // lengths for N0, N1, ...
        const auto nLengths = get_container_subset(b_ns_ks_lengths, nDimIds);

        // naive tensor B[N0, N1, N2, ..., K0, K1, K2, ...]
        const auto b_grid_desc_ns_ks =
            make_naive_tensor_descriptor(b_ns_ks_lengths, b_ns_ks_strides);

        // transformed tensor B[NRaw = N0 * N1 * N2 * ..., KRaw = K0 * K1 * K2 * ...]
        const auto b_grid_desc_nraw_kraw = transform_tensor_descriptor(
            b_grid_desc_ns_ks,
            make_tuple(make_merge_transform(nLengths), make_merge_transform(kLengths)),
            make_tuple(nDimIds, kDimIds),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return matrix_padder.PadBDescriptor_N_K(b_grid_desc_nraw_kraw);
    }

    __host__ __device__ static auto
    MakeBsGridDescriptor_N_K(const std::array<std::vector<index_t>, NumBTensor>& bs_ns_ks_lengths,
                             const std::array<std::vector<index_t>, NumBTensor>& bs_ns_ks_strides)
    {
        return generate_tuple(
            [&](auto i) {
                return MakeBGridDescriptor_N_K(bs_ns_ks_lengths[i], bs_ns_ks_strides[i]);
            },
            Number<NumBTensor>{});
    }

    // assume E[M0, M1, M2, ..., N0, N1, N2...]
    static auto MakeEGridDescriptor_M_N(const std::vector<index_t>& e_ms_ns_lengths_,
                                        const std::vector<index_t>& e_ms_ns_strides_)
    {
        assert(e_ms_ns_lengths_.size() == NumDimM + NumDimN &&
               e_ms_ns_strides_.size() == NumDimM + NumDimN);

        const auto to_tuple = [&](auto& vec, auto num) {
            return generate_tuple([&](auto i) { return vec[i]; }, num);
        };

        const auto e_ms_ns_lengths = to_tuple(e_ms_ns_lengths_, Number<NumDimM + NumDimN>{});
        const auto e_ms_ns_strides = to_tuple(e_ms_ns_strides_, Number<NumDimM + NumDimN>{});

        // dimension Ids for M0, M1, ...
        constexpr auto mDimIds = typename arithmetic_sequence_gen<0, NumDimM, 1>::type{};

        // dimension Ids for N0, N1, ...
        constexpr auto nDimIds =
            typename arithmetic_sequence_gen<NumDimM, NumDimM + NumDimN, 1>::type{};

        // lengths for M0, M1, ...
        const auto mLengths = get_container_subset(e_ms_ns_lengths, mDimIds);

        // lengths for K0, K1, ...
        const auto nLengths = get_container_subset(e_ms_ns_lengths, nDimIds);

        // naive tensor E[M0, M1, M2, ..., N0, N1, N2...]
        const auto e_grid_desc_ms_ns =
            make_naive_tensor_descriptor(e_ms_ns_lengths, e_ms_ns_strides);

        // transformed tensor E[MRaw = M0 * M1 * M2 * ... , NRaw = N0 * N1 * N2 * ...]
        const auto e_grid_desc_mraw_nraw = transform_tensor_descriptor(
            e_grid_desc_ms_ns,
            make_tuple(make_merge_transform(mLengths), make_merge_transform(nLengths)),
            make_tuple(mDimIds, nDimIds),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return matrix_padder.PadCDescriptor_M_N(e_grid_desc_mraw_nraw);
    }

    static auto
    MakeDsGridDescriptor_M_N(const std::array<std::vector<index_t>, NumDTensor>& ds_ms_ns_lengths,
                             const std::array<std::vector<index_t>, NumDTensor>& ds_ms_ns_strides)
    {
        return generate_tuple(
            [&](auto i) {
                return MakeEGridDescriptor_M_N(ds_ms_ns_lengths[i], ds_ms_ns_strides[i]);
            },
            Number<NumDTensor>{});
    }

    // desc for problem definition
    using AsGridDesc_M_K = remove_cvref_t<decltype(MakeAsGridDescriptor_M_K({}, {}))>;
    using BsGridDesc_N_K = remove_cvref_t<decltype(MakeBsGridDescriptor_N_K({}, {}))>;
    using DsGridDesc_M_N = remove_cvref_t<decltype(MakeDsGridDescriptor_M_N({}, {}))>;
    using EGridDesc_M_N  = remove_cvref_t<decltype(MakeEGridDescriptor_M_N({}, {}))>;

    // desc for blockwise copy
    using AsGridDesc_AK0_M_AK1 =
        remove_cvref_t<decltype(GridwiseGemm::MakeDefaultAsGridDescriptor_AK0_M_AK1(
            AsGridDesc_M_K{}))>;
    using BsGridDesc_BK0_N_BK1 =
        remove_cvref_t<decltype(GridwiseGemm::MakeDefaultBsGridDescriptor_BK0_N_BK1(
            BsGridDesc_N_K{}))>;
    using DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock = remove_cvref_t<
        decltype(GridwiseGemm::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            DsGridDesc_M_N{}))>;
    using EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock =
        remove_cvref_t<decltype(GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            EGridDesc_M_N{}))>;

    // block-to-e-tile map
    using Block2ETileMap =
        remove_cvref_t<decltype(GridwiseGemm::MakeDefaultBlock2ETileMap(EGridDesc_M_N{}))>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(std::array<const void*, NumATensor> p_as_grid,
                 std::array<const void*, NumBTensor> p_bs_grid,
                 std::array<const void*, NumDTensor> p_ds_grid,
                 void* p_e_grid,
                 const std::array<std::vector<index_t>, NumATensor>& a_ms_ks_lengths,
                 const std::array<std::vector<index_t>, NumATensor>& a_ms_ks_strides,
                 const std::array<std::vector<index_t>, NumBTensor>& b_ns_ks_lengths,
                 const std::array<std::vector<index_t>, NumBTensor>& b_ns_ks_strides,
                 const std::array<std::vector<index_t>, NumDTensor>& d_ms_ns_lengths,
                 const std::array<std::vector<index_t>, NumDTensor>& d_ms_ns_strides,
                 const std::vector<index_t>& e_ms_ns_length,
                 const std::vector<index_t>& e_ms_ns_stride,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation cde_element_op)
            : p_as_grid_{},
              p_bs_grid_{},
              p_ds_grid_{},
              p_e_grid_{static_cast<EDataType*>(p_e_grid)},
              as_grid_desc_m_k_{},
              bs_grid_desc_n_k_{},
              ds_grid_desc_m_n_{},
              e_grid_desc_m_n_{MakeEGridDescriptor_M_N(e_ms_ns_length, e_ms_ns_stride)},
              as_grid_desc_ak0_m_ak1_{},
              bs_grid_desc_bk0_n_bk1_{},
              ds_grid_desc_mblock_mperblock_nblock_nperblock_{},
              e_grid_desc_mblock_mperblock_nblock_nperblock_{},
              block_2_etile_map_{GridwiseGemm::MakeDefaultBlock2ETileMap(e_grid_desc_m_n_)},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op}
        {
            // populate pointer, desc for As
            static_for<0, NumATensor, 1>{}([&](auto i) {
                // using ALayout   = remove_cvref_t<tuple_element_t<i.value, AsLayout>>;
                using ADataType = remove_cvref_t<tuple_element_t<i.value, AsDataType>>;

                // A pointer
                p_as_grid_(i) = static_cast<const ADataType*>(p_as_grid[i]);

                // A desc
                as_grid_desc_m_k_(i) =
                    MakeAGridDescriptor_M_K(a_ms_ks_lengths[i], a_ms_ks_strides[i]);
            });

            // populate pointer, desc for Bs
            static_for<0, NumBTensor, 1>{}([&](auto i) {
                // using BLayout   = remove_cvref_t<tuple_element_t<i.value, BsLayout>>;
                using BDataType = remove_cvref_t<tuple_element_t<i.value, BsDataType>>;

                // B pointer
                p_bs_grid_(i) = static_cast<const BDataType*>(p_bs_grid[i]);

                // B desc
                bs_grid_desc_n_k_(i) =
                    MakeBGridDescriptor_N_K(b_ns_ks_lengths[i], b_ns_ks_strides[i]);
            });

            // populate pointer, desc for Ds
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                // using DLayout   = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                // D pointer
                p_ds_grid_(i) = static_cast<const DDataType*>(p_ds_grid[i]);

                // D desc
                ds_grid_desc_m_n_(i) =
                    MakeEGridDescriptor_M_N(d_ms_ns_lengths[i], d_ms_ns_strides[i]);
            });

            // populate desc for Ds/E
            if(GridwiseGemm::CheckValidity(as_grid_desc_m_k_,
                                           bs_grid_desc_n_k_,
                                           ds_grid_desc_m_n_,
                                           e_grid_desc_m_n_,
                                           block_2_etile_map_))
            {
                as_grid_desc_ak0_m_ak1_ =
                    GridwiseGemm::MakeDefaultAsGridDescriptor_AK0_M_AK1(as_grid_desc_m_k_);

                bs_grid_desc_bk0_n_bk1_ =
                    GridwiseGemm::MakeDefaultBsGridDescriptor_BK0_N_BK1(bs_grid_desc_n_k_);

                ds_grid_desc_mblock_mperblock_nblock_nperblock_ =
                    GridwiseGemm::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        ds_grid_desc_m_n_);

                e_grid_desc_mblock_mperblock_nblock_nperblock_ =
                    GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        e_grid_desc_m_n_);
            }

            // for sanity check of vector memory access
            for(index_t i = 0; i < NumATensor; ++i)
            {
                a_mz_stride_[i] = a_ms_ks_strides[i][NumDimM - 1];
                a_kz_stride_[i] = a_ms_ks_strides[i][NumDimM + NumDimK - 1];
            }

            for(index_t i = 0; i < NumBTensor; ++i)
            {
                b_nz_stride_[i] = b_ns_ks_strides[i][NumDimN - 1];
                b_kz_stride_[i] = b_ns_ks_strides[i][NumDimN + NumDimK - 1];
            }

            for(index_t i = 0; i < NumDTensor; ++i)
            {
                ds_nz_stride_[i] = d_ms_ns_strides[i][NumDimM + NumDimN - 1];
            }

            e_nz_stride_ = e_ms_ns_stride[NumDimM + NumDimN - 1];
        }

        // pointers
        typename GridwiseGemm::AsGridPointer p_as_grid_;
        typename GridwiseGemm::BsGridPointer p_bs_grid_;
        typename GridwiseGemm::DsGridPointer p_ds_grid_;
        EDataType* p_e_grid_;

        // tensor descriptors for problem definiton
        AsGridDesc_M_K as_grid_desc_m_k_;
        BsGridDesc_N_K bs_grid_desc_n_k_;
        DsGridDesc_M_N ds_grid_desc_m_n_;
        EGridDesc_M_N e_grid_desc_m_n_;

        // tensor descriptors for block/thread-wise copy
        AsGridDesc_AK0_M_AK1 as_grid_desc_ak0_m_ak1_;
        BsGridDesc_BK0_N_BK1 bs_grid_desc_bk0_n_bk1_;
        DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
            ds_grid_desc_mblock_mperblock_nblock_nperblock_;
        EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock e_grid_desc_mblock_mperblock_nblock_nperblock_;

        // block-to-e-tile map
        Block2ETileMap block_2_etile_map_;

        // element-wise op
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;

        // Strides for the last M/N/K dimensions of A/B/Ds/E
        //   for sanity check of vector load/store
        std::array<index_t, NumATensor> a_mz_stride_;
        std::array<index_t, NumATensor> a_kz_stride_;

        std::array<index_t, NumBTensor> b_nz_stride_;
        std::array<index_t, NumBTensor> b_kz_stride_;

        std::array<index_t, NumDTensor> ds_nz_stride_;
        index_t e_nz_stride_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(!GridwiseGemm::CheckValidity(arg.as_grid_desc_m_k_,
                                            arg.bs_grid_desc_n_k_,
                                            arg.ds_grid_desc_m_n_,
                                            arg.e_grid_desc_m_n_,
                                            arg.block_2_etile_map_))
            {
                throw std::runtime_error("wrong! GridwiseGemm has invalid setting");
            }

            const index_t grid_size =
                arg.block_2_etile_map_.CalculateGridSize(arg.e_grid_desc_m_n_);

            auto launch_kernel = [&](auto has_main_k_block_loop) {
                constexpr bool has_main_loop = has_main_k_block_loop.value;

                const auto kernel = kernel_contraction_multiple_abd_xdl_cshuffle<
                    GridwiseGemm,
                    typename GridwiseGemm::AsGridPointer,
                    typename GridwiseGemm::BsGridPointer,
                    typename GridwiseGemm::DsGridPointer,
                    EDataType,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CDEElementwiseOperation,
                    DeviceOp::AsGridDesc_AK0_M_AK1,
                    DeviceOp::BsGridDesc_BK0_N_BK1,
                    DeviceOp::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                    DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                    DeviceOp::Block2ETileMap,
                    has_main_loop>;

                return launch_and_time_kernel(stream_config,
                                              kernel,
                                              dim3(grid_size),
                                              dim3(BlockSize),
                                              0,
                                              arg.p_as_grid_,
                                              arg.p_bs_grid_,
                                              arg.p_ds_grid_,
                                              arg.p_e_grid_,
                                              arg.a_element_op_,
                                              arg.b_element_op_,
                                              arg.cde_element_op_,
                                              arg.as_grid_desc_ak0_m_ak1_,
                                              arg.bs_grid_desc_bk0_n_bk1_,
                                              arg.ds_grid_desc_mblock_mperblock_nblock_nperblock_,
                                              arg.e_grid_desc_mblock_mperblock_nblock_nperblock_,
                                              arg.block_2_etile_map_);
            };

            const auto K = arg.as_grid_desc_m_k_[I0].GetLength(I1);

            if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
            {
                return launch_kernel(integral_constant<bool, true>{});
            }
            else
            {
                return launch_kernel(integral_constant<bool, false>{});
            }
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(!ck::is_xdl_supported())
        {
            return false;
        }

        // check vector load/store
        {
            bool all_valid = true;

            static_for<0, NumATensor, 1>{}([&](auto i) {
                // vector memory access of A: could be on M or AK1 dimension
                if constexpr(ABlockTransferSrcVectorDim == 1)
                {
                    if(!(arg.a_mz_stride_[i] == 1 && arg.as_grid_desc_ak0_m_ak1_[i].GetLength(I1) %
                                                             ABlockTransferSrcScalarPerVector ==
                                                         0))
                    {
                        all_valid = false;
                    }
                }
                else
                {
                    if(!(arg.a_kz_stride_[i] == 1 && arg.as_grid_desc_ak0_m_ak1_[i].GetLength(I2) %
                                                             ABlockTransferSrcScalarPerVector ==
                                                         0))
                    {
                        all_valid = false;
                    }
                }
            });

            // vector memory access of B: could be on N or BK1 dimension
            static_for<0, NumBTensor, 1>{}([&](auto i) {
                if constexpr(BBlockTransferSrcVectorDim == 1)
                {
                    if(!(arg.b_nz_stride_[i] == 1 && arg.bs_grid_desc_bk0_n_bk1_[i].GetLength(I1) %
                                                             BBlockTransferSrcScalarPerVector ==
                                                         0))
                    {
                        all_valid = false;
                    }
                }
                else
                {
                    if(!(arg.b_kz_stride_[i] == 1 && arg.bs_grid_desc_bk0_n_bk1_[i].GetLength(I2) %
                                                             BBlockTransferSrcScalarPerVector ==
                                                         0))
                    {
                        all_valid = false;
                    }
                }
            });

            // check vector load of Ds
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                if(!(arg.ds_nz_stride_[i] == 1 &&
                     arg.ds_grid_desc_mblock_mperblock_nblock_nperblock_[i].GetLength(I3) %
                             CDEBlockTransferScalarPerVector_NPerBlock ==
                         0))
                {
                    all_valid = false;
                }
            });

            // vector memory access of E: always on NPerBlock dimension
            if(!(arg.e_nz_stride_ == 1 &&
                 arg.e_grid_desc_mblock_mperblock_nblock_nperblock_.GetLength(I3) %
                         CDEBlockTransferScalarPerVector_NPerBlock ==
                     0))
            {
                all_valid = false;
            }

            if(!all_valid)
            {
                return false;
            }
        }

        return GridwiseGemm::CheckValidity(arg.as_grid_desc_m_k_,
                                           arg.bs_grid_desc_n_k_,
                                           arg.ds_grid_desc_m_n_,
                                           arg.e_grid_desc_m_n_,
                                           arg.block_2_etile_map_);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(std::array<const void*, NumATensor> p_as,
                             std::array<const void*, NumBTensor> p_bs,
                             std::array<const void*, NumDTensor> p_ds,
                             void* p_e,
                             const std::array<std::vector<index_t>, NumATensor>& a_ms_ks_lengths,
                             const std::array<std::vector<index_t>, NumATensor>& a_ms_ks_strides,
                             const std::array<std::vector<index_t>, NumBTensor>& b_ns_ks_lengths,
                             const std::array<std::vector<index_t>, NumBTensor>& b_ns_ks_strides,
                             const std::array<std::vector<index_t>, NumDTensor>& d_ms_ns_lengths,
                             const std::array<std::vector<index_t>, NumDTensor>& d_ms_ns_strides,
                             const std::vector<index_t>& e_ms_ns_length,
                             const std::vector<index_t>& e_ms_ns_stride,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CDEElementwiseOperation cde_element_op)
    {
        return Argument{p_as,
                        p_bs,
                        p_ds,
                        p_e,
                        a_ms_ks_lengths,
                        a_ms_ks_strides,
                        b_ns_ks_lengths,
                        b_ns_ks_strides,
                        d_ms_ns_lengths,
                        d_ms_ns_strides,
                        e_ms_ns_length,
                        e_ms_ns_stride,
                        a_element_op,
                        b_element_op,
                        cde_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::array<const void*, NumATensor> p_as,
                        std::array<const void*, NumBTensor> p_bs,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_e,
                        const std::array<std::vector<index_t>, NumATensor>& as_ms_ks_lengths,
                        const std::array<std::vector<index_t>, NumATensor>& as_ms_ks_strides,
                        const std::array<std::vector<index_t>, NumBTensor>& bs_ns_ks_lengths,
                        const std::array<std::vector<index_t>, NumBTensor>& bs_ns_ks_strides,
                        const std::array<std::vector<index_t>, NumDTensor>& ds_ms_ns_lengths,
                        const std::array<std::vector<index_t>, NumDTensor>& ds_ms_ns_strides,
                        const std::vector<index_t>& e_ms_ns_length,
                        const std::vector<index_t>& e_ms_ns_stride,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) override
    {
        return std::make_unique<Argument>(p_as,
                                          p_bs,
                                          p_ds,
                                          p_e,
                                          as_ms_ks_lengths,
                                          as_ms_ks_strides,
                                          bs_ns_ks_lengths,
                                          bs_ns_ks_strides,
                                          ds_ms_ns_lengths,
                                          ds_ms_ns_strides,
                                          e_ms_ns_length,
                                          e_ms_ns_stride,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        std::map<LoopScheduler, std::string> LoopSchedToString{
            {LoopScheduler::Default, "Default"}, {LoopScheduler::Interwave, "Interwave"}};

        std::map<PipelineVersion, std::string> PipelineVersionToString{{PipelineVersion::v1, "v1"},
                                                                       {PipelineVersion::v2, "v2"}};

        // clang-format off
        str << "DeviceContractionMultipleABD_Xdl_CShuffle"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1 << ", "
            << MPerXDL << ", "
            << NPerXDL << ", "
            << MXdlPerWave << ", "
            << NXdlPerWave << ", "
            << ABlockTransferSrcScalarPerVector << ", "
            << BBlockTransferSrcScalarPerVector << ", "
            << CShuffleMXdlPerWavePerShuffle << ", "
            << CShuffleNXdlPerWavePerShuffle << ", "
            << getGemmSpecializationString(GemmSpec)
            << ">"
            << " LoopScheduler: "
            << LoopSchedToString[LoopSched] << ", "
            << "PipelineVersion: "
            << PipelineVersionToString[PipelineVer];
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
