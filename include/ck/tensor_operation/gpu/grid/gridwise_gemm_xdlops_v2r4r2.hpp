// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/env.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_common.hpp"

#if __clang_major__ >= 23
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
#endif
namespace ck {

template <typename GridwiseGemm,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          typename Block2CTileMap,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(GridwiseGemm::MaxBlockSize, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_gemm_xdlops_v2r4r2_simplified(typename GridwiseGemm::Argument karg,
                                         const Block2CTileMap& b2c_map,
                                         const AElementwiseOperation a_element_op,
                                         const BElementwiseOperation b_element_op,
                                         const CElementwiseOperation c_element_op)
{
#if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx94__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<CGlobalMemoryDataOperation>())
    {
        constexpr index_t shared_size =
            GridwiseGemm::GetSharedMemoryNumberOfByte(get_device_arch());

        __shared__ uint8_t p_shared[shared_size];

        GridwiseGemm::template Run<HasMainKBlockLoop, CGlobalMemoryDataOperation>(
            karg, static_cast<void*>(p_shared), b2c_map, a_element_op, b_element_op, c_element_op);
    }
#else
    ignore = karg;
    ignore = b2c_map;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

template <index_t BlockSize,
          typename FloatA,
          typename FloatB,
          typename FloatAcc,
          typename FloatC,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          tensor_operation::device::GemmSpecialization GemmSpec,
          index_t NumGemmKPrefetchStage,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t K0PerBlock,
          index_t MPerXdl,
          index_t NPerXdl,
          index_t K1Value,
          index_t MRepeat,
          index_t NRepeat,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_K1,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_K1,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          bool BBlockLdsExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          index_t CBlockTransferScalarPerVector_NWaveNPerXDL,
          typename CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          LoopScheduler LoopSched     = make_default_loop_scheduler(),
          PipelineVersion PipelineVer = PipelineVersion::v1,
          typename ComputeTypeA       = FloatC,
          typename ComputeTypeB       = ComputeTypeA,
          typename LDSTypeA           = ComputeTypeA,
          typename LDSTypeB           = ComputeTypeB>
struct GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2
    : public GridwiseGemm_xdl_cshuffle_base<
          tensor_layout::gemm::RowMajor,
          tensor_layout::gemm::ColumnMajor,
          tensor_layout::gemm::RowMajor,
          LDSTypeA,
          LDSTypeB,
          FloatAcc,
          FloatC,
          Tuple<>,
          FloatC,
          AElementwiseOperation,
          BElementwiseOperation,
          BlockSize,
          MPerBlock,
          NPerBlock,
          K0PerBlock * K1Value,
          K1Value,
          K1Value,
          MPerXdl,
          NPerXdl,
          MRepeat,
          NRepeat,
          ABlockTransferThreadClusterLengths_K0_M_K1,
          ABlockTransferThreadClusterArrangeOrder,
          ABlockTransferSrcAccessOrder,
          ABlockTransferSrcVectorDim,
          ABlockTransferSrcScalarPerVector,
          ABlockTransferDstScalarPerVector_K1,
          AThreadTransferSrcResetCoordinateAfterRun,
          ABlockLdsExtraM,
          BBlockTransferThreadClusterLengths_K0_N_K1,
          BBlockTransferThreadClusterArrangeOrder,
          BBlockTransferSrcAccessOrder,
          BBlockTransferSrcVectorDim,
          BBlockTransferSrcScalarPerVector,
          BBlockTransferDstScalarPerVector_K1,
          BThreadTransferSrcResetCoordinateAfterRun,
          BBlockLdsExtraN,
          CShuffleMRepeatPerShuffle,
          CShuffleNRepeatPerShuffle,
          CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          Sequence<CBlockTransferScalarPerVector_NWaveNPerXDL>,
          ComputeTypeA,
          ComputeTypeB,
          true> // ForceNaiveLayout
{
    using Base = GridwiseGemm_xdl_cshuffle_base<
        tensor_layout::gemm::RowMajor,
        tensor_layout::gemm::ColumnMajor,
        tensor_layout::gemm::RowMajor,
        LDSTypeA,
        LDSTypeB,
        FloatAcc,
        FloatC,
        Tuple<>,
        FloatC,
        AElementwiseOperation,
        BElementwiseOperation,
        BlockSize,
        MPerBlock,
        NPerBlock,
        K0PerBlock * K1Value,
        K1Value,
        K1Value,
        MPerXdl,
        NPerXdl,
        MRepeat,
        NRepeat,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsExtraN,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        Sequence<CBlockTransferScalarPerVector_NWaveNPerXDL>,
        ComputeTypeA,
        ComputeTypeB,
        true>; // ForceNaiveLayout
    using Base::I0;
    using Base::I1;
    using Base::I2;
    using Base::I3;
    using ThisThreadBlock = typename Base::ThisThreadBlock;
    using Base::GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1;
    using Base::GetABlockDescriptor_AKB_AK0PerBlock_MPerBlock_AK1;
    using Base::GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1;
    using Base::GetBBlockDescriptor_BKB_BK0PerBlock_NPerBlock_BK1;

    // K1 should be Number<...>
    static constexpr auto K1  = Number<K1Value>{};
    static constexpr auto M01 = 1;
    static constexpr auto N01 = 1;

    static constexpr auto gemm_padder =
        tensor_operation::device::GemmPadder<GemmSpec, index_t, index_t, index_t>{
            MPerBlock, NPerBlock, K1* K0PerBlock};

    using GridwiseGemmPipe = remove_cvref_t<
        decltype(GridwiseGemmPipeline_Selector<PipelineVer, NumGemmKPrefetchStage, LoopSched>())>;

    struct Argument : public ck::tensor_operation::device::BaseArgument
    {
        const FloatA* p_a_grid;
        const FloatB* p_b_grid;
        FloatC* p_c_grid;
        index_t M;
        index_t N;
        index_t K;
        index_t StrideA;
        index_t StrideB;
        index_t StrideC;
        index_t MPadded;
        index_t NPadded;
        index_t KPadded;
        index_t K0Padded;
        index_t k_batch;

        Argument(const FloatA* p_a_grid_,
                 const FloatB* p_b_grid_,
                 FloatC* p_c_grid_,
                 index_t M_,
                 index_t N_,
                 index_t K_,
                 index_t StrideA_,
                 index_t StrideB_,
                 index_t StrideC_,
                 index_t MPadded_,
                 index_t NPadded_,
                 index_t KPadded_,
                 index_t K0Padded_,
                 index_t k_batch_)
            : p_a_grid(p_a_grid_),
              p_b_grid(p_b_grid_),
              p_c_grid(p_c_grid_),
              M(M_),
              N(N_),
              K(K_),
              StrideA(StrideA_),
              StrideB(StrideB_),
              StrideC(StrideC_),
              MPadded(MPadded_),
              NPadded(NPadded_),
              KPadded(KPadded_),
              K0Padded(K0Padded_),
              k_batch(k_batch_)
        {
        }

        void Print() const
        {
            std::cout << "arg {" << "M:" << M << ", " << "N:" << N << ", " << "K:" << K << ", "
                      << "SA:" << StrideA << ", " << "SB:" << StrideB << ", " << "SC:" << StrideC
                      << ", " << "MP:" << MPadded << ", " << "NP:" << NPadded << ", "
                      << "KP:" << KPadded << ", " << "K0Padded:" << K0Padded << ", "
                      << "KB:" << k_batch << "}" << std::endl;
        }
    };

    __host__ __device__ static auto CalculateGridSize(const Argument& karg)
    {
        return std::make_tuple(math::integer_divide_ceil(karg.N, NPerBlock),
                               math::integer_divide_ceil(karg.M, MPerBlock),
                               karg.k_batch);
    }

    // prefer this to be called on host
    __host__ __device__ static auto CalculateMPadded(index_t M)
    {
        return math::integer_least_multiple(M, MPerBlock);
    }

    __host__ __device__ static auto CalculateNPadded(index_t N)
    {
        return math::integer_least_multiple(N, NPerBlock);
    }

    __host__ __device__ static auto CalculateK0Padded(index_t K, index_t K_Batch = 1)
    {
        // k_batch * k0 * k0_per_block * k1
        auto K_t = K_Batch * K0PerBlock * K1;
        return (K + K_t - 1) / K_t * K0PerBlock;
    }

    __host__ __device__ static auto CalculateKPadded(index_t K, index_t K_Batch = 1)
    {
        auto K0Padded = CalculateK0Padded(K, K_Batch);
        return K_Batch * K0Padded * K1;
    }

    __host__ __device__ static auto MakeAGridDescriptor_KBatch_K0_M_K1(index_t M,
                                                                       index_t MPad,
                                                                       index_t K,
                                                                       index_t StrideA,
                                                                       index_t KBatch,
                                                                       index_t K0Padded,
                                                                       index_t KPad)
    {
        const auto a_grid_desc_m_k = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(StrideA, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(I1, StrideA));
            }
        }();

        if constexpr(GemmSpec == tensor_operation::device::GemmSpecialization::MPadding ||
                     GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                     GemmSpec == tensor_operation::device::GemmSpecialization::MKPadding ||
                     GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding)
        {

            const auto a_grid_desc_m_kpad = transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_pass_through_transform(M), make_right_pad_transform(K, KPad - K)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            // const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;
            return transform_tensor_descriptor(
                a_grid_desc_m_kpad,
                make_tuple(make_unmerge_transform(make_tuple(KBatch, K0Padded, K1)),
                           make_right_pad_transform(M, MPad - M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));
        }
        else if constexpr(GemmSpec == tensor_operation::device::GemmSpecialization::MPadding ||
                          GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding)
        {
            // const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;
            return transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(KBatch, K0Padded, K1)),
                           make_right_pad_transform(M, MPad - M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));
        }
        else if constexpr(GemmSpec == tensor_operation::device::GemmSpecialization::KPadding)
        {
            const auto a_grid_desc_m_kpad = transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_pass_through_transform(M), make_right_pad_transform(K, KPad - K)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return transform_tensor_descriptor(
                a_grid_desc_m_kpad,
                make_tuple(make_unmerge_transform(make_tuple(KBatch, K0Padded, K1)),
                           make_pass_through_transform(M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(KBatch, K0Padded, K1)),
                           make_pass_through_transform(M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));
        }
    }

    __host__ __device__ static auto MakeBGridDescriptor_KBatch_K0_N_K1(index_t K,
                                                                       index_t NPad,
                                                                       index_t N,
                                                                       index_t StrideB,
                                                                       index_t KBatch,
                                                                       index_t K0Padded,
                                                                       index_t KPad)
    {
        const auto b_grid_desc_k_n = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(StrideB, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(I1, StrideB));
            }
        }();

        if constexpr(GemmSpec == tensor_operation::device::GemmSpecialization::NPadding ||
                     GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                     GemmSpec == tensor_operation::device::GemmSpecialization::NKPadding ||
                     GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding)
        {

            const auto b_grid_desc_kpad_n = transform_tensor_descriptor(
                b_grid_desc_k_n,
                make_tuple(make_right_pad_transform(K, KPad - K), make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            // const auto PadN = (NPerBlock - N % NPerBlock) % NPerBlock;
            return transform_tensor_descriptor(
                b_grid_desc_kpad_n,
                make_tuple(make_unmerge_transform(make_tuple(KBatch, K0Padded, K1)),
                           make_right_pad_transform(N, NPad - N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));
        }
        else if constexpr(GemmSpec == tensor_operation::device::GemmSpecialization::NPadding ||
                          GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding)
        {
            // const auto PadN = (NPerBlock - N % NPerBlock) % NPerBlock;
            return transform_tensor_descriptor(
                b_grid_desc_k_n,
                make_tuple(make_unmerge_transform(make_tuple(KBatch, K0Padded, K1)),
                           make_right_pad_transform(N, NPad - N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));
        }
        else if constexpr(GemmSpec == tensor_operation::device::GemmSpecialization::KPadding)
        {
            const auto b_grid_desc_kpad_n = transform_tensor_descriptor(
                b_grid_desc_k_n,
                make_tuple(make_right_pad_transform(K, KPad - K), make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return transform_tensor_descriptor(
                b_grid_desc_kpad_n,
                make_tuple(make_unmerge_transform(make_tuple(KBatch, K0Padded, K1)),
                           make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                b_grid_desc_k_n,
                make_tuple(make_unmerge_transform(make_tuple(KBatch, K0Padded, K1)),
                           make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));
        }
    }

    __host__ __device__ static auto MakeCGridDescriptor_M_N(index_t M, index_t N, index_t StrideC)
    {
        const auto c_grid_desc_m_n = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(StrideC, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(I1, StrideC));
            }
        }();

        return gemm_padder.PadCDescriptor_M_N(c_grid_desc_m_n);
    }

    template <
        InMemoryDataOperationEnum CGlobalMemoryDataOperation_ = InMemoryDataOperationEnum::Set>
    __device__ static bool constexpr IsValidCompilationParameter()
    {
        constexpr bool valid = tensor_operation::device::IsValidGemmCompilationParameter<
            BlockSize,
            MPerBlock,
            NPerBlock,
            MPerXdl,
            NPerXdl,
            MRepeat,
            NRepeat,
            FloatC,
            CGlobalMemoryDataOperation_>();
        if constexpr(!valid)
        {
            return false;
        }

        if constexpr(K1Value %
                         MfmaSelector<ComputeTypeA, MPerXdl, NPerXdl, ComputeTypeB>::selected_mfma
                             .k_per_blk !=
                     0)
        {
            return false;
        }
        return true;
    }

    __host__ static bool CheckValidity(const Argument& karg)
    {
        if(!is_xdl_wmma_k_supported<ComputeTypeA, K1Value * K0PerBlock, K1Value>())
        {
            return false;
        }

        if constexpr(!(GemmSpec == tensor_operation::device::GemmSpecialization::MPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding))
        {
            if(!(karg.M % MPerBlock == 0))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg M value is not a multiple of MPerBlock! M: " << karg.M << " "
                              << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                              << std::endl;
                }
                return false;
            }
        }

        if constexpr(!(GemmSpec == tensor_operation::device::GemmSpecialization::NPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::NKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding))
        {
            if(!(karg.N % NPerBlock == 0))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg N value is not a multiple of NPerBlock! N: " << karg.N << " "
                              << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                              << std::endl;
                }
                return false;
            }
        }

        if constexpr(!(GemmSpec == tensor_operation::device::GemmSpecialization::KPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::NKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding))
        {

            auto K_t = karg.k_batch * K0PerBlock * K1;
            if(!(karg.K % K_t == 0))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg K value is not a multiple of K_Batch * K0PerBlock * K1! K: "
                              << karg.K << " " << __FILE__ << ":" << __LINE__
                              << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
        {
            if(karg.K % ABlockTransferSrcScalarPerVector != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg K (" << karg.K
                              << ") value is not a multiple of ABlockTransferSrcScalarPerVector ("
                              << ABlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }
        else
        {
            if(karg.M % ABlockTransferSrcScalarPerVector != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg M (" << karg.M
                              << ") value is not a multiple of ABlockTransferSrcScalarPerVector ("
                              << ABlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
        {
            if(karg.N % BBlockTransferSrcScalarPerVector != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg N (" << karg.N
                              << ") value is not a multiple of BBlockTransferSrcScalarPerVector ("
                              << BBlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }
        else
        {
            if(karg.K % BBlockTransferSrcScalarPerVector != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg K (" << karg.K
                              << ") value is not a multiple of BBlockTransferSrcScalarPerVector ("
                              << BBlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
        {
            if(karg.N % CBlockTransferScalarPerVector_NWaveNPerXDL != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg N (" << karg.N
                              << ") value is not a multiple of "
                                 "CBlockTransferScalarPerVector_NWaveNPerXDL ("
                              << CBlockTransferScalarPerVector_NWaveNPerXDL << " )! " << __FILE__
                              << ":" << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }
        else
        {
            if(karg.M % CBlockTransferScalarPerVector_NWaveNPerXDL != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg M (" << karg.M
                              << ") value is not a multiple of "
                                 "CBlockTransferScalarPerVector_NWaveNPerXDL ("
                              << CBlockTransferScalarPerVector_NWaveNPerXDL << " )! " << __FILE__
                              << ":" << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }

        const auto num_k_loop = karg.K0Padded / K0PerBlock;
        if(!GridwiseGemmPipe::IsSupported(num_k_loop))
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "The number of k loops (" << num_k_loop
                          << ") value is not supported by GridwiseGemm Pipeline."
                          << " K0Padded: " << karg.K0Padded << ", K0PerBlock: " << K0PerBlock << " "
                          << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                          << std::endl;
            }
            return false;
        }

        return true;
    }

    __host__ __device__ static auto GetKPad(index_t K, index_t KBatch)
    {
        const index_t K0Padded =
            math::integer_divide_ceil(K, K1 * K0PerBlock * KBatch) * K0PerBlock;
        const index_t KPad = KBatch * K0Padded * K1;
        return KPad;
    }

    __host__ __device__ static constexpr bool CalculateHasMainK0BlockLoop(index_t K0Padded)
    {
        const index_t num_loop = K0Padded / K0PerBlock;
        return GridwiseGemmPipe::CalculateHasMainLoop(num_loop);
    }

    template <typename CGridDesc>
    __host__ __device__ static constexpr auto
    MakeCGridDesc_MBlock_MPerBlock_NBlock_NPerBlock(const CGridDesc& c_m_n_grid_desc)
    {
        const auto M = c_m_n_grid_desc.GetLength(I0);
        const auto N = c_m_n_grid_desc.GetLength(I1);

        const auto MBlock = M / MPerBlock;
        const auto NBlock = N / NPerBlock;

        return transform_tensor_descriptor(
            c_m_n_grid_desc,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                       make_unmerge_transform(make_tuple(NBlock, Number<NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));
    }

    // return block_id to C matrix tile idx (m0, n0) mapping
    template <typename CGridDesc>
    __host__ __device__ static constexpr auto MakeCBlockClusterAdaptor(
        const CGridDesc& c_m_n_grid_desc, index_t /* M01 */, index_t /* N01 */, index_t KBatch)
    {
        return BlockToCTileMap_KSplit_M00_N0_M01Adapt<MPerBlock, NPerBlock, CGridDesc>(
            c_m_n_grid_desc, 8, KBatch);
    }

    __host__ __device__ static constexpr auto
    GetCBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock()
    {
        constexpr index_t MWave = MPerBlock / (MRepeat * MPerXdl);
        constexpr index_t NWave = NPerBlock / (NRepeat * NPerXdl);

        return make_naive_tensor_descriptor_packed(
            make_tuple(I1,
                       Number<CShuffleMRepeatPerShuffle * MWave * MPerXdl>{},
                       I1,
                       Number<CShuffleNRepeatPerShuffle * NWave * NPerXdl>{}));
    }

    // return block_id to C matrix tile idx (m0, n0, k_split) mapping
    __host__ __device__ static constexpr auto MakeDefaultBlock2CTileMap()
    {
        return BlockToCTileMap_3DGrid_KSplit<MPerBlock, NPerBlock>();
    }

    using CGridDesc_M_N         = remove_cvref_t<decltype(MakeCGridDescriptor_M_N(1, 1, 1))>;
    using DefaultBlock2CTileMap = remove_cvref_t<decltype(MakeDefaultBlock2CTileMap())>;

    template <bool HasMainKBlockLoop,
              InMemoryDataOperationEnum CGlobalMemoryDataOperation,
              typename Block2CTileMap>
    __device__ static void Run(const Argument& karg,
                               void* __restrict__ p_shared_block,
                               const Block2CTileMap& block_2_ctile_map,
                               const AElementwiseOperation a_element_op = AElementwiseOperation{},
                               const BElementwiseOperation b_element_op = BElementwiseOperation{},
                               const CElementwiseOperation c_element_op = CElementwiseOperation{})
    {
        const FloatA* p_a_grid           = karg.p_a_grid;
        const FloatB* p_b_grid           = karg.p_b_grid;
        FloatC* p_c_grid                 = karg.p_c_grid;
        const auto a_b_k0_m_k1_grid_desc = MakeAGridDescriptor_KBatch_K0_M_K1(
            karg.M, karg.MPadded, karg.K, karg.StrideA, karg.k_batch, karg.K0Padded, karg.KPadded);
        const auto b_b_k0_n_k1_grid_desc = MakeBGridDescriptor_KBatch_K0_N_K1(
            karg.K, karg.NPadded, karg.N, karg.StrideB, karg.k_batch, karg.K0Padded, karg.KPadded);
        const auto c_grid_desc_m_n = MakeCGridDescriptor_M_N(karg.M, karg.N, karg.StrideC);

        const auto c_grid_desc_mblock_mperblock_nblock_nperblock =
            MakeCGridDesc_MBlock_MPerBlock_NBlock_NPerBlock(c_grid_desc_m_n);

        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_b_k0_m_k1_grid_desc.GetElementSpaceSize());
        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_b_k0_n_k1_grid_desc.GetElementSpaceSize());

        // divide block work by [KBatch, M, N]
        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        if(!block_2_ctile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0),
                          c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2))))
        {
            return;
        }

        const index_t block_m_id = __builtin_amdgcn_readfirstlane(block_work_idx[I1]);
        const index_t block_n_id = __builtin_amdgcn_readfirstlane(block_work_idx[I2]);
        const index_t k_batch_id = __builtin_amdgcn_readfirstlane(block_work_idx[I0]);

        // HACK: this force m/n_block_data_idx_on_grid into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_m_id * MPerBlock);

        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_n_id * NPerBlock);

        // lds max alignment
        constexpr auto max_lds_align = K1;

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_k0_m_k1_block_desc =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(get_device_arch());
        constexpr auto a_b_k0_m_k1_block_desc =
            GetABlockDescriptor_AKB_AK0PerBlock_MPerBlock_AK1(a_k0_m_k1_block_desc);
        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_k0_n_k1_block_desc =
            GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1(get_device_arch());
        constexpr auto b_b_k0_n_k1_block_desc =
            GetBBlockDescriptor_BKB_BK0PerBlock_NPerBlock_BK1(b_k0_n_k1_block_desc);
        // A matrix blockwise copy
        auto a_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                AElementwiseOperation,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<1, K0PerBlock, MPerBlock, K1>,
                                                ABlockTransferThreadClusterLengths_K0_M_K1,
                                                ABlockTransferThreadClusterArrangeOrder,
                                                FloatA,
                                                LDSTypeA,
                                                decltype(a_b_k0_m_k1_grid_desc),
                                                decltype(a_b_k0_m_k1_block_desc),
                                                ABlockTransferSrcAccessOrder,
                                                Sequence<0, 2, 1, 3>,
                                                ABlockTransferSrcVectorDim,
                                                3,
                                                ABlockTransferSrcScalarPerVector,
                                                ABlockTransferDstScalarPerVector_K1,
                                                1,
                                                1,
                                                AThreadTransferSrcResetCoordinateAfterRun,
                                                true>(
                a_b_k0_m_k1_grid_desc,
                make_multi_index(k_batch_id, 0, m_block_data_idx_on_grid, 0),
                a_element_op,
                a_b_k0_m_k1_block_desc,
                make_multi_index(0, 0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

        // B matrix blockwise copy
        auto b_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                BElementwiseOperation,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<1, K0PerBlock, NPerBlock, K1>,
                                                BBlockTransferThreadClusterLengths_K0_N_K1,
                                                BBlockTransferThreadClusterArrangeOrder,
                                                FloatB,
                                                LDSTypeB,
                                                decltype(b_b_k0_n_k1_grid_desc),
                                                decltype(b_b_k0_n_k1_block_desc),
                                                BBlockTransferSrcAccessOrder,
                                                Sequence<0, 2, 1, 3>,
                                                BBlockTransferSrcVectorDim,
                                                3,
                                                BBlockTransferSrcScalarPerVector,
                                                BBlockTransferDstScalarPerVector_K1,
                                                1,
                                                1,
                                                BThreadTransferSrcResetCoordinateAfterRun,
                                                true>(
                b_b_k0_n_k1_grid_desc,
                make_multi_index(k_batch_id, 0, n_block_data_idx_on_grid, 0),
                b_element_op,
                b_b_k0_n_k1_block_desc,
                make_multi_index(0, 0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

        // GEMM definition
        //   c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[K0PerBlock, MPerBlock] is in LDS
        //     b_mtx[K0PerBlock, NPerBlock] is in LDS
        //     c_mtx[MPerBlock, NPerBlock] is distributed among threads, and saved in
        //       register
        // sanity check

        auto blockwise_gemm = BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_Selector<
            BlockSize,
            LDSTypeA,
            LDSTypeB,
            FloatAcc,
            decltype(a_k0_m_k1_block_desc),
            decltype(b_k0_n_k1_block_desc),
            MPerXdl,
            NPerXdl,
            MRepeat,
            NRepeat,
            K1,
            LoopSched,
            ComputeTypeA,
            ComputeTypeB>();

        auto c_thread_buf = blockwise_gemm.GetCThreadBuffer();

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size =
            math::integer_least_multiple(a_k0_m_k1_block_desc.GetElementSpaceSize(), max_lds_align);

        auto p_a_block = reinterpret_cast<LDSTypeA*>(p_shared_block);
        auto p_b_block = reinterpret_cast<LDSTypeB*>(p_a_block + a_block_space_size);

        constexpr auto a_block_slice_copy_step = make_multi_index(0, K0PerBlock, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(0, K0PerBlock, 0, 0);

        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            p_a_block, a_k0_m_k1_block_desc.GetElementSpaceSize());
        auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            p_b_block, b_k0_n_k1_block_desc.GetElementSpaceSize());

        // gridwise GEMM pipeline
        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(
            (a_b_k0_m_k1_grid_desc.GetLength(I1) * a_b_k0_m_k1_grid_desc.GetLength(I3)) /
            (K0PerBlock * K1));

        const auto gridwise_gemm_pipeline = GridwiseGemmPipe{};

        gridwise_gemm_pipeline.template Run<HasMainKBlockLoop>(a_b_k0_m_k1_grid_desc,
                                                               a_b_k0_m_k1_block_desc,
                                                               a_blockwise_copy,
                                                               a_grid_buf,
                                                               a_block_buf,
                                                               a_block_slice_copy_step,
                                                               b_b_k0_n_k1_grid_desc,
                                                               b_b_k0_n_k1_block_desc,
                                                               b_blockwise_copy,
                                                               b_grid_buf,
                                                               b_block_buf,
                                                               b_block_slice_copy_step,
                                                               blockwise_gemm,
                                                               c_thread_buf,
                                                               num_k_block_main_loop);

        // output: register to global memory
        Base::template RunEpilogue<CGlobalMemoryDataOperation, false, false>(
            blockwise_gemm,
            c_grid_desc_mblock_mperblock_nblock_nperblock,
            c_thread_buf,
            block_m_id,
            block_n_id,
            p_shared_block,
            p_c_grid,
            c_element_op);
    }

    static std::string GetTypeString()
    {
        auto str = std::stringstream();

        // clang-format off
        str << "GemmXdlSplitKCShuffle_"
            << getGemmSpecializationString(GemmSpec) << "_"
            << std::string(ALayout::name)[0]
            << std::string(BLayout::name)[0]
            << std::string(CLayout::name)[0]
            << "_"
            << "B" << BlockSize << "_"
            << "Vec" << ABlockTransferSrcScalarPerVector << "x"
            << BBlockTransferSrcScalarPerVector << "x"
            << CBlockTransferScalarPerVector_NWaveNPerXDL << "_"
            << MPerBlock << "x"
            << NPerBlock << "x"
            << K0PerBlock << "x"
            << K1 ;
        // clang-format on

        return str.str();
    }
};

} // namespace ck

#if __clang_major__ >= 23
#pragma clang diagnostic pop
#endif
