#pragma once
#include <iostream>
#include <sstream>
#include "device.hpp"
#include "device_gemm_reduce.hpp"
#include "common_header.hpp"
#include "tensor_layout.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm_reduce_xdl_cshuffle_v1.hpp"
#include "gemm_specialization.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename FloatD,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename D0ReduceOperation,
          typename D1ReduceOperation,
          typename AGridDesc_AK0_M_AK1,
          typename BGridDesc_BK0_N_BK1,
          typename CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename DGridDescriptor_MBlock_MPerBlock,
          typename ComputeBasePrtOfBatch,
          typename Block2CTileMap,
          bool HasMainK0BlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_batched_gemm_reduce_xdl_cshuffle_v1(
            const FloatAB* __restrict__ p_a_grid,
            const FloatAB* __restrict__ p_b_grid,
            FloatC* __restrict__ p_c_grid,
            FloatD* __restrict__ p_d0_grid,
            FloatD* __restrict__ p_d1_grid,
            const index_t batch_count,
            const AElementwiseOperation a_element_op,
            const BElementwiseOperation b_element_op,
            const CElementwiseOperation c_element_op,
            const D0ReduceOperation d0_reduce_op,
            const D1ReduceOperation d1_reduce_op,
            const AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1,
            const BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1,
            const CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                c_grid_desc_mblock_mperblock_nblock_nperblock,
            const DGridDescriptor_MBlock_MPerBlock d_grid_desc_mblock_mperblock,
            const ComputeBasePrtOfBatch compute_base_ptr_of_batch_,
            const Block2CTileMap block_2_ctile_map)
{
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t a_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_base_ptr_of_batch_.GetABasePtr(g_idx)));
    const long_index_t b_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_base_ptr_of_batch_.GetBBasePtr(g_idx)));
    const long_index_t c_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_base_ptr_of_batch_.GetCBasePtr(g_idx)));

    const long_index_t d0_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_base_ptr_of_batch_.GetD0BasePtr(g_idx)));
    const long_index_t d1_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_base_ptr_of_batch_.GetD1BasePtr(g_idx)));

    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    GridwiseGemm::template Run<HasMainK0BlockLoop>(p_a_grid + a_batch_offset,
                                                   p_b_grid + b_batch_offset,
                                                   p_c_grid + c_batch_offset,
                                                   p_d0_grid + d0_batch_offset,
                                                   p_d1_grid + d1_batch_offset,
                                                   p_shared,
                                                   a_element_op,
                                                   b_element_op,
                                                   c_element_op,
                                                   d0_reduce_op,
                                                   d1_reduce_op,
                                                   a_grid_desc_ak0_m_ak1,
                                                   b_grid_desc_bk0_n_bk1,
                                                   c_grid_desc_mblock_mperblock_nblock_nperblock,
                                                   d_grid_desc_mblock_mperblock,
                                                   block_2_ctile_map);
}

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename GemmAccDataType,
          typename CShuffleDataType,
          typename ReduceAccDataType,
          typename DDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename D0ReduceOperation,
          typename D1ReduceOperation,
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
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          typename CReduceThreadClusterLengths_MPerBlock_NPerBlock,
          index_t CReduceThreadLds2VGprCopySrcDstScalarPerVector_NPerBlock,
          index_t CReduceThreadVgpr2GlobalCopySrcDstScalarPerVector_MPerBlock>
struct DeviceBatchedGemmReduce_Xdl_CShuffle : public DeviceGemmReduce<AElementwiseOperation,
                                                                      BElementwiseOperation,
                                                                      CElementwiseOperation,
                                                                      D0ReduceOperation,
                                                                      D1ReduceOperation>
{
    using DeviceOp = DeviceBatchedGemmReduce_Xdl_CShuffle;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static auto MakeAGridDescriptor_AK0_M_AK1(index_t MRaw, index_t KRaw, index_t StrideA)
    {
        const auto a_grid_desc_mraw_kraw = [&]() {
            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, KRaw),
                                                    make_tuple(StrideA, I1));
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, KRaw),
                                                    make_tuple(I1, StrideA));
            }
        }();

        const auto M = math::integer_divide_ceil(MRaw, MPerBlock) * MPerBlock;
        const auto K = math::integer_divide_ceil(KRaw, KPerBlock) * KPerBlock;

        const auto MPad = M - MRaw;
        const auto KPad = K - KRaw;

        if constexpr(GemmSpec == GemmSpecialization::MKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad both M and K
            assert(K % AK1 == 0);

            const auto AK0 = K / AK1;

            const auto a_grid_desc_m_k =
                transform_tensor_descriptor(a_grid_desc_mraw_kraw,
                                            make_tuple(make_right_pad_transform(MRaw, MPad),
                                                       make_right_pad_transform(KRaw, KPad)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto a_grid_desc_ak0_m_ak1 =
                transform_tensor_descriptor(a_grid_desc_m_k,
                                            make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                       make_pass_through_transform(M)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::MPadding ||
                          GemmSpec == GemmSpecialization::MNPadding)
        {
            // pad M, but not K
            assert(KRaw % AK1 == 0);

            const auto AK0 = KRaw / AK1;

            const auto a_grid_desc_ak0_m_ak1 =
                transform_tensor_descriptor(a_grid_desc_mraw_kraw,
                                            make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                       make_right_pad_transform(MRaw, MPad)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::KPadding ||
                          GemmSpec == GemmSpecialization::NKPadding)
        {
            // pad K, but not M
            assert(K % AK1 == 0);

            const auto AK0 = K / AK1;

            const auto a_grid_desc_m_k = transform_tensor_descriptor(
                a_grid_desc_mraw_kraw,
                make_tuple(make_pass_through_transform(MRaw), make_right_pad_transform(KRaw, KPad)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto a_grid_desc_ak0_m_ak1 =
                transform_tensor_descriptor(a_grid_desc_m_k,
                                            make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                       make_pass_through_transform(MRaw)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else
        {
            // not pad M or K
            assert(KRaw % AK1 == 0);

            const auto AK0 = KRaw / AK1;

            const auto a_grid_desc_ak0_m_ak1 =
                transform_tensor_descriptor(a_grid_desc_mraw_kraw,
                                            make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                       make_pass_through_transform(MRaw)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
    }

    static auto MakeBGridDescriptor_BK0_N_BK1(index_t KRaw, index_t NRaw, index_t StrideB)
    {
        const auto b_grid_desc_nraw_kraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(I1, StrideB));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(StrideB, I1));
            }
        }();

        const auto N = math::integer_divide_ceil(NRaw, NPerBlock) * NPerBlock;
        const auto K = math::integer_divide_ceil(KRaw, KPerBlock) * KPerBlock;

        const auto NPad = N - NRaw;
        const auto KPad = K - KRaw;

        if constexpr(GemmSpec == GemmSpecialization::NKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad both N and K
            assert(K % BK1 == 0);

            const auto BK0 = K / BK1;

            const auto b_grid_desc_n_k =
                transform_tensor_descriptor(b_grid_desc_nraw_kraw,
                                            make_tuple(make_right_pad_transform(NRaw, NPad),
                                                       make_right_pad_transform(KRaw, KPad)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto b_grid_desc_bk0_n_bk1 =
                transform_tensor_descriptor(b_grid_desc_n_k,
                                            make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                       make_pass_through_transform(N)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::NPadding ||
                          GemmSpec == GemmSpecialization::MNPadding)
        {
            // pad N, but not K
            assert(KRaw % BK1 == 0);

            const auto BK0 = KRaw / BK1;

            const auto b_grid_desc_bk0_n_bk1 =
                transform_tensor_descriptor(b_grid_desc_nraw_kraw,
                                            make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                       make_right_pad_transform(NRaw, NPad)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::KPadding ||
                          GemmSpec == GemmSpecialization::MKPadding)
        {
            // pad K, but not N
            assert(K % BK1 == 0);

            const auto BK0 = K / BK1;

            const auto b_grid_desc_n_k = transform_tensor_descriptor(
                b_grid_desc_nraw_kraw,
                make_tuple(make_pass_through_transform(NRaw), make_right_pad_transform(KRaw, KPad)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto b_grid_desc_bk0_n_bk1 =
                transform_tensor_descriptor(b_grid_desc_n_k,
                                            make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                       make_pass_through_transform(NRaw)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else
        {
            // not pad N or K
            assert(KRaw % BK1 == 0);

            const auto BK0 = KRaw / BK1;

            const auto b_grid_desc_bk0_n_bk1 =
                transform_tensor_descriptor(b_grid_desc_nraw_kraw,
                                            make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                       make_pass_through_transform(NRaw)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
    }

    static auto MakeCGridDescriptor_M_N(index_t MRaw, index_t NRaw, index_t StrideC)
    {
        const auto c_grid_desc_mraw_nraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(StrideC, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(I1, StrideC));
            }
        }();

        const auto M = math::integer_divide_ceil(MRaw, MPerBlock) * MPerBlock;
        const auto N = math::integer_divide_ceil(NRaw, NPerBlock) * NPerBlock;

        const auto MPad = M - MRaw;
        const auto NPad = N - NRaw;

        if constexpr(GemmSpec == GemmSpecialization::MNPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad M and N
            return transform_tensor_descriptor(c_grid_desc_mraw_nraw,
                                               make_tuple(make_right_pad_transform(MRaw, MPad),
                                                          make_right_pad_transform(NRaw, NPad)),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr(GemmSpec == GemmSpecialization::MPadding ||
                          GemmSpec == GemmSpecialization::MKPadding)
        {
            // pad M, but not N
            return transform_tensor_descriptor(
                c_grid_desc_mraw_nraw,
                make_tuple(make_right_pad_transform(MRaw, MPad), make_pass_through_transform(NRaw)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr(GemmSpec == GemmSpecialization::NPadding ||
                          GemmSpec == GemmSpecialization::NKPadding)
        {
            // pad N, but not M
            return transform_tensor_descriptor(
                c_grid_desc_mraw_nraw,
                make_tuple(make_pass_through_transform(MRaw), make_right_pad_transform(NRaw, NPad)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else
        {
            // not pad M or N
            return c_grid_desc_mraw_nraw;
        }
    }

    // assume D is packed tensor
    static auto MakeDGridDescriptor_M(index_t MRaw)
    {
        const auto d_grid_desc_mraw = make_naive_tensor_descriptor_packed(make_tuple(MRaw));

        const auto M    = math::integer_divide_ceil(MRaw, MPerBlock) * MPerBlock;
        const auto MPad = M - MRaw;

        if constexpr(GemmSpec == GemmSpecialization::MPadding ||
                     GemmSpec == GemmSpecialization::MNPadding ||
                     GemmSpec == GemmSpecialization::MKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad M
            return transform_tensor_descriptor(d_grid_desc_mraw,
                                               make_tuple(make_right_pad_transform(MRaw, MPad)),
                                               make_tuple(Sequence<0>{}),
                                               make_tuple(Sequence<0>{}));
        }
        else
        {
            // not pad M
            return d_grid_desc_mraw;
        }
    }

    using AGridDesc_AK0_M_AK1 = decltype(MakeAGridDescriptor_AK0_M_AK1(1, 1, 1));
    using BGridDesc_BK0_N_BK1 = decltype(MakeBGridDescriptor_BK0_N_BK1(1, 1, 1));
    using CGridDesc_M_N       = decltype(MakeCGridDescriptor_M_N(1, 1, 1));
    using DGridDesc_M         = decltype(MakeDGridDescriptor_M(1));

    static constexpr auto MakeBlock2CTileMap(index_t batch_count,
                                             const CGridDesc_M_N& c_grid_desc_m_n,
                                             index_t M01,
                                             index_t N01)
    {
        const auto M = c_grid_desc_m_n.GetLength(I0);
        const auto N = c_grid_desc_m_n.GetLength(I1);

        constexpr auto M1 = Number<MPerBlock>{};
        constexpr auto N1 = Number<NPerBlock>{};

        const auto M0 = M / M1;
        const auto N0 = N / N1;

        const auto M00 = M0 / M01;
        const auto N00 = N0 / N01;

        const auto g_m00_m01_n00_n01_to_m0_n0_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_insert_transform(batch_count),
                           make_unmerge_transform(make_tuple(M00, M01)),
                           make_unmerge_transform(make_tuple(N00, N01))),
                make_tuple(Sequence<>{}, Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2, 4>{}));

        const auto globalblockid_to_m00_m01_n00_n01_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(batch_count, M00, N00, M01, N01))),
                make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                make_tuple(Sequence<0>{}));

        const auto globalblockid_to_m0_n0_block_cluster_adaptor =
            chain_tensor_adaptors(g_m00_m01_n00_n01_to_m0_n0_block_cluster_adaptor,
                                  globalblockid_to_m00_m01_n00_n01_block_cluster_adaptor);

        return globalblockid_to_m0_n0_block_cluster_adaptor;
    }

    struct ComputeBasePtrOfStridedBatch
    {
        ComputeBasePtrOfStridedBatch(index_t BatchStrideA,
                                     index_t BatchStrideB,
                                     index_t BatchStrideC,
                                     index_t BatchStrideD0,
                                     index_t BatchStrideD1)
            : BatchStrideA_(BatchStrideA),
              BatchStrideB_(BatchStrideB),
              BatchStrideC_(BatchStrideC),
              BatchStrideD0_(BatchStrideD0),
              BatchStrideD1_(BatchStrideD1)
        {
        }

        __host__ __device__ constexpr long_index_t GetABasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideA_);
        }

        __host__ __device__ constexpr long_index_t GetBBasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideB_);
        }

        __host__ __device__ constexpr long_index_t GetCBasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideC_);
        }

        __host__ __device__ constexpr long_index_t GetD0BasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideD0_);
        }

        __host__ __device__ constexpr long_index_t GetD1BasePtr(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideD1_);
        }

        private:
        index_t BatchStrideA_;
        index_t BatchStrideB_;
        index_t BatchStrideC_;
        index_t BatchStrideD0_;
        index_t BatchStrideD1_;
    };

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemmReduce_k0mk1_k0nk1_mn_xdl_cshuffle_v1<
        ADataType, // TODO: distinguish A/B datatype
        GemmAccDataType,
        CShuffleDataType,
        CDataType,
        ReduceAccDataType,
        DDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        D0ReduceOperation,
        D1ReduceOperation,
        InMemoryDataOperationEnum::Set,
        InMemoryDataOperationEnum::AtomicAdd,
        AGridDesc_AK0_M_AK1,
        BGridDesc_BK0_N_BK1,
        CGridDesc_M_N,
        DGridDesc_M,
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
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CShuffleBlockTransferScalarPerVector_NPerBlock,
        CReduceThreadClusterLengths_MPerBlock_NPerBlock,
        CReduceThreadLds2VGprCopySrcDstScalarPerVector_NPerBlock,
        CReduceThreadVgpr2GlobalCopySrcDstScalarPerVector_MPerBlock>;

    using Block2CTileMap = decltype(MakeBlock2CTileMap(1, CGridDesc_M_N{}, 1, 1));

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const ADataType* p_a_grid,
                 const BDataType* p_b_grid,
                 CDataType* p_c_grid,
                 DDataType* p_d0_grid,
                 DDataType* p_d1_grid,
                 index_t MRaw,
                 index_t NRaw,
                 index_t KRaw,
                 index_t StrideA,
                 index_t StrideB,
                 index_t StrideC,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op,
                 D0ReduceOperation d0_reduce_op,
                 D1ReduceOperation d1_reduce_op,
                 index_t BatchCount)
            : p_a_grid_{p_a_grid},
              p_b_grid_{p_b_grid},
              p_c_grid_{p_c_grid},
              p_d0_grid_{p_d0_grid},
              p_d1_grid_{p_d1_grid},
              BatchCount_(BatchCount),
              a_grid_desc_ak0_m_ak1_{DeviceOp::MakeAGridDescriptor_AK0_M_AK1(MRaw, KRaw, StrideA)},
              b_grid_desc_bk0_n_bk1_{DeviceOp::MakeBGridDescriptor_BK0_N_BK1(KRaw, NRaw, StrideB)},
              c_grid_desc_m_n_{DeviceOp::MakeCGridDescriptor_M_N(MRaw, NRaw, StrideC)},
              d_grid_desc_m_{DeviceOp::MakeDGridDescriptor_M(MRaw)},
              c_grid_desc_mblock_mperblock_nblock_nperblock_{},
              d_grid_desc_mblock_mperblock_{},
              compute_base_ptr_of_batch_{a_grid_desc_ak0_m_ak1_.GetElementSpaceSize(),
                                         b_grid_desc_bk0_n_bk1_.GetElementSpaceSize(),
                                         c_grid_desc_m_n_.GetElementSpaceSize(),
                                         d_grid_desc_m_.GetElementSpaceSize(),
                                         d_grid_desc_m_.GetElementSpaceSize()},
              block_2_ctile_map_{},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op},
              d0_reduce_op_{d0_reduce_op},
              d1_reduce_op_{d1_reduce_op}
        {
            if(GridwiseGemm::CheckValidity(
                   a_grid_desc_ak0_m_ak1_, b_grid_desc_bk0_n_bk1_, c_grid_desc_m_n_))
            {
                c_grid_desc_mblock_mperblock_nblock_nperblock_ =
                    GridwiseGemm::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        c_grid_desc_m_n_);

                d_grid_desc_mblock_mperblock_ =
                    GridwiseGemm::MakeDGridDescriptor_MBlock_MPerBlock(d_grid_desc_m_);

                block_2_ctile_map_ = MakeBlock2CTileMap(BatchCount, c_grid_desc_m_n_, 1, 1);
            }
        }

        //  private:
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;
        DDataType* p_d0_grid_;
        DDataType* p_d1_grid_;
        index_t BatchCount_;
        AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1_;
        BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1_;
        CGridDesc_M_N c_grid_desc_m_n_;
        DGridDesc_M d_grid_desc_m_;
        typename GridwiseGemm::CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            c_grid_desc_mblock_mperblock_nblock_nperblock_;
        typename GridwiseGemm::DGridDescriptor_MBlock_MPerBlock d_grid_desc_mblock_mperblock_;
        ComputeBasePtrOfStridedBatch compute_base_ptr_of_batch_;
        Block2CTileMap block_2_ctile_map_;
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
        D0ReduceOperation d0_reduce_op_;
        D1ReduceOperation d1_reduce_op_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, int /* nrepeat */ = 1)
        {
#if 0
            {
                std::cout << "arg.BatchCount_ = " << arg.BatchCount_ << std::endl;

                std::cout << "arg.a_grid_desc_ak0_m_ak1_{"
                          << arg.a_grid_desc_ak0_m_ak1_.GetLength(I0) << ", "
                          << arg.a_grid_desc_ak0_m_ak1_.GetLength(I1) << ", "
                          << arg.a_grid_desc_ak0_m_ak1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.b_grid_desc_bk0_n_bk1_{"
                          << arg.b_grid_desc_bk0_n_bk1_.GetLength(I0) << ", "
                          << arg.b_grid_desc_bk0_n_bk1_.GetLength(I1) << ", "
                          << arg.b_grid_desc_bk0_n_bk1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.c_grid_desc_m_n_{ " << arg.c_grid_desc_m_n_.GetLength(I0) << ", "
                          << arg.c_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;

                std::cout << "arg.d_grid_desc_m_{ " << arg.d_grid_desc_m_.GetLength(I0) << "}"
                          << std::endl;
            }
#endif

            if(!GridwiseGemm::CheckValidity(
                   arg.a_grid_desc_ak0_m_ak1_, arg.b_grid_desc_bk0_n_bk1_, arg.c_grid_desc_m_n_))
            {
                throw std::runtime_error("wrong! GridwiseGemm has invalid setting");
            }

            const index_t grid_size =
                GridwiseGemm::CalculateGridSize(arg.c_grid_desc_m_n_) * arg.BatchCount_;

            const auto K0 = arg.a_grid_desc_ak0_m_ak1_.GetLength(I0);

            const bool has_main_k0_block_loop = GridwiseGemm::CalculateHasMainK0BlockLoop(K0);

            if(has_main_k0_block_loop)
            {
                const auto kernel = kernel_batched_gemm_reduce_xdl_cshuffle_v1<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    DDataType,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CElementwiseOperation,
                    D0ReduceOperation,
                    D1ReduceOperation,
                    DeviceOp::AGridDesc_AK0_M_AK1,
                    DeviceOp::BGridDesc_BK0_N_BK1,
                    typename GridwiseGemm::CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                    typename GridwiseGemm::DGridDescriptor_MBlock_MPerBlock,
                    ComputeBasePtrOfStridedBatch,
                    remove_reference_t<Block2CTileMap>,
                    true>;

                launch_kernel(kernel,
                              dim3(grid_size),
                              dim3(BlockSize),
                              0,
                              arg.p_a_grid_,
                              arg.p_b_grid_,
                              arg.p_c_grid_,
                              arg.p_d0_grid_,
                              arg.p_d1_grid_,
                              arg.BatchCount_,
                              arg.a_element_op_,
                              arg.b_element_op_,
                              arg.c_element_op_,
                              arg.d0_reduce_op_,
                              arg.d1_reduce_op_,
                              arg.a_grid_desc_ak0_m_ak1_,
                              arg.b_grid_desc_bk0_n_bk1_,
                              arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                              arg.d_grid_desc_mblock_mperblock_,
                              arg.compute_base_ptr_of_batch_,
                              arg.block_2_ctile_map_);
            }
            else
            {
                const auto kernel = kernel_batched_gemm_reduce_xdl_cshuffle_v1<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    DDataType,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CElementwiseOperation,
                    D0ReduceOperation,
                    D1ReduceOperation,
                    DeviceOp::AGridDesc_AK0_M_AK1,
                    DeviceOp::BGridDesc_BK0_N_BK1,
                    typename GridwiseGemm::CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                    typename GridwiseGemm::DGridDescriptor_MBlock_MPerBlock,
                    ComputeBasePtrOfStridedBatch,
                    remove_reference_t<Block2CTileMap>,
                    false>;

                launch_kernel(kernel,
                              dim3(grid_size),
                              dim3(BlockSize),
                              0,
                              arg.p_a_grid_,
                              arg.p_b_grid_,
                              arg.p_c_grid_,
                              arg.p_d0_grid_,
                              arg.p_d1_grid_,
                              arg.BatchCount_,
                              arg.a_element_op_,
                              arg.b_element_op_,
                              arg.c_element_op_,
                              arg.d0_reduce_op_,
                              arg.d1_reduce_op_,
                              arg.a_grid_desc_ak0_m_ak1_,
                              arg.b_grid_desc_bk0_n_bk1_,
                              arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                              arg.d_grid_desc_mblock_mperblock_,
                              arg.compute_base_ptr_of_batch_,
                              arg.block_2_ctile_map_);
            }

            return 0;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg, int nrepeat = 1) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), nrepeat);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        return GridwiseGemm::CheckValidity(
            arg.a_grid_desc_ak0_m_ak1_, arg.b_grid_desc_bk0_n_bk1_, arg.c_grid_desc_m_n_);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        auto casted_p_arg = dynamic_cast<const Argument*>(p_arg);
        if(casted_p_arg == nullptr)
        {
            return false;
        }
        else
        {
            return IsSupportedArgument(*casted_p_arg);
        }
    }

    static auto MakeArgument(const ADataType* p_a,
                             const BDataType* p_b,
                             CDataType* p_c,
                             DDataType* p_d0,
                             DDataType* p_d1,
                             index_t MRaw,
                             index_t NRaw,
                             index_t KRaw,
                             index_t StrideA,
                             index_t StrideB,
                             index_t StrideC,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op,
                             D0ReduceOperation d0_reduce_op,
                             D1ReduceOperation d1_reduce_op,
                             index_t BatchCount)
    {
        return Argument{p_a,
                        p_b,
                        p_c,
                        p_d0,
                        p_d1,
                        MRaw,
                        NRaw,
                        KRaw,
                        StrideA,
                        StrideB,
                        StrideC,
                        a_element_op,
                        b_element_op,
                        c_element_op,
                        d0_reduce_op,
                        d1_reduce_op,
                        BatchCount};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      void* p_c,
                                                      void* p_d0,
                                                      void* p_d1,
                                                      index_t MRaw,
                                                      index_t NRaw,
                                                      index_t KRaw,
                                                      index_t StrideA,
                                                      index_t StrideB,
                                                      index_t StrideC,
                                                      AElementwiseOperation a_element_op,
                                                      BElementwiseOperation b_element_op,
                                                      CElementwiseOperation c_element_op,
                                                      D0ReduceOperation d0_reduce_op,
                                                      D1ReduceOperation d1_reduce_op,
                                                      index_t BatchCount) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          static_cast<CDataType*>(p_c),
                                          static_cast<DDataType*>(p_d0),
                                          static_cast<DDataType*>(p_d1),
                                          MRaw,
                                          NRaw,
                                          KRaw,
                                          StrideA,
                                          StrideB,
                                          StrideC,
                                          a_element_op,
                                          b_element_op,
                                          c_element_op,
                                          d0_reduce_op,
                                          d1_reduce_op,
                                          BatchCount);
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

        // clang-format off
        str << "DeviceBatchedGemmReduce_Xdl_CShuffle"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
