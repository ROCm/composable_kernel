#ifndef DEVICE_GEMM_XDL_SPLITK_HPP
#define DEVICE_GEMM_XDL_SPLITK_HPP

#include <cstdio>
#include <iostream>
#include <sstream>
#include <utility>
#include "device.hpp"
#include "device_base.hpp"
#include "device_gemm.hpp"
#include "common_header.hpp"
#include "tensor_layout.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm_xdlops_v2r3.hpp"
#include "gemm_specialization.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

/*
 * \brief Wrapper function of GridwiseGemm::Run to realize BatchedGEMM.
 *
 * \see \link device_batched_gemm_xdl.hpp kernel_batched_gemm_xdlops_v2r3
 */
template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename AGridDesc_K0_M_K1,
          typename BGridDesc_K0_N_K1,
          typename AGridDesc_K0_M_K1_Tail,
          typename BGridDesc_K0_N_K1_Tail,
          typename CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename ComputePtrOffsetOfBatch,
          typename Block2CTileMap,
          bool HasMainKBlockLoop,
          bool TailHasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_batched_gemm_xdlops_v2r3(
            const FloatAB* __restrict__ p_a_grid,
            const FloatAB* __restrict__ p_b_grid,
            FloatC* __restrict__ p_c_grid,
            const index_t batch_count,
            const AGridDesc_K0_M_K1 a_grid_desc_k0_m_k1,
            const BGridDesc_K0_N_K1 b_grid_desc_k0_n_k1,
            const AGridDesc_K0_M_K1_Tail a_grid_desc_k0_m_k1_tail,
            const BGridDesc_K0_N_K1_Tail b_grid_desc_k0_n_k1_tail,
            const CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2 c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2,
            const AElementwiseOperation a_element_op,
            const BElementwiseOperation b_element_op,
            const CElementwiseOperation c_element_op,
            const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch,
            const Block2CTileMap block_2_ctile_map)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__))
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t a_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx)));
    const long_index_t b_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx)));
    const long_index_t c_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetCPtrOffset(g_idx)));

    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    if(g_idx < batch_count - 1)
    {
        GridwiseGemm::template Run<HasMainKBlockLoop>(p_a_grid + a_batch_offset,
                                                      p_b_grid + b_batch_offset,
                                                      p_c_grid + c_batch_offset,
                                                      p_shared,
                                                      a_grid_desc_k0_m_k1,
                                                      b_grid_desc_k0_n_k1,
                                                      c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                                                      a_element_op,
                                                      b_element_op,
                                                      c_element_op,
                                                      block_2_ctile_map);
    }
    else
    {
        GridwiseGemm::template Run<TailHasMainKBlockLoop>(p_a_grid + a_batch_offset,
                                                          p_b_grid + b_batch_offset,
                                                          p_c_grid + c_batch_offset,
                                                          p_shared,
                                                          a_grid_desc_k0_m_k1_tail,
                                                          b_grid_desc_k0_n_k1_tail,
                                                          c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                                                          a_element_op,
                                                          b_element_op,
                                                          c_element_op,
                                                          block_2_ctile_map);
    }
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_c_grid;
    ignore = batch_count;
    ignore = a_grid_desc_k0_m_k1;
    ignore = b_grid_desc_k0_n_k1;
    ignore = a_grid_desc_k0_m_k1_tail;
    ignore = b_grid_desc_k0_n_k1_tail;
    ignore = c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
    ignore = compute_ptr_offset_of_batch;
    ignore = block_2_ctile_map;
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t K0PerBlock,
          ck::index_t K1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_K1,
          bool ABlockLdsAddExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_K1,
          bool BBlockLdsAddExtraN,
          ck::index_t CThreadTransferSrcDstVectorDim,
          ck::index_t CThreadTransferDstScalarPerVector>
struct DeviceGemmXdlSplitK
    : public DeviceGemm<AElementwiseOperation, BElementwiseOperation, CElementwiseOperation>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static constexpr auto K1Number = Number<K1>{};

    static auto GetActualBatchAndKSplitted(index_t K, index_t KBatch)
    {
        const index_t K0 = math::integer_divide_ceil(K, K1 * K0PerBlock * KBatch) * K0PerBlock;
        const index_t KSplitted    = K0 * K1;
        const index_t actual_batch = math::integer_divide_ceil(K, KSplitted);

        return std::make_pair(actual_batch, KSplitted);
    }

    static auto MakeAGridDescriptor_K0_M_K1(index_t M, index_t K, index_t StrideA)
    {
        assert(K % (K1 * K0PerBlock) == 0);

        const index_t K0 = K / K1;

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

        if constexpr(GemmSpec == GemmSpecialization::MNPadding)
        {
            const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;

            return transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_right_pad_transform(M, PadM)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_pass_through_transform(M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
    }

    static auto MakeBGridDescriptor_K0_N_K1(index_t K, index_t N, index_t StrideB)
    {
        assert(K % (K1 * K0PerBlock) == 0);

        const index_t K0 = K / K1;

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

        if constexpr(GemmSpec == GemmSpecialization::MNPadding)
        {
            const auto PadN = (NPerBlock - N % NPerBlock) % NPerBlock;

            return transform_tensor_descriptor(
                b_grid_desc_k_n,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_right_pad_transform(N, PadN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                b_grid_desc_k_n,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
    }

    static auto MakeAGridDescriptor_K0_M_K1_Tail(index_t M, index_t K, index_t StrideA)
    {
        const index_t KPadded = math::integer_divide_ceil(K, K1 * K0PerBlock) * K1 * K0PerBlock;
        const index_t K0   = KPadded / K1;

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

        const auto a_grid_desc_m_kpad = transform_tensor_descriptor(
            a_grid_desc_m_k,
            make_tuple(make_pass_through_transform(M), make_right_pad_transform(K, KPadded - K)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        if constexpr(GemmSpec == GemmSpecialization::MNPadding)
        {
            const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;

            return transform_tensor_descriptor(
                a_grid_desc_m_kpad,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_right_pad_transform(M, PadM)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                a_grid_desc_m_kpad,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_pass_through_transform(M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
    }

    static auto MakeBGridDescriptor_K0_N_K1_Tail(index_t K, index_t N, index_t StrideB)
    {
        const index_t KPadded = math::integer_divide_ceil(K, K1 * K0PerBlock) * K1 * K0PerBlock;

        const index_t K0 = KPadded / K1;

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

        const auto b_grid_desc_kpad_n = transform_tensor_descriptor(
            b_grid_desc_k_n,
            make_tuple(make_right_pad_transform(K, KPadded - K), make_pass_through_transform(N)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        if constexpr(GemmSpec == GemmSpecialization::MNPadding)
        {
            const auto PadN = (NPerBlock - N % NPerBlock) % NPerBlock;

            return transform_tensor_descriptor(
                b_grid_desc_kpad_n,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_right_pad_transform(N, PadN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                b_grid_desc_kpad_n,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
    }

    static auto MakeCGridDescriptor_M_N(index_t M, index_t N, index_t StrideC)
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

        if constexpr(GemmSpec == GemmSpecialization::MNPadding)
        {
            const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;
            const auto PadN = (NPerBlock - N % NPerBlock) % NPerBlock;

            return transform_tensor_descriptor(
                c_grid_desc_m_n,
                make_tuple(make_right_pad_transform(M, PadM), make_right_pad_transform(N, PadN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else
        {

            return transform_tensor_descriptor(
                c_grid_desc_m_n,
                make_tuple(make_pass_through_transform(M), make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
    }

    using AGridDesc_K0_M_K1      = decltype(MakeAGridDescriptor_K0_M_K1(1, 1, 1));
    using BGridDesc_K0_N_K1      = decltype(MakeBGridDescriptor_K0_N_K1(1, 1, 1));
    using AGridDesc_K0_M_K1_Tail = decltype(MakeAGridDescriptor_K0_M_K1_Tail(1, 1, 1));
    using BGridDesc_K0_N_K1_Tail = decltype(MakeBGridDescriptor_K0_N_K1_Tail(1, 1, 1));
    using CGridDesc_M_N          = decltype(MakeCGridDescriptor_M_N(1, 1, 1));

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

    struct ComputePtrOffsetOfStridedBatch
    {
        ComputePtrOffsetOfStridedBatch(const index_t BatchStrideA, const index_t BatchStrideB)
            : BatchStrideA_(BatchStrideA), BatchStrideB_(BatchStrideB)
        {
        }

        __host__ __device__ constexpr long_index_t GetAPtrOffset(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideA_);
        }

        __host__ __device__ constexpr long_index_t GetBPtrOffset(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideB_);
        }

        __host__ __device__ constexpr long_index_t GetCPtrOffset(index_t g_idx) const
        {
            ignore = g_idx;
            return 0;
        }

        private:
        index_t BatchStrideA_;
        index_t BatchStrideB_;
        // index_t BatchStrideC_; // always zero
    };

    using GridwiseGemm =
        GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3<BlockSize,
                                                ADataType, // TODO: distinguish A/B datatype
                                                AccDataType,
                                                CDataType,
                                                InMemoryDataOperationEnum::AtomicAdd,
                                                // AGridDesc_K0_M_K1,
                                                // BGridDesc_K0_N_K1,
                                                // CGridDesc_M_N,
                                                AElementwiseOperation,
                                                BElementwiseOperation,
                                                CElementwiseOperation,
                                                MPerBlock,
                                                NPerBlock,
                                                K0PerBlock,
                                                MPerXDL,
                                                NPerXDL,
                                                K1,
                                                MXdlPerWave,
                                                NXdlPerWave,
                                                ABlockTransferThreadClusterLengths_K0_M_K1,
                                                ABlockTransferThreadClusterArrangeOrder,
                                                ABlockTransferSrcAccessOrder,
                                                ABlockTransferSrcVectorDim,
                                                ABlockTransferSrcScalarPerVector,
                                                ABlockTransferDstScalarPerVector_K1,
                                                false, // AThreadTransferSrcResetCoordinateAfterRun,
                                                ABlockLdsAddExtraM,
                                                BBlockTransferThreadClusterLengths_K0_N_K1,
                                                BBlockTransferThreadClusterArrangeOrder,
                                                BBlockTransferSrcAccessOrder,
                                                BBlockTransferSrcVectorDim,
                                                BBlockTransferSrcScalarPerVector,
                                                BBlockTransferDstScalarPerVector_K1,
                                                false, // BThreadTransferSrcResetCoordinateAfterRun,
                                                BBlockLdsAddExtraN,
                                                Sequence<2, 3, 0, 1, 7, 5, 4, 6>,
                                                CThreadTransferSrcDstVectorDim,
                                                CThreadTransferDstScalarPerVector>;

    using CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2 =
        decltype(GridwiseGemm::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(CGridDesc_M_N{}));
    using Block2CTileMap = decltype(MakeBlock2CTileMap(1, CGridDesc_M_N{}, 1, 1));

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const ADataType* p_a_grid,
                 const BDataType* p_b_grid,
                 CDataType* p_c_grid,
                 index_t M,
                 index_t N,
                 index_t K,
                 index_t StrideA,
                 index_t StrideB,
                 index_t StrideC,
                 index_t M01,
                 index_t N01,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op,
                 index_t k_batch)
            : p_a_grid_{p_a_grid},
              p_b_grid_{p_b_grid},
              p_c_grid_{p_c_grid},
              BatchCount_(k_batch),
              has_tail_(false),
              compute_ptr_offset_of_batch_{0, 0},
              M01_{M01},
              N01_{N01},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op}
        {
            const auto actual_batch_and_ksplitted = GetActualBatchAndKSplitted(K, k_batch);
            BatchCount_                           = actual_batch_and_ksplitted.first;
            const auto KSplitted                  = actual_batch_and_ksplitted.second;

            a_grid_desc_k0_m_k1_ =
                DeviceGemmXdlSplitK::MakeAGridDescriptor_K0_M_K1(M, KSplitted, StrideA);
            b_grid_desc_k0_n_k1_ =
                DeviceGemmXdlSplitK::MakeBGridDescriptor_K0_N_K1(KSplitted, N, StrideB);
            c_grid_desc_m_n_ = DeviceGemmXdlSplitK::MakeCGridDescriptor_M_N(M, N, StrideC);

            bool is_valid = GridwiseGemm::CheckValidity(
                a_grid_desc_k0_m_k1_, b_grid_desc_k0_n_k1_, c_grid_desc_m_n_, M01_, N01_);

            if(K != KSplitted * BatchCount_)
            {
                has_tail_        = true;
                const auto KTail = K - KSplitted * (BatchCount_ - 1);
                a_grid_desc_k0_m_k1_tail_ =
                    DeviceGemmXdlSplitK::MakeAGridDescriptor_K0_M_K1_Tail(M, KTail, StrideA);
                b_grid_desc_k0_n_k1_tail_ =
                    DeviceGemmXdlSplitK::MakeBGridDescriptor_K0_N_K1_Tail(KTail, N, StrideB);

                is_valid &= GridwiseGemm::CheckValidity(a_grid_desc_k0_m_k1_tail_,
                                                        b_grid_desc_k0_n_k1_tail_,
                                                        c_grid_desc_m_n_,
                                                        M01_,
                                                        N01_);
            }

            if(is_valid)
            {
                c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_ =
                    GridwiseGemm::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(c_grid_desc_m_n_);

                const index_t a_batch_stride = [KSplitted, StrideA]() {
                    if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
                    {
                        ignore = StrideA;
                        return KSplitted;
                    }
                    else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ALayout>::value)
                    {
                        return KSplitted * StrideA;
                    }
                }();

                const index_t b_batch_stride = [KSplitted, StrideB]() {
                    if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
                    {
                        return KSplitted * StrideB;
                    }
                    else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
                    {
                        ignore = StrideB;
                        return KSplitted;
                    }
                }();

                compute_ptr_offset_of_batch_ =
                    ComputePtrOffsetOfStridedBatch{a_batch_stride, b_batch_stride};
                block_2_ctile_map_ = MakeBlock2CTileMap(BatchCount_, c_grid_desc_m_n_, M01, N01);
            }
        }

        //  private:
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;
        index_t BatchCount_;
        bool has_tail_;
        AGridDesc_K0_M_K1 a_grid_desc_k0_m_k1_;
        BGridDesc_K0_N_K1 b_grid_desc_k0_n_k1_;
        AGridDesc_K0_M_K1_Tail a_grid_desc_k0_m_k1_tail_;
        BGridDesc_K0_N_K1_Tail b_grid_desc_k0_n_k1_tail_;
        CGridDesc_M_N c_grid_desc_m_n_;
        CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2 c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_;
        ComputePtrOffsetOfStridedBatch compute_ptr_offset_of_batch_;
        Block2CTileMap block_2_ctile_map_;
        index_t M01_;
        index_t N01_;
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceGemmXdlSplitK::Argument;

        float Run(const Argument& arg, int nrepeat = 1)
        {
            {
                std::cout << "k_batch = " << arg.BatchCount_ << "\n";
                std::cout << "arg.a_grid_desc_k0_m_k1_{" << arg.a_grid_desc_k0_m_k1_.GetLength(I0)
                          << ", " << arg.a_grid_desc_k0_m_k1_.GetLength(I1) << ", "
                          << arg.a_grid_desc_k0_m_k1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.b_grid_desc_k0_n_k1_{" << arg.b_grid_desc_k0_n_k1_.GetLength(I0)
                          << ", " << arg.b_grid_desc_k0_n_k1_.GetLength(I1) << ", "
                          << arg.b_grid_desc_k0_n_k1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.c_grid_desc_m_n_{" << arg.c_grid_desc_m_n_.GetLength(I0) << ", "
                          << arg.c_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;

                if(arg.has_tail_)
                {
                    std::cout << "arg.a_grid_desc_k0_m_k1_tail_{"
                              << arg.a_grid_desc_k0_m_k1_tail_.GetLength(I0) << ", "
                              << arg.a_grid_desc_k0_m_k1_tail_.GetLength(I1) << ", "
                              << arg.a_grid_desc_k0_m_k1_.GetLength(I2) << "}" << std::endl;

                    std::cout << "arg.b_grid_desc_k0_n_k1_tail_{"
                              << arg.b_grid_desc_k0_n_k1_tail_.GetLength(I0) << ", "
                              << arg.b_grid_desc_k0_n_k1_tail_.GetLength(I1) << ", "
                              << arg.b_grid_desc_k0_n_k1_tail_.GetLength(I2) << "}" << std::endl;
                }
            }

            bool is_valid = GridwiseGemm::CheckValidity(arg.a_grid_desc_k0_m_k1_,
                                                        arg.b_grid_desc_k0_n_k1_,
                                                        arg.c_grid_desc_m_n_,
                                                        arg.M01_,
                                                        arg.N01_);
            if(arg.has_tail_)
            {

                is_valid &= GridwiseGemm::CheckValidity(arg.a_grid_desc_k0_m_k1_tail_,
                                                        arg.b_grid_desc_k0_n_k1_tail_,
                                                        arg.c_grid_desc_m_n_,
                                                        arg.M01_,
                                                        arg.N01_);
            }
            if(!is_valid)
            {
                throw std::runtime_error(
                    "wrong! GridwiseBatchedGemm_km_kn_m0m1n0n1_xdlops_v2r3 has invalid setting");
            }

            const index_t grid_size =
                GridwiseGemm::CalculateGridSize(arg.c_grid_desc_m_n_) * arg.BatchCount_;

            float ave_time = 0;

            if(arg.has_tail_)
            {
                const auto K0                     = arg.a_grid_desc_k0_m_k1_.GetLength(I0);
                const bool has_main_k0_block_loop = GridwiseGemm::CalculateHasMainK0BlockLoop(K0);
                const auto K0_tail                = arg.a_grid_desc_k0_m_k1_.GetLength(I0);
                const bool tail_has_main_k0_block_loop =
                    GridwiseGemm::CalculateHasMainK0BlockLoop(K0_tail);

                const auto Run = [&](const auto& kernel)
                {
                    return launch_and_time_kernel(kernel,
                                                      nrepeat,
                                                      dim3(grid_size),
                                                      dim3(BlockSize),
                                                      0,
                                                      arg.p_a_grid_,
                                                      arg.p_b_grid_,
                                                      arg.p_c_grid_,
                                                      arg.BatchCount_,
                                                      arg.a_grid_desc_k0_m_k1_,
                                                      arg.b_grid_desc_k0_n_k1_,
                                                      arg.a_grid_desc_k0_m_k1_tail_,
                                                      arg.b_grid_desc_k0_n_k1_tail_,
                                                      arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_,
                                                      arg.a_element_op_,
                                                      arg.b_element_op_,
                                                      arg.c_element_op_,
                                                      arg.compute_ptr_offset_of_batch_,
                                                      arg.block_2_ctile_map_);

                };

                if(has_main_k0_block_loop && tail_has_main_k0_block_loop)
                {
                    const auto kernel = kernel_batched_gemm_xdlops_v2r3<
                        GridwiseGemm,
                        ADataType, // TODO: distiguish A/B datatype
                        CDataType,
                        remove_reference_t<DeviceGemmXdlSplitK::AGridDesc_K0_M_K1>,
                        remove_reference_t<DeviceGemmXdlSplitK::BGridDesc_K0_N_K1>,
                        remove_reference_t<DeviceGemmXdlSplitK::AGridDesc_K0_M_K1_Tail>,
                        remove_reference_t<DeviceGemmXdlSplitK::BGridDesc_K0_N_K1_Tail>,
                        remove_reference_t<CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2>,
                        AElementwiseOperation,
                        BElementwiseOperation,
                        CElementwiseOperation,
                        ComputePtrOffsetOfStridedBatch,
                        remove_reference_t<Block2CTileMap>,
                        true,
                        true>;

                    ave_time = Run(kernel);

                }
                else if(has_main_k0_block_loop && !tail_has_main_k0_block_loop)
                {
                    const auto kernel = kernel_batched_gemm_xdlops_v2r3<
                        GridwiseGemm,
                        ADataType, // TODO: distiguish A/B datatype
                        CDataType,
                        remove_reference_t<DeviceGemmXdlSplitK::AGridDesc_K0_M_K1>,
                        remove_reference_t<DeviceGemmXdlSplitK::BGridDesc_K0_N_K1>,
                        remove_reference_t<DeviceGemmXdlSplitK::AGridDesc_K0_M_K1_Tail>,
                        remove_reference_t<DeviceGemmXdlSplitK::BGridDesc_K0_N_K1_Tail>,
                        remove_reference_t<CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2>,
                        AElementwiseOperation,
                        BElementwiseOperation,
                        CElementwiseOperation,
                        ComputePtrOffsetOfStridedBatch,
                        remove_reference_t<Block2CTileMap>,
                        true,
                        false>;

                    ave_time = Run(kernel);
                }
                else if(!has_main_k0_block_loop && tail_has_main_k0_block_loop)
                {
                    const auto kernel = kernel_batched_gemm_xdlops_v2r3<
                        GridwiseGemm,
                        ADataType, // TODO: distiguish A/B datatype
                        CDataType,
                        remove_reference_t<DeviceGemmXdlSplitK::AGridDesc_K0_M_K1>,
                        remove_reference_t<DeviceGemmXdlSplitK::BGridDesc_K0_N_K1>,
                        remove_reference_t<DeviceGemmXdlSplitK::AGridDesc_K0_M_K1_Tail>,
                        remove_reference_t<DeviceGemmXdlSplitK::BGridDesc_K0_N_K1_Tail>,
                        remove_reference_t<CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2>,
                        AElementwiseOperation,
                        BElementwiseOperation,
                        CElementwiseOperation,
                        ComputePtrOffsetOfStridedBatch,
                        remove_reference_t<Block2CTileMap>,
                        false,
                        true>;

                    ave_time = Run(kernel);
                }
                else
                {
                    const auto kernel = kernel_batched_gemm_xdlops_v2r3<
                        GridwiseGemm,
                        ADataType, // TODO: distiguish A/B datatype
                        CDataType,
                        remove_reference_t<DeviceGemmXdlSplitK::AGridDesc_K0_M_K1>,
                        remove_reference_t<DeviceGemmXdlSplitK::BGridDesc_K0_N_K1>,
                        remove_reference_t<DeviceGemmXdlSplitK::AGridDesc_K0_M_K1_Tail>,
                        remove_reference_t<DeviceGemmXdlSplitK::BGridDesc_K0_N_K1_Tail>,
                        remove_reference_t<CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2>,
                        AElementwiseOperation,
                        BElementwiseOperation,
                        CElementwiseOperation,
                        ComputePtrOffsetOfStridedBatch,
                        remove_reference_t<Block2CTileMap>,
                        false,
                        false>;

                    ave_time = Run(kernel);
                }
            }
            else
            {
                const auto K0                     = arg.a_grid_desc_k0_m_k1_.GetLength(I0);
                const bool has_main_k0_block_loop = GridwiseGemm::CalculateHasMainK0BlockLoop(K0);

                if(has_main_k0_block_loop)
                {
                    const auto kernel = ck::kernel_batched_gemm_xdlops_v2r3<
                        GridwiseGemm,
                        ADataType, // TODO: distiguish A/B datatype
                        CDataType,
                        remove_reference_t<DeviceGemmXdlSplitK::AGridDesc_K0_M_K1>,
                        remove_reference_t<DeviceGemmXdlSplitK::BGridDesc_K0_N_K1>,
                        remove_reference_t<CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2>,
                        AElementwiseOperation,
                        BElementwiseOperation,
                        CElementwiseOperation,
                        ComputePtrOffsetOfStridedBatch,
                        remove_reference_t<Block2CTileMap>,
                        true>;

                    ave_time = launch_and_time_kernel(kernel,
                                                      nrepeat,
                                                      dim3(grid_size),
                                                      dim3(BlockSize),
                                                      0,
                                                      arg.p_a_grid_,
                                                      arg.p_b_grid_,
                                                      arg.p_c_grid_,
                                                      arg.BatchCount_,
                                                      arg.a_grid_desc_k0_m_k1_,
                                                      arg.b_grid_desc_k0_n_k1_,
                                                      arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_,
                                                      arg.a_element_op_,
                                                      arg.b_element_op_,
                                                      arg.c_element_op_,
                                                      arg.compute_ptr_offset_of_batch_,
                                                      arg.block_2_ctile_map_);
                }
                else
                {
                    const auto kernel = ck::kernel_batched_gemm_xdlops_v2r3<
                        GridwiseGemm,
                        ADataType, // TODO: distiguish A/B datatype
                        CDataType,
                        remove_reference_t<DeviceGemmXdlSplitK::AGridDesc_K0_M_K1>,
                        remove_reference_t<DeviceGemmXdlSplitK::BGridDesc_K0_N_K1>,
                        remove_reference_t<CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2>,
                        AElementwiseOperation,
                        BElementwiseOperation,
                        CElementwiseOperation,
                        ComputePtrOffsetOfStridedBatch,
                        remove_reference_t<Block2CTileMap>,
                        false>;

                    ave_time = launch_and_time_kernel(kernel,
                                                      nrepeat,
                                                      dim3(grid_size),
                                                      dim3(BlockSize),
                                                      0,
                                                      arg.p_a_grid_,
                                                      arg.p_b_grid_,
                                                      arg.p_c_grid_,
                                                      arg.BatchCount_,
                                                      arg.a_grid_desc_k0_m_k1_,
                                                      arg.b_grid_desc_k0_n_k1_,
                                                      arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_,
                                                      arg.a_element_op_,
                                                      arg.b_element_op_,
                                                      arg.c_element_op_,
                                                      arg.compute_ptr_offset_of_batch_,
                                                      arg.block_2_ctile_map_);
                }
            }

            return ave_time;
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
        return GridwiseGemm::CheckValidity(arg.a_grid_desc_k0_m_k1_,
                                           arg.b_grid_desc_k0_n_k1_,
                                           arg.c_grid_desc_m_n_,
                                           arg.M01_,
                                           arg.N01_);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const ADataType* p_a,
                             const BDataType* p_b,
                             CDataType* p_c,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideB,
                             index_t StrideC,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op,
                             index_t BatchCount)
    {
        return Argument{p_a,
                        p_b,
                        p_c,
                        M,
                        N,
                        K,
                        StrideA,
                        StrideB,
                        StrideC,
                        1,
                        1,
                        a_element_op,
                        b_element_op,
                        c_element_op,
                        BatchCount};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      void* p_c,
                                                      index_t M,
                                                      index_t N,
                                                      index_t K,
                                                      index_t StrideA,
                                                      index_t StrideB,
                                                      index_t StrideC,
                                                      AElementwiseOperation a_element_op,
                                                      BElementwiseOperation b_element_op,
                                                      CElementwiseOperation c_element_op,
                                                      index_t BatchCount) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          static_cast<CDataType*>(p_c),
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideC,
                                          1,
                                          1,
                                          a_element_op,
                                          b_element_op,
                                          c_element_op,
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
        str << "DeviceGemmXdlSplitK"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << K0PerBlock
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
