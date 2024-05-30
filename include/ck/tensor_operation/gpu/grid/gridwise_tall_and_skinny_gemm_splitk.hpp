// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once 

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v1.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_dl_v2r3.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_tall_and_skinny_gemm.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_tensor_slice_transfer_v5r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_set.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

template <typename GridwiseTsmm,
          typename FloatAB,
          typename FloatC,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          bool HasMainKBlockLoop,
          bool HasTripleTailKBlockLoop,
          typename Block2CTileMap>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_tsmm_dl_v1r3(
            typename GridwiseTsmm::Argument karg) //: in __global__ functions, struct is
                                                  // better for reduced load overhead
{

    GridwiseTsmm::template Run<HasMainKBlockLoop,
                               HasTripleTailKBlockLoop,
                               GridwiseTsmm,
                               CGlobalMemoryDataOperation>(karg);
}

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          tensor_operation::device::GemmSpecialization GemmSpec,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t K0PerBlock,
          index_t K1Value,
          index_t MPerThread,
          index_t NPerThread,
          index_t KPerThread,
          typename ABlockTransferThreadSliceLengths_KBatch_K0_M0_M1_K1,
          typename ABlockTransferThreadClusterLengths_KBatch_K0_M0_M1_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          typename ABlockTransferSrcVectorTensorLengths_KBatch_K0_M0_M1_K1,
          typename ABlockTransferSrcVectorTensorContiguousDimOrder,
          typename ABlockTransferDstVectorTensorLengths_KBatch_K0_M0_M1_K1,
          typename BThreadTransferSrcDstAccessOrder,
          index_t BThreadTransferSrcVectorDim,
          index_t BThreadTransferSrcScalarPerVector,
          typename CThreadTransferSrcDstAccessOrder,
          index_t CThreadTransferSrcDstVectorDim,
          index_t CThreadTransferDstScalarPerVector>
struct GridwiseTsmmDl_km_kn_mn
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    // K1 should be Number<...>
    static constexpr auto K1 = Number<K1Value>{};

    //  Argument
    struct Argument : public tensor_operation::device::BaseArgument //
    {
        Argument(const FloatAB* p_a_grid_,
                 const FloatAB* p_b_grid_,
                 FloatC* p_c_grid_,
                 index_t M_,
                 index_t N_,
                 index_t K_,
                 index_t StrideA_,
                 index_t StrideB_,
                 index_t StrideC_,
                 //  index_t MPadded_,
                 //  index_t NPadded_,
                 // index_t KPadded_,
                 index_t K0_,
                 index_t k_batch_)
            : p_a_grid{p_a_grid_},
              p_b_grid{p_b_grid_},
              p_c_grid{p_c_grid_},
              M{M_},
              N{N_},
              K{K_},
              StrideA{StrideA_},
              StrideB{StrideB_},
              StrideC{StrideC_},
              // MPadded(MPadded_),
              // NPadded(NPadded_),
              // KPadded(KPadded_),
              K0(K0_),
              k_batch(k_batch_)
        {
        }

        //  private:
        const FloatAB* p_a_grid;
        const FloatAB* p_b_grid;
        FloatC* p_c_grid;

        index_t M, N, K;
        index_t StrideA, StrideB, StrideC;
        //:
        // index_t MPadded;
        // index_t NPadded;
        // index_t KPadded;
        index_t K0;
        index_t k_batch;
    };

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        // TODO: change this. I think it needs multi-dimensional alignment
        constexpr auto max_lds_align = K1;

        // TODO: check alignment
        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_k_m = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<K0PerBlock>{}, Number<MPerBlock>{}, K1), max_lds_align);

        // TODO: check alignment
        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_aligned_space_size =
            math::integer_least_multiple(a_block_desc_k_m.GetElementSpaceSize(), max_lds_align);

        return 3 * (a_block_aligned_space_size) * sizeof(FloatAB);
    }

    __host__ __device__ static constexpr index_t
    CalculateGridSize(index_t M, index_t N, index_t k_batch) //
    {
        const index_t grid_size = math::integer_divide_ceil(N, NPerBlock) *
                                  math::integer_divide_ceil(M, MPerBlock) * k_batch;

        return grid_size;
    }

    __host__ __device__ static constexpr bool CalculateHasMainKBlockLoop(index_t K0)
    {
        const bool has_main_k_block_loop = (K0 + K0PerBlock) / (3 * K0PerBlock) > 1;

        return has_main_k_block_loop;
    }

    __host__ __device__ static constexpr bool CalculateHasTripleTailKBlockLoop(index_t K0)

    {
        const bool has_triple_tail_k_block_loop = (K0 / K0PerBlock) % 3 == 0;

        return has_triple_tail_k_block_loop;
    }

    __host__ __device__ static auto CalculateMPadded(index_t M)
    {
        return math::integer_least_multiple(M, MPerBlock);
    }

    __host__ __device__ static auto CalculateNPadded(index_t N)
    {
        return math::integer_least_multiple(N, NPerBlock);
    }

    __host__ __device__ static auto CalculateK0(index_t K, index_t K_Batch = 1)
    {
        // k_batch * k0 * k0_per_block * k1
        auto K_t = K_Batch * K0PerBlock * K1;
        return (K + K_t - 1) / K_t * K0PerBlock;
    }

    __host__ __device__ static auto CalculateKPadded(index_t K, index_t K_Batch = 1)
    {
        auto K0 = CalculateK0(K, K_Batch);
        return K_Batch * K0 * K1;
    }

    static constexpr auto K1Number = Number<K1>{};

    // M, K -> KBatch, K0, M, K1: M -> MPad, K->KBatch, K0, K1
    __host__ __device__ static auto MakeAGridDescriptor_KBatch_K0_M_K1(
        index_t M, index_t MPad, index_t K, index_t StrideA, index_t KBatch, index_t K0)
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

        if constexpr(GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding)
        {

            return transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(
                               make_tuple(KBatch, K0, K1Number)),  // unmerge is split 1D to 3D
                           make_right_pad_transform(M, MPad - M)), //
                make_tuple(Sequence<1>{}, Sequence<0>{}), // mapped to input M & K; sequence 0 is M;
                                                          // 1 is K; make unmerge is working on K;
                make_tuple(Sequence<0, 1, 3>{}, // input is M,K; output we want is Kbatch, K0 and K1
                                                // -> 0, 1, 3; output is transformed from 2D to 4D
                           Sequence<2>{}));     // 2->M
        }
        else
        {
            return transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(KBatch, K0, K1Number)),
                           make_pass_through_transform(M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));
        }
    }

    __host__ __device__ static auto MakeBGridDescriptor_KBatch_K0_N_K1(
        index_t K, index_t NPad, index_t N, index_t StrideB, index_t KBatch, index_t K0)
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

        if constexpr(GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding)
        {

            return transform_tensor_descriptor(
                b_grid_desc_k_n,
                make_tuple(make_unmerge_transform(make_tuple(KBatch, K0, K1Number)),
                           make_right_pad_transform(N, NPad - N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                b_grid_desc_k_n,
                make_tuple(make_unmerge_transform(make_tuple(KBatch, K0, K1Number)),
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

        if constexpr(GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding)
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

    __host__ __device__ static auto GetKPad(index_t K, index_t KBatch)
    {
        const index_t K0   = math::integer_divide_ceil(K, K1 * K0PerBlock * KBatch) * K0PerBlock;
        const index_t KPad = KBatch * K0 * K1;
        return KPad;
    }

    using AGridDesc_Kbatch_K0_M_K1 = decltype(MakeAGridDescriptor_KBatch_K0_M_K1(1, 1, 1, 1, 1, 1));
    using BGridDesc_Kbatch_K0_N_K1 = decltype(MakeBGridDescriptor_KBatch_K0_N_K1(1, 1, 1, 1, 1, 1));
    using CGridDesc_M_N            = decltype(MakeCGridDescriptor_M_N(1, 1, 1));

    __host__ __device__ static constexpr bool CheckValidity(const Argument& karg)
    {

        const auto MPadded                    = CalculateMPadded(karg.M);
        const auto NPadded                    = CalculateNPadded(karg.N);
        const auto a_grid_desc_kbatch_k0_m_k1 = MakeAGridDescriptor_KBatch_K0_M_K1(
            karg.M, MPadded, karg.K, karg.StrideA, karg.k_batch, karg.K0);
        const auto b_grid_desc_kbatch_k0_n_k1 = MakeBGridDescriptor_KBatch_K0_N_K1(
            karg.K, NPadded, karg.N, karg.StrideB, karg.k_batch, karg.K0);
        const auto c_grid_desc_m_n = MakeCGridDescriptor_M_N(karg.M, karg.N, karg.StrideC);

        const auto KBatch_a = a_grid_desc_kbatch_k0_m_k1.GetLength(I0);
        const auto KBatch_b = b_grid_desc_kbatch_k0_n_k1.GetLength(I0);
        const auto K0_      = a_grid_desc_kbatch_k0_m_k1.GetLength(I1);
        const auto M_       = a_grid_desc_kbatch_k0_m_k1.GetLength(I2);
        const auto N_       = b_grid_desc_kbatch_k0_n_k1.GetLength(I2);

        return (M_ % MPerBlock == 0 && N_ % NPerBlock == 0 && K0_ % K0PerBlock == 0 &&
                M_ == c_grid_desc_m_n.GetLength(I0) && N_ == c_grid_desc_m_n.GetLength(I1) &&
                a_grid_desc_kbatch_k0_m_k1.GetLength(I3) ==
                    b_grid_desc_kbatch_k0_n_k1.GetLength(I3) &&
                karg.k_batch >= 1 && KBatch_a == karg.k_batch && KBatch_b == karg.k_batch);
    }

    // KBatch, K0, M, K1 -> KBatch, K0, M0, M1 (MPerBlock), K1
    __host__ __device__ static constexpr auto MakeAGridDescriptor_Kbatch_K0_M0_M1_K1(
        const AGridDesc_Kbatch_K0_M_K1& a_grid_desc_kbatch_k0_m_k1)
    {
        const auto KBatch = a_grid_desc_kbatch_k0_m_k1.GetLength(I0);
        const auto K0     = a_grid_desc_kbatch_k0_m_k1.GetLength(I1);
        const auto M      = a_grid_desc_kbatch_k0_m_k1.GetLength(I2);

        const auto M1 = Number<MPerBlock>{};
        const auto M0 = M / M1;

        const auto a_grid_desc_kbatch_k0_m0_m1_k1 = transform_tensor_descriptor(
            a_grid_desc_kbatch_k0_m_k1,
            make_tuple(make_pass_through_transform(KBatch),
                       make_pass_through_transform(K0),
                       make_unmerge_transform(make_tuple(M0, M1)),
                       make_pass_through_transform(K1)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),     // IP
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{})); // OP

        return a_grid_desc_kbatch_k0_m0_m1_k1;
    }

    __host__ __device__ static constexpr auto MakeBGridDescriptor_Kbatch_K0_N0_N1_K1(
        const BGridDesc_Kbatch_K0_N_K1& b_grid_desc_kbatch_k0_n_k1)
    {
        const auto KBatch = b_grid_desc_kbatch_k0_n_k1.GetLength(I0);
        const auto K0     = b_grid_desc_kbatch_k0_n_k1.GetLength(I1);
        const auto N      = b_grid_desc_kbatch_k0_n_k1.GetLength(I2);

        const auto N1 = Number<NPerBlock>{};
        const auto N0 = N / N1;

        const auto b_grid_desc_kbatch_k0_n0_n1_k1 = transform_tensor_descriptor(
            b_grid_desc_kbatch_k0_n_k1,
            make_tuple(make_pass_through_transform(KBatch),
                       make_pass_through_transform(K0),
                       make_unmerge_transform(make_tuple(N0, N1)),
                       make_pass_through_transform(K1)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}));

        return b_grid_desc_kbatch_k0_n0_n1_k1;
    }

    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_M0_M10_M11_N0_N10_N11(const CGridDesc_M_N& c_grid_desc_m_n)
    {
        const auto M = c_grid_desc_m_n.GetLength(I0);
        const auto N = c_grid_desc_m_n.GetLength(I1);

        constexpr auto M1 = Number<MPerBlock>{};
        constexpr auto N1 = Number<NPerBlock>{};

        const auto M0 = M / M1;
        const auto N0 = N / N1;

        constexpr auto M11 = Number<MPerThread>{};
        constexpr auto N11 = Number<NPerThread>{};

        constexpr auto M10 = M1 / M11;
        constexpr auto N10 = N1 / N11;

        const auto c_grid_desc_m0_m10_m11_n0_n10_n11 = transform_tensor_descriptor(
            c_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(M0, M10, M11)),
                       make_unmerge_transform(make_tuple(N0, N10, N11))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1, 2>{}, Sequence<3, 4, 5>{}));

        return c_grid_desc_m0_m10_m11_n0_n10_n11;
    }

    // return block_id to C matrix tile idx (m0, n0) mapping
    __host__ __device__ static constexpr auto MakeDefaultBlock2CTileMap()
    {
        //: 3d ksplit for C
        return BlockToCTileMap_3DGrid_KSplit<MPerBlock, NPerBlock>();
    }
    using DefaultBlock2CTileMap = remove_cvref_t<decltype(MakeDefaultBlock2CTileMap())>; //
    using AGridDesc_K0_M0_M1_K1 =
        decltype(MakeAGridDescriptor_Kbatch_K0_M0_M1_K1(AGridDesc_Kbatch_K0_M_K1{}));
    using BGridDesc_K0_N0_N1_K1 =
        decltype(MakeBGridDescriptor_Kbatch_K0_N0_N1_K1(BGridDesc_Kbatch_K0_N_K1{}));
    using CGridDesc_M0_M10_M11_N0_N10_N11 =
        decltype(MakeCGridDescriptor_M0_M10_M11_N0_N10_N11(CGridDesc_M_N{})); //
    using Block2CTileMap = decltype(MakeDefaultBlock2CTileMap());             //

    template <bool HasMainKBlockLoop,
              bool HasTripleTailKBlockLoop,
              typename GridwiseTsmm,
              InMemoryDataOperationEnum CGlobalMemoryDataOperation>
    __device__ static void Run(const Argument& karg)
    {
        constexpr index_t shared_block_size =
            GridwiseTsmm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

        __shared__ FloatAB p_shared_block[shared_block_size];

        const Block2CTileMap& block_2_ctile_map = Block2CTileMap{};

        const auto MPadded = CalculateMPadded(karg.M);
        const auto NPadded = CalculateNPadded(karg.N);

        const FloatAB* p_a_grid               = karg.p_a_grid;
        const FloatAB* p_b_grid               = karg.p_b_grid;
        FloatC* p_c_grid                      = karg.p_c_grid;
        const auto a_grid_desc_kbatch_k0_m_k1 = GridwiseTsmm::MakeAGridDescriptor_KBatch_K0_M_K1(
            karg.M, MPadded, karg.K, karg.StrideA, karg.k_batch, karg.K0); //
        const auto b_grid_desc_kbatch_k0_n_k1 = GridwiseTsmm::MakeBGridDescriptor_KBatch_K0_N_K1(
            karg.K, NPadded, karg.N, karg.StrideB, karg.k_batch, karg.K0); //
        const auto c_grid_desc_m_n =
            GridwiseTsmm::MakeCGridDescriptor_M_N(karg.M, karg.N, karg.StrideC);

        const auto a_grid_desc_kbatch_k0_m0_m1_k1 =
            GridwiseTsmm::MakeAGridDescriptor_Kbatch_K0_M0_M1_K1(a_grid_desc_kbatch_k0_m_k1); //
        const auto b_grid_desc_kbatch_k0_n0_n1_k1 =
            GridwiseTsmm::MakeBGridDescriptor_Kbatch_K0_N0_N1_K1(b_grid_desc_kbatch_k0_n_k1); //
        const auto c_grid_desc_m0_m10_m11_n0_n10_n11 =
            GridwiseTsmm::MakeCGridDescriptor_M0_M10_M11_N0_N10_N11(c_grid_desc_m_n);

        const auto a_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc_kbatch_k0_m0_m1_k1.GetElementSpaceSize());
        const auto b_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_grid_desc_kbatch_k0_n0_n1_k1.GetElementSpaceSize());
        ignore          = b_global_buf;
        auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_grid, c_grid_desc_m0_m10_m11_n0_n10_n11.GetElementSpaceSize());

        const auto c_m0_n0_block_cluster_idx = block_2_ctile_map.convert_1D_block_idx_to_3D_tuple(
            get_block_1d_id(), karg.N, karg.k_batch);

        // HACK: this force index data into SGPR
        const index_t im0       = __builtin_amdgcn_readfirstlane(c_m0_n0_block_cluster_idx[I0]);
        const index_t in0       = __builtin_amdgcn_readfirstlane(c_m0_n0_block_cluster_idx[I1]);
        const index_t kbatch_id = __builtin_amdgcn_readfirstlane(c_m0_n0_block_cluster_idx[I2]);

        if(!block_2_ctile_map.ValidCTileIndex(
               make_tuple(im0, in0),
               make_tuple(c_grid_desc_m0_m10_m11_n0_n10_n11.GetLength(I0),
                          c_grid_desc_m0_m10_m11_n0_n10_n11.GetLength(I3))))
        {
            return;
        }

        // TODO: change this. I think it needs multi-dimensional alignment
        constexpr auto max_lds_align = K1;

        constexpr auto a_block_desc_copy_kbatch_k0_m0_m1_k1 = make_naive_tensor_descriptor_aligned(
            make_tuple(I1, Number<K0PerBlock>{}, I1, Number<MPerBlock>{}, K1), max_lds_align);

        // A matrix blockwise copy
        auto a_blockwise_copy = BlockwiseTensorSliceTransfer_v5r1<
            BlockSize,
            InMemoryDataOperationEnum::Set,
            Sequence<1, K0PerBlock, 1, MPerBlock, K1.value>, //: 5 dimensions; kbatch for each
                                                             // dimension is 1
            ABlockTransferThreadSliceLengths_KBatch_K0_M0_M1_K1,
            ABlockTransferThreadClusterLengths_KBatch_K0_M0_M1_K1,
            ABlockTransferThreadClusterArrangeOrder, // 0, 1, 2, 3, 4
            FloatAB,
            FloatAB,
            remove_reference_t<decltype(a_grid_desc_kbatch_k0_m0_m1_k1)>, // Global tensor desc
            decltype(a_block_desc_copy_kbatch_k0_m0_m1_k1),               // block tensor desc
            ABlockTransferSrcAccessOrder,                                 // 5-dim
            Sequence<0, 1, 2, 3, 4>,
            ABlockTransferSrcVectorTensorLengths_KBatch_K0_M0_M1_K1, // SrcVectorTensorLengths
            ABlockTransferDstVectorTensorLengths_KBatch_K0_M0_M1_K1, // DstVectorTensorLengths
            ABlockTransferSrcVectorTensorContiguousDimOrder, // SrcVectorTensorContiguousDimOrder
            Sequence<0, 1, 2, 3, 4>,                         // DstVectorTensorContiguousDimOrder
            false,
            true>(a_grid_desc_kbatch_k0_m0_m1_k1,            // for src desc
                  make_multi_index(kbatch_id, 0, im0, 0, 0), //: calculate start index of K
                  a_block_desc_copy_kbatch_k0_m0_m1_k1,      // for dst desc
                  make_multi_index(0, 0, 0, 0, 0));

        static constexpr auto b_thread_desc_copy_kbatch_k0_n0_n1_k1 =
            make_naive_tensor_descriptor_packed(
                make_tuple(I1,
                           Number<K0PerBlock>{},
                           I1,
                           Number<NPerThread>{},
                           Number<K1>{})); //: this descriptor is used only for copy

        static constexpr auto b_thread_desc_copy_k0_n0_n1_k1 = make_naive_tensor_descriptor_packed(
            make_tuple(I1, Number<K0PerBlock>{}, I1, Number<NPerThread>{}, Number<K1>{}));

        auto b_threadwise_copy = ThreadwiseTensorSliceTransfer_v2<
            FloatAB,
            FloatAB,
            remove_reference_t<decltype(b_grid_desc_kbatch_k0_n0_n1_k1)>,
            decltype(b_thread_desc_copy_kbatch_k0_n0_n1_k1), //
            Sequence<1, K0PerBlock, 1, NPerThread, K1.value>,
            BThreadTransferSrcDstAccessOrder,
            BThreadTransferSrcVectorDim,
            BThreadTransferSrcScalarPerVector,
            1,
            false,
            true>(b_grid_desc_kbatch_k0_n0_n1_k1,
                  make_multi_index(kbatch_id, 0, in0, get_thread_local_1d_id() * NPerThread, 0));

        static constexpr auto b_k0_n_k1_thread_desc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<K0PerBlock>{}, Number<NPerThread>{}, Number<K1>{}));

        // TODO: check alignment
        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_block_desc_k0_m0_m1_k1 = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<K0PerBlock>{}, I1, Number<MPerBlock>{}, K1), max_lds_align);

        // TODO: check alignment
        // A matrix in LDS memory, for blockwise GEMM
        constexpr auto a_k0_m_k1_block_desc = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<K0PerBlock>{}, Number<MPerBlock>{}, K1), max_lds_align);

        static_assert(a_block_desc_k0_m0_m1_k1.GetElementSpaceSize() ==
                          a_k0_m_k1_block_desc.GetElementSpaceSize() &&
                      "wrong!");

        const auto blockwise_tsmm =
            BlockwiseGemmDlops_km_kn_m0m1n0n1_v3<BlockSize,
                                                 FloatAB,
                                                 FloatAB,
                                                 FloatAcc,
                                                 decltype(a_k0_m_k1_block_desc),
                                                 decltype(b_k0_n_k1_thread_desc),
                                                 MPerThread,
                                                 NPerBlock,
                                                 KPerThread>{};

        constexpr auto c_m10_m11_n10_n11_thread_tensor_lengths =
            decltype(blockwise_tsmm)::GetCThreadTensorLengths_BM0_BM1_BN0_BN1();

        constexpr auto c_thread_desc_m10_m11_n10_n11 = make_naive_tensor_descriptor_packed(
            sequence_to_tuple_of_number(c_m10_m11_n10_n11_thread_tensor_lengths));

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_aligned_space_size = math::integer_least_multiple(
            a_block_desc_k0_m0_m1_k1.GetElementSpaceSize(), max_lds_align);

        FloatAB* p_a_block_triple = p_shared_block;

        auto b_thread_buf1 = make_static_buffer<AddressSpaceEnum::Vgpr, FloatAB>(
            b_k0_n_k1_thread_desc.GetElementSpaceSize());

        auto b_thread_buf2 = make_static_buffer<AddressSpaceEnum::Vgpr, FloatAB>(
            b_k0_n_k1_thread_desc.GetElementSpaceSize());

        auto b_thread_buf3 = make_static_buffer<AddressSpaceEnum::Vgpr, FloatAB>(
            b_k0_n_k1_thread_desc.GetElementSpaceSize());

        // register allocation for output
        auto c_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatAcc>(
            c_thread_desc_m10_m11_n10_n11.GetElementSpaceSize());

        // Initialize C
        c_thread_buf.Clear();

        constexpr auto a_block_slice_copy_step  = make_multi_index(0, K0PerBlock, 0, 0, 0);
        constexpr auto b_thread_slice_copy_step = make_multi_index(0, K0PerBlock, 0, 0, 0);

        auto a_block_buf1 = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            p_a_block_triple, a_block_desc_copy_kbatch_k0_m0_m1_k1.GetElementSpaceSize());

        auto a_block_buf2 = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            p_a_block_triple + a_block_aligned_space_size,
            a_block_desc_copy_kbatch_k0_m0_m1_k1.GetElementSpaceSize());

        auto a_block_buf3 = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            p_a_block_triple + 2 * a_block_aligned_space_size,
            a_block_desc_copy_kbatch_k0_m0_m1_k1.GetElementSpaceSize());

        // LDS triple buffer: preload data into LDS
        {
            a_blockwise_copy.RunRead(a_grid_desc_kbatch_k0_m0_m1_k1,
                                     a_global_buf); // a_global_buf -> reg_tmp_buf
            a_blockwise_copy.RunWrite(a_block_desc_copy_kbatch_k0_m0_m1_k1,
                                      a_block_buf1); // reg_tmp_buf->a_block_buf1

            b_threadwise_copy.Run(b_grid_desc_kbatch_k0_n0_n1_k1,
                                  b_global_buf,
                                  b_thread_desc_copy_k0_n0_n1_k1,
                                  make_tuple(I0, I0, I0, I0, I0),
                                  b_thread_buf1);
        }

        if constexpr(HasMainKBlockLoop)
        {
            const auto K0 = a_grid_desc_kbatch_k0_m0_m1_k1.GetLength(I1);

            index_t k_block_data_begin = 0;

            // LDS triple buffer: main body
            // use Do-While loop instead of For loop to simplify control flow
            do
            {
                #pragma unroll 2
                for (int unroll_idx = 0; unroll_idx < 2; ++unroll_idx)
                {
                    // First iteration
                    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc_kbatch_k0_m0_m1_k1,
                                                        a_block_slice_copy_step);

                    b_threadwise_copy.MoveSrcSliceWindow(b_grid_desc_kbatch_k0_n0_n1_k1,
                                                        b_thread_slice_copy_step);

                    // LDS triple buffer: load next data from device mem
                    a_blockwise_copy.RunRead(a_grid_desc_kbatch_k0_m0_m1_k1, a_global_buf);

                    b_threadwise_copy.Run(b_grid_desc_kbatch_k0_n0_n1_k1,
                                        b_global_buf,
                                        b_thread_desc_copy_k0_n0_n1_k1,
                                        make_tuple(I0, I0, I0, I0, I0),
                                        b_thread_buf2);

                    block_sync_lds();

                    // LDS triple buffer: GEMM on current data
                    blockwise_tsmm.Run(a_block_buf1, b_thread_buf1, c_thread_buf);

                    // LDS triple buffer: store next data to LDS
                    a_blockwise_copy.RunWrite(a_block_desc_copy_kbatch_k0_m0_m1_k1, a_block_buf2);

                    // Second iteration
                    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc_kbatch_k0_m0_m1_k1,
                                                        a_block_slice_copy_step);

                    b_threadwise_copy.MoveSrcSliceWindow(b_grid_desc_kbatch_k0_n0_n1_k1,
                                                        b_thread_slice_copy_step);

                    // LDS triple buffer: load next data from device mem
                    a_blockwise_copy.RunRead(a_grid_desc_kbatch_k0_m0_m1_k1, a_global_buf);

                    b_threadwise_copy.Run(b_grid_desc_kbatch_k0_n0_n1_k1,
                                        b_global_buf,
                                        b_thread_desc_copy_k0_n0_n1_k1,
                                        make_tuple(I0, I0, I0, I0, I0),
                                        b_thread_buf3);

                    block_sync_lds();

                    // LDS triple buffer: GEMM on current data
                    blockwise_tsmm.Run(a_block_buf2, b_thread_buf2, c_thread_buf);

                    // LDS triple buffer: store next data to LDS
                    a_blockwise_copy.RunWrite(a_block_desc_copy_kbatch_k0_m0_m1_k1, a_block_buf3);

                    // Third iteration
                    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc_kbatch_k0_m0_m1_k1,
                                                        a_block_slice_copy_step);

                    b_threadwise_copy.MoveSrcSliceWindow(b_grid_desc_kbatch_k0_n0_n1_k1,
                                                        b_thread_slice_copy_step);

                    // LDS triple buffer: load next data from device mem
                    a_blockwise_copy.RunRead(a_grid_desc_kbatch_k0_m0_m1_k1, a_global_buf);

                    b_threadwise_copy.Run(b_grid_desc_kbatch_k0_n0_n1_k1,
                                        b_global_buf,
                                        b_thread_desc_copy_k0_n0_n1_k1,
                                        make_tuple(I0, I0, I0, I0, I0),
                                        b_thread_buf1);

                    block_sync_lds();

                    // LDS triple buffer: GEMM on current data
                    blockwise_tsmm.Run(a_block_buf3, b_thread_buf3, c_thread_buf);

                    // LDS triple buffer: store next data to LDS
                    a_blockwise_copy.RunWrite(a_block_desc_copy_kbatch_k0_m0_m1_k1, a_block_buf1);

                    k_block_data_begin += 3 * K0PerBlock;
                }
            } while(k_block_data_begin < K0 - 6 * K0PerBlock);
        }


        // LDS triple buffer: tail
        if constexpr(HasTripleTailKBlockLoop) // if has 3 iterations left
        {
            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc_kbatch_k0_m0_m1_k1,
                                                a_block_slice_copy_step);

            b_threadwise_copy.MoveSrcSliceWindow(b_grid_desc_kbatch_k0_n0_n1_k1,
                                                 b_thread_slice_copy_step);

            block_sync_lds();

            // LDS triple buffer: load second last data from device mem
            a_blockwise_copy.RunRead(a_grid_desc_kbatch_k0_m0_m1_k1, a_global_buf);

            b_threadwise_copy.Run(b_grid_desc_kbatch_k0_n0_n1_k1,
                                  b_global_buf,
                                  b_thread_desc_copy_k0_n0_n1_k1,
                                  make_tuple(I0, I0, I0, I0, I0),
                                  b_thread_buf2);

            // LDS triple buffer: GEMM on 3rd last data
            blockwise_tsmm.Run(a_block_buf1, b_thread_buf1, c_thread_buf);

            // LDS triple buffer: store second last data to LDS
            a_blockwise_copy.RunWrite(a_block_desc_copy_kbatch_k0_m0_m1_k1, a_block_buf2);

            block_sync_lds();

            // LDS triple buffer: load last data from device mem
            a_blockwise_copy.RunRead(a_grid_desc_kbatch_k0_m0_m1_k1, a_global_buf);

            b_threadwise_copy.Run(b_grid_desc_kbatch_k0_n0_n1_k1,
                                  b_global_buf,
                                  b_thread_desc_copy_k0_n0_n1_k1,
                                  make_tuple(I0, I0, I0, I0, I0),
                                  b_thread_buf3);

            // LDS triple buffer: GEMM on 2nd last data
            blockwise_tsmm.Run(a_block_buf2, b_thread_buf2, c_thread_buf);

            // LDS triple buffer: store last data to LDS
            a_blockwise_copy.RunWrite(a_block_desc_copy_kbatch_k0_m0_m1_k1, a_block_buf3);

            block_sync_lds();

            // LDS triple buffer: GEMM on last data
            blockwise_tsmm.Run(a_block_buf3, b_thread_buf3, c_thread_buf);
        }
        else // if has less than 3 iterations left
        {
            __syncthreads();

            // LDS triple buffer: GEMM on remaining data
            blockwise_tsmm.Run(a_block_buf1, b_thread_buf1, c_thread_buf);
        }

        // output: register to global memory
        {
            constexpr auto c_thread_desc_m0_m10_m11_n0_n10_n11 =
                make_naive_tensor_descriptor_packed(
                    make_tuple(I1,
                               Number<c_m10_m11_n10_n11_thread_tensor_lengths[I0]>{},
                               Number<c_m10_m11_n10_n11_thread_tensor_lengths[I1]>{},
                               I1,
                               Number<c_m10_m11_n10_n11_thread_tensor_lengths[I2]>{},
                               Number<c_m10_m11_n10_n11_thread_tensor_lengths[I3]>{}));

            const auto c_m10_m11_n10_n11_thread_origin_idx_on_block =
                blockwise_tsmm.CalculateCThreadOriginOnBlock_BM0_BM1_BN0_BN1(
                    get_thread_local_1d_id());

            ThreadwiseTensorSliceTransfer_v1r3<
                FloatAcc,
                FloatC,
                decltype(c_thread_desc_m0_m10_m11_n0_n10_n11),
                decltype(c_grid_desc_m0_m10_m11_n0_n10_n11),
                ck::tensor_operation::element_wise::PassThrough,
                Sequence<1,
                         c_m10_m11_n10_n11_thread_tensor_lengths[I0],
                         c_m10_m11_n10_n11_thread_tensor_lengths[I1],
                         1,
                         c_m10_m11_n10_n11_thread_tensor_lengths[I2],
                         c_m10_m11_n10_n11_thread_tensor_lengths[I3]>,
                CThreadTransferSrcDstAccessOrder,
                CThreadTransferSrcDstVectorDim,
                CThreadTransferDstScalarPerVector,
                CGlobalMemoryDataOperation,
                1,
                true>{c_grid_desc_m0_m10_m11_n0_n10_n11,
                      make_multi_index(im0,
                                       c_m10_m11_n10_n11_thread_origin_idx_on_block[I0],
                                       c_m10_m11_n10_n11_thread_origin_idx_on_block[I1],
                                       in0,
                                       c_m10_m11_n10_n11_thread_origin_idx_on_block[I2],
                                       c_m10_m11_n10_n11_thread_origin_idx_on_block[I3]),
                      ck::tensor_operation::element_wise::PassThrough{}}
                .Run(c_thread_desc_m0_m10_m11_n0_n10_n11,
                     make_tuple(I0, I0, I0, I0, I0, I0),
                     c_thread_buf,
                     c_grid_desc_m0_m10_m11_n0_n10_n11,
                     c_grid_buf);
        }
    }
};
} // namespace ck
