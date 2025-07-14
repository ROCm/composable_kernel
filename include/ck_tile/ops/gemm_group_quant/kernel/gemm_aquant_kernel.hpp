// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/host/concat.hpp"

namespace ck_tile {

struct AQuantGemmProblem
{
    CK_TILE_HOST AQuantGemmProblem() = default;
    CK_TILE_HOST AQuantGemmProblem(index_t M_,
                                   index_t N_,
                                   index_t K_,
                                   index_t QK_,
                                   index_t stride_A_,
                                   index_t stride_B_,
                                   index_t stride_C_,
                                   index_t stride_AQ_)
        : M(M_),
          N(N_),
          K(K_),
          QK(QK_),
          stride_A(stride_A_),
          stride_B(stride_B_),
          stride_C(stride_C_),
          stride_AQ(stride_AQ_)
    {
    }

    index_t M;
    index_t N;
    index_t K;
    index_t QK;
    index_t stride_A;
    index_t stride_B;
    index_t stride_C;
    index_t stride_AQ;
};

struct AQuantGemmHostArgs : public AQuantGemmProblem
{
    CK_TILE_HOST AQuantGemmHostArgs() = default;
    CK_TILE_HOST AQuantGemmHostArgs(const void* a_ptr_,
                                    const void* b_ptr_,
                                    void* c_ptr_,
                                    const void* aq_ptr_,
                                    index_t k_batch_,
                                    index_t M_,
                                    index_t N_,
                                    index_t K_,
                                    index_t QK_,
                                    index_t stride_A_,
                                    index_t stride_B_,
                                    index_t stride_C_,
                                    index_t stride_AQ_)
        : AQuantGemmProblem(M_, N_, K_, QK_, stride_A_, stride_B_, stride_C_, stride_AQ_),
          a_ptr(a_ptr_),
          b_ptr(b_ptr_),
          aq_ptr(aq_ptr_),
          c_ptr(c_ptr_),
          k_batch(k_batch_)
    {
    }

    const void* a_ptr;
    const void* b_ptr;
    const void* aq_ptr;
    void* c_ptr;
    index_t k_batch;
};

struct AQuantGemmKernelArgs
{
    const void* a_ptr;
    const void* b_ptr;
    const void* aq_ptr;
    void* c_ptr;
    index_t M;
    index_t N;
    index_t K;
    index_t QK;
    index_t stride_A;
    index_t stride_B;
    index_t stride_C;
    index_t stride_AQ;
    index_t k_batch;
};

template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct AQuantGemmKernel
{
    using TilePartitioner                    = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline                       = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline                   = remove_cvref_t<EpiloguePipeline_>;
    using ALayout                            = remove_cvref_t<typename GemmPipeline::ALayout>;
    using AQLayout                           = remove_cvref_t<typename GemmPipeline::AQLayout>;
    using BLayout                            = remove_cvref_t<typename GemmPipeline::BLayout>;
    using CLayout                            = remove_cvref_t<typename GemmPipeline::CLayout>;
    static constexpr index_t KernelBlockSize = GemmPipeline::BlockSize;

    using ADataType  = remove_cvref_t<typename GemmPipeline::ADataType>;
    using AQDataType = remove_cvref_t<typename GemmPipeline::AQDataType>;
    using BDataType  = remove_cvref_t<typename GemmPipeline::BDataType>;
    using CDataType  = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    static constexpr auto I0 = number<0>();
    static constexpr auto I1 = number<1>();
    static constexpr auto I2 = number<2>();
    static constexpr auto I3 = number<3>();

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "gemm", gemm_prec_str<ADataType, BDataType>, GemmPipeline::GetName());
        // clang-format on
    }

    CK_TILE_HOST static constexpr auto GridSize(index_t M, index_t N, index_t KBatch)
    {
        return dim3(TilePartitioner::GridSize(M, N), 1, KBatch);
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(KernelBlockSize); }

    CK_TILE_HOST static constexpr AQuantGemmKernelArgs
    MakeKernelArgs(const AQuantGemmHostArgs& hostArgs)
    {
        return AQuantGemmKernelArgs{hostArgs.a_ptr,
                                    hostArgs.b_ptr,
                                    hostArgs.aq_ptr,
                                    hostArgs.c_ptr,
                                    hostArgs.M,
                                    hostArgs.N,
                                    hostArgs.K,
                                    hostArgs.QK,
                                    hostArgs.stride_A,
                                    hostArgs.stride_B,
                                    hostArgs.stride_C,
                                    hostArgs.stride_AQ,
                                    hostArgs.k_batch};
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    struct SplitKBatchOffset
    {
        __device__ SplitKBatchOffset(const AQuantGemmKernelArgs& kargs,
                                     const std::size_t k_id = blockIdx.z)
        {
            constexpr auto K1   = TilePartitioner::BlockGemmShape::WarpTile::at(number<2>{});
            const index_t K_t   = __builtin_amdgcn_readfirstlane(kargs.k_batch * K1);
            const index_t KRead = __builtin_amdgcn_readfirstlane((kargs.K + K_t - 1) / K_t * K1);

            if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                a_k_split_offset = __builtin_amdgcn_readfirstlane(k_id * KRead);
            }
            else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                a_k_split_offset = __builtin_amdgcn_readfirstlane(k_id * KRead * kargs.stride_A);
            }

            if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, BLayout>)
            {
                b_k_split_offset = __builtin_amdgcn_readfirstlane(k_id * KRead * kargs.stride_B);
            }
            else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, BLayout>)
            {
                b_k_split_offset = __builtin_amdgcn_readfirstlane(k_id * KRead);
            }

            if(k_id < static_cast<uint32_t>(kargs.k_batch - 1))
            {
                splitted_k = __builtin_amdgcn_readfirstlane(KRead);
            }
            else
            {
                splitted_k = __builtin_amdgcn_readfirstlane(kargs.K - KRead * (kargs.k_batch - 1));
            }
        }

        index_t a_k_split_offset;
        index_t b_k_split_offset;
        index_t splitted_k;
    };

    CK_TILE_HOST static bool IsSupportedArgument(const AQuantGemmKernelArgs& kargs)
    {
        if(kargs.k_batch != 1)
        {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR("Conditions not met for Kbatch >1 !");
            }
            return false;
        }

        static_assert(std::is_same_v<AQLayout, tensor_layout::gemm::RowMajor>);
        if(kargs.QK % GemmPipeline::GetVectorSizeAQ() != 0)
        {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR("K is not a multiple of vector load size for A tensor!");
            }
            return false;
        }

        if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.K % (TilePartitioner::KPerBlock * kargs.k_batch) != 0 &&
               GemmPipeline::kPadK == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("Can't support K that is not a multiple of k_batch * KPerBlock "
                                  "without padding!");
                }
                return false;
            }
            if(kargs.K % GemmPipeline::GetVectorSizeA() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("K is not a multiple of vector load size for A tensor!");
                }
                return false;
            }
        }
        else
        {
            if(kargs.M % TilePartitioner::MPerBlock != 0 && GemmPipeline::kPadM == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Can't support M that is not a multiple of MPerBlock without padding!");
                }
                return false;
            }
            if(kargs.M % GemmPipeline::GetVectorSizeA() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("M is not a multiple of vector load size for A tensor!");
                }
                return false;
            }
        }

        if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.N % TilePartitioner::NPerBlock != 0 && GemmPipeline::kPadN == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Can't support N that is not a multiple of NPerBlock without padding!");
                }
                return false;
            }
            if(kargs.N % GemmPipeline::GetVectorSizeB() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("N is not a multiple of vector load size for B tensor!");
                }
                return false;
            }
        }
        else
        {
            if(kargs.K % (TilePartitioner::KPerBlock * kargs.k_batch) != 0 &&
               GemmPipeline::kPadK == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("Can't support K that is not a multiple of k_batch * KPerBlock "
                                  "without padding!");
                }
                return false;
            }
            if(kargs.K % GemmPipeline::GetVectorSizeB() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("K is not a multiple of vector load size for B tensor!");
                }
                return false;
            }
        }

        if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.N % TilePartitioner::NPerBlock != 0 && GemmPipeline::kPadN == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Can't support N that is not a multiple of NPerBlock without padding!");
                }
                return false;
            }
            if(kargs.N % EpiloguePipeline::GetVectorSizeC() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("N is not a multiple of vector load size for C tensor!");
                }
                return false;
            }
        }
        else
        {
            if(kargs.M % TilePartitioner::MPerBlock != 0 && GemmPipeline::kPadM == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Can't support M that is not a multiple of MPerBlock without padding!");
                }
                return false;
            }
            if(kargs.M % EpiloguePipeline::GetVectorSizeC() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("M is not a multiple of vector load size for C tensor!");
                }
                return false;
            }
        }
        return true;
    }

    template <memory_operation_enum DstInMemOp = memory_operation_enum::set>
    CK_TILE_DEVICE static auto MakeGemmTensorViews(const ADataType* a_ptr,
                                                   const BDataType* b_ptr,
                                                   const AQDataType* aq_ptr,
                                                   CDataType* c_ptr,
                                                   const AQuantGemmKernelArgs& kargs,
                                                   const SplitKBatchOffset& splitk_batch_offset)
    {
        static_assert(!TilePartitioner::BlockGemmShape::PermuteA, "Not implemented!");
        const auto& a_tensor_view = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_ptr,
                    make_tuple(kargs.M, splitk_batch_offset.splitted_k),
                    make_tuple(kargs.stride_A, 1),
                    number<GemmPipeline::GetVectorSizeA()>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_ptr,
                    make_tuple(splitk_batch_offset.splitted_k, kargs.M),
                    make_tuple(kargs.stride_A, 1),
                    number<GemmPipeline::GetVectorSizeA()>{},
                    number<1>{});
            }
        }();

        const auto& aq_tensor_view = [&]() {
            static_assert(std::is_same_v<AQLayout, tensor_layout::gemm::RowMajor>);
            return make_naive_tensor_view<address_space_enum::global>(
                aq_ptr,
                make_tuple(kargs.M, kargs.QK),
                make_tuple(kargs.stride_AQ, 1),
                number<GemmPipeline::GetVectorSizeAQ()>{},
                number<1>{});
        }();

        const auto& b_tensor_view = [&]() {
            if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
            {
                if constexpr(TilePartitioner::BlockGemmShape::PermuteB)
                {
                    constexpr index_t K1          = GemmPipeline::GetSmemPackB();
                    const index_t K0              = splitk_batch_offset.splitted_k / K1;
                    constexpr index_t VectorSizeB = std::min(K1, GemmPipeline::GetVectorSizeB());
                    const auto b_k0_n_k1_desc =
                        make_naive_tensor_descriptor(make_tuple(K0, kargs.N, K1),
                                                     make_tuple(kargs.N * K1, K1, I1),
                                                     number<VectorSizeB>{},
                                                     number<1>{});
                    const auto b_n_k_desc = transform_tensor_descriptor(
                        b_k0_n_k1_desc,
                        make_tuple(make_merge_transform(make_tuple(K0, K1)),
                                   make_pass_through_transform(kargs.N)),
                        make_tuple(sequence<0, 2>{}, sequence<1>{}),
                        make_tuple(sequence<0>{}, sequence<1>{}));
                    return make_tensor_view<address_space_enum::global>(b_ptr, b_n_k_desc);
                }
                else
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        b_ptr,
                        make_tuple(splitk_batch_offset.splitted_k, kargs.N),
                        make_tuple(kargs.stride_B, 1),
                        number<GemmPipeline::GetVectorSizeB()>{},
                        number<1>{});
                }
            }
            else
            {
                if constexpr(TilePartitioner::BlockGemmShape::PermuteB)
                {
                    constexpr index_t K1          = GemmPipeline::GetSmemPackB();
                    const index_t K0              = splitk_batch_offset.splitted_k / K1;
                    constexpr index_t VectorSizeB = std::min(K1, GemmPipeline::GetVectorSizeB());
                    const auto b_k0_n_k1_desc =
                        make_naive_tensor_descriptor(make_tuple(K0, kargs.N, K1),
                                                     make_tuple(kargs.N * K1, K1, I1),
                                                     number<VectorSizeB>{},
                                                     number<1>{});
                    const auto b_n_k_desc = transform_tensor_descriptor(
                        b_k0_n_k1_desc,
                        make_tuple(make_merge_transform(make_tuple(K0, K1)),
                                   make_pass_through_transform(kargs.N)),
                        make_tuple(sequence<0, 2>{}, sequence<1>{}),
                        make_tuple(sequence<1>{}, sequence<0>{}));
                    return make_tensor_view<address_space_enum::global>(b_ptr, b_n_k_desc);
                }
                else
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        b_ptr,
                        make_tuple(kargs.N, splitk_batch_offset.splitted_k),
                        make_tuple(kargs.stride_B, 1),
                        number<GemmPipeline::GetVectorSizeB()>{},
                        number<1>{});
                }
            }
        }();

        // TODO: enable vector write for C in ColMajor
        const auto& c_tensor_view = [&]() {
            if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    c_ptr,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(kargs.stride_C, 1),
                    number<EpiloguePipeline::GetVectorSizeC()>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    c_ptr,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(1, kargs.stride_C),
                    number<1>{},
                    number<1>{});
            }
        }();

        return make_tuple(a_tensor_view, aq_tensor_view, b_tensor_view, c_tensor_view);
    }

    template <typename TensorView>
    CK_TILE_DEVICE static auto MakeGemmPadViews(const TensorView& views)
    {
        const auto& a_pad_view = [&]() {
            const auto& a_tensor_view = views.at(I0);
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(a_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::KPerBlock>{}),
                                       sequence<false, GemmPipeline::kPadK>{});
            }
            else
            {
                return pad_tensor_view(a_tensor_view,
                                       make_tuple(number<TilePartitioner::KPerBlock>{},
                                                  number<TilePartitioner::MPerBlock>{}),
                                       sequence<false, GemmPipeline::kPadM>{});
            }
        }();

        const auto& aq_pad_view = [&]() {
            const auto& aq_tensor_view = views.at(I1);
            static_assert(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>);
            return pad_tensor_view(
                aq_tensor_view,
                make_tuple(number<TilePartitioner::MPerBlock>{},
                           number<TilePartitioner::KPerBlock / GemmPipeline::QuantGroupSize>{}),
                // TODO: Add support for padding.
                sequence<false, false>{});
        }();

        const auto& b_pad_view = [&]() {
            const auto& b_tensor_view = views.at(I2);
            if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>)
            {
                return pad_tensor_view(b_tensor_view,
                                       make_tuple(number<TilePartitioner::NPerBlock>{},
                                                  number<TilePartitioner::KPerBlock>{}),
                                       sequence<false, GemmPipeline::kPadK>{});
            }
            else
            {
                return pad_tensor_view(b_tensor_view,
                                       make_tuple(number<TilePartitioner::KPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<false, GemmPipeline::kPadN>{});
            }
        }();

        // TODO vector write in for C in ColMajor
        const auto& c_pad_view = [&]() {
            const auto& c_tensor_view = views.at(I3);
            if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(c_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<false, GemmPipeline::kPadN>{});
            }
            else
            {
                return pad_tensor_view(c_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<GemmPipeline::kPadM, false>{});
            }
        }();

        return make_tuple(a_pad_view, aq_pad_view, b_pad_view, c_pad_view);
    }

    template <typename PadView>
    CK_TILE_DEVICE static auto
    MakeGemmTileWindows(const PadView& views, const index_t i_m, const index_t i_n)
    {
        const auto& a_pad_view  = views.at(I0);
        const auto& aq_pad_view = views.at(I1);
        const auto& b_pad_view  = views.at(I2);
        const auto& c_pad_view  = views.at(I3);

        const auto& a_block_window = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_tile_window(a_pad_view,
                                        make_tuple(number<TilePartitioner::MPerBlock>{},
                                                   number<TilePartitioner::KPerBlock>{}),
                                        {i_m, 0});
            }
            else
            {
                return make_tile_window(a_pad_view,
                                        make_tuple(number<TilePartitioner::KPerBlock>{},
                                                   number<TilePartitioner::MPerBlock>{}),
                                        {0, i_m});
            }
        }();

        const auto& aq_block_window = [&]() {
            static_assert(std::is_same_v<AQLayout, tensor_layout::gemm::RowMajor>);
            return make_tile_window(
                aq_pad_view,
                make_tuple(number<TilePartitioner::MPerBlock>{},
                           number<TilePartitioner::KPerBlock / GemmPipeline::QuantGroupSize>{}),
                {i_m, 0});
        }();

        const auto& b_block_window = [&]() {
            if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>)
            {
                return make_tile_window(b_pad_view,
                                        make_tuple(number<TilePartitioner::NPerBlock>{},
                                                   number<TilePartitioner::KPerBlock>{}),
                                        {i_n, 0});
            }
            else
            {
                return make_tile_window(b_pad_view,
                                        make_tuple(number<TilePartitioner::KPerBlock>{},
                                                   number<TilePartitioner::NPerBlock>{}),
                                        {0, i_n});
            }
        }();

        auto c_block_window = make_tile_window(
            c_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {i_m, i_n});

        return make_tuple(a_block_window, aq_block_window, b_block_window, c_block_window);
    }

    /**
     * @brief Runs single GEMM problem cooperatively by whole workgroup.
     *
     * @param a_ptr input A pointer
     * @param b_ptr input B pointer
     * @param aq_ptr input AQ pointer
     * @param c_ptr output C pointer
     * @param smem_ptr_0 The start memory pointer of the shared memory block.
     * @param kargs GEMM kernel arguments
     * @param splitk_batch_offset splitk_batch_offset Utility structure used to calculate k batch.
     * @param block_idx_m The GEMM's output M dimension tile index processed by this workgroup.
     * @param block_idx_n The GEMM's output N dimension tile index processed by this workgroup.
     *
     * @tparam DstInMemOp Destination memory operation (default: set).
     */
    template <memory_operation_enum DstInMemOp = memory_operation_enum::set>
    CK_TILE_DEVICE static void RunGemm(const ADataType* a_ptr,
                                       const BDataType* b_ptr,
                                       const AQDataType* aq_ptr,
                                       CDataType* c_ptr,
                                       void* smem_ptr_0,
                                       const AQuantGemmKernelArgs& kargs,
                                       const SplitKBatchOffset& splitk_batch_offset,
                                       const index_t block_idx_m,
                                       const index_t block_idx_n)
    {
        // Create Gemm tensor views, pad views and tile windows
        const auto& gemm_tensor_views_tuple = MakeGemmTensorViews<DstInMemOp>(
            a_ptr, b_ptr, aq_ptr, c_ptr, kargs, splitk_batch_offset);

        const auto& gemm_pad_views = MakeGemmPadViews(gemm_tensor_views_tuple);
        auto gemm_tile_windows     = MakeGemmTileWindows(gemm_pad_views, block_idx_m, block_idx_n);

        const index_t num_loop = __builtin_amdgcn_readfirstlane(
            TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k));

        // Run GEMM cooperatively by whole workgroup.
        const auto& a_block_window  = gemm_tile_windows.at(I0);
        const auto& aq_block_window = gemm_tile_windows.at(I1);
        const auto& b_block_window  = gemm_tile_windows.at(I2);

        const auto& c_block_tile = GemmPipeline{}.template operator()(
            a_block_window, b_block_window, aq_block_window, num_loop, smem_ptr_0);

        // Run Epilogue Pipeline
        auto& c_block_window = gemm_tile_windows.at(I3);

        EpiloguePipeline{}.template
        operator()<decltype(c_block_window), decltype(c_block_tile), decltype(c_block_window)>(
            c_block_window, c_block_tile, c_block_window, smem_ptr_0);
    }

    CK_TILE_DEVICE void operator()(AQuantGemmKernelArgs kargs) const
    {
        const auto blockId  = __builtin_amdgcn_readfirstlane(blockIdx.x);
        const auto [iM, iN] = TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(blockId);
        const index_t i_m   = __builtin_amdgcn_readfirstlane(iM * TilePartitioner::MPerBlock);
        const index_t i_n   = __builtin_amdgcn_readfirstlane(iN * TilePartitioner::NPerBlock);

        const SplitKBatchOffset splitk_batch_offset(kargs);
        // options
        const ADataType* a_ptr   = static_cast<const ADataType*>(kargs.a_ptr);
        const BDataType* b_ptr   = static_cast<const BDataType*>(kargs.b_ptr);
        const AQDataType* aq_ptr = static_cast<const AQDataType*>(kargs.aq_ptr);
        CDataType* c_ptr         = static_cast<CDataType*>(kargs.c_ptr);

        // allocate LDS
        __shared__ char smem_ptr_0[GetSmemSize()];

        assert(kargs.k_batch == 1);
        RunGemm(a_ptr, b_ptr, aq_ptr, c_ptr, smem_ptr_0, kargs, splitk_batch_offset, i_m, i_n);
    }
};

} // namespace ck_tile
