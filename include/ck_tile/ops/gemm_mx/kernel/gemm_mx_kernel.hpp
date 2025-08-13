// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/host/concat.hpp"

namespace ck_tile {

struct GemmMXProblem
{
    CK_TILE_HOST GemmMXProblem() = default;
    CK_TILE_HOST GemmMXProblem(index_t M_,
                               index_t N_,
                               index_t K_,
                               index_t stride_A_,
                               index_t stride_B_,
                               index_t stride_C_,
                               index_t stride_scale_A_,
                               index_t stride_scale_B_)
        : M(M_),
          N(N_),
          K(K_),
          stride_A(stride_A_),
          stride_B(stride_B_),
          stride_C(stride_C_),
          stride_scale_A(stride_scale_A_),
          stride_scale_B(stride_scale_B_)
    {
    }

    index_t M;
    index_t N;
    index_t K;
    index_t stride_A;
    index_t stride_B;
    index_t stride_C;
    index_t stride_scale_A;
    index_t stride_scale_B;
};

struct GemmMXHostArgs : public GemmMXProblem
{
    CK_TILE_HOST GemmMXHostArgs() = default;
    CK_TILE_HOST GemmMXHostArgs(const void* a_ptr_,
                                const void* a_scale_ptr_,
                                const void* b_ptr_,
                                const void* b_scale_ptr_,
                                void* c_ptr_,
                                index_t k_batch_,
                                index_t M_,
                                index_t N_,
                                index_t K_,
                                index_t stride_A_,
                                index_t stride_scale_A_,
                                index_t stride_B_,
                                index_t stride_scale_B_,
                                index_t stride_C_)
        : GemmMXProblem(
              M_, N_, K_, stride_A_, stride_B_, stride_C_, stride_scale_A_, stride_scale_B_),
          a_ptr(a_ptr_),
          a_scale_ptr(a_scale_ptr_),
          b_ptr(b_ptr_),
          b_scale_ptr(b_scale_ptr_),
          c_ptr(c_ptr_),
          k_batch(k_batch_)
    {
    }

    const void* a_ptr;
    const void* a_scale_ptr;
    const void* b_ptr;
    const void* b_scale_ptr;
    void* c_ptr;
    index_t k_batch;
};

struct GemmMXKernelArgs
{
    const void* a_ptr;
    const void* a_scale_ptr;
    const void* b_ptr;
    const void* b_scale_ptr;
    void* c_ptr;
    index_t M;
    index_t N;
    index_t K;
    index_t stride_A;
    index_t stride_scale_A;
    index_t stride_B;
    index_t stride_scale_B;
    index_t stride_C;
    index_t k_batch;
};

template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct GemmMXKernel
{
    using TilePartitioner                    = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline                       = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline                   = remove_cvref_t<EpiloguePipeline_>;
    using ALayout                            = remove_cvref_t<typename GemmPipeline::ALayout>;
    using AScaleLayout                       = remove_cvref_t<typename GemmPipeline::AScaleLayout>;
    using BLayout                            = remove_cvref_t<typename GemmPipeline::BLayout>;
    using BScaleLayout                       = remove_cvref_t<typename GemmPipeline::BScaleLayout>;
    using CLayout                            = remove_cvref_t<typename GemmPipeline::CLayout>;
    static constexpr index_t KernelBlockSize = GemmPipeline::BlockSize;

    using ADataType      = remove_cvref_t<typename GemmPipeline::ADataType>;
    using AScaleDataType = remove_cvref_t<typename GemmPipeline::AScaleDataType>;
    using BDataType      = remove_cvref_t<typename GemmPipeline::BDataType>;
    using BScaleDataType = remove_cvref_t<typename GemmPipeline::BScaleDataType>;
    using CDataType      = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    using APackedSize    = remove_cvref_t<typename GemmPipeline::PackedSize>;
    using BPackedSize    = remove_cvref_t<typename GemmPipeline::PackedSize>;
    using BlockScaleSize = remove_cvref_t<typename GemmPipeline::BlockScaleSize>;

    using BlockGemm     = remove_cvref_t<typename GemmPipeline::BlockGemm>;
    using MThreadPerXdl = BlockGemm::WarpGemm::kM;
    using NThreadPerXdl = BlockGemm::WarpGemm::kN;
    using KThreadPerXdl = get_warp_size() / MThreadPerXdl; // 64 is the number of threads in a wave

    using MXdlPack = remove_cvref_t<typename GemmPipeline::MXdlPack>;
    using NXdlPack = remove_cvref_t<typename GemmPipeline::NXdlPack>;
    using KXdlPack = remove_cvref_t<typename GemmPipeline::KXdlPack>;

    using mx_scale_t                           = ck_tile::e8m0_bexp_t;
    static constexpr index_t scale_pack_size_a = sizeof(AScaleDataType) / sizeof(mx_scale_t);
    static constexpr index_t scale_pack_size_b = sizeof(BScaleDataType) / sizeof(mx_scale_t);
    static_assert(KXdlPack * MXdlPack % scale_pack_size_a == 0,
                  "KXdlPack * MXdlPack must be a multiple of scale_pack_size_a");
    static_assert(KXdlPack * NXdlPack % scale_pack_size_b == 0,
                  "KXdlPack * NXdlPack must be a multiple of scale_pack_size_b");

    static constexpr auto I0 = number<0>();
    static constexpr auto I1 = number<1>();
    static constexpr auto I2 = number<2>();
    static constexpr auto I3 = number<3>();
    static constexpr auto I4 = number<4>();

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

    CK_TILE_HOST static constexpr GemmMXKernelArgs MakeKernelArgs(const GemmMXHostArgs& hostArgs)
    {
        return GemmMXKernelArgs{hostArgs.a_ptr,
                                hostArgs.a_scale_ptr,
                                hostArgs.b_ptr,
                                hostArgs.b_scale_ptr,
                                hostArgs.c_ptr,
                                hostArgs.M,
                                hostArgs.N,
                                hostArgs.K / APackedSize,
                                hostArgs.stride_A / APackedSize,
                                hostArgs.stride_scale_A,
                                hostArgs.stride_B / BPackedSize,
                                hostArgs.stride_scale_B,
                                hostArgs.stride_C,
                                hostArgs.k_batch};
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    struct SplitKBatchOffset
    {
        __device__ SplitKBatchOffset(const GemmMXKernelArgs& kargs,
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

            // Calculate A scale offset
            a_scale_k_split_offset = __builtin_amdgcn_readfirstlane(
                k_id * KRead / (BlockScaleSize / APackedSize) * MXdlPack * MThreadPerXdl)

                // Caluculate B scale offset
                b_scale_k_split_offset = __builtin_amdgcn_readfirstlane(
                    k_id * KRead / (BlockScaleSize / BPackedSize) * NXdlPack * NThreadPerXdl);
        }

        index_t a_k_split_offset;
        index_t b_k_split_offset;
        index_t a_scale_k_split_offset;
        index_t b_scale_k_split_offset;
        index_t splitted_k;
    };

    CK_TILE_HOST static bool IsSupportedArgument(const GemmMXKernelArgs& kargs)
    {
        if(kargs.k_batch != 1)
        {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR("Conditions not met for Kbatch >1 !");
            }
            return false;
        }

        // static_assert(std::is_same_v<AQLayout, tensor_layout::gemm::RowMajor>);
        // if(kargs.QK % GemmPipeline::GetVectorSizeAQ() != 0)
        // {
        //     if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
        //     {
        //         CK_TILE_ERROR("K is not a multiple of vector load size for A tensor!");
        //     }
        //     return false;
        // }

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
                                                   const AScaleDataType* a_scale_ptr,
                                                   const BDataType* b_ptr,
                                                   const BScaleDataType* b_scale_ptr,
                                                   CDataType* c_ptr,
                                                   const GemmMXKernelArgs& kargs,
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

        // A scale tensor view
        const auto Padded_Scale_M = integer_divide_ceil(kargs.M, BlockScaleSize) * BlockScaleSize;
        const auto& a_scale_tensor_view = [&]() {
            static_assert(std::is_same_v<AScaleLayout, tensor_layout::gemm::RowMajor>);
            // Pack 2x2 e8m0 over M/K dimension into 1 int32_t to trigger dword width load
            const auto a_naive_desc = make_naive_tensor_descriptor_packed(
                make_tuple(Padded_Scale_M / (MXdlPack * MThreadPerXdl),
                           (kargs.K * APackedSize) / BlockScaleSize / (KXdlPack * KThreadPerXdl),
                           KThreadPerXdl,
                           MThreadPerXdl));
            const auto a_m_k_desc = transform_tensor_descriptor(
                a_naive_desc,
                make_tuple(make_merge_transform(make_tuple(
                               Padded_Scale_M / (MXdlPack * MThreadPerXdl), MThreadPerXdl)),
                           make_merge_transform(make_tuple(kargs.K * APackedSize / BlockScaleSize /
                                                               (KXdlPack * KThreadPerXdl),
                                                           KThreadPerXdl))),
                make_tuple(sequence<0, 3>{}, sequence<1, 2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return make_tensor_view<address_space_enum::global>(a_scale_ptr, a_m_k_desc);
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

        // B scale tensor view
        const auto& b_scale_tensor_view = [&]() {
            static_assert(std::is_same_v<BScaleLayout, tensor_layout::gemm::ColumnMajor>);
            const auto b_navie_desc = make_naive_tensor_descriptor_packed(
                make_tuple(kargs.N / (NXdlPack * NThreadPerXdl),
                           (kargs.K * BPackedSize) / BlockScaleSize / (KXdlPack * KThreadPerXdl),
                           KThreadPerXdl,
                           NThreadPerXdl));
            const auto b_n_k_desc = transform_tensor_descriptor(
                b_navie_desc,
                make_tuple(
                    make_merge_transform(
                        make_tuple(kargs.N / (NXdlPack * NThreadPerXdl), NThreadPerXdl)),
                    make_merge_transform(make_tuple((kargs.K * BPackedSize) / BlockScaleSize /
                                                        (KXdlPack * KThreadPerXdl),
                                                    KThreadPerXdl))),
                make_tuple(sequence<0, 3>{}, sequence<1, 2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return make_tensor_view<address_space_enum::global>(b_scale_ptr, b_n_k_desc);
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

        return make_tuple(
            a_tensor_view, a_scale_tensor_view, b_tensor_view, b_scale_tensor_view, c_tensor_view);
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

        const auto& a_scale_pad_view = [&]() {
            const auto& a_scale_tensor_view = views.at(I1);
            static_assert(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>);
            return pad_tensor_view(a_scale_tensor_view,
                                   make_tuple(number<TilePartitioner::MPerBlock / MXdlPack>{},
                                              number<TilePartitioner::KPerBlock * APackedSize /
                                                     (BlockScaleSize * KXdlPack)>{}),
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

        const auto& b_scale_pad_view = [&]() {
            const auto& b_scale_tensor_view = views.at(I3);
            static_assert(std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>);
            return pad_tensor_view(b_scale_tensor_view,
                                   make_tuple(number<TilePartitioner::NPerBlock / NXdlPack>{},
                                              number<TilePartitioner::KPerBlock * BPackedSize /
                                                     (BlockScaleSize * KXdlPack)>{}),
                                   sequence<false, false>{});
        }();

        // TODO vector write in for C in ColMajor
        const auto& c_pad_view = [&]() {
            const auto& c_tensor_view = views.at(I4);
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

        return make_tuple(a_pad_view, a_scale_pad_view, b_pad_view, b_scale_pad_view, c_pad_view);
    }

    template <typename PadView>
    CK_TILE_DEVICE static auto
    MakeGemmTileWindows(const PadView& views, const index_t i_m, const index_t i_n)
    {
        const auto& a_pad_view       = views.at(I0);
        const auto& a_scale_pad_view = views.at(I1);
        const auto& b_pad_view       = views.at(I2);
        const auto& b_scale_pad_view = views.at(I3);
        const auto& c_pad_view       = views.at(I4);

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

        const auto& a_scale_block_window = [&]() {
            static_assert(std::is_same_v<AScaleLayout, tensor_layout::gemm::RowMajor>);
            return make_tile_window(a_scale_pad_view,
                                    make_tuple(number<TilePartitioner::MPerBlock / MXdlPack>{},
                                               number<TilePartitioner::KPerBlock * APackedSize /
                                                      (BlockScaleSize * KXdlPack)>{}),
                                    {i_m / MXdlPack, 0});
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

        const auto& b_scale_block_window = [&]() {
            static_assert(std::is_same_v<BScaleLayout, tensor_layout::gemm::ColumnMajor>);
            return make_tile_window(b_scale_pad_view,
                                    make_tuple(number<TilePartitioner::NPerBlock / NXdlPack>{},
                                               number<TilePartitioner::KPerBlock * BPackedSize /
                                                      (BlockScaleSize * KXdlPack)>{}),
                                    {i_n / NXdlPack, 0});
        }();

        auto c_block_window = make_tile_window(
            c_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {i_m, i_n});

        return make_tuple(a_block_window,
                          a_scale_block_window,
                          b_block_window,
                          b_scale_block_window,
                          c_block_window);
    }

    /**
     * @brief Runs single GEMM problem cooperatively by whole workgroup.
     *
     * @param a_ptr input A pointer
     * @param a_scale input A scale pointer
     * @param b_ptr input B pointer
     * @param b_scale_ptr input B scale pointer
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
                                       const AScaleDataType* a_scale_ptr,
                                       const BDataType* b_ptr,
                                       const BScaleDataType* b_scale_ptr,
                                       CDataType* c_ptr,
                                       void* smem_ptr_0,
                                       const GemmMXKernelArgs& kargs,
                                       const SplitKBatchOffset& splitk_batch_offset,
                                       const index_t block_idx_m,
                                       const index_t block_idx_n)
    {
        // Create Gemm tensor views, pad views and tile windows
        const auto& gemm_tensor_views_tuple = MakeGemmTensorViews<DstInMemOp>(
            a_ptr, a_scale_ptr, b_ptr, b_scale_ptr, c_ptr, kargs, splitk_batch_offset);

        const auto& gemm_pad_views = MakeGemmPadViews(gemm_tensor_views_tuple);
        auto gemm_tile_windows     = MakeGemmTileWindows(gemm_pad_views, block_idx_m, block_idx_n);

        const index_t num_loop = __builtin_amdgcn_readfirstlane(
            TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k));

        // Run GEMM cooperatively by whole workgroup.
        const auto& a_block_window       = gemm_tile_windows.at(I0);
        const auto& a_scale_block_window = gemm_tile_windows.at(I1);
        const auto& b_block_window       = gemm_tile_windows.at(I2);
        const auto& b_scale_block_window = gemm_tile_windows.at(I3);

        const auto& c_block_tile = GemmPipeline{}.template operator()(a_block_window,
                                                                      a_scale_block_window,
                                                                      b_block_window,
                                                                      b_scale_block_window,
                                                                      num_loop,
                                                                      smem_ptr_0);

        // Run Epilogue Pipeline
        auto& c_block_window = gemm_tile_windows.at(I4);

        EpiloguePipeline{}.template
        operator()<decltype(c_block_window), decltype(c_block_tile), decltype(c_block_window)>(
            c_block_window, c_block_tile, c_block_window, smem_ptr_0);
    }

    CK_TILE_DEVICE void operator()(GemmMXKernelArgs kargs) const
    {
        const auto blockId  = __builtin_amdgcn_readfirstlane(blockIdx.x);
        const auto [iM, iN] = TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(blockId);
        const index_t i_m   = __builtin_amdgcn_readfirstlane(iM * TilePartitioner::MPerBlock);
        const index_t i_n   = __builtin_amdgcn_readfirstlane(iN * TilePartitioner::NPerBlock);

        const SplitKBatchOffset splitk_batch_offset(kargs);
        // options
        const ADataType* a_ptr            = static_cast<const ADataType*>(kargs.a_ptr);
        const AScaleDataType* a_scale_ptr = static_cast<const AScaleDataType*>(kargs.a_scale_ptr);
        const BDataType* b_ptr            = static_cast<const BDataType*>(kargs.b_ptr);
        const BScaleDataType* b_scale_ptr = static_cast<const BScaleDataType*>(kargs.b_scale_ptr);
        CDataType* c_ptr                  = static_cast<CDataType*>(kargs.c_ptr);

        // allocate LDS
        __shared__ char smem_ptr_0[GetSmemSize()];

        assert(kargs.k_batch == 1);
        RunGemm(a_ptr,
                a_scale_ptr,
                b_ptr,
                b_scale_ptr,
                c_ptr,
                smem_ptr_0,
                kargs,
                splitk_batch_offset,
                i_m,
                i_n);
    }
};

} // namespace ck_tile
