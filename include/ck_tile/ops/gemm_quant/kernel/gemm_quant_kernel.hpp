// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/tile_gemm_quant_traits.hpp"

#if __clang_major__ >= 23
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
#endif
namespace ck_tile {

namespace detail {
// Helper templates for safe type extraction
template <typename, typename Default, typename = void>
struct get_aq_layout_or
{
    using type = Default;
};

template <typename T, typename Default>
struct get_aq_layout_or<T, Default, std::void_t<typename T::AQLayout>>
{
    using type = typename T::AQLayout;
};

template <typename, typename Default, typename = void>
struct get_bq_layout_or
{
    using type = Default;
};

template <typename T, typename Default>
struct get_bq_layout_or<T, Default, std::void_t<typename T::BQLayout>>
{
    using type = typename T::BQLayout;
};

template <typename, typename Default, typename = void>
struct get_aq_data_type_or
{
    using type = Default;
};

template <typename T, typename Default>
struct get_aq_data_type_or<T, Default, std::void_t<typename T::AQDataType>>
{
    using type = typename T::AQDataType;
};

template <typename, typename Default, typename = void>
struct get_bq_data_type_or
{
    using type = Default;
};

template <typename T, typename Default>
struct get_bq_data_type_or<T, Default, std::void_t<typename T::BQDataType>>
{
    using type = typename T::BQDataType;
};

template <typename, typename = void>
struct is_Aquantpreshuffle_enabled
{
    static constexpr bool value = false;
};

template <typename T>
struct is_Aquantpreshuffle_enabled<T, std::void_t<decltype(T::APreshuffleQuant)>>
{
    static constexpr bool value = T::APreshuffleQuant;
};

template <typename, typename = void>
struct is_Bquantpreshuffle_enabled
{
    static constexpr bool value = false;
};

template <typename T>
struct is_Bquantpreshuffle_enabled<T, std::void_t<decltype(T::BPreshuffleQuant)>>
{
    static constexpr bool value = T::BPreshuffleQuant;
};

template <typename, typename = void>
struct is_preshuffleB_enabled
{
    static constexpr bool value = false;
};

template <typename T>
struct is_preshuffleB_enabled<T, std::void_t<decltype(T::PreshuffleB)>>
{
    static constexpr bool value = T::PreshuffleB;
};
} // namespace detail

struct QuantGemmProblem
{
    CK_TILE_HOST QuantGemmProblem() = default;
    CK_TILE_HOST QuantGemmProblem(index_t M_,
                                  index_t N_,
                                  index_t K_,
                                  index_t QK_A_,
                                  index_t QK_B_,
                                  index_t stride_A_,
                                  index_t stride_B_,
                                  index_t stride_C_,
                                  index_t stride_AQ_,
                                  index_t stride_BQ_)
        : M(M_),
          N(N_),
          K(K_),
          QK_A(QK_A_),
          QK_B(QK_B_),
          stride_A(stride_A_),
          stride_B(stride_B_),
          stride_C(stride_C_),
          stride_AQ(stride_AQ_),
          stride_BQ(stride_BQ_)
    {
    }

    index_t M;
    index_t N;
    index_t K;
    index_t QK_A;
    index_t QK_B;
    index_t stride_A;
    index_t stride_B;
    index_t stride_C;
    index_t stride_AQ;
    index_t stride_BQ;
};

struct QuantGemmHostArgs : public QuantGemmProblem
{
    CK_TILE_HOST QuantGemmHostArgs() = default;
    CK_TILE_HOST QuantGemmHostArgs(const void* a_ptr_,
                                   const void* b_ptr_,
                                   void* c_ptr_,
                                   const void* aq_ptr_,
                                   const void* bq_ptr_,
                                   index_t k_batch_,
                                   index_t M_,
                                   index_t N_,
                                   index_t K_,
                                   index_t QK_A_,
                                   index_t QK_B_,
                                   index_t stride_A_,
                                   index_t stride_B_,
                                   index_t stride_C_,
                                   index_t stride_AQ_,
                                   index_t stride_BQ_)
        : QuantGemmProblem(
              M_, N_, K_, QK_A_, QK_B_, stride_A_, stride_B_, stride_C_, stride_AQ_, stride_BQ_),
          a_ptr(a_ptr_),
          b_ptr(b_ptr_),
          aq_ptr(aq_ptr_),
          bq_ptr(bq_ptr_),
          c_ptr(c_ptr_),
          k_batch(k_batch_)
    {
    }

    const void* a_ptr  = nullptr;
    const void* b_ptr  = nullptr;
    const void* aq_ptr = nullptr;
    const void* bq_ptr = nullptr;
    void* c_ptr        = nullptr;
    // k_batch must be a positive integer; defaults to 1 (no split-K).
    index_t k_batch = 1;
};

struct QuantGemmKernelArgs
{
    const void* a_ptr;
    const void* b_ptr;
    const void* aq_ptr;
    const void* bq_ptr;
    void* c_ptr;
    index_t M;
    index_t N;
    index_t K;
    index_t QK_A;
    index_t QK_B;
    index_t stride_A;
    index_t stride_B;
    index_t stride_C;
    index_t stride_AQ;
    index_t stride_BQ;
    index_t k_batch;
};

CK_TILE_HOST_DEVICE auto
get_splitk_batch_k_read(index_t K, index_t k_batch, index_t k_unit) noexcept -> index_t
{
    // k_batch and k_unit must be positive integers.  Callers are expected to
    // validate via IsSupportedArgument(); this fallback returns K so a
    // misconfigured launch behaves as a no-split kernel.
    if(k_batch <= 0 || k_unit <= 0)
    {
        return K;
    }
    const index_t k_t = k_batch * k_unit;
    return (K + k_t - 1) / k_t * k_unit;
}

CK_TILE_HOST_DEVICE auto
get_splitk_last_batch_k(index_t K, index_t k_batch, index_t k_read) noexcept -> index_t
{
    if(k_batch <= 0)
    {
        return K;
    }
    return K - k_read * (k_batch - 1);
}

template <typename TilePartitioner_,
          typename GemmPipeline_,
          typename EpiloguePipeline_,
          QuantType QuantType_,
          bool RuntimeSplitKTail_ = false>
struct QuantGemmKernel
{
    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline     = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;
    using ALayout          = remove_cvref_t<typename GemmPipeline::ALayout>;
    using BLayout          = remove_cvref_t<typename GemmPipeline::BLayout>;
    using CLayout          = remove_cvref_t<typename GemmPipeline::CLayout>;

    using AQLayout = remove_cvref_t<
        typename detail::get_aq_layout_or<GemmPipeline, typename GemmPipeline::ALayout>::type>;
    using BQLayout = remove_cvref_t<
        typename detail::get_bq_layout_or<GemmPipeline, typename GemmPipeline::BLayout>::type>;

    static constexpr index_t kBlockSize = GemmPipeline::BlockSize;
    static constexpr bool APreshuffleQuant =
        detail::is_Aquantpreshuffle_enabled<GemmPipeline_>::value;
    static constexpr bool BPreshuffleQuant =
        detail::is_Bquantpreshuffle_enabled<GemmPipeline_>::value;
    static constexpr bool PreshuffleB = detail::is_preshuffleB_enabled<GemmPipeline_>::value;

    using ADataType   = remove_cvref_t<typename GemmPipeline::ADataType>;
    using BDataType   = remove_cvref_t<typename GemmPipeline::BDataType>;
    using CDataType   = remove_cvref_t<typename EpiloguePipeline::ODataType>;
    using AccDataType = remove_cvref_t<typename EpiloguePipeline::AccDataType>;

    using AQDataType =
        remove_cvref_t<typename detail::get_aq_data_type_or<GemmPipeline, AccDataType>::type>;
    using BQDataType =
        remove_cvref_t<typename detail::get_bq_data_type_or<GemmPipeline, AccDataType>::type>;

    static constexpr auto I0 = number<0>(); // A Tensor
    static constexpr auto I1 = number<1>(); // AQ Tensor
    static constexpr auto I2 = number<2>(); // B Tensor
    static constexpr auto I3 = number<3>(); // BQ Tensor
    static constexpr auto I4 = number<4>(); // C Tensor

    static constexpr auto kQuantType        = QuantType_;
    static constexpr bool RuntimeSplitKTail = RuntimeSplitKTail_;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "gemm_quant", gemm_prec_str<ADataType, BDataType>(), GemmPipeline::GetName());
        // clang-format on
    }

    CK_TILE_HOST static constexpr auto GridSize(index_t M, index_t N, index_t KBatch)
    {
        return dim3(TilePartitioner::GridSize(M, N), 1, KBatch);
    }

    CK_TILE_HOST static auto BlockSize()
    {
        return is_wave32() ? dim3(kBlockSize / 2) : dim3(kBlockSize);
    }

    CK_TILE_HOST static constexpr QuantGemmKernelArgs
    MakeKernelArgs(const QuantGemmHostArgs& hostArgs)
    {
        return QuantGemmKernelArgs{hostArgs.a_ptr,
                                   hostArgs.b_ptr,
                                   hostArgs.aq_ptr,
                                   hostArgs.bq_ptr,
                                   hostArgs.c_ptr,
                                   hostArgs.M,
                                   hostArgs.N,
                                   hostArgs.K,
                                   hostArgs.QK_A,
                                   hostArgs.QK_B,
                                   hostArgs.stride_A,
                                   hostArgs.stride_B,
                                   hostArgs.stride_C,
                                   hostArgs.stride_AQ,
                                   hostArgs.stride_BQ,
                                   hostArgs.k_batch};
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    private:
    CK_TILE_DEVICE static constexpr index_t get_padding_size(index_t length, index_t alignment)
    {
        return ck_tile::integer_least_multiple(length, alignment) - length;
    };
    // ===================================================================
    // Helper: Create Pre-shuffled Quantization Tensor Descriptor
    // ===================================================================
    template <index_t KPerBlockBQ,
              index_t NPerBlockBQ,
              index_t NPerBlock,
              index_t WarpTileN,
              index_t GetVectorSizeBQ,
              typename BQDataType_>
    CK_TILE_DEVICE static auto
    MakePreshuffledQuantTensorView(const BQDataType_* bq_ptr, index_t N, index_t QN_B, index_t QK_B)
    {
        // Step 1: Calculate base BQ tensor dimensions
        // ----------------------------------------------------------
        // bq_x: Number of quantization groups in N dimension
        //       = N * KPerBlockBQ, where KPerBlockBQ is the number of
        //       K-dimension groups per block
        // bq_y: Number of quantization groups in K dimension
        //       = Total K groups (QK_B) / groups per block
        const auto bq_x = N * KPerBlockBQ;
        const auto bq_y = QK_B / KPerBlockBQ;

        const auto bq_desc = make_naive_tensor_descriptor(
            make_tuple(bq_y, bq_x), make_tuple(bq_x, 1), number<GetVectorSizeBQ>{}, number<1>{});

        // Step 2: First padding transformation (block-level alignment)
        // ----------------------------------------------------------
        // Pad the X dimension to be a multiple of block_tile_size to ensure
        // each thread block can process complete tiles without edge cases
        const auto block_tile_size = NPerBlockBQ * KPerBlockBQ;

        const auto bq_pad0_desc = transform_tensor_descriptor(
            bq_desc,
            make_tuple(make_pass_through_transform(bq_y),
                       make_right_pad_transform(bq_x, get_padding_size(bq_x, block_tile_size))),
            make_tuple(sequence<0>{}, sequence<1>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        // Step 3: Unmerge transformation (wave-level decomposition)
        // ----------------------------------------------------------
        // Split the X dimension into [wave_tile_count_x, wave_tile_size]
        // This separates the work into tiles that can be processed by
        // individual warps/waves
        const auto pad_bq_x          = bq_pad0_desc.get_lengths()[I1];
        const auto wave_tile_size    = ((QN_B <= WarpTileN) ? (WarpTileN / QN_B) : 1) * KPerBlockBQ;
        const auto wave_tile_count_x = ck_tile::integer_divide_ceil(pad_bq_x, wave_tile_size);

        const auto bq_unmerge_pad0_desc = transform_tensor_descriptor(
            bq_pad0_desc,
            make_tuple(make_pass_through_transform(bq_y),
                       make_unmerge_transform(make_tuple(wave_tile_count_x, wave_tile_size))),
            make_tuple(sequence<0>{}, sequence<1>{}),
            make_tuple(sequence<0>{}, sequence<1, 2>{}));

        // Step 4: Second padding transformation (warp-level alignment)
        // ----------------------------------------------------------
        // Pad wave_tile_size to be a multiple of warp_size (typically 32 or 64)
        // This ensures coalesced memory accesses within each warp
        const auto bq_pad1_desc = transform_tensor_descriptor(
            bq_unmerge_pad0_desc,
            make_tuple(make_pass_through_transform(bq_y),
                       make_pass_through_transform(wave_tile_count_x),
                       make_right_pad_transform(wave_tile_size,
                                                get_padding_size(wave_tile_size, get_warp_size()))),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

        // Step 5: Final merge transformation (prepare for indexing)
        // ----------------------------------------------------------
        // Merge [bq_y, wave_tile_count_x] into a single outer dimension
        // This creates a 2D layout: [merged_outer_dim, pad_wave_size]
        // where merged_outer_dim = bq_y * wave_tile_count_x
        // This layout facilitates efficient block-to-data mapping
        const auto pad_wave_size = ck_tile::integer_least_multiple(wave_tile_size, get_warp_size());
        const auto bq_merge_pad1_desc = transform_tensor_descriptor(
            bq_pad1_desc,
            make_tuple(make_merge_transform(make_tuple(bq_y, wave_tile_count_x)),
                       make_pass_through_transform(pad_wave_size)),
            make_tuple(sequence<0, 1>{}, sequence<2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return make_tensor_view<address_space_enum::global>(bq_ptr, bq_merge_pad1_desc);
    }

    public:
    struct SplitKBatchOffset
    {
        __device__ SplitKBatchOffset(const QuantGemmKernelArgs& kargs,
                                     const std::size_t k_id = blockIdx.z)
        {
            constexpr auto K1 =
                GemmPipeline::BlockGemmShape::WarpTile::at(I2); // smallest unit of K work per block
            const index_t KRead =
                amd_wave_read_first_lane(get_splitk_batch_k_read(kargs.K, kargs.k_batch, K1));
            // total k elements to be read in this batch
            // offset not necessarily = KRead, because B can have packed elements (e.g. fp8i4)
            constexpr index_t BPackedSize =
                ck_tile::numeric_traits<remove_cvref_t<BDataType>>::PackedSize;
            const index_t b_k_offset_elements =
                amd_wave_read_first_lane(k_id * KRead / BPackedSize);

            if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                a_k_split_offset = amd_wave_read_first_lane(k_id * KRead);
            }
            else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                a_k_split_offset = amd_wave_read_first_lane(k_id * KRead * kargs.stride_A);
            }

            if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, BLayout>)
            {
                b_k_split_offset = amd_wave_read_first_lane(b_k_offset_elements * kargs.stride_B);
            }
            else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, BLayout>)
            {
                if constexpr(PreshuffleB)
                {
                    // Preshuffled B is laid out as [N/N_Warp_Tile, K_outer, N_Warp_Tile, K_inner]
                    // (see shuffle_b<>), where each "N_outer" row spans N_Warp_Tile * full_K
                    // linear elements.  MakeBBlockWindow already builds the descriptor with
                    // stride [N_Warp_Tile * kargs.K, 1], so to advance the K starting position
                    // by k_id * KRead within row 0 we need to advance the pointer by
                    // (k_id * KRead) * N_Warp_Tile -- not just (k_id * KRead).
                    constexpr index_t N_Warp_Tile = GemmPipeline::BlockGemmShape::WarpTile::at(I1);
                    b_k_split_offset = amd_wave_read_first_lane(b_k_offset_elements * N_Warp_Tile);
                }
                else
                {
                    b_k_split_offset = amd_wave_read_first_lane(b_k_offset_elements);
                }
            }

            if(k_id < static_cast<uint32_t>(kargs.k_batch - 1))
            {
                splitted_k = amd_wave_read_first_lane(KRead);
            }
            else
            {
                splitted_k = amd_wave_read_first_lane(kargs.K - KRead * (kargs.k_batch - 1));
            }

            // Compute BQ offset for BQuantGrouped mode (non-preshuffle only)
            // Note: With the alignment validation in IsSupportedArgument, KRead is always
            // a multiple of BQuantGroupSize::kK, so bq_k_split_offset will be correctly aligned.
            if constexpr(kQuantType == QuantType::BQuantGrouped && !BPreshuffleQuant)
            {
                using BQuantGroupSize = remove_cvref_t<typename GemmPipeline::BQuantGroupSize>;
                // Compute the K offset for this batch (in terms of K elements)
                const index_t k_offset = amd_wave_read_first_lane(k_id * KRead);
                // Convert K offset to BQ group offset (logical offset in K/kK dimension)
                bq_group_offset = amd_wave_read_first_lane(k_offset / BQuantGroupSize::kK);

                // BQ tensor layout:
                // RowMajor: [K/kK, N/kN] with stride [N/kN, 1]
                // ColumnMajor: [N/kN, K/kK] with stride [K/kK, 1]
                if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, BQLayout>)
                {
                    // For RowMajor BQ, K is the row dimension
                    // offset = bq_group_offset * stride_BQ
                    const index_t stride_bq =
                        amd_wave_read_first_lane(integer_divide_ceil(kargs.N, BQuantGroupSize::kN));
                    bq_k_split_offset = amd_wave_read_first_lane(bq_group_offset * stride_bq);
                }
                else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, BQLayout>)
                {
                    // For ColumnMajor BQ, K is the column dimension
                    // offset = bq_group_offset
                    bq_k_split_offset = amd_wave_read_first_lane(bq_group_offset);
                }

                aq_group_offset   = 0;
                aq_k_split_offset = 0;
            }
            else if constexpr(kQuantType == QuantType::ABQuantGrouped && !APreshuffleQuant)
            {
                using AQuantGroupSize = remove_cvref_t<typename GemmPipeline::AQuantGroupSize>;
                using BQuantGroupSize = remove_cvref_t<typename GemmPipeline::BQuantGroupSize>;

                // Compute AQ K-group offset for this split-K batch.
                const index_t k_offset_aq = amd_wave_read_first_lane(k_id * KRead);
                aq_group_offset = amd_wave_read_first_lane(k_offset_aq / AQuantGroupSize::kK);
                if constexpr(std::is_same_v<AQLayout, tensor_layout::gemm::RowMajor>)
                {
                    // RowMajor AQ is [M, QK_A] with stride [stride_AQ, 1].
                    // Advancing to K-group column g is a pointer offset of g.
                    aq_k_split_offset = amd_wave_read_first_lane(aq_group_offset);
                }
                else if constexpr(std::is_same_v<AQLayout, tensor_layout::gemm::ColumnMajor>)
                {
                    // ColumnMajor AQ is [QK_A, M] with K-group row stride stride_AQ.
                    // Advancing to K-group row g is a pointer offset of g * stride_AQ.
                    aq_k_split_offset = amd_wave_read_first_lane(aq_group_offset * kargs.stride_AQ);
                }

                // Compute BQ K-group offset for this split-K batch.
                // BQ tensor layout is ColumnMajor [N/kN, K/kK] with stride [K/kK, 1] for
                // ABQuantGrouped. Advancing to column bq_group_offset means a pointer offset of
                // bq_group_offset elements (column stride = 1).
                const index_t k_offset_bq = amd_wave_read_first_lane(k_id * KRead);
                bq_group_offset   = amd_wave_read_first_lane(k_offset_bq / BQuantGroupSize::kK);
                bq_k_split_offset = amd_wave_read_first_lane(bq_group_offset);
            }
            else
            {
                bq_group_offset   = 0;
                bq_k_split_offset = 0;
                aq_group_offset   = 0;
                aq_k_split_offset = 0;
            }
        }

        index_t a_k_split_offset;
        index_t b_k_split_offset;
        index_t aq_group_offset;   // Logical offset in K-groups for AQ (K/kK dimension)
        index_t aq_k_split_offset; // Memory pointer offset for AQ
        index_t bq_group_offset;   // Logical offset in K-groups for BQ (K/kK dimension)
        index_t bq_k_split_offset; // Memory pointer offset for BQ (accounting for layout/stride)
        index_t splitted_k;
    };

    CK_TILE_DEVICE static auto MakeABlockWindow(const ADataType* a_ptr,
                                                const QuantGemmKernelArgs& kargs,
                                                const index_t k_size,
                                                const index_t i_m)
    {
        // Step 1: Create tensor view for A
        const auto& a_tensor_view = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_ptr,
                    make_tuple(kargs.M, k_size),
                    make_tuple(kargs.stride_A, 1),
                    number<GemmPipeline::GetVectorSizeA()>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_ptr,
                    make_tuple(k_size, kargs.M),
                    make_tuple(kargs.stride_A, 1),
                    number<GemmPipeline::GetVectorSizeA()>{},
                    number<1>{});
            }
        }();

        // Step 2: Create padded view
        const auto& a_pad_view = [&]() {
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

        // Step 3: Create tile window
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

        return a_block_window;
    }

    CK_TILE_DEVICE static auto MakeAQBlockWindow(const AQDataType* aq_ptr,
                                                 const QuantGemmKernelArgs& kargs,
                                                 const index_t i_m,
                                                 const index_t i_n,
                                                 const index_t aq_group_offset = 0)
    {
        // Step 1: Create tensor view for AQ
        const auto& aq_tensor_view = [&]() {
            if constexpr(kQuantType == QuantType::AQuantGrouped && APreshuffleQuant)
            {
                static_assert(std::is_same_v<AQLayout, tensor_layout::gemm::RowMajor>);
                const auto aq_x = kargs.M * GemmPipeline::KPerBlockAQ;
                const auto aq_y = kargs.QK_A / GemmPipeline::KPerBlockAQ;
                const auto aq_desc =
                    make_naive_tensor_descriptor(make_tuple(aq_y, aq_x),
                                                 make_tuple(aq_x, 1),
                                                 number<GemmPipeline::GetVectorSizeAQ()>{},
                                                 number<1>{});

                const auto block_tile_size = GemmPipeline::MPerBlock * GemmPipeline::KPerBlockAQ;
                const auto aq_pad0_desc    = transform_tensor_descriptor(
                    aq_desc,
                    make_tuple(
                        make_pass_through_transform(aq_y),
                        make_right_pad_transform(aq_x, get_padding_size(aq_x, block_tile_size))),
                    make_tuple(sequence<0>{}, sequence<1>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));

                const auto pad_aq_x = aq_pad0_desc.get_lengths()[I1];
                const auto wave_tile_size =
                    GemmPipeline::BlockGemmShape::WarpTile::at(I0) * GemmPipeline::KPerBlockAQ;
                const auto wave_tile_count_x =
                    ck_tile::integer_divide_ceil(pad_aq_x, wave_tile_size);

                const auto aq_unmerge_pad0_desc = transform_tensor_descriptor(
                    aq_pad0_desc,
                    make_tuple(
                        make_pass_through_transform(aq_y),
                        make_unmerge_transform(make_tuple(wave_tile_count_x, wave_tile_size))),
                    make_tuple(sequence<0>{}, sequence<1>{}),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}));

                const auto aq_pad1_desc = transform_tensor_descriptor(
                    aq_unmerge_pad0_desc,
                    make_tuple(
                        make_pass_through_transform(aq_y),
                        make_pass_through_transform(wave_tile_count_x),
                        make_right_pad_transform(
                            wave_tile_size, get_padding_size(wave_tile_size, get_warp_size()))),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

                const auto pad_wave_size =
                    ck_tile::integer_least_multiple(wave_tile_size, get_warp_size());
                const auto aq_merge_pad1_desc = transform_tensor_descriptor(
                    aq_pad1_desc,
                    make_tuple(make_merge_transform(make_tuple(aq_y, wave_tile_count_x)),
                               make_pass_through_transform(pad_wave_size)),
                    make_tuple(sequence<0, 1>{}, sequence<2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));

                return make_tensor_view<address_space_enum::global>(aq_ptr, aq_merge_pad1_desc);
            }
            else if constexpr(kQuantType == QuantType::AQuantGrouped && !APreshuffleQuant)
            {
                if constexpr(std::is_same_v<AQLayout, tensor_layout::gemm::RowMajor>)
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        aq_ptr,
                        make_tuple(kargs.M, kargs.QK_A),
                        make_tuple(kargs.stride_AQ, 1),
                        number<GemmPipeline::GetVectorSizeAQ()>{},
                        number<1>{});
                }
                else // Column major AQ
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        aq_ptr,
                        make_tuple(kargs.QK_A, kargs.M),
                        make_tuple(kargs.stride_AQ, 1),

                        number<GemmPipeline::GetVectorSizeAQ()>{},
                        number<1>{});
                }
            }
            else if constexpr(kQuantType == QuantType::ABQuantGrouped && !APreshuffleQuant)
            {
                // For split-K, aq_ptr is already offset by aq_k_split_offset elements.
                // The remaining K-groups from this offset position = QK_A - aq_group_offset.
                const index_t remaining_qk_a = kargs.QK_A - aq_group_offset;
                if constexpr(std::is_same_v<AQLayout, tensor_layout::gemm::RowMajor>)
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        aq_ptr,
                        make_tuple(kargs.M, remaining_qk_a),
                        make_tuple(kargs.stride_AQ, 1),
                        number<GemmPipeline::GetVectorSizeAQ()>{},
                        number<1>{});
                }
                else // Column major AQ
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        aq_ptr,
                        make_tuple(kargs.M, remaining_qk_a),
                        make_tuple(1, kargs.stride_AQ),
                        number<GemmPipeline::GetVectorSizeAQ()>{},
                        number<1>{});
                }
            }
            else if constexpr(kQuantType == QuantType::RowColQuant)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    aq_ptr,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(1, 0), // broadcasting over n
                    number<1>{},
                    number<1>{});
            }
            else
            {
                return nullptr;
            }
        }();

        // Step 2: Create tile window (no padding for AQ)
        const auto& aq_block_window = [&]() {
            if constexpr(kQuantType == QuantType::AQuantGrouped && APreshuffleQuant)
            {
                static_assert(std::is_same_v<AQLayout, tensor_layout::gemm::RowMajor>);
                using AQuantGroupSize  = remove_cvref_t<typename GemmPipeline::AQuantGroupSize>;
                constexpr auto block_m = TilePartitioner::MPerBlock;
                constexpr auto warp_m  = GemmPipeline::BlockGemmShape::WarpTile::at(I0);
                constexpr auto aqk_per_block = TilePartitioner::KPerBlock / AQuantGroupSize::kK;
                constexpr auto tile_window_width =
                    ck_tile::integer_least_multiple(warp_m * aqk_per_block, get_warp_size());
                constexpr auto tile_window_height = block_m / warp_m;
                auto block_m_idx                  = i_m / block_m;
                return make_tile_window(
                    aq_tensor_view,
                    make_tuple(number<tile_window_height>{}, number<tile_window_width>{}),
                    {block_m_idx * tile_window_height, 0});
            }
            else if constexpr(kQuantType == QuantType::AQuantGrouped && !APreshuffleQuant)
            {
                using AQuantGroupSize = remove_cvref_t<typename GemmPipeline::AQuantGroupSize>;
                constexpr auto aqk_per_block = TilePartitioner::KPerBlock / AQuantGroupSize::kK;
                constexpr auto block_m       = TilePartitioner::MPerBlock;

                if constexpr(std::is_same_v<AQLayout, tensor_layout::gemm::RowMajor>)
                {
                    return make_tile_window(aq_tensor_view,
                                            make_tuple(number<block_m>{}, number<aqk_per_block>{}),
                                            {i_m, 0});
                }
                else // Column major AQ
                {
                    return make_tile_window(aq_tensor_view,
                                            make_tuple(number<aqk_per_block>{}, number<block_m>{}),
                                            {0, i_m});
                }
            }
            else if constexpr(kQuantType == QuantType::ABQuantGrouped && !APreshuffleQuant)
            {
                using QuantGroupSize   = remove_cvref_t<typename GemmPipeline::AQuantGroupSize>;
                constexpr auto block_m = TilePartitioner::MPerBlock;
                constexpr auto block_k = TilePartitioner::KPerBlock;
                return make_tile_window(
                    aq_tensor_view,
                    make_tuple(number<block_m>{}, number<block_k / QuantGroupSize::kK>{}),
                    {i_m, 0});
            }
            else if constexpr(kQuantType == QuantType::RowColQuant)
            {
                return make_tile_window(aq_tensor_view,
                                        make_tuple(number<TilePartitioner::MPerBlock>{},
                                                   number<TilePartitioner::NPerBlock>{}),
                                        {i_m, i_n});
            }
            else
            {
                return nullptr;
            }
        }();

        return aq_block_window;
    }

    CK_TILE_DEVICE static auto MakeBBlockWindow(const BDataType* b_ptr,
                                                const QuantGemmKernelArgs& kargs,
                                                const index_t k_size,
                                                const index_t i_n)
    {
        // Step 1: Create tensor view for B
        const auto& b_tensor_view = [&]() {
            if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
            {
                if constexpr(GemmPipeline::BlockGemmShape::PermuteB)
                {
                    constexpr index_t K1          = GemmPipeline::GetSmemPackB();
                    const index_t K0              = k_size / K1;
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
                        make_tuple(k_size, kargs.N),
                        make_tuple(kargs.stride_B, 1),
                        number<GemmPipeline::GetVectorSizeB()>{},
                        number<1>{});
                }
            }
            else
            {
                if constexpr(GemmPipeline::BlockGemmShape::PermuteB)
                {
                    constexpr index_t K1          = GemmPipeline::GetSmemPackB();
                    const index_t K0              = k_size / K1;
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
                    if constexpr(PreshuffleB)
                    {
                        constexpr auto warp_k = GemmPipeline::BlockGemmShape::WarpTile::at(I2);
                        index_t kFlatKSplit   = GemmPipeline::flatKPerWarp * (k_size / warp_k);
                        index_t kFlatK        = GemmPipeline::flatKPerWarp * (kargs.K / warp_k);
                        index_t kFlatN        = kargs.N * kargs.K / kFlatK;
                        return make_naive_tensor_view<address_space_enum::global>(
                            b_ptr,
                            make_tuple(kFlatN, kFlatKSplit),
                            make_tuple(kFlatK, 1),
                            number<GemmPipeline::GetVectorSizeB()>{},
                            number<1>{});
                    }
                    else
                    {
                        return make_naive_tensor_view<address_space_enum::global>(
                            b_ptr,
                            make_tuple(kargs.N, k_size),
                            make_tuple(kargs.stride_B, 1),
                            number<GemmPipeline::GetVectorSizeB()>{},
                            number<1>{});
                    }
                }
            }
        }();

        // Step 2: Create padded view (or flat view for PreshuffleB)
        const auto& b_pad_view = [&]() {
            if constexpr(PreshuffleB)
            {
                return b_tensor_view; // no padding for preshuffle
            }
            else if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>)
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

        // Step 3: Create tile window
        const auto& b_block_window = [&]() {
            if constexpr(PreshuffleB)
            {
                return make_tile_window(
                    b_pad_view,
                    make_tuple(number<GemmPipeline::flatNPerWarp>{},
                               number<GemmPipeline::flatKPerWarp>{}),
                    {static_cast<int>(i_n / GemmPipeline::BlockGemmShape::WarpTile::at(I1)), 0});
            }
            else
            {
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
            }
        }();

        return b_block_window;
    }

    CK_TILE_DEVICE static auto MakeBQBlockWindow(const BQDataType* bq_ptr,
                                                 const QuantGemmKernelArgs& kargs,
                                                 const index_t bq_group_offset,
                                                 const index_t i_m,
                                                 const index_t i_n)
    {
        // Step 1: Create tensor view for BQ
        // Note: For split-K, the bq_ptr is already offset by bq_k_split_offset (pointer offset).
        // The dimension should use the remaining K-groups from this offset position.
        const auto& bq_tensor_view = [&]() {
            if constexpr(kQuantType == QuantType::RowColQuant)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    bq_ptr,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(0, 1), // broadcasting over m
                    number<1>{},
                    number<1>{});
            }
            else if constexpr(kQuantType == QuantType::BQuantGrouped ||
                              kQuantType == QuantType::ABQuantGrouped)
            {
                if constexpr(BPreshuffleQuant)
                {
                    static_assert(std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>,
                                  "PreshuffleQuant with BQuantGrouped currently only supports "
                                  "ColumnMajor BQ layout");
                    using BQuantGroupSize = remove_cvref_t<typename GemmPipeline::BQuantGroupSize>;

                    return MakePreshuffledQuantTensorView<
                        GemmPipeline::KPerBlockBQ,
                        GemmPipeline::NPerBlockBQ,
                        GemmPipeline::NPerBlock,
                        TilePartitioner::BlockGemmShape::WarpTile::at(I1),
                        GemmPipeline::GetVectorSizeBQ()>(
                        bq_ptr,
                        ck_tile::integer_divide_ceil(kargs.N, BQuantGroupSize::kN),
                        BQuantGroupSize::kN,
                        kargs.QK_B);
                }
                else
                {
                    using BQuantGroupSize = remove_cvref_t<typename GemmPipeline::BQuantGroupSize>;

                    if constexpr(kQuantType == QuantType::ABQuantGrouped)
                    {
                        static_assert(std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>,
                                      "ABQuantGrouped requires ColumnMajor BQ layout");
                    }

                    using BQuantGroupSize = remove_cvref_t<typename GemmPipeline::BQuantGroupSize>;
                    if constexpr(std::is_same_v<BQLayout, tensor_layout::gemm::RowMajor>)
                    {
                        return make_naive_tensor_view<address_space_enum::global>(
                            bq_ptr,
                            make_tuple(kargs.QK_B - bq_group_offset,
                                       integer_divide_ceil(kargs.N, BQuantGroupSize::kN)),
                            make_tuple(integer_divide_ceil(kargs.N, BQuantGroupSize::kN), 1),
                            number<GemmPipeline::GetVectorSizeBQ()>{},
                            number<1>{});
                    }
                    else
                    {
                        return make_naive_tensor_view<address_space_enum::global>(
                            bq_ptr,
                            make_tuple(integer_divide_ceil(kargs.N, BQuantGroupSize::kN),
                                       kargs.QK_B - bq_group_offset),
                            make_tuple(kargs.QK_B, 1),
                            number<GemmPipeline::GetVectorSizeBQ()>{},
                            number<1>{});
                    }
                }
            }
            else
            {
                return nullptr;
            }
        }();

        // Step 2: Create tile window (no padding for BQ)
        const auto& bq_block_window = [&]() {
            if constexpr(kQuantType == QuantType::RowColQuant)
            {
                return make_tile_window(bq_tensor_view,
                                        make_tuple(number<TilePartitioner::MPerBlock>{},
                                                   number<TilePartitioner::NPerBlock>{}),
                                        {i_m, i_n});
            }
            else if constexpr(kQuantType == QuantType::BQuantGrouped ||
                              kQuantType == QuantType::ABQuantGrouped)
            {
                using BQuantGroupSize = remove_cvref_t<typename GemmPipeline::BQuantGroupSize>;
                if constexpr(BPreshuffleQuant)
                {
                    static_assert(std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>);

                    // Number of N-dimension quantization groups per block
                    constexpr auto block_n = (BQuantGroupSize::kN <= TilePartitioner::NPerBlock)
                                                 ? TilePartitioner::NPerBlock / BQuantGroupSize::kN
                                                 : BQuantGroupSize::kN / TilePartitioner::NPerBlock;

                    // Number of N-dimension elements per warp
                    constexpr auto warp_n = TilePartitioner::BlockGemmShape::WarpTile::at(I1);

                    // Determine how many warps share the same scale in N-dimension
                    constexpr auto warp_per_group = (BQuantGroupSize::kN < warp_n)
                                                        ? (warp_n / BQuantGroupSize::kN)
                                                        : (BQuantGroupSize::kN / warp_n);

                    // Number of K-dimension quantization groups per block
                    constexpr auto bqk_per_block = TilePartitioner::KPerBlock / BQuantGroupSize::kK;

                    // The pre-shuffled layout flattens warp_n x
                    // bqk_per_block scales per row, Padded up to warp_size
                    // to ensure coalesced memory access.
                    constexpr auto tile_window_width =
                        ck_tile::integer_least_multiple(warp_n * bqk_per_block, get_warp_size());

                    // Adapts based on fine vs coarse quantization granularity:
                    //   - Fine-grained (BQuantGroupSize::kN < warp_n):
                    //       Multiple quant groups per warp -> fewer rows needed per block.
                    //       height = block_n / warp_per_group
                    //
                    //   - Coarse-grained (BQuantGroupSize::kN >= warp_n):
                    //       Each row represents one quant group.
                    //       height = block_n
                    constexpr auto tile_window_height =
                        (BQuantGroupSize::kN < warp_n) ? block_n / warp_per_group : block_n;

                    auto block_n_idx = i_n / TilePartitioner::NPerBlock;

                    // For decode shapes GN: 128, Blocks needs to repeat 0,0,1,1,2,2 ...
                    if(BQuantGroupSize::kN > TilePartitioner::NPerBlock)
                    {
                        block_n_idx = block_n_idx >> 1;
                    }

                    if(BQuantGroupSize::kN > TilePartitioner::NPerBlock)
                    {
                        return make_tile_window(
                            bq_tensor_view,
                            make_tuple(number<tile_window_height>{}, number<tile_window_width>{}),
                            {block_n_idx, 0});
                    }
                    else
                    {
                        return make_tile_window(
                            bq_tensor_view,
                            make_tuple(number<tile_window_height>{}, number<tile_window_width>{}),
                            {block_n_idx * tile_window_height, 0});
                    }
                }
                else
                {
                    if constexpr(kQuantType == QuantType::ABQuantGrouped)
                    {
                        static_assert(std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>,
                                      "ABQuantGrouped requires RowMajor AQ layout");
                    }
                    constexpr auto tensor_dim =
                        (BQuantGroupSize::kN <= TilePartitioner::NPerBlock)
                            ? TilePartitioner::NPerBlock / BQuantGroupSize::kN
                            : 1;
                    if constexpr(std::is_same_v<BQLayout, tensor_layout::gemm::RowMajor>)
                    {
                        return make_tile_window(
                            bq_tensor_view,
                            make_tuple(number<TilePartitioner::KPerBlock / BQuantGroupSize::kK>{},
                                       number<tensor_dim>{}),
                            {0, i_n / BQuantGroupSize::kN});
                    }
                    else
                    {
                        static_assert(std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>);
                        return make_tile_window(
                            bq_tensor_view,
                            make_tuple(number<tensor_dim>{},
                                       number<TilePartitioner::KPerBlock / BQuantGroupSize::kK>{}),
                            {i_n / BQuantGroupSize::kN, 0});
                    }
                }
            }
            else
            {
                return nullptr;
            }
        }();

        return bq_block_window;
    }

    template <memory_operation_enum DstInMemOp = memory_operation_enum::set>
    CK_TILE_DEVICE static auto MakeCBlockWindow(CDataType* c_ptr,
                                                const QuantGemmKernelArgs& kargs,
                                                const index_t i_m,
                                                const index_t i_n)
    {
        // Step 1: Create tensor view for C
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

        // Step 2: Create padded view
        const auto& c_pad_view = [&]() {
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

        // Step 3: Create tile window
        auto c_block_window = make_tile_window(
            c_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {i_m, i_n});

        return c_block_window;
    }

    CK_TILE_HOST static bool IsSupportedArgument(const QuantGemmKernelArgs& kargs)
    {
        // k_batch must be a positive integer.
        if(kargs.k_batch <= 0)
        {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR("k_batch must be a positive integer (got " +
                              std::to_string(kargs.k_batch) + ")!");
            }
            return false;
        }

        // The split-K K-unit (warp-tile K dimension) must be positive too;
        // it is a compile-time constant taken from the pipeline shape.
        static_assert(GemmPipeline::BlockGemmShape::WarpTile::at(I2) > 0,
                      "Pipeline warp-tile K dimension (k_unit) must be positive.");

        // ABQuantGrouped does not currently support RowMajor BQ layout: the
        // BQ tensor view, tile window, and split-K offset code are all
        // written for ColumnMajor BQ.  The deeper static_asserts in
        // MakeBQBlockWindow enforce this at instantiation time; surface it
        // here at the host-arg entry point too so the limitation is visible
        // before the first device-side instantiation.
        static_assert(!(kQuantType == QuantType::ABQuantGrouped &&
                        std::is_same_v<BQLayout, tensor_layout::gemm::RowMajor>),
                      "ABQuantGrouped does not currently support RowMajor BQ layout. "
                      "Use ColumnMajor BQ (or extend MakeBQBlockWindow and the split-K "
                      "BQ offset path to handle RowMajor BQ).");

        // Split-K is supported for BQuantGrouped (without preshuffle) and
        // ABQuantGrouped (without APreshuffleQuant) modes.
        if(kargs.k_batch != 1)
        {
            constexpr bool is_bquant_non_preshuffle =
                (kQuantType == QuantType::BQuantGrouped) && !BPreshuffleQuant;
            constexpr bool is_abquant_non_preshuffle =
                (kQuantType == QuantType::ABQuantGrouped) && !APreshuffleQuant;
            constexpr bool is_splitk_supported =
                is_bquant_non_preshuffle || is_abquant_non_preshuffle;

            if constexpr(!is_splitk_supported)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("Conditions not met for Kbatch >1 ! "
                                  "Split-K is supported for BQuantGrouped without preshuffle "
                                  "and ABQuantGrouped without APreshuffleQuant.");
                }
                return false;
            }
            else
            {
                constexpr auto K1 = GemmPipeline::BlockGemmShape::WarpTile::at(I2);
                const index_t KRead =
                    get_splitk_batch_k_read(kargs.K, kargs.k_batch, K1); // per-batch K read size
                const index_t KLast = get_splitk_last_batch_k(kargs.K, kargs.k_batch, KRead);
                constexpr index_t BPackedSize =
                    ck_tile::numeric_traits<remove_cvref_t<BDataType>>::PackedSize;

                if(KLast <= 0)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("Split-K configuration produces an empty final K batch!");
                    }
                    return false;
                }

                // Constraint 1: KRead must align with B packing requirements.
                // For packed data types, multiple K elements are stored in each storage unit.
                // Split-K advances the B pointer by (KRead / BPackedSize) storage units per batch.
                // If KRead is not divisible by BPackedSize, this division produces a fractional
                // offset, making it impossible to start reading from a valid storage unit boundary.
                if(KRead % BPackedSize != 0)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("KRead must be a multiple of B packed size for split-K!");
                    }
                    return false;
                }

                // Constraint 2: KRead must align with B quantization group boundaries.
                if constexpr(is_bquant_non_preshuffle || is_abquant_non_preshuffle)
                {
                    using BQuantGroupSize = remove_cvref_t<typename GemmPipeline::BQuantGroupSize>;
                    if(KRead % BQuantGroupSize::kK != 0)
                    {
                        if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                        {
                            CK_TILE_ERROR(
                                "Split-K batch size must be aligned with B quantization group "
                                "size! KRead=" +
                                std::to_string(KRead) +
                                " is not divisible by BQuantGroupSize::kK=" +
                                std::to_string(BQuantGroupSize::kK));
                        }
                        return false;
                    }
                }

                // Constraint 3: KRead must align with A quantization group boundaries
                // (only needed for ABQuantGrouped since AQ also indexes into K).
                if constexpr(is_abquant_non_preshuffle)
                {
                    using AQuantGroupSize = remove_cvref_t<typename GemmPipeline::AQuantGroupSize>;
                    if(KRead % AQuantGroupSize::kK != 0)
                    {
                        if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                        {
                            CK_TILE_ERROR(
                                "Split-K batch size must be aligned with A quantization group "
                                "size! KRead=" +
                                std::to_string(KRead) +
                                " is not divisible by AQuantGroupSize::kK=" +
                                std::to_string(AQuantGroupSize::kK));
                        }
                        return false;
                    }
                }

                // Constraint 4: per-batch K must span at least 2 K_Tile iterations.
                // The software-pipelined GEMM kernels (CompV3 family) prefetch one tile
                // ahead and require num_loop >= 2 per batch.  When KRead == KPerBlock
                // (i.e. per_batch_num_loop == 1) the prefetch would read the tile
                // belonging to the next split-K batch, producing incorrect results.
                {
                    const index_t per_batch_num_loop = TilePartitioner::GetLoopNum(KRead);
                    if(per_batch_num_loop < 2)
                    {
                        if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                        {
                            CK_TILE_ERROR(
                                "Split-K requires at least 2 K-tile iterations per batch. "
                                "KRead=" +
                                std::to_string(KRead) + " < 2 * KPerBlock=" +
                                std::to_string(2 *
                                               static_cast<index_t>(TilePartitioner::KPerBlock)) +
                                ". Increase K or decrease k_batch.");
                        }
                        return false;
                    }
                }

                // Host-side fixed tail selection is only valid when all split-K batches have
                // the same hot-loop/tail classification. Earlier batches use KRead; the final
                // batch may be shorter due to split rounding.
                {
                    const index_t first_num_loop = TilePartitioner::GetLoopNum(KRead);
                    const index_t last_num_loop  = TilePartitioner::GetLoopNum(KLast);
                    const bool first_hot_loop    = GemmPipeline::BlockHasHotloop(first_num_loop);
                    const bool last_hot_loop     = GemmPipeline::BlockHasHotloop(last_num_loop);
                    const auto first_tail = GemmPipeline::GetBlockLoopTailNum(first_num_loop);
                    const auto last_tail  = GemmPipeline::GetBlockLoopTailNum(last_num_loop);

                    if constexpr(!RuntimeSplitKTail)
                    {
                        if(first_hot_loop != last_hot_loop || first_tail != last_tail)
                        {
                            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                            {
                                CK_TILE_ERROR(
                                    "Split-K batches require different hot-loop/tail handling. "
                                    "Use a K/k_batch combination that gives matching pipeline "
                                    "tails or enable runtime split-K tail dispatch.");
                            }
                            return false;
                        }
                    }
                }
            }
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
            // For RowMajor C, M is the row dimension - check M alignment here because
            // ALayout=RowMajor does not check M (it only checks K), leaving a gap for
            // the RowMajorA + RowMajorC combination.
            if(kargs.M % TilePartitioner::MPerBlock != 0 && GemmPipeline::kPadM == false &&
               GemmPipeline::BlockGemmShape::NumWarps != 8)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Can't support M that is not a multiple of MPerBlock without padding!");
                }
                return false;
            }
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

    template <typename ADramBlockWindow, typename BDramBlockWindow, typename BQDramBlockWindow>
    CK_TILE_DEVICE static auto CallBQuantGemmPipeline(const ADramBlockWindow& a_block_window,
                                                      const BDramBlockWindow& b_block_window,
                                                      const BQDramBlockWindow& bq_block_window,
                                                      const index_t num_loop,
                                                      void* smem_ptr,
                                                      const index_t n)
    {
        if constexpr(RuntimeSplitKTail)
        {
            static_assert(!PreshuffleB,
                          "RuntimeSplitKTail is not implemented for preshuffle-B BQuant "
                          "pipelines.");
            const bool has_hot_loop   = GemmPipeline::BlockHasHotloop(num_loop);
            const TailNumber tail_num = GemmPipeline::GetBlockLoopTailNum(num_loop);
            return GemmPipeline{}(a_block_window,
                                  b_block_window,
                                  bq_block_window,
                                  num_loop,
                                  has_hot_loop,
                                  tail_num,
                                  smem_ptr,
                                  n);
        }
        else
        {
            return GemmPipeline{}(
                a_block_window, b_block_window, bq_block_window, num_loop, smem_ptr, n);
        }
    }

    template <typename ADramBlockWindow,
              typename BDramBlockWindow,
              typename AQDramBlockWindow,
              typename BQDramBlockWindow>
    CK_TILE_DEVICE static auto CallABQuantGemmPipeline(const ADramBlockWindow& a_block_window,
                                                       const BDramBlockWindow& b_block_window,
                                                       const AQDramBlockWindow& aq_block_window,
                                                       const BQDramBlockWindow& bq_block_window,
                                                       const index_t num_loop,
                                                       void* smem_ptr,
                                                       const index_t m,
                                                       const index_t n)
    {
        if constexpr(RuntimeSplitKTail)
        {
            const bool has_hot_loop   = GemmPipeline::BlockHasHotloop(num_loop);
            const TailNumber tail_num = GemmPipeline::GetBlockLoopTailNum(num_loop);
            return GemmPipeline{}(a_block_window,
                                  b_block_window,
                                  aq_block_window,
                                  bq_block_window,
                                  num_loop,
                                  has_hot_loop,
                                  tail_num,
                                  smem_ptr,
                                  m,
                                  n);
        }
        else
        {
            return GemmPipeline{}(a_block_window,
                                  b_block_window,
                                  aq_block_window,
                                  bq_block_window,
                                  num_loop,
                                  smem_ptr,
                                  m,
                                  n);
        }
    }

    /**
     * @brief Runs single GEMM problem cooperatively by whole workgroup.
     *
     * @param a_ptr input A pointer
     * @param b_ptr input B pointer
     * @param aq_ptr input AQ pointer
     * @param bq_ptr input BQ pointer
     * @param c_ptr output C pointer
     * @param smem_ptr The start memory pointer of the shared memory block.
     * @param kargs GEMM kernel arguments
     * @param splitk_batch_offset splitk_batch_offset Utility structure used to calculate k batch.
     * @param block_idx_m The GEMM's output M dimension tile index processed by this workgroup.
     * @param block_idx_n The GEMM's output N dimension tile index processed by this workgroup.
     *
     */
    CK_TILE_DEVICE static void RunGemm(const ADataType* a_ptr,
                                       const BDataType* b_ptr,
                                       const AQDataType* aq_ptr,
                                       const BQDataType* bq_ptr,
                                       CDataType* c_ptr,
                                       void* smem_ptr,
                                       const QuantGemmKernelArgs& kargs,
                                       const SplitKBatchOffset& splitk_batch_offset,
                                       const index_t block_idx_m,
                                       const index_t block_idx_n)
    {
        // Create block windows using specialized methods
        const auto& a_block_window =
            MakeABlockWindow(a_ptr, kargs, splitk_batch_offset.splitted_k, block_idx_m);
        const auto& b_block_window =
            MakeBBlockWindow(b_ptr, kargs, splitk_batch_offset.splitted_k, block_idx_n);
        // Note: Pass aq_group_offset so the tensor view dimension reflects
        // the remaining K-groups from the split-K offset position.
        const auto& aq_block_window = MakeAQBlockWindow(
            aq_ptr, kargs, block_idx_m, block_idx_n, splitk_batch_offset.aq_group_offset);
        // Note: Pass bq_group_offset so the tensor view dimension reflects
        // the remaining K-groups from the split-K offset position.
        const auto& bq_block_window = MakeBQBlockWindow(
            bq_ptr, kargs, splitk_batch_offset.bq_group_offset, block_idx_m, block_idx_n);

        const index_t num_loop =
            amd_wave_read_first_lane(TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k));
        // Run GEMM cooperatively by whole workgroup.
        const auto& c_block_tile = [&]() {
            if constexpr(kQuantType == QuantType::AQuantGrouped)
            {
                index_t m = 0;
                if constexpr(APreshuffleQuant)
                {
                    m = kargs.M;
                }
                return GemmPipeline{}(
                    a_block_window, b_block_window, aq_block_window, num_loop, smem_ptr, m);
            }
            else if constexpr(kQuantType == QuantType::BQuantGrouped)
            {
                index_t n = 0;
                if constexpr(BPreshuffleQuant)
                {
                    n = kargs.N;
                }
                return CallBQuantGemmPipeline(
                    a_block_window, b_block_window, bq_block_window, num_loop, smem_ptr, n);
            }
            else if constexpr(kQuantType == QuantType::ABQuantGrouped)
            {
                index_t m = 0;
                index_t n = 0;
                if constexpr(BPreshuffleQuant)
                {
                    // m = kargs.M;
                    n = kargs.N;
                }
                return CallABQuantGemmPipeline(a_block_window,
                                               b_block_window,
                                               aq_block_window,
                                               bq_block_window,
                                               num_loop,
                                               smem_ptr,
                                               m,
                                               n);
            }
            else if constexpr(kQuantType == QuantType::RowColQuant ||
                              kQuantType == QuantType::TensorQuant)
            {
                return GemmPipeline{}(a_block_window, b_block_window, num_loop, smem_ptr);
            }
        }();

        const index_t k_batch = amd_wave_read_first_lane(kargs.k_batch);

        // Run Epilogue Pipeline with k_batch dispatch
        if(k_batch == 1)
        {
            auto c_block_window = MakeCBlockWindow<memory_operation_enum::set>(
                c_ptr, kargs, block_idx_m, block_idx_n);

            if constexpr(kQuantType == QuantType::ABQuantGrouped ||
                         kQuantType == QuantType::AQuantGrouped ||
                         kQuantType == QuantType::BQuantGrouped)
            {
                EpiloguePipeline{}(c_block_window, c_block_tile, c_block_window, smem_ptr);
            }
            else if constexpr(kQuantType == QuantType::RowColQuant)
            {
                EpiloguePipeline{}(c_block_window,
                                   c_block_tile,
                                   c_block_window,
                                   smem_ptr,
                                   aq_block_window,
                                   bq_block_window);
            }
            else if constexpr(kQuantType == QuantType::TensorQuant)
            {
                const AccDataType aq_scale = type_convert<AccDataType>(*aq_ptr);
                const AccDataType bq_scale = type_convert<AccDataType>(*bq_ptr);
                EpiloguePipeline{}(
                    c_block_window, c_block_tile, c_block_window, smem_ptr, aq_scale, bq_scale);
            }
        }
        else
        {
            auto c_block_window = MakeCBlockWindow<memory_operation_enum::atomic_add>(
                c_ptr, kargs, block_idx_m, block_idx_n);

            if constexpr(kQuantType == QuantType::ABQuantGrouped ||
                         kQuantType == QuantType::AQuantGrouped ||
                         kQuantType == QuantType::BQuantGrouped)
            {
                EpiloguePipeline{}(c_block_window, c_block_tile, c_block_window, smem_ptr);
            }
            else if constexpr(kQuantType == QuantType::RowColQuant)
            {
                EpiloguePipeline{}(c_block_window,
                                   c_block_tile,
                                   c_block_window,
                                   smem_ptr,
                                   aq_block_window,
                                   bq_block_window);
            }
            else if constexpr(kQuantType == QuantType::TensorQuant)
            {
                const AccDataType aq_scale = type_convert<AccDataType>(*aq_ptr);
                const AccDataType bq_scale = type_convert<AccDataType>(*bq_ptr);
                EpiloguePipeline{}(
                    c_block_window, c_block_tile, c_block_window, smem_ptr, aq_scale, bq_scale);
            }
        }
    }

    CK_TILE_DEVICE void Run_(const QuantGemmKernelArgs& kargs) const
    {
        const auto blockId  = amd_wave_read_first_lane(blockIdx.x);
        const auto [iM, iN] = TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(blockId);
        const index_t i_m   = amd_wave_read_first_lane(iM * TilePartitioner::MPerBlock);
        const index_t i_n   = amd_wave_read_first_lane(iN * TilePartitioner::NPerBlock);
        const SplitKBatchOffset splitk_batch_offset(kargs);

        // Apply splitk offset to input pointers
        const ADataType* a_ptr =
            static_cast<const ADataType*>(kargs.a_ptr) + splitk_batch_offset.a_k_split_offset;
        const BDataType* b_ptr =
            static_cast<const BDataType*>(kargs.b_ptr) + splitk_batch_offset.b_k_split_offset;
        // For ABQuantGrouped split-K, aq_ptr is offset by aq_k_split_offset elements to point
        // to the start of this batch's AQ K-groups (aq_group_offset columns in RowMajor AQ).
        const AQDataType* aq_ptr =
            static_cast<const AQDataType*>(kargs.aq_ptr) + splitk_batch_offset.aq_k_split_offset;
        const BQDataType* bq_ptr =
            static_cast<const BQDataType*>(kargs.bq_ptr) + splitk_batch_offset.bq_k_split_offset;
        CDataType* c_ptr = static_cast<CDataType*>(kargs.c_ptr);

        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];
        RunGemm(
            a_ptr, b_ptr, aq_ptr, bq_ptr, c_ptr, smem_ptr, kargs, splitk_batch_offset, i_m, i_n);
    }

    template <typename T, typename = void>
    static constexpr bool kIsAvailableV = true;
    template <typename T>
    static constexpr bool kIsAvailableV<T, std::void_t<decltype(T::kIsAvailable)>> =
        T::kIsAvailable;

    CK_TILE_DEVICE void operator()(const QuantGemmKernelArgs& kargs) const
    {
        if constexpr(!kIsAvailableV<GemmPipeline>)
            ignore = kargs;
        else
            Run_(kargs);
    }
};

} // namespace ck_tile
#if __clang_major__ >= 23
#pragma clang diagnostic pop
#endif
