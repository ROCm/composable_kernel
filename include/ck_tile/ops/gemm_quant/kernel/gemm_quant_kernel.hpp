// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

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
struct is_quantpreshuffle_enabled
{
    static constexpr bool value = false;
};

template <typename T>
struct is_quantpreshuffle_enabled<T, std::void_t<decltype(T::PreshuffleQuant)>>
{
    static constexpr bool value = T::PreshuffleQuant;
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
    index_t k_batch    = 0;
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

template <typename TilePartitioner_,
          typename GemmPipeline_,
          typename EpiloguePipeline_,
          QuantType QuantType_>
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
    static constexpr bool PreshuffleQuant =
        detail::is_quantpreshuffle_enabled<GemmPipeline_>::value;
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

    static constexpr auto kQuantType = QuantType_;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "gemm_quant", gemm_prec_str<ADataType, BDataType>, GemmPipeline::GetName());
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

    struct SplitKBatchOffset
    {
        __device__ SplitKBatchOffset(const QuantGemmKernelArgs& kargs,
                                     const std::size_t k_id = blockIdx.z)
        {
            constexpr auto K1   = TilePartitioner::BlockGemmShape::WarpTile::at(I2);
            const index_t K_t   = amd_wave_read_first_lane(kargs.k_batch * K1);
            const index_t KRead = amd_wave_read_first_lane((kargs.K + K_t - 1) / K_t * K1);

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
                b_k_split_offset = amd_wave_read_first_lane(k_id * KRead * kargs.stride_B);
            }
            else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, BLayout>)
            {
                b_k_split_offset = amd_wave_read_first_lane(k_id * KRead);
            }

            if(k_id < static_cast<uint32_t>(kargs.k_batch - 1))
            {
                splitted_k = amd_wave_read_first_lane(KRead);
            }
            else
            {
                splitted_k = amd_wave_read_first_lane(kargs.K - KRead * (kargs.k_batch - 1));
            }
        }

        index_t a_k_split_offset;
        index_t b_k_split_offset;
        index_t splitted_k;
    };

    CK_TILE_HOST static bool IsSupportedArgument(const QuantGemmKernelArgs& kargs)
    {
        if(kargs.k_batch != 1)
        {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR("Conditions not met for Kbatch >1 !");
            }
            return false;
        }

        if constexpr(kQuantType == QuantType::AQuantGrouped)
        {
            static_assert(std::is_same_v<AQLayout, tensor_layout::gemm::RowMajor>);
            if(kargs.QK_A % GemmPipeline::GetVectorSizeAQ() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("K_A is not a multiple of vector load size for A tensor!");
                }
                return false;
            }
        }

        if constexpr(kQuantType == QuantType::BQuantGrouped)
        {
            static_assert(std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>);
            if(kargs.QK_B % GemmPipeline::GetVectorSizeBQ() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("K_B is not a multiple of vector load size for B tensor!");
                }
                return false;
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
                                                   const BQDataType* bq_ptr,
                                                   CDataType* c_ptr,
                                                   const QuantGemmKernelArgs& kargs,
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

        const auto get_padding_size = [](index_t length, index_t alignment) {
            return ck_tile::integer_least_multiple(length, alignment) - length;
        };

        const auto& aq_tensor_view = [&]() {
            if constexpr(kQuantType == QuantType::AQuantGrouped && PreshuffleQuant)
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
                    TilePartitioner::BlockGemmShape::WarpTile::at(I0) * GemmPipeline::KPerBlockAQ;
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
            else if constexpr(kQuantType == QuantType::AQuantGrouped && !PreshuffleQuant)
            {
                static_assert(std::is_same_v<AQLayout, tensor_layout::gemm::RowMajor>);
                return make_naive_tensor_view<address_space_enum::global>(
                    aq_ptr,
                    make_tuple(kargs.M, kargs.QK_A),
                    make_tuple(kargs.stride_AQ, 1),
                    number<GemmPipeline::GetVectorSizeAQ()>{},
                    number<1>{});
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
                return nullptr; // TODO: use some other "empty" type for this
            }
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
                    if constexpr(PreshuffleB)
                    {
                        index_t kFlatK =
                            GemmPipeline::flatKPerWarp *
                            (splitk_batch_offset.splitted_k /
                             TilePartitioner::BlockGemmShape::WarpTile::at(number<2>{}));
                        index_t kFlatN = kargs.N * kargs.K / kFlatK;

                        return make_naive_tensor_view<address_space_enum::global>(
                            b_ptr,
                            make_tuple(kFlatN, kFlatK),
                            make_tuple(kFlatK, 1),
                            number<GemmPipeline::GetVectorSizeB()>{},
                            number<1>{});
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
            }
        }();

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
            else if constexpr(kQuantType == QuantType::BQuantGrouped)
            {
                static_assert(std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>);
                using QuantGroupSize = remove_cvref_t<typename GemmPipeline::QuantGroupSize>;
                return make_naive_tensor_view<address_space_enum::global>(
                    bq_ptr,
                    make_tuple(kargs.QK_B, integer_divide_ceil(kargs.N, QuantGroupSize::kN)),
                    make_tuple(1, kargs.stride_BQ),
                    number<GemmPipeline::GetVectorSizeBQ()>{},
                    number<1>{});
            }
            else
            {
                return nullptr; // TODO: use some other "empty" type for this
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

        return make_tuple(
            a_tensor_view, aq_tensor_view, b_tensor_view, bq_tensor_view, c_tensor_view);
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

        // no padding
        const auto& aq_pad_view = [&]() { return views.at(I1); }();

        const auto& b_flat_view = views.at(I2); // not applying any padding to flat B view

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

        // no padding
        const auto& bq_pad_view = [&]() { return views.at(I3); }();

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
        if constexpr(PreshuffleB)
        {

            return make_tuple(a_pad_view, aq_pad_view, b_flat_view, bq_pad_view, c_pad_view);
        }
        else
        {
            return make_tuple(a_pad_view, aq_pad_view, b_pad_view, bq_pad_view, c_pad_view);
        }
    }

    template <typename PadView>
    CK_TILE_DEVICE static auto
    MakeGemmTileWindows(const PadView& views, const index_t i_m, const index_t i_n)
    {

        const auto& a_pad_view     = views.at(I0);
        const auto& aq_pad_view    = views.at(I1);
        const auto& b_pad_view     = views.at(I2);
        const auto& bq_pad_view    = views.at(I3);
        const auto& c_pad_view     = views.at(I4);
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
            if constexpr(kQuantType == QuantType::AQuantGrouped && PreshuffleQuant)
            {
                static_assert(std::is_same_v<AQLayout, tensor_layout::gemm::RowMajor>);
                using QuantGroupSize   = remove_cvref_t<typename GemmPipeline::QuantGroupSize>;
                constexpr auto block_m = TilePartitioner::MPerBlock;
                constexpr auto warp_m  = TilePartitioner::BlockGemmShape::WarpTile::at(I0);
                constexpr auto aqk_per_block = TilePartitioner::KPerBlock / QuantGroupSize::kK;
                constexpr auto tile_window_width =
                    ck_tile::integer_least_multiple(warp_m * aqk_per_block, get_warp_size());
                constexpr auto tile_window_height = block_m / warp_m;
                auto block_m_idx                  = i_m / block_m;
                return make_tile_window(
                    aq_pad_view,
                    make_tuple(number<tile_window_height>{}, number<tile_window_width>{}),
                    {block_m_idx * tile_window_height, 0});
            }
            else if constexpr(kQuantType == QuantType::AQuantGrouped && !PreshuffleQuant)
            {
                static_assert(std::is_same_v<AQLayout, tensor_layout::gemm::RowMajor>);
                using QuantGroupSize   = remove_cvref_t<typename GemmPipeline::QuantGroupSize>;
                constexpr auto block_m = TilePartitioner::MPerBlock;
                constexpr auto block_k = TilePartitioner::KPerBlock;
                return make_tile_window(
                    aq_pad_view,
                    make_tuple(number<block_m>{}, number<block_k / QuantGroupSize::kK>{}),
                    {i_m, 0});
            }
            else if constexpr(kQuantType == QuantType::RowColQuant)
            {
                return make_tile_window(aq_pad_view,
                                        make_tuple(number<TilePartitioner::MPerBlock>{},
                                                   number<TilePartitioner::NPerBlock>{}),
                                        {i_m, i_n});
            }
            else
            {
                return nullptr; // TODO: use some other "empty" type?
            }
        }();

        const auto& b_block_window = [&]() {
            if constexpr(PreshuffleB)
            {

                return make_tile_window(
                    b_pad_view,
                    make_tuple(number<GemmPipeline::flatNPerWarp>{},
                               number<GemmPipeline::flatKPerWarp>{}),
                    {static_cast<int>(i_n / TilePartitioner::BlockGemmShape::WarpTile::at(I1)), 0});
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

        const auto& bq_block_window = [&]() {
            if constexpr(kQuantType == QuantType::RowColQuant)
            {
                return make_tile_window(bq_pad_view,
                                        make_tuple(number<TilePartitioner::MPerBlock>{},
                                                   number<TilePartitioner::NPerBlock>{}),
                                        {i_m, i_n});
            }
            else if constexpr(kQuantType == QuantType::BQuantGrouped)
            {
                static_assert(std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>);
                using QuantGroupSize = remove_cvref_t<typename GemmPipeline::QuantGroupSize>;
                return make_tile_window(
                    bq_pad_view,
                    make_tuple(number<TilePartitioner::KPerBlock / QuantGroupSize::kK>{},
                               number<TilePartitioner::NPerBlock / QuantGroupSize::kN>{}),
                    {0, i_n / QuantGroupSize::kN});
            }
            else
            {
                return nullptr; // TODO: use some other "empty" type here
            }
        }();

        auto c_block_window = make_tile_window(
            c_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {i_m, i_n});

        return make_tuple(
            a_block_window, aq_block_window, b_block_window, bq_block_window, c_block_window);
    }

    /**
     * @brief Runs single GEMM problem cooperatively by whole workgroup.
     *
     * @param a_ptr input A pointer
     * @param b_ptr input B pointer
     * @param aq_ptr input AQ pointer
     * @param bq_ptr input BQ pointer
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
                                       const BQDataType* bq_ptr,
                                       CDataType* c_ptr,
                                       void* smem_ptr_0,
                                       const QuantGemmKernelArgs& kargs,
                                       const SplitKBatchOffset& splitk_batch_offset,
                                       const index_t block_idx_m,
                                       const index_t block_idx_n)
    {
        // Create Gemm tensor views, pad views and tile windows
        const auto& gemm_tensor_views_tuple = MakeGemmTensorViews<DstInMemOp>(
            a_ptr, b_ptr, aq_ptr, bq_ptr, c_ptr, kargs, splitk_batch_offset);

        const auto& gemm_pad_views = MakeGemmPadViews(gemm_tensor_views_tuple);
        auto gemm_tile_windows     = MakeGemmTileWindows(gemm_pad_views, block_idx_m, block_idx_n);

        const index_t num_loop =
            amd_wave_read_first_lane(TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k));

        // Run GEMM cooperatively by whole workgroup.
        const auto& a_block_window = gemm_tile_windows.at(I0);
        const auto& b_block_window = gemm_tile_windows.at(I2);

        const auto& c_block_tile = [&]() {
            if constexpr(kQuantType == QuantType::AQuantGrouped)
            {
                const auto& aq_block_window = gemm_tile_windows.at(I1);
                return GemmPipeline{}.template operator()(
                    a_block_window, b_block_window, aq_block_window, kargs.M, num_loop, smem_ptr_0);
            }
            else if constexpr(kQuantType == QuantType::BQuantGrouped)
            {
                const auto& bq_block_window = gemm_tile_windows.at(I3);
                return GemmPipeline{}.template operator()(
                    a_block_window, b_block_window, bq_block_window, num_loop, smem_ptr_0);
            }
            else if constexpr(kQuantType == QuantType::RowColQuant ||
                              kQuantType == QuantType::TensorQuant)
            {
                return GemmPipeline{}.template operator()(
                    a_block_window, b_block_window, num_loop, smem_ptr_0);
            }
        }();

        // Run Epilogue Pipeline
        auto& c_block_window = gemm_tile_windows.at(I4);

        if constexpr(kQuantType == QuantType::AQuantGrouped ||
                     kQuantType == QuantType::BQuantGrouped)
        {
            EpiloguePipeline{}(c_block_window, c_block_tile, c_block_window, smem_ptr_0);
        }
        else if constexpr(kQuantType == QuantType::RowColQuant)
        {
            const auto& aq_block_window = gemm_tile_windows.at(I1);
            const auto& bq_block_window = gemm_tile_windows.at(I3);
            EpiloguePipeline{}(c_block_window,
                               c_block_tile,
                               c_block_window,
                               smem_ptr_0,
                               aq_block_window,
                               bq_block_window);
        }
        else if constexpr(kQuantType == QuantType::TensorQuant)
        {
            // TODO: why doesn't readfirstlane work here?
            // const AccDataType aq_scale =
            //     __builtin_amdgcn_readfirstlane(type_convert<AccDataType>(*aq_ptr));
            // const AccDataType bq_scale =
            //     __builtin_amdgcn_readfirstlane(type_convert<AccDataType>(*bq_ptr));
            const AccDataType aq_scale = type_convert<AccDataType>(*aq_ptr);
            const AccDataType bq_scale = type_convert<AccDataType>(*bq_ptr);
            EpiloguePipeline{}(
                c_block_window, c_block_tile, c_block_window, smem_ptr_0, aq_scale, bq_scale);
        }
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
    CK_TILE_DEVICE static void RunGemm2LDS(const ADataType* a_ptr,
                                           const BDataType* b_ptr,
                                           const AQDataType* aq_ptr,
                                           const BQDataType* bq_ptr,
                                           CDataType* c_ptr,
                                           void* smem_ptr_0,
                                           void* smem_ptr_1,
                                           const QuantGemmKernelArgs& kargs,
                                           const SplitKBatchOffset& splitk_batch_offset,
                                           const index_t block_idx_m,
                                           const index_t block_idx_n)
    {
        // Create Gemm tensor views, pad views and tile windows
        const auto& gemm_tensor_views_tuple = MakeGemmTensorViews<DstInMemOp>(
            a_ptr, b_ptr, aq_ptr, bq_ptr, c_ptr, kargs, splitk_batch_offset);

        const auto& gemm_pad_views = MakeGemmPadViews(gemm_tensor_views_tuple);
        auto gemm_tile_windows     = MakeGemmTileWindows(gemm_pad_views, block_idx_m, block_idx_n);

        const index_t num_loop = __builtin_amdgcn_readfirstlane(
            TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k));

        // Run GEMM cooperatively by whole workgroup.
        const auto& a_block_window = gemm_tile_windows.at(I0);
        const auto& b_block_window = gemm_tile_windows.at(I2);

        const auto& c_block_tile = [&]() {
            if constexpr(kQuantType == QuantType::BQuantGrouped)
            {
                const auto& bq_block_window = gemm_tile_windows.at(I3);
                return GemmPipeline{}.template operator()(a_block_window,
                                                          b_block_window,
                                                          bq_block_window,
                                                          num_loop,
                                                          smem_ptr_0,
                                                          smem_ptr_1);
            }
            else
            {
                return nullptr;
            }
        }();

        // Run Epilogue Pipeline
        auto& c_block_window = gemm_tile_windows.at(I4);

        if constexpr(kQuantType == QuantType::BQuantGrouped)
        {
            EpiloguePipeline{}(c_block_window, c_block_tile, c_block_window, smem_ptr_0);
        }
        else
        {
            return;
            // throw std::runtime_error("DoubleSmemBuffer Not implemented for AQuantGrouped or
            // RowColQuant"); static_assert(kQuantType == QuantType::BQuantGrouped,
            // "DoubleSmemBuffer Not implemented");
        }
    }

    CK_TILE_DEVICE void operator()(QuantGemmKernelArgs kargs) const
    {
        const auto blockId  = amd_wave_read_first_lane(blockIdx.x);
        const auto [iM, iN] = TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(blockId);
        const index_t i_m   = amd_wave_read_first_lane(iM * TilePartitioner::MPerBlock);
        const index_t i_n   = amd_wave_read_first_lane(iN * TilePartitioner::NPerBlock);

        const SplitKBatchOffset splitk_batch_offset(kargs);
        // options
        const ADataType* a_ptr   = static_cast<const ADataType*>(kargs.a_ptr);
        const BDataType* b_ptr   = static_cast<const BDataType*>(kargs.b_ptr);
        const AQDataType* aq_ptr = static_cast<const AQDataType*>(kargs.aq_ptr);
        const BQDataType* bq_ptr = static_cast<const BQDataType*>(kargs.bq_ptr);
        CDataType* c_ptr         = static_cast<CDataType*>(kargs.c_ptr);

        // allocate LDS
        __shared__ char smem_ptr_0[GetSmemSize()];
        assert(kargs.k_batch == 1);
        if constexpr(GemmPipeline::DoubleSmemBuffer == true)
        {
            __shared__ char smem_ptr_1[GetSmemSize()];

            RunGemm2LDS(a_ptr,
                        b_ptr,
                        aq_ptr,
                        bq_ptr,
                        c_ptr,
                        smem_ptr_0,
                        smem_ptr_1,
                        kargs,
                        splitk_batch_offset,
                        i_m,
                        i_n);
        }
        else
        {
            RunGemm(a_ptr,
                    b_ptr,
                    aq_ptr,
                    bq_ptr,
                    c_ptr,
                    smem_ptr_0,
                    kargs,
                    splitk_batch_offset,
                    i_m,
                    i_n);
        }
    }
};

} // namespace ck_tile
