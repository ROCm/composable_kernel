// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_universal_gemm_as_bs_cr.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

#include <tuple>
#include <type_traits>

namespace ck_tile {

template <typename T, typename = void>
struct has_a_tile_access_pattern : std::false_type
{
};

template <typename T>
struct has_a_tile_access_pattern<T, std::void_t<decltype(T::ATileAccessPattern)>> : std::true_type
{
};

template <typename T, typename = void>
struct has_b_tile_access_pattern : std::false_type
{
};

template <typename T>
struct has_b_tile_access_pattern<T, std::void_t<decltype(T::BTileAccessPattern)>> : std::true_type
{
};

template <typename D, typename P, typename = void>
struct GetLdsADataType
{
    using type =
        std::conditional_t<std::is_same_v<remove_cvref_t<typename P::ADataType>, pk_int4_t>,
                           remove_cvref_t<typename P::BDataType>,
                           remove_cvref_t<typename P::ADataType>>;
};

template <typename D, typename P>
struct GetLdsADataType<D, P, std::void_t<typename D::template LdsADataType<P>>>
{
    using type = typename D::template LdsADataType<P>;
};

template <typename D, typename P, typename = void>
struct GetLdsBDataType
{
    using type =
        std::conditional_t<std::is_same_v<remove_cvref_t<typename P::BDataType>, pk_int4_t> ||
                               std::is_same_v<remove_cvref_t<typename P::BDataType>, pk_fp4_raw_t>,
                           remove_cvref_t<typename P::ADataType>,
                           remove_cvref_t<typename P::BDataType>>;
};

template <typename D, typename P>
struct GetLdsBDataType<D, P, std::void_t<typename D::template LdsBDataType<P>>>
{
    using type = typename D::template LdsBDataType<P>;
};

// Trait combining both LDS data types
template <typename D, typename P>
struct LdsDataTypeTraits
{
    using AType = typename GetLdsADataType<D, P>::type;
    using BType = typename GetLdsBDataType<D, P>::type;
};

template <typename Derived>
struct UniversalGemmBasePolicy
{
    // Trait for LDS data types: use Derived's version if defined, otherwise use default
    template <typename Problem>
    using LdsDataTypes_ = LdsDataTypeTraits<Derived, Problem>;

    template <typename Problem>
    using ALdsDataType_ = typename LdsDataTypes_<Problem>::AType;

    template <typename Problem>
    using BLdsDataType_ = typename LdsDataTypes_<Problem>::BType;

#if defined(__gfx950__) || defined(__gfx125__)
    // The combination of pk_int4_t and transposed loading causes numerical errors.
    // Therefore do not use transposed loading in this case.
    // Also, transpose load (ds_read_tr) requires specific tile distribution patterns
    // that only work for certain K warp tile sizes based on data type size:
    // - For 1-byte types (fp8/bf8): K warp tile <= 64
    // - For 2-byte types (fp16/bf16): K warp tile <= 32
    template <typename T>
    static constexpr bool supports_transpose_load =
        std::is_same_v<T, pk_fp4_t> || std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t> ||
        std::is_same_v<T, fp8_t> || std::is_same_v<T, bf8_t> || std::is_same_v<T, int8_t> ||
        std::is_same_v<T, uint8_t>;

    template <typename Problem>
    static constexpr bool is_a_load_tr = []() {
        using ADataType = remove_cvref_t<typename Problem::ADataType>;
        using BDataType = remove_cvref_t<typename Problem::BDataType>;
        if constexpr(!supports_transpose_load<ADataType> || std::is_same_v<BDataType, pk_int4_t>)
            return false;
        else if constexpr(!std::is_same_v<remove_cvref_t<typename Problem::ALayout>,
                                          tensor_layout::gemm::ColumnMajor>)
            return false;
        else
        {
#if defined(__gfx950__)
            using WarpTile                  = typename Problem::BlockGemmShape::WarpTile;
            constexpr index_t kKWarpTile    = WarpTile::at(number<2>{});
            constexpr index_t kMaxKWarpTile = (sizeof(ADataType) == 1) ? 64 : 32;
            return kKWarpTile <= kMaxKWarpTile;
#else
            return true;
#endif
        }
    }();

    template <typename Problem>
    static constexpr bool is_b_load_tr = []() {
        using BLdsDataType = BLdsDataType_<Problem>;
        using BDataType    = remove_cvref_t<typename Problem::BDataType>;
        if constexpr(!supports_transpose_load<BLdsDataType> || std::is_same_v<BDataType, pk_int4_t>)
            return false;
        else if constexpr(!std::is_same_v<remove_cvref_t<typename Problem::BLayout>,
                                          tensor_layout::gemm::RowMajor>)
            return false;
        else
        {
#if defined(__gfx950__)
            using WarpTile                  = typename Problem::BlockGemmShape::WarpTile;
            constexpr index_t kKWarpTile    = WarpTile::at(number<2>{});
            constexpr index_t kMaxKWarpTile = (sizeof(BLdsDataType) == 1) ? 64 : 32;
            return kKWarpTile <= kMaxKWarpTile;
#else
            return true;
#endif
        }
    }();
#else
    template <typename Problem>
    static constexpr bool is_a_load_tr = false;
    template <typename Problem>
    static constexpr bool is_b_load_tr = false;
#endif

    template <typename T>
    using has_bcastpolicy_type = decltype(T::BCastPolicy);

    template <typename Problem>
    static constexpr bool IsBCastPolicyBeforeLDSWrite_v = [] {
        if constexpr(is_detected<has_bcastpolicy_type, Problem>{})
        {
            return Problem::BCastPolicy == CastPolicy::BeforeLDSWrite;
        }
        else
        {
            return false;
        }
    }();

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};

#if defined(__gfx125__)
    // change to warp raked for lds write bank conflict elimination
    static constexpr auto DefaultATileAccessPattern = tile_distribution_pattern::warp_raked;
    static constexpr auto DefaultBTileAccessPattern = tile_distribution_pattern::warp_raked;
#else
    // Default tile access patterns
    static constexpr auto DefaultATileAccessPattern = tile_distribution_pattern::thread_raked;
    static constexpr auto DefaultBTileAccessPattern = tile_distribution_pattern::thread_raked;
#endif
    static constexpr auto getATileAccessPattern()
    {
        if constexpr(has_a_tile_access_pattern<Derived>::value)
            return Derived::ATileAccessPattern;
        else
            return DefaultATileAccessPattern;
    }

    static constexpr auto getBTileAccessPattern()
    {
        if constexpr(has_b_tile_access_pattern<Derived>::value)
            return Derived::BTileAccessPattern;
        else
            return DefaultBTileAccessPattern;
    }

    // =====================================================
    // Architecture-specific A LDS Block Descriptor implementations
    // =====================================================

    // Default implementation for gfx9 (Wave64) with XOR swizzle
    template <typename Problem, typename ArchTag>
    CK_TILE_DEVICE static constexpr auto MakeALdsBlockDescriptorImpl(ArchTag)
    {
        using ALayout               = remove_cvref_t<typename Problem::ALayout>;
        using ADataType             = ALdsDataType_<Problem>;
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        if constexpr(is_a_load_tr<Problem>)
        {
            // TODO: better lds descriptor for performance
            constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor( //
                make_tuple(number<KPerBlock>{}, number<MPerBlock>{}),
                make_tuple(number<MPerBlock>{}, number<1>{}),
                number<MPerBlock>{},
                number<1>{});
            return a_lds_block_desc_0;
        }
        else
        {
            // Only use this ColumnMajor layout for Wave64 mode (gfx9)
            constexpr auto Wave64 = get_warp_size() == 64;
            if constexpr(Wave64 &&
                         std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::ColumnMajor>)
            {
                // kfold and mpair dimension is not always required.
                // more dimension in merge_transform increase the difficulty of generating immarg
                // offset for compiler.
                constexpr index_t BlockSize   = Problem::kBlockSize;
                constexpr index_t VecLoadSize = GetVectorSizeA<Problem>();
                using TileEncodingPattern =
                    tile_distribution_encoding_pattern_2d<BlockSize,
                                                          KPerBlock,
                                                          MPerBlock,
                                                          VecLoadSize,
                                                          getATileAccessPattern()>;
                // AK1: the shuffled tile dstr has shape <X1, Y2>, use Y2 as AK1
                constexpr auto AK1 = number<TileEncodingPattern::Y2>{};
                constexpr auto AK0 = number<KPerBlock / AK1>{};
                // How the M dimension is split across threads
                constexpr auto M0 = TileEncodingPattern::X0; // # of threads in M dim
                constexpr auto M1 = number<MPerBlock / M0>{};

                // Get the warp tile size
                using WarpTile         = typename Problem::BlockGemmShape::WarpTile;
                constexpr auto MPerXdl = number<WarpTile::at(I0)>{};

                // Number of threads covering K dimension
                constexpr auto KThreadWrite     = TileEncodingPattern::Y0 * TileEncodingPattern::Y1;
                constexpr auto K0PerThreadWrite = AK0 / KThreadWrite;
                constexpr auto KThreadRead      = get_warp_size() / MPerXdl;
                constexpr auto K0PerThreadRead  = AK0 / KThreadRead;

                // check if we exceed all LDS banks
                constexpr auto LdsBanksWidth = get_n_lds_banks() * get_n_dwords_per_128b();
                constexpr auto kfold         = (AK1 * M0 * sizeof(ADataType) > LdsBanksWidth)
                                                   ? 1
                                                   : LdsBanksWidth / (AK1 * M0 * sizeof(ADataType));
                constexpr auto KThreadReadPerm =
                    (kfold * K0PerThreadWrite / K0PerThreadRead) > 1
                        ? KThreadRead / (kfold * K0PerThreadWrite / K0PerThreadRead)
                        : KThreadRead;

                // 1<=mpair<=n0
                constexpr auto mpair =
                    (AK1 * MPerXdl * sizeof(ADataType) > LdsBanksWidth)
                        ? 1
                        : ((LdsBanksWidth / (AK1 * MPerXdl * sizeof(ADataType))) > M0
                               ? M0
                               : LdsBanksWidth / (AK1 * MPerXdl * sizeof(ADataType)));

                constexpr auto a_lds_block_desc = make_naive_tensor_descriptor_packed(
                    make_tuple(number<KThreadWrite / kfold / KThreadReadPerm>{},
                               number<K0PerThreadWrite>{},
                               number<KThreadReadPerm * M1>{},
                               number<kfold * M0 / mpair>{},
                               number<mpair>{},
                               AK1),
                    AK1);

                constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
                    a_lds_block_desc,
                    make_tuple(make_pass_through_transform(
                                   number<KThreadWrite / kfold / KThreadReadPerm>{}),
                               make_pass_through_transform(number<K0PerThreadWrite>{}),
                               make_xor_transform(make_tuple(number<KThreadReadPerm * M1>{},
                                                             number<kfold * M0 / mpair>{})),
                               make_pass_through_transform(number<mpair>{}),
                               make_pass_through_transform(AK1)),
                    make_tuple(sequence<0>{},
                               sequence<1>{},
                               sequence<2, 3>{},
                               sequence<4>{},
                               sequence<5>{}),
                    make_tuple(sequence<0>{},
                               sequence<1>{},
                               sequence<2, 3>{},
                               sequence<4>{},
                               sequence<5>{}));

                constexpr auto a_lds_block_desc_unmerged = transform_tensor_descriptor(
                    a_lds_block_desc_permuted,
                    make_tuple(
                        make_pass_through_transform(
                            number<KThreadWrite / kfold / KThreadReadPerm>{}),
                        make_pass_through_transform(number<K0PerThreadWrite>{}),
                        make_unmerge_transform(make_tuple(number<KThreadReadPerm>{}, number<M1>{})),
                        make_unmerge_transform(make_tuple(number<kfold>{}, number<M0 / mpair>{})),
                        make_pass_through_transform(number<mpair>{}),
                        make_pass_through_transform(AK1)),
                    make_tuple(sequence<0>{},
                               sequence<1>{},
                               sequence<2>{},
                               sequence<3>{},
                               sequence<4>{},
                               sequence<5>{}),
                    make_tuple(sequence<1>{},
                               sequence<2>{},
                               sequence<0, 3>{},
                               sequence<4, 5>{},
                               sequence<6>{},
                               sequence<7>{}));

                constexpr auto a_lds_block_desc_ak0_m_ak1 = transform_tensor_descriptor(
                    a_lds_block_desc_unmerged,
                    make_tuple(make_merge_transform_v3_division_mod(
                                   make_tuple(number<KThreadReadPerm>{},
                                              number<KThreadWrite / kfold / KThreadReadPerm>{},
                                              number<kfold>{},
                                              number<K0PerThreadWrite>{},
                                              AK1)),
                               make_merge_transform_v3_division_mod(make_tuple(
                                   number<M0 / mpair>{}, number<mpair>{}, number<M1>{}))),
                    make_tuple(sequence<0, 1, 4, 2, 7>{}, sequence<5, 6, 3>{}),
                    make_tuple(sequence<1>{}, sequence<0>{}));

                return a_lds_block_desc_ak0_m_ak1;
            }
            else // A is in RowMajor
            {
                constexpr index_t KPack     = Derived::template GetSmemPackA<Problem>();
                constexpr auto DataTypeSize = sizeof(ADataType);
                constexpr index_t MLdsLayerRequired =
                    get_n_lds_banks() * get_n_dwords_per_128b() / KPerBlock / DataTypeSize;
                constexpr auto MLdsLayer = max(1, MLdsLayerRequired);

                constexpr index_t NBanks = get_n_lds_banks();
                static_assert(NBanks == 32 || NBanks == 64, "Unexpected LDS bank count");
                constexpr index_t RowMul = (NBanks == 64) ? 2 : 1;

                constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
                    make_tuple(number<KPerBlock / KPack * MLdsLayer>{},
                               number<MPerBlock / MLdsLayer>{},
                               number<KPack>{}),
                    make_tuple(number<KPack>{}, number<KPerBlock * MLdsLayer>{}, number<1>{}),
                    number<KPack>{},
                    number<1>{});

                constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
                    a_lds_block_desc_0,
                    make_tuple(
                        make_xor_transform(make_tuple(number<MPerBlock / MLdsLayer * RowMul>{},
                                                      number<KPerBlock / KPack * MLdsLayer>{})),
                        make_pass_through_transform(number<KPack>{})),
                    make_tuple(sequence<1, 0>{}, sequence<2>{}),
                    make_tuple(sequence<1, 0>{}, sequence<2>{}));

                constexpr auto a_lds_block_desc_xk0_mnldslayer_mn_xk1 = transform_tensor_descriptor(
                    a_lds_block_desc_permuted,
                    make_tuple(make_unmerge_transform(
                                   make_tuple(number<MLdsLayer>{}, number<KPerBlock / KPack>{})),
                               make_pass_through_transform(number<MPerBlock / MLdsLayer>{}),
                               make_pass_through_transform(number<KPack>{})),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                    make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

                constexpr auto a_lds_block_desc = transform_tensor_descriptor(
                    a_lds_block_desc_xk0_mnldslayer_mn_xk1,
                    make_tuple(make_merge_transform_v3_division_mod(make_tuple(
                                   number<MPerBlock / MLdsLayer>{}, number<MLdsLayer>{})),
                               make_merge_transform_v3_division_mod(
                                   make_tuple(number<KPerBlock / KPack>{}, number<KPack>{}))),
                    make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));

                return a_lds_block_desc;
            }
        }
    }

    // gfx125 specific implementation (uses padding instead of XOR for bank conflict avoidance)
    // TODO: need support fp4 transpose load
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeALdsBlockDescriptorImpl(gfx125_t)
    {
        using ADataType             = ALdsDataType_<Problem>;
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        constexpr auto DataTypeSize = sizeof(ADataType);

        constexpr index_t PackedSize = numeric_traits<ADataType>::PackedSize;

        // for gfx1250, always use KPack based on 128bits
        constexpr index_t BytesPerDword = sizeof(int32_t);
        constexpr index_t KPack =
            get_n_dwords_per_128b() * BytesPerDword / DataTypeSize * PackedSize;

        if constexpr(is_a_load_tr<Problem>)
        {
            return MakeALdsBlockDescriptorForTrLoad<Problem>();
        }
        else
        {

            constexpr auto LdsPaddingConfigA = GetLdsPaddingConfig<Problem, true>();

            constexpr auto IsNeedPadding = LdsPaddingConfigA[I0];
            // set to -1 to make sure PaddingDataAmount = 0 when IsNeedPadding = false
            constexpr auto PaddingAmount = IsNeedPadding ? LdsPaddingConfigA[I1] : -1;

            constexpr index_t MLdsLayerRequired =
                get_n_lds_banks() * get_n_dwords_per_128b() / KPerBlock / DataTypeSize;

            constexpr auto MLdsLayer = max(1, MLdsLayerRequired);

            constexpr auto PaddingDataAmount = (PaddingAmount + 1) * BytesPerDword / DataTypeSize;

            // gfx125: use simple layout without XOR (relies on padding in descriptor)
            constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<MPerBlock / MLdsLayer>{},
                           number<KPerBlock / KPack * MLdsLayer>{},
                           number<KPack>{}),
                make_tuple(number<KPerBlock * MLdsLayer + PaddingDataAmount>{},
                           number<KPack>{},
                           number<1>{}),
                number<KPack>{},
                number<1>{});

            constexpr auto a_lds_block_desc_1 = transform_tensor_descriptor(
                a_lds_block_desc_0,
                make_tuple(make_pass_through_transform(number<MPerBlock / MLdsLayer>{}),
                           make_unmerge_transform(
                               make_tuple(number<MLdsLayer>{}, number<KPerBlock / KPack>{})),
                           make_pass_through_transform(number<KPack>{})),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

            constexpr auto a_lds_block_desc = transform_tensor_descriptor(
                a_lds_block_desc_1,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(number<MPerBlock / MLdsLayer>{}, number<MLdsLayer>{})),
                           make_merge_transform_v3_division_mod(
                               make_tuple(number<KPerBlock / KPack>{}, number<KPack>{}))),
                make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return a_lds_block_desc;
        }
    }

    // =====================================================
    // Main entry point: dispatches based on architecture
    // =====================================================
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        return MakeALdsBlockDescriptorImpl<Problem>(get_device_arch());
    }

    // =====================================================
    // Architecture-specific B LDS Block Descriptor implementations
    // =====================================================

    // Default implementation for gfx9 (Wave64) with XOR swizzle
    template <typename Problem, typename ArchTag>
    CK_TILE_DEVICE static constexpr auto MakeBLdsBlockDescriptorImpl(ArchTag)
    {
        using BLayout                              = remove_cvref_t<typename Problem::BLayout>;
        constexpr bool IsBCastPolicyBeforeLDSWrite = IsBCastPolicyBeforeLDSWrite_v<Problem>;
        using BDataType                            = std::conditional_t<IsBCastPolicyBeforeLDSWrite,
                                                                        typename Problem::ADataType,
                                                                        BLdsDataType_<Problem>>;
        constexpr index_t NPerBlock                = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock                = Problem::BlockGemmShape::kK;

        if constexpr(is_b_load_tr<Problem>)
        {

            // TODO: better lds descriptor for performance
            constexpr auto b_lds_block_desc_0 =
                make_naive_tensor_descriptor(make_tuple(number<KPerBlock>{}, number<NPerBlock>{}),
                                             make_tuple(number<NPerBlock>{}, number<1>{}),
                                             number<NPerBlock>{},
                                             number<1>{});
            return b_lds_block_desc_0;
        }
        else
        {
            // Only use this RowMajor layout for Wave64 mode (gfx9)
            constexpr auto Wave64 = get_warp_size() == 64;
            if constexpr(Wave64 && std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                constexpr index_t BlockSize   = Problem::kBlockSize;
                constexpr index_t VecLoadSize = GetVectorSizeB<Problem>();
                using TileEncodingPattern =
                    tile_distribution_encoding_pattern_2d<BlockSize,
                                                          KPerBlock,
                                                          NPerBlock,
                                                          VecLoadSize,
                                                          getBTileAccessPattern()>;
                // BK1: the shuffled tile dstr has shape <X1, Y2>, use Y2 as BK1
                constexpr auto BK1 = number<TileEncodingPattern::Y2>{};
                constexpr auto BK0 = number<KPerBlock / BK1>{};
                // How threads access data on N dim
                constexpr auto N0 = TileEncodingPattern::X0; // # of threads in N dim
                constexpr auto N1 = number<NPerBlock / N0>{};

                // Get NPerXdl, the warp tile size
                using WarpTile         = typename Problem::BlockGemmShape::WarpTile;
                constexpr auto NPerXdl = number<WarpTile::at(I1)>{};

                // Number of threads covering K dimension
                constexpr auto KThreadWrite     = TileEncodingPattern::Y0 * TileEncodingPattern::Y1;
                constexpr auto K0PerThreadWrite = BK0 / KThreadWrite;
                constexpr auto KThreadRead      = get_warp_size() / NPerXdl;
                constexpr auto K0PerThreadRead  = BK0 / KThreadRead;

                // check if we exceed all LDS banks
                constexpr auto LdsBanksWidth = get_n_lds_banks() * get_n_dwords_per_128b();
                constexpr auto kfold         = (BK1 * N0 * sizeof(BDataType) > LdsBanksWidth)
                                                   ? 1
                                                   : LdsBanksWidth / (BK1 * N0 * sizeof(BDataType));
                constexpr auto KThreadReadPerm =
                    (kfold * K0PerThreadWrite / K0PerThreadRead) > 1
                        ? KThreadRead / (kfold * K0PerThreadWrite / K0PerThreadRead)
                        : KThreadRead;

                // 1<=npair<=n0
                constexpr auto npair =
                    (BK1 * NPerXdl * sizeof(BDataType) > LdsBanksWidth)
                        ? 1
                        : ((LdsBanksWidth / (BK1 * NPerXdl * sizeof(BDataType))) > N0
                               ? N0
                               : LdsBanksWidth / (BK1 * NPerXdl * sizeof(BDataType)));

                constexpr auto b_lds_block_desc = make_naive_tensor_descriptor_packed(
                    make_tuple(number<KThreadWrite / kfold / KThreadReadPerm>{},
                               number<K0PerThreadWrite>{},
                               number<KThreadReadPerm * N1>{},
                               number<kfold * N0 / npair>{},
                               number<npair>{},
                               BK1),
                    BK1);

                constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
                    b_lds_block_desc,
                    make_tuple(make_pass_through_transform(
                                   number<KThreadWrite / kfold / KThreadReadPerm>{}),
                               make_pass_through_transform(number<K0PerThreadWrite>{}),
                               make_xor_transform(make_tuple(number<KThreadReadPerm * N1>{},
                                                             number<kfold * N0 / npair>{})),
                               make_pass_through_transform(number<npair>{}),
                               make_pass_through_transform(BK1)),
                    make_tuple(sequence<0>{},
                               sequence<1>{},
                               sequence<2, 3>{},
                               sequence<4>{},
                               sequence<5>{}),
                    make_tuple(sequence<0>{},
                               sequence<1>{},
                               sequence<2, 3>{},
                               sequence<4>{},
                               sequence<5>{}));

                constexpr auto b_lds_block_desc_unmerged = transform_tensor_descriptor(
                    b_lds_block_desc_permuted,
                    make_tuple(
                        make_pass_through_transform(
                            number<KThreadWrite / kfold / KThreadReadPerm>{}),
                        make_pass_through_transform(number<K0PerThreadWrite>{}),
                        make_unmerge_transform(make_tuple(number<KThreadReadPerm>{}, number<N1>{})),
                        make_unmerge_transform(make_tuple(number<kfold>{}, number<N0 / npair>{})),
                        make_pass_through_transform(number<npair>{}),
                        make_pass_through_transform(BK1)),
                    make_tuple(sequence<0>{},
                               sequence<1>{},
                               sequence<2>{},
                               sequence<3>{},
                               sequence<4>{},
                               sequence<5>{}),
                    make_tuple(
                        sequence<1>{},    // 0: K0PerThreadWrite
                        sequence<2>{},    // 1: KThreadReadPerm
                        sequence<0, 3>{}, // 2: KThreadWrite / kfold / KThreadReadPerm,  3: N1
                        sequence<4, 5>{}, // 4: kfold,  5: N0 / npair
                        sequence<6>{},    // 6: npair
                        sequence<7>{}));  // 7: BK1

                constexpr auto b_lds_block_desc_nk = transform_tensor_descriptor(
                    b_lds_block_desc_unmerged,
                    make_tuple(make_merge_transform_v3_division_mod(
                                   make_tuple(number<KThreadReadPerm>{},
                                              number<KThreadWrite / kfold / KThreadReadPerm>{},
                                              number<kfold>{},
                                              number<K0PerThreadWrite>{},
                                              BK1)),
                               make_merge_transform_v3_division_mod(make_tuple(
                                   number<N0 / npair>{}, number<npair>{}, number<N1>{}))),
                    make_tuple(sequence<0, 1, 4, 2, 7>{}, sequence<5, 6, 3>{}),
                    make_tuple(sequence<1>{}, sequence<0>{}));

                return b_lds_block_desc_nk;
            }
            else // B is Column Major
            {
                constexpr index_t KPack        = GetSmemPackB<Problem>();
                constexpr auto BK0             = number<KPerBlock / KPack>{};
                constexpr auto DataTypeSize    = sizeof(BDataType);
                constexpr uint64_t MinLdsLayer = 1ULL;
                constexpr auto NLdsLayer =
                    max(MinLdsLayer,
                        get_n_lds_banks() * get_n_dwords_per_128b() / KPerBlock / DataTypeSize);

                constexpr index_t NBanks = get_n_lds_banks();
                static_assert(NBanks == 32 || NBanks == 64, "Unexpected LDS bank count");
                constexpr index_t RowMul = (NBanks == 64) ? 2 : 1;

                constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
                    make_tuple(BK0 * number<NLdsLayer>{},
                               number<NPerBlock / NLdsLayer>{},
                               number<KPack>{}),
                    make_tuple(number<KPack>{}, number<KPerBlock * NLdsLayer>{}, number<1>{}),
                    number<KPack>{},
                    number<1>{});

                constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
                    b_lds_block_desc_0,
                    make_tuple(
                        make_xor_transform(make_tuple(number<NPerBlock / NLdsLayer * RowMul>{},
                                                      BK0 * number<NLdsLayer>{})),
                        make_pass_through_transform(number<KPack>{})),
                    make_tuple(sequence<1, 0>{}, sequence<2>{}),
                    make_tuple(sequence<1, 0>{}, sequence<2>{}));

                constexpr auto b_lds_block_desc_bk0_nldslayer_n_bk1 = transform_tensor_descriptor(
                    b_lds_block_desc_permuted,
                    make_tuple(make_unmerge_transform(make_tuple(number<NLdsLayer>{}, BK0)),
                               make_pass_through_transform(number<NPerBlock / NLdsLayer>{}),
                               make_pass_through_transform(number<KPack>{})),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                    make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

                constexpr auto b_lds_block_desc = transform_tensor_descriptor(
                    b_lds_block_desc_bk0_nldslayer_n_bk1,
                    make_tuple(
                        make_merge_transform_v3_division_mod(
                            make_tuple(number<NPerBlock / NLdsLayer>{}, number<NLdsLayer>{})),
                        make_merge_transform_v3_division_mod(make_tuple(BK0, number<KPack>{}))),
                    make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
                return b_lds_block_desc;
            }
        }
    }

    // gfx125 specific implementation (uses padding instead of XOR for bank conflict avoidance)
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeBLdsBlockDescriptorImpl(gfx125_t)
    {
        using BDataType = BLdsDataType_<Problem>;

        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        constexpr auto DataTypeSize = sizeof(BDataType);

        constexpr index_t PackedSize = numeric_traits<BDataType>::PackedSize;

        // for gfx1250, always use KPack based on 128bits
        constexpr index_t BytesPerDword = sizeof(int32_t);
        constexpr index_t KPack =
            get_n_dwords_per_128b() * BytesPerDword / DataTypeSize * PackedSize;

        if constexpr(is_b_load_tr<Problem>)
        {
            return MakeBLdsBlockDescriptorForTrLoad<Problem>();
        }
        else
        {
            constexpr auto LdsPaddingConfigB = GetLdsPaddingConfig<Problem, false>();

            constexpr auto IsNeedPadding = LdsPaddingConfigB[I0];
            // set to -1 to make sure PaddingDataAmount = 0 when IsNeedPadding = false
            constexpr auto PaddingAmount = IsNeedPadding ? LdsPaddingConfigB[I1] : -1;

            constexpr index_t NLdsLayerRequired =
                get_n_lds_banks() * get_n_dwords_per_128b() / KPerBlock / DataTypeSize;
            constexpr auto NLdsLayer = max(1, NLdsLayerRequired);

            constexpr auto PaddingDataAmount = (PaddingAmount + 1) * BytesPerDword / DataTypeSize;

            // gfx125: use simple layout without XOR (relies on padding in descriptor)
            constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<NPerBlock / NLdsLayer>{},
                           number<KPerBlock / KPack * NLdsLayer>{},
                           number<KPack>{}),
                make_tuple(number<KPerBlock * NLdsLayer + PaddingDataAmount>{},
                           number<KPack>{},
                           number<1>{}),
                number<KPack>{},
                number<1>{});

            constexpr auto b_lds_block_desc_1 = transform_tensor_descriptor(
                b_lds_block_desc_0,
                make_tuple(make_pass_through_transform(number<NPerBlock / NLdsLayer>{}),
                           make_unmerge_transform(
                               make_tuple(number<NLdsLayer>{}, number<KPerBlock / KPack>{})),
                           make_pass_through_transform(number<KPack>{})),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

            constexpr auto b_lds_block_desc = transform_tensor_descriptor(
                b_lds_block_desc_1,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(number<NPerBlock / NLdsLayer>{}, number<NLdsLayer>{})),
                           make_merge_transform_v3_division_mod(
                               make_tuple(number<KPerBlock / KPack>{}, number<KPack>{}))),
                make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return b_lds_block_desc;
        }
    }

    // =====================================================
    // Main entry point: dispatches based on architecture
    // =====================================================
    /**
     * @brief Create LDS block descriptor for B tensor.
     *
     * @tparam Problem  Gemm pipeline problem.
     * @return B tensor LDS block descriptor.
     */
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        return MakeBLdsBlockDescriptorImpl<Problem>(get_device_arch());
    }

    /**
     * @brief Get the maximum global memory vector load size.
     *
     * @tparam Problem      The UniversalGemmPipelineProblem object.
     * @tparam DataType     The tensor data type we're considering.
     * @tparam MNPerBlock   The MPerBlock or NPerBlock value depending on tensor (A/B).
     * @tparam XPerTile     The contiguous Tile dimension size.
     * @return Maximum DRAM vector load size.
     */
    template <typename Problem,
              typename DataType,
              index_t MNPerBlock,
              index_t XPerTile,
              bool IsWave32Host>
    CK_TILE_HOST_DEVICE static constexpr auto GetGlobalVectorLoadSize()
    {
        constexpr index_t BlockSize = IsWave32Host ? Problem::kBlockSize / 2 : Problem::kBlockSize;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t elements_per_thread = MNPerBlock * KPerBlock / BlockSize;
        constexpr index_t PackedSize =
            ck_tile::numeric_traits<remove_cvref_t<DataType>>::PackedSize;

        // Assume DataType is even!
        if constexpr(XPerTile % (PackedSize * 16 / sizeof(DataType)) == 0 &&
                     elements_per_thread % (PackedSize * 16 / sizeof(DataType)) == 0)
        {
            return (PackedSize * 16 / sizeof(DataType));
        }
        else if constexpr(XPerTile % (PackedSize * 8 / sizeof(DataType)) == 0 &&
                          elements_per_thread % (PackedSize * 8 / sizeof(DataType)) == 0)
        {
            return (PackedSize * 8 / sizeof(DataType));
        }
        else if constexpr(sizeof(DataType) >= PackedSize * 4 &&
                          XPerTile % (PackedSize * 4 / sizeof(DataType)) == 0 &&
                          elements_per_thread % (PackedSize * 4 / sizeof(DataType)) == 0)
        {
            return (PackedSize * 4 / sizeof(DataType));
        }
        else if constexpr(sizeof(DataType) >= PackedSize * 2 &&
                          XPerTile % (PackedSize * 2 / sizeof(DataType)) == 0 &&
                          elements_per_thread % (PackedSize * 2 / sizeof(DataType)) == 0)
        {
            return (PackedSize * 2 / sizeof(DataType));
        }
        else
        {
            return PackedSize;
        }
    }

    template <typename Problem, bool IsWave32Host = false>
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeA()
    {
        using AsLayout              = problem_as_layout_t<Problem>;
        using AsDataType            = problem_as_data_type_t<Problem>;
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        using ALayout   = remove_cvref_t<std::tuple_element_t<number<0>{}, AsLayout>>;
        using ADataType = remove_cvref_t<std::tuple_element_t<number<0>{}, AsDataType>>;

        if constexpr(problem_fixed_vector_size_v<Problem>)
        {
            return Problem::VectorSizeA;
        }
        else if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            return GetGlobalVectorLoadSize<Problem,
                                           ADataType,
                                           MPerBlock,
                                           KPerBlock,
                                           IsWave32Host>();
        }
        else
        {
            return GetGlobalVectorLoadSize<Problem,
                                           ADataType,
                                           MPerBlock,
                                           MPerBlock,
                                           IsWave32Host>();
        }
    }

    template <typename Problem, bool IsWave32Host = false>
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeB()
    {
        using BsLayout              = problem_bs_layout_t<Problem>;
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        using BLayout               = remove_cvref_t<std::tuple_element_t<number<0>{}, BsLayout>>;

        constexpr bool IsBCastPolicyBeforeLDSWrite = IsBCastPolicyBeforeLDSWrite_v<Problem>;
        using BDataType                            = std::conditional_t<IsBCastPolicyBeforeLDSWrite,
                                                                        typename Problem::ADataType,
                                                                        typename Problem::BDataType>;

        if constexpr(Problem::FixedVectorSize)
        {
            return Problem::VectorSizeB;
        }
        else if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            return GetGlobalVectorLoadSize<Problem,
                                           BDataType,
                                           NPerBlock,
                                           NPerBlock,
                                           IsWave32Host>();
        }
        else
        {
            return GetGlobalVectorLoadSize<Problem,
                                           BDataType,
                                           NPerBlock,
                                           KPerBlock,
                                           IsWave32Host>();
        }
    }

    /**
     * @brief Get the vector store size for C tensor.
     *
     * @tparam Problem - Gemm pipeline problem class.
     *
     * @note The vector store size for output C tensor would depend on multiple factors
     *       like its data layout and warp gemm C transposition. In general it would
     *       be the number of consecutive elements in contiguous C dimension hold by
     *       single thread.
     *
     * @return The vector store size for C tensor.
     */
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeC()
    {
        using BlockGemm = remove_cvref_t<decltype(Derived::template GetBlockGemm<Problem>())>;
        using WG        = typename BlockGemm::WarpGemm;

        constexpr bool TransposeC = Problem::TransposeC;
        using CLayout             = typename Problem::CLayout;
        using CWarpDstr           = typename WG::CWarpDstr;

        // N is contiguous dimension
        if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
        {
            if constexpr(TransposeC)
            {
                // In this case each thread has multiple consecutive elements in
                // N dimension, however consecutive threads' elements have stride.
                constexpr index_t NDimY = CWarpDstr::NDimY;
                constexpr auto c_warp_y_lengths =
                    CWarpDstr{}.get_ys_to_d_descriptor().get_lengths();
                static_assert(WG::WarpGemmAttribute::Impl::kCM1PerLane ==
                              c_warp_y_lengths.get(number<NDimY - 1>{}));
                return c_warp_y_lengths.get(number<NDimY - 1>{});
            }
            else
            {
                // In this case each thread has just a single item in Ndim
                return WG::WarpGemmAttribute::Impl::kCNLane / WG::kN;
            }
        }
        // M is contiguous dimension
        else if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::ColumnMajor>)
        {
            if constexpr(TransposeC)
            {
                // In this case each thread has just a single item in Mdim
                return WG::WarpGemmAttribute::Impl::kCNLane / WG::kN;
            }
            else
            {
                // In this case each thread has multiple consecutive elements in
                // M dimension, however consecutive threads' elements have stride.
                constexpr index_t NDimY = CWarpDstr::NDimY;
                constexpr auto c_warp_y_lengths =
                    CWarpDstr{}.get_ys_to_d_descriptor().get_lengths();
                static_assert(WG::WarpGemmAttribute::Impl::kCM1PerLane ==
                              c_warp_y_lengths.get(number<NDimY - 1>{}));
                return c_warp_y_lengths.get(number<NDimY - 1>{});
            }
        }
        else
        {
            static_assert(false, "Unsupported CLayout!");
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto IsTransposeC()
    {
        return Problem::TransposeC;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeADramTileDistribution()
    {
        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t VecLoadSize =
            problem_fixed_vector_size_v<Problem> ? Problem::VectorSizeA : GetVectorSizeA<Problem>();
        constexpr index_t NumWaveGroups = Problem::NumWaveGroups;

        using ALayout = remove_cvref_t<
            std::tuple_element_t<number<0>{}, remove_cvref_t<typename Problem::AsLayoutTuple>>>;
        // Tile: MPerBlock X KPerBlock
        if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            using TileEncodingPattern =
                tile_distribution_encoding_pattern_2d<BlockSize,
                                                      MPerBlock,
                                                      KPerBlock,
                                                      VecLoadSize,
                                                      getATileAccessPattern(),
                                                      NumWaveGroups>;
            return TileEncodingPattern::make_2d_static_tile_distribution();
        }
        // Tile: KPerBlock X MPerBlock
        else
        {
            using TileEncodingPattern =
                tile_distribution_encoding_pattern_2d<BlockSize,
                                                      KPerBlock,
                                                      MPerBlock,
                                                      VecLoadSize,
                                                      getATileAccessPattern(),
                                                      NumWaveGroups>;
            return TileEncodingPattern::make_2d_static_tile_distribution();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBDramTileDistribution()
    {
        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        // If we cast before writing to LDS, the vectorsize is defined by the A type
        // since the assumption is that A type is going to be the B LDS type
        constexpr bool IsBCastPolicyBeforeLDSWrite = IsBCastPolicyBeforeLDSWrite_v<Problem>;
        constexpr index_t VecLoadSize =
            IsBCastPolicyBeforeLDSWrite
                ? (problem_fixed_vector_size_v<Problem> ? Problem::VectorSizeA
                                                        : GetVectorSizeA<Problem>())
                : (problem_fixed_vector_size_v<Problem> ? Problem::VectorSizeB
                                                        : GetVectorSizeB<Problem>());
        constexpr index_t NumWaveGroups = Problem::NumWaveGroups;
        using BLayout                   = remove_cvref_t<
                              std::tuple_element_t<number<0>{}, remove_cvref_t<typename Problem::BsLayoutTuple>>>;
        // Tile: KPerBlock X NPerBlock
        if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            using TileEncodingPattern =
                tile_distribution_encoding_pattern_2d<BlockSize,
                                                      KPerBlock,
                                                      NPerBlock,
                                                      VecLoadSize,
                                                      getBTileAccessPattern(),
                                                      NumWaveGroups>;
            return TileEncodingPattern::make_2d_static_tile_distribution();
        }
        // Tile: NPerBlock X KPerBlock
        else
        {
            using TileEncodingPattern =
                tile_distribution_encoding_pattern_2d<BlockSize,
                                                      NPerBlock,
                                                      KPerBlock,
                                                      VecLoadSize,
                                                      getBTileAccessPattern(),
                                                      NumWaveGroups>;
            return TileEncodingPattern::make_2d_static_tile_distribution();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledARegTileDistribution()
    {
        using ALayout = remove_cvref_t<
            std::tuple_element_t<number<0>{}, remove_cvref_t<typename Problem::AsLayoutTuple>>>;
        static_assert(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::ColumnMajor>);
        constexpr index_t BlockSize     = Problem::kBlockSize;
        constexpr index_t MPerBlock     = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock     = Problem::BlockGemmShape::kK;
        constexpr index_t VecLoadSize   = GetVectorSizeA<Problem>();
        constexpr index_t NumWaveGroups = Problem::NumWaveGroups;

        using TileEncodingPattern = tile_distribution_encoding_pattern_2d<BlockSize,
                                                                          KPerBlock,
                                                                          MPerBlock,
                                                                          VecLoadSize,
                                                                          getATileAccessPattern(),
                                                                          NumWaveGroups>;
        return TileEncodingPattern::make_shuffled_2d_static_tile_distribution();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledBRegTileDistribution()
    {
        using BLayout = remove_cvref_t<
            std::tuple_element_t<number<0>{}, remove_cvref_t<typename Problem::BsLayoutTuple>>>;
        static_assert(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>);
        constexpr index_t BlockSize     = Problem::kBlockSize;
        constexpr index_t NPerBlock     = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock     = Problem::BlockGemmShape::kK;
        constexpr index_t VecLoadSize   = GetVectorSizeB<Problem>();
        constexpr index_t NumWaveGroups = Problem::NumWaveGroups;

        using TileEncodingPattern = tile_distribution_encoding_pattern_2d<BlockSize,
                                                                          KPerBlock,
                                                                          NPerBlock,
                                                                          VecLoadSize,
                                                                          getBTileAccessPattern(),
                                                                          NumWaveGroups>;
        return TileEncodingPattern::make_shuffled_2d_static_tile_distribution();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemPackA()
    {
        using A         = ALdsDataType_<Problem>;
        using BlockGemm = remove_cvref_t<decltype(Derived::template GetBlockGemm<Problem>())>;

        constexpr index_t PackedSize = numeric_traits<A>::PackedSize;

        constexpr index_t KPack = static_cast<index_t>(BlockGemm::Traits::KPackA);
        constexpr index_t VecElems =
            static_cast<index_t>(Problem::VectorLoadSize / sizeof(A) * PackedSize);

        return (KPack < VecElems) ? KPack : VecElems;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemPackB()
    {
        using B         = BLdsDataType_<Problem>;
        using BlockGemm = remove_cvref_t<decltype(Derived::template GetBlockGemm<Problem>())>;

        constexpr index_t PackedSize = numeric_traits<B>::PackedSize;

        constexpr index_t KPack = static_cast<index_t>(BlockGemm::Traits::KPackB);
        constexpr index_t VecElems =
            static_cast<index_t>(Problem::VectorLoadSize / sizeof(B) * PackedSize);

        return (KPack < VecElems) ? KPack : VecElems;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeA()
    {
        using ADataType                 = ALdsDataType_<Problem>;
        constexpr index_t PackedSize    = numeric_traits<ADataType>::PackedSize;
        constexpr auto a_lds_block_desc = Derived::template MakeALdsBlockDescriptor<Problem>();
        constexpr index_t smem_size_a   = integer_least_multiple(
            a_lds_block_desc.get_element_space_size() * sizeof(ADataType) / PackedSize, 16);
        return smem_size_a;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeB()
    {
        constexpr bool IsBCastPolicyBeforeLDSWrite = IsBCastPolicyBeforeLDSWrite_v<Problem>;
        using BDataType                            = std::conditional_t<IsBCastPolicyBeforeLDSWrite,
                                                                        typename Problem::ADataType,
                                                                        BLdsDataType_<Problem>>;
        constexpr index_t PackedSize               = numeric_traits<BDataType>::PackedSize;
        constexpr auto b_lds_block_desc = Derived::template MakeBLdsBlockDescriptor<Problem>();
        constexpr index_t smem_size_b   = integer_least_multiple(
            b_lds_block_desc.get_element_space_size() * sizeof(BDataType) / PackedSize, 16);
        return smem_size_b;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        constexpr index_t smem_size_a = GetSmemSizeA<Problem>();
        constexpr index_t smem_size_b = GetSmemSizeB<Problem>();

        return smem_size_a + smem_size_b;
    }

    // GetLdsPaddingConfig,  MakeALdsBlockDescriptorForTrLoad, MakeBLdsBlockDescriptorForTrLoad
    // functions are used in gfx1250
    template <typename Problem, bool IsA>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfig()
    {
        auto constexpr_log2_floor = [](index_t x) constexpr {
            index_t result = 0;
            while(x > 1)
            {
                x >>= 1;
                result++;
            }
            return result;
        };
        using DataType =
            remove_cvref_t<std::conditional_t<IsA, ALdsDataType_<Problem>, BLdsDataType_<Problem>>>;
        constexpr index_t MNPerBlock =
            IsA ? Problem::BlockGemmShape::kM : Problem::BlockGemmShape::kN;

        constexpr index_t BytesPerDword = sizeof(int32_t);
        constexpr auto DataTypeSize     = sizeof(DataType);

        constexpr auto is_tr_load = IsA ? is_a_load_tr<Problem> : is_b_load_tr<Problem>;
        constexpr auto PackedSize = numeric_traits<DataType>::PackedSize;
        if constexpr(is_tr_load)
        {
            constexpr index_t banks_per_mblk =
                MNPerBlock * DataTypeSize / PackedSize / BytesPerDword;
            // 8 * PackedSize means 8 * PackedSize columns which is in gfx1250 tr load instructions
            // layout; this value is the column number that will access simultaneously in one cycle
            if constexpr(banks_per_mblk * 8 * PackedSize <= get_n_lds_banks())
            {
                return make_tuple(number<false>{}, number<0>{}, number<0>{});
            }
            else
            {
                // check tr load instructions layout
                constexpr index_t bank_of_vecs = 16 * sizeof(DataType) / PackedSize / BytesPerDword;
                constexpr index_t pad_amount   = bank_of_vecs - 1;
                constexpr index_t pad_interval = (banks_per_mblk < get_n_lds_banks())
                                                     ? constexpr_log2_floor(get_n_lds_banks()) - 1
                                                     : constexpr_log2_floor(banks_per_mblk) - 1;

                return make_tuple(number<true>{}, number<pad_amount>{}, number<pad_interval>{});
            }
        }
        else
        {
            constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
            constexpr index_t banks_per_kblk =
                KPerBlock * DataTypeSize / PackedSize / BytesPerDword;
            // log2 minus 1 of the number of dwords to store into the destination before adding
            // padding; one bank is 1 dword size
            constexpr index_t pad_interval = (banks_per_kblk < get_n_lds_banks())
                                                 ? constexpr_log2_floor(get_n_lds_banks()) - 1
                                                 : constexpr_log2_floor(banks_per_kblk) - 1;
            // always use b128 to ds_load; this value calculate the bank number per 128 bits
            constexpr index_t banks_per_128b = get_n_dwords_per_128b();
            // amount of padding to add in dwords 0 means 1 dword padding; 1 means 2 dwords
            // padding
            // ...
            constexpr index_t pad_amount = banks_per_128b - 1;

            return make_tuple(number<true>{}, number<pad_amount>{}, number<pad_interval>{});
        }
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeALdsBlockDescriptorForTrLoad()
    {
        static_assert(is_a_load_tr<Problem>,
                      "MakeALdsBlockDescriptorForTrLoad function is only for A tr load case");
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        constexpr auto LdsPaddingConfigA = GetLdsPaddingConfig<Problem, true>();
        constexpr auto IsPadding         = LdsPaddingConfigA[I0];
        constexpr auto PaddingAmount     = LdsPaddingConfigA[I1];
        constexpr auto PaddingInterval   = LdsPaddingConfigA[I2];
        using ADataType                  = ALdsDataType_<Problem>;
        constexpr auto DataTypeSize      = sizeof(ADataType);
        constexpr auto PackedSize        = numeric_traits<ADataType>::PackedSize;
        if constexpr(!IsPadding)
        {
            constexpr index_t KPack = GetSmemPackA<Problem>();
            constexpr auto a_lds_block_desc_0 =
                make_naive_tensor_descriptor(make_tuple(number<KPerBlock>{}, number<MPerBlock>{}),
                                             make_tuple(number<MPerBlock>{}, number<1>{}),
                                             number<KPack>{},
                                             number<1>{});
            return a_lds_block_desc_0;
        }
        else
        {
            constexpr index_t BytesPerDword = sizeof(int32_t);
            constexpr index_t KPack         = GetSmemPackA<Problem>();
            constexpr index_t PaddingStride =
                (1 << (PaddingInterval + 1)) * BytesPerDword / DataTypeSize * PackedSize;
            constexpr index_t PaddingDataAmount =
                (PaddingAmount + 1) * BytesPerDword / DataTypeSize * PackedSize;
            // which means lds bank number > MPerBlock
            if constexpr(PaddingStride > MPerBlock)
            {
                constexpr index_t KLdsLayerRequired =
                    get_n_lds_banks() * BytesPerDword / MPerBlock / DataTypeSize * PackedSize;
                constexpr auto KLdsLayer = max(1, KLdsLayerRequired);

                constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
                    make_tuple(number<KPerBlock / KLdsLayer>{},
                               number<MPerBlock / KPack * KLdsLayer>{},
                               number<KPack>{}),
                    make_tuple(number<MPerBlock * KLdsLayer + PaddingDataAmount>{},
                               number<KPack>{},
                               number<1>{}),
                    number<KPack>{},
                    number<1>{});
                constexpr auto a_lds_block_desc_1 = transform_tensor_descriptor(
                    a_lds_block_desc_0,
                    make_tuple(make_pass_through_transform(number<KPerBlock / KLdsLayer>{}),
                               make_unmerge_transform(
                                   make_tuple(number<KLdsLayer>{}, number<MPerBlock / KPack>{})),
                               make_pass_through_transform(number<KPack>{})),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));
                constexpr auto a_lds_block_desc = transform_tensor_descriptor(
                    a_lds_block_desc_1,
                    make_tuple(make_merge_transform_v3_division_mod(make_tuple(
                                   number<KPerBlock / KLdsLayer>{}, number<KLdsLayer>{})),
                               make_merge_transform_v3_division_mod(
                                   make_tuple(number<MPerBlock / KPack>{}, number<KPack>{}))),
                    make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
                return a_lds_block_desc;
            }
            else
            {
                constexpr auto MLdsLayer          = MPerBlock / PaddingStride;
                constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
                    make_tuple(number<KPerBlock * MLdsLayer>{},
                               number<MPerBlock / KPack / MLdsLayer>{},
                               number<KPack>{}),
                    make_tuple(number<MPerBlock / MLdsLayer + PaddingDataAmount>{},
                               number<KPack>{},
                               number<1>{}),
                    number<KPack>{},
                    number<1>{});
                constexpr auto a_lds_block_desc_1 = transform_tensor_descriptor(
                    a_lds_block_desc_0,
                    make_tuple(make_unmerge_transform(
                                   make_tuple(number<KPerBlock>{}, number<MLdsLayer>{})),
                               make_pass_through_transform(number<MPerBlock / KPack / MLdsLayer>{}),
                               make_pass_through_transform(number<KPack>{})),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                    make_tuple(sequence<0, 1>{}, sequence<2>{}, sequence<3>{}));
                constexpr auto a_lds_block_desc = transform_tensor_descriptor(
                    a_lds_block_desc_1,
                    make_tuple(make_pass_through_transform(number<KPerBlock>{}),
                               make_merge_transform_v3_division_mod(
                                   make_tuple(number<MLdsLayer>{},
                                              number<MPerBlock / KPack / MLdsLayer>{},
                                              number<KPack>{}))),
                    make_tuple(sequence<0>{}, sequence<1, 2, 3>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
                return a_lds_block_desc;
            }
        }
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeBLdsBlockDescriptorForTrLoad()
    {
        static_assert(is_b_load_tr<Problem>,
                      "MakeBLdsBlockDescriptorForTrLoad function is only for B tr load case");
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        constexpr auto LdsPaddingConfigB = GetLdsPaddingConfig<Problem, false>();
        constexpr auto IsPadding         = LdsPaddingConfigB[I0];
        constexpr auto PaddingAmount     = LdsPaddingConfigB[I1];
        constexpr auto PaddingInterval   = LdsPaddingConfigB[I2];
        using BDataType                  = BLdsDataType_<Problem>;
        constexpr auto DataTypeSize      = sizeof(BDataType);
        constexpr auto PackedSize        = numeric_traits<BDataType>::PackedSize;
        if constexpr(!IsPadding)
        {
            constexpr index_t KPack = GetSmemPackB<Problem>();
            constexpr auto b_lds_block_desc_0 =
                make_naive_tensor_descriptor(make_tuple(number<KPerBlock>{}, number<NPerBlock>{}),
                                             make_tuple(number<NPerBlock>{}, number<1>{}),
                                             number<KPack>{},
                                             number<1>{});
            return b_lds_block_desc_0;
        }
        else
        {
            constexpr index_t BytesPerDword = sizeof(int32_t);
            constexpr index_t KPack         = GetSmemPackB<Problem>();
            constexpr index_t PaddingStride =
                (1 << (PaddingInterval + 1)) * BytesPerDword / DataTypeSize * PackedSize;
            constexpr index_t PaddingDataAmount =
                (PaddingAmount + 1) * BytesPerDword / DataTypeSize * PackedSize;
            // which means lds bank number > NPerBlock
            if constexpr(PaddingStride > NPerBlock)
            {
                constexpr index_t KLdsLayerRequired =
                    get_n_lds_banks() * BytesPerDword / NPerBlock / DataTypeSize * PackedSize;
                constexpr auto KLdsLayer = max(1, KLdsLayerRequired);

                constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
                    make_tuple(number<KPerBlock / KLdsLayer>{},
                               number<NPerBlock / KPack * KLdsLayer>{},
                               number<KPack>{}),
                    make_tuple(number<NPerBlock * KLdsLayer + PaddingDataAmount>{},
                               number<KPack>{},
                               number<1>{}),
                    number<KPack>{},
                    number<1>{});
                constexpr auto b_lds_block_desc_1 = transform_tensor_descriptor(
                    b_lds_block_desc_0,
                    make_tuple(make_pass_through_transform(number<KPerBlock / KLdsLayer>{}),
                               make_unmerge_transform(
                                   make_tuple(number<KLdsLayer>{}, number<NPerBlock / KPack>{})),
                               make_pass_through_transform(number<KPack>{})),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));
                constexpr auto b_lds_block_desc = transform_tensor_descriptor(
                    b_lds_block_desc_1,
                    make_tuple(make_merge_transform_v3_division_mod(make_tuple(
                                   number<KPerBlock / KLdsLayer>{}, number<KLdsLayer>{})),
                               make_merge_transform_v3_division_mod(
                                   make_tuple(number<NPerBlock / KPack>{}, number<KPack>{}))),
                    make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
                return b_lds_block_desc;
            }
            else
            {
                constexpr auto NLdsLayer          = NPerBlock / PaddingStride;
                constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
                    make_tuple(number<KPerBlock * NLdsLayer>{},
                               number<NPerBlock / KPack / NLdsLayer>{},
                               number<KPack>{}),
                    make_tuple(number<NPerBlock / NLdsLayer + PaddingDataAmount>{},
                               number<KPack>{},
                               number<1>{}),
                    number<KPack>{},
                    number<1>{});
                constexpr auto b_lds_block_desc_1 = transform_tensor_descriptor(
                    b_lds_block_desc_0,
                    make_tuple(make_unmerge_transform(
                                   make_tuple(number<KPerBlock>{}, number<NLdsLayer>{})),
                               make_pass_through_transform(number<NPerBlock / KPack / NLdsLayer>{}),
                               make_pass_through_transform(number<KPack>{})),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                    make_tuple(sequence<0, 1>{}, sequence<2>{}, sequence<3>{}));
                constexpr auto b_lds_block_desc = transform_tensor_descriptor(
                    b_lds_block_desc_1,
                    make_tuple(make_pass_through_transform(number<KPerBlock>{}),
                               make_merge_transform_v3_division_mod(
                                   make_tuple(number<NLdsLayer>{},
                                              number<NPerBlock / KPack / NLdsLayer>{},
                                              number<KPack>{}))),
                    make_tuple(sequence<0>{}, sequence<1, 2, 3>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
                return b_lds_block_desc;
            }
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool isClusterLaunch()
    {
        constexpr index_t clusterM = Problem::BlockGemmShape::kclusterM;
        constexpr index_t clusterN = Problem::BlockGemmShape::kclusterN;
        constexpr index_t clusterK = Problem::BlockGemmShape::kclusterK;
        // cluster launch is enabled only when TilePartitioner uses cluster tile gemm shape and
        // cluster size > 1
        return is_cluster_tile_gemm_shape<typename Problem::BlockGemmShape>::value &&
               (clusterM * clusterN * clusterK > 1);
    }
};

// UniversalGemm Policy
struct UniversalGemmPipelineAgBgCrPolicy
    : public UniversalGemmBasePolicy<UniversalGemmPipelineAgBgCrPolicy>
{
    template <typename Problem>
    using LdsADataType = typename Problem::ADataType;

    template <typename Problem>
    using LdsBDataType = typename Problem::BDataType;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;

#if defined(__gfx950__)
        constexpr index_t vector_size =
            DS_READ_TR_SIZE() / sizeof(typename Problem::AComputeDataType);
        constexpr index_t thread_elements = WarpTile::at(I1) * WarpTile::at(I2) / get_warp_size();
        constexpr auto wg_attr_num_access =
            !(is_a_load_tr<Problem> || is_b_load_tr<Problem>) ? WGAttrNumAccessEnum::Single
            : vector_size == thread_elements                  ? WGAttrNumAccessEnum::Single
            : vector_size * 2 == thread_elements              ? WGAttrNumAccessEnum::Double
            : vector_size * 4 == thread_elements              ? WGAttrNumAccessEnum::Quad
                                                              : WGAttrNumAccessEnum::Invalid;
#else
        constexpr auto wg_attr_num_access = WGAttrNumAccessEnum::Default;
#endif

        using ATypeToUse = typename Problem::AComputeDataType;
        using BTypeToUse = typename Problem::BComputeDataType;

        using WarpGemm = WarpGemmDispatcher<typename Problem::AComputeDataType,
                                            typename Problem::BComputeDataType,
                                            typename Problem::CDataType,
                                            WarpTile::at(I0),
                                            WarpTile::at(I1),
                                            WarpTile::at(I2),
                                            Problem::TransposeC,
                                            false,
                                            Problem::UseStructuredSparsity,
                                            wg_attr_num_access>;

        using BlockGemmPolicy = BlockGemmASmemBSmemCRegV1CustomPolicy<ATypeToUse,
                                                                      BTypeToUse,
                                                                      typename Problem::CDataType,
                                                                      BlockWarps,
                                                                      WarpGemm>;
        return BlockUniversalGemmAsBsCr<Problem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
