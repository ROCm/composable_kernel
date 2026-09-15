// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

#include "hstu_attention_config.hpp"

namespace ck_tile {

namespace detail {

template <index_t X>
CK_TILE_HOST_DEVICE constexpr index_t f_integer_log2()
{
    static_assert(X > 0 && (X & (X - 1)) == 0, "integer_log2 requires a positive power of two");

    index_t value  = X;
    index_t result = 0;
    while(value > 1)
    {
        value >>= 1;
        ++result;
    }
    return result;
};

#if HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE
template <index_t PadIntervalBytes, index_t PadLengthBytes>
CK_TILE_HOST_DEVICE constexpr bool IsTdmPaddingValid()
{
    constexpr bool is_interval_valid = PadIntervalBytes >= 8 && PadIntervalBytes <= 1024 &&
                                       (PadIntervalBytes & (PadIntervalBytes - 1)) == 0;
    constexpr bool is_length_valid =
        PadLengthBytes >= 4 && PadLengthBytes <= 512 && PadLengthBytes % 4 == 0;

    return is_interval_valid && is_length_valid;
};

template <index_t PadIntervalBytes, index_t PadLengthBytes>
CK_TILE_HOST_DEVICE constexpr auto GetTdmRawPaddingConfig()
{
    static_assert(IsTdmPaddingValid<PadIntervalBytes, PadLengthBytes>(),
                  "Invalid tdm padding config!");

    constexpr index_t raw_pad_interval = f_integer_log2<PadIntervalBytes / 4>() - 1;
    constexpr index_t raw_pad_length   = PadLengthBytes / 4 - 1;

    return make_tuple(number<raw_pad_interval>{}, number<raw_pad_length>{});
};

template <typename WarpGemm, bool IsGemmInputA, index_t kKPerBlock>
CK_TILE_HOST_DEVICE constexpr auto GetTdmLdsPaddingConfigForNormalRead()
{
    // Tdm only supported by gfx1250, which has 64 banks
    constexpr index_t BankSpanBytes = 64 * 4;

    if constexpr(IsGemmInputA)
    { // access gemm inputA from LDS
        using AEncoding = typename WarpGemm::AWarpDstrEncoding;
        using ADataType = typename WarpGemm::ADataType;

        constexpr index_t ScalarPerVector = AEncoding::detail::ys_lengths_[AEncoding::NDimY - 1];

        constexpr index_t kKBytesPerBlock  = kKPerBlock * sizeof(ADataType);
        constexpr index_t PadIntervalBytes = max(kKBytesPerBlock, BankSpanBytes);
        constexpr index_t PadLengthBytes   = ScalarPerVector * sizeof(ADataType);

        return make_tuple(number<PadIntervalBytes>{}, number<PadLengthBytes>{});
    }
    else
    { // acccess gemm inputB from LDS
        using BEncoding = typename WarpGemm::BWarpDstrEncoding;
        using BDataType = typename WarpGemm::BDataType;

        constexpr index_t ScalarPerVector = BEncoding::detail::ys_lengths_[BEncoding::NDimY - 1];

        constexpr index_t kKBytesPerBlock  = kKPerBlock * sizeof(BDataType);
        constexpr index_t PadIntervalBytes = max(kKBytesPerBlock, BankSpanBytes);
        constexpr index_t PadLengthBytes   = ScalarPerVector * sizeof(BDataType);

        return make_tuple(number<PadIntervalBytes>{}, number<PadLengthBytes>{});
    }
}

template <typename WarpGemm, bool IsGemmInputA, index_t kKPerBlock>
CK_TILE_HOST_DEVICE constexpr auto GetTdmLdsPaddingConfigForTrLoadRead()
{
    // Tdm only supported by gfx1250, which has 64 banks
    constexpr index_t BankSpanBytes = 64 * 4;

    if constexpr(IsGemmInputA)
    { // access gemm inputA from LDS
        using ADataType = typename WarpGemm::ADataType;
        using AEncoding = typename WarpGemm::AWarpDstrEncoding;
        using AEncodingForTrLoad =
            typename InputTileDistributionTraits<AEncoding, ADataType>::TransposedDstrEncode;

        constexpr index_t ScalarPerVector =
            AEncodingForTrLoad::detail::ys_lengths_[AEncoding::NDimY - 1];

        // kABKLane is same as kCMLane
        constexpr index_t kAKLane = WarpGemm::kCMLane;

        constexpr index_t kKBytesPerBlock  = kKPerBlock * sizeof(ADataType);
        constexpr index_t PadIntervalBytes = max(kKBytesPerBlock, BankSpanBytes);
        constexpr index_t PadLengthBytes   = kAKLane * ScalarPerVector * sizeof(ADataType);

        return make_tuple(number<PadIntervalBytes>{}, number<PadLengthBytes>{});
    }
    else
    { // acccess gemm inputB from LDS
        using BDataType = typename WarpGemm::BDataType;
        using BEncoding = typename WarpGemm::BWarpDstrEncoding;
        using BEncodingForTrLoad =
            typename InputTileDistributionTraits<BEncoding, BDataType>::TransposedDstrEncode;

        constexpr index_t ScalarPerVector =
            BEncodingForTrLoad::detail::ys_lengths_[BEncoding::NDimY - 1];

        // kABKLane is same as kCMLane
        constexpr index_t kBKLane = WarpGemm::kCMLane;

        constexpr index_t kKBytesPerBlock  = kKPerBlock * sizeof(BDataType);
        constexpr index_t PadIntervalBytes = max(kKBytesPerBlock, BankSpanBytes);
        constexpr index_t PadLengthBytes   = kBKLane * ScalarPerVector * sizeof(BDataType);

        return make_tuple(number<PadIntervalBytes>{}, number<PadLengthBytes>{});
    }
}
#endif

template <typename WarpGemm, bool IsGemmInputA, index_t kKPerBlock>
CK_TILE_HOST_DEVICE constexpr auto GetLdsPaddingConfigForNormalRead()
{
#if defined(__hstu_gfx95__) || defined(__hstu_gfx125__)
    constexpr index_t BankSpanBytes = 64 * 4;
#else
    constexpr index_t BankSpanBytes = 32 * 4;
#endif

    if constexpr(IsGemmInputA)
    { // access gemm inputA from LDS
        using AEncoding = typename WarpGemm::AWarpDstrEncoding;
        using ADataType = typename WarpGemm::ADataType;

        constexpr index_t ScalarPerVector = AEncoding::detail::ys_lengths_[AEncoding::NDimY - 1];

        constexpr index_t BankSpanElements = BankSpanBytes / sizeof(ADataType);
        constexpr index_t PadInterval      = max(kKPerBlock, BankSpanElements);
        constexpr index_t PadLength        = ScalarPerVector;

        return make_tuple(number<PadInterval>{}, number<PadLength>{});
    }
    else
    { // acccess gemm inputB from LDS
        using BEncoding = typename WarpGemm::BWarpDstrEncoding;
        using BDataType = typename WarpGemm::BDataType;

        constexpr index_t ScalarPerVector = BEncoding::detail::ys_lengths_[BEncoding::NDimY - 1];

        constexpr index_t BankSpanElements = BankSpanBytes / sizeof(BDataType);
        constexpr index_t PadInterval      = max(kKPerBlock, BankSpanElements);
        constexpr index_t PadLength        = ScalarPerVector;

        return make_tuple(number<PadInterval>{}, number<PadLength>{});
    }
}

template <typename WarpGemm, bool IsGemmInputA, index_t kKPerBlock>
CK_TILE_HOST_DEVICE constexpr auto GetLdsPaddingConfigForTrLoadRead()
{
#if defined(__hstu_gfx95__) || defined(__hstu_gfx125__)
    constexpr index_t BankSpanBytes = 64 * 4;
#else
    constexpr index_t BankSpanBytes = 32 * 4;
#endif

    if constexpr(IsGemmInputA)
    { // access gemm inputA from LDS
        using ADataType = typename WarpGemm::ADataType;
        using AEncoding = typename WarpGemm::AWarpDstrEncoding;
        using AEncodingForTrLoad =
            typename InputTileDistributionTraits<AEncoding, ADataType>::TransposedDstrEncode;

        constexpr index_t ScalarPerVector =
            AEncodingForTrLoad::detail::ys_lengths_[AEncoding::NDimY - 1];

        // kABKLane is same as kCMLane
        constexpr index_t kAKLane = WarpGemm::kCMLane;

        constexpr index_t BankSpanElements = BankSpanBytes / sizeof(ADataType);
        constexpr index_t PadInterval      = max(kKPerBlock, BankSpanElements);
        constexpr index_t PadLength        = kAKLane * ScalarPerVector;

        return make_tuple(number<PadInterval>{}, number<PadLength>{});
    }
    else
    { // acccess gemm inputB from LDS
        using BDataType = typename WarpGemm::BDataType;
        using BEncoding = typename WarpGemm::BWarpDstrEncoding;
        using BEncodingForTrLoad =
            typename InputTileDistributionTraits<BEncoding, BDataType>::TransposedDstrEncode;

        constexpr index_t ScalarPerVector =
            BEncodingForTrLoad::detail::ys_lengths_[BEncoding::NDimY - 1];

        // kABKLane is same as kCMLane
        constexpr index_t kBKLane = WarpGemm::kCMLane;

        constexpr index_t BankSpanElements = BankSpanBytes / sizeof(BDataType);
        constexpr index_t PadInterval      = max(kKPerBlock, BankSpanElements);
        constexpr index_t PadLength        = kBKLane * ScalarPerVector;

        return make_tuple(number<PadInterval>{}, number<PadLength>{});
    }
}

template <index_t NumBuffers, index_t Rows, index_t Cols, index_t PadInterval, index_t PadLength>
CK_TILE_HOST_DEVICE constexpr auto MakeRowMajorLdsPaddedBlockDescriptor()
{
    constexpr index_t LogicBufferSize = Rows * Cols;

    static_assert(LogicBufferSize >= PadInterval, "Check failed!");

    constexpr index_t NumIntervals     = LogicBufferSize / PadInterval;
    constexpr index_t SingleBufferSize = NumIntervals * (PadInterval + PadLength);

    constexpr auto interval_desc = make_naive_tensor_descriptor(
        make_tuple(number<NumBuffers>{}, number<NumIntervals>{}, number<PadInterval>{}),
        make_tuple(number<SingleBufferSize>{}, number<PadInterval + PadLength>{}, number<1>{}),
        number<8>{},
        number<1>{});

    constexpr auto flat_desc =
        transform_tensor_descriptor(interval_desc,
                                    make_tuple(make_pass_through_transform(number<NumBuffers>{}),
                                               make_merge_transform_v3_division_mod(make_tuple(
                                                   number<NumIntervals>{}, number<PadInterval>{}))),
                                    make_tuple(sequence<0>{}, sequence<1, 2>{}),
                                    make_tuple(sequence<0>{}, sequence<1>{}));

    constexpr auto normal_layout_desc = transform_tensor_descriptor(
        flat_desc,
        make_tuple(make_pass_through_transform(number<NumBuffers>{}),
                   make_unmerge_transform(make_tuple(number<Rows>{}, number<Cols>{}))),
        make_tuple(sequence<0>{}, sequence<1>{}),
        make_tuple(sequence<0>{}, sequence<1, 2>{}));

    return transform_tensor_descriptor(normal_layout_desc,
                                       make_tuple(make_merge_transform_v3_division_mod(make_tuple(
                                                      number<NumBuffers>{}, number<Rows>{})),
                                                  make_pass_through_transform(number<Cols>{})),
                                       make_tuple(sequence<0, 1>{}, sequence<2>{}),
                                       make_tuple(sequence<0>{}, sequence<1>{}));
};

template <typename DataType,
          index_t NumBuffers,
          index_t Rows,
          index_t Cols,
          index_t PadIntervalBytes,
          index_t PadLengthBytes>
CK_TILE_HOST_DEVICE constexpr auto MakeRowMajorLdsPaddedBlockDescriptor()
{
    constexpr index_t PadInterval = PadIntervalBytes / sizeof(DataType);
    constexpr index_t PadLength   = PadLengthBytes / sizeof(DataType);

    return MakeRowMajorLdsPaddedBlockDescriptor<NumBuffers, Rows, Cols, PadInterval, PadLength>();
};

template <index_t NumBuffers, index_t Rows, index_t Cols>
CK_TILE_HOST_DEVICE constexpr auto MakeRowMajorLdsPlainBlockDescriptor()
{
    constexpr index_t SingleBufferSize = Rows * Cols;

    constexpr auto naive_desc = make_naive_tensor_descriptor(
        make_tuple(number<NumBuffers>{}, number<Rows>{}, number<Cols>{}),
        make_tuple(number<SingleBufferSize>{}, number<Cols>{}, number<1>{}),
        number<8>{},
        number<1>{});

    return transform_tensor_descriptor(
        naive_desc,
        make_tuple(make_merge_transform(make_tuple(number<NumBuffers>{}, number<Rows>{})),
                   make_pass_through_transform(number<Cols>{})),
        make_tuple(sequence<0, 1>{}, sequence<2>{}),
        make_tuple(sequence<0>{}, sequence<1>{}));
}

template <typename Problem, index_t NumBuffers, index_t kN, index_t kK, index_t kKPack>
CK_TILE_HOST_DEVICE static constexpr auto MakeSwizzledNativeDesc()
{
    constexpr index_t ElementBytes = sizeof(typename Problem::QKVDataType);

#if defined(__hstu_gfx95__) || defined(__hstu_gfx125__)
    constexpr index_t BankSpanBytes = 64 * 4;
#else
    constexpr index_t BankSpanBytes = 32 * 4;
#endif

    constexpr index_t SingleBufferSize = kN * kK;

    if constexpr(kK * ElementBytes < BankSpanBytes)
    {
        constexpr index_t NLdsLayer = BankSpanBytes / (kK * ElementBytes);

        // 4D packed physical layout [NumBuffers, kN/NLdsLayer, (kK/kKPack)*NLdsLayer, kKPack].
        constexpr auto desc_0 =
            make_naive_tensor_descriptor(make_tuple(number<NumBuffers>{},
                                                    number<kN / NLdsLayer>{},
                                                    number<kK / kKPack * NLdsLayer>{},
                                                    number<kKPack>{}),
                                         make_tuple(number<SingleBufferSize>{},
                                                    number<kK * NLdsLayer>{},
                                                    number<kKPack>{},
                                                    number<1>{}),
                                         number<kKPack>{},
                                         number<1>{});

        // XOR-swizzle the (kN/NLdsLayer, kK-group*NLdsLayer) dims -> scatter banks.
        constexpr auto desc_permuted = transform_tensor_descriptor(
            desc_0,
            make_tuple(make_pass_through_transform(number<NumBuffers>{}),
                       make_xor_transform(
                           make_tuple(number<kN / NLdsLayer>{}, number<kK / kKPack * NLdsLayer>{})),
                       make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}),
            make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

        // Split the kK-group dim back into [kK/kKPack, NLdsLayer].
        constexpr auto desc_split = transform_tensor_descriptor(
            desc_permuted,
            make_tuple(
                make_pass_through_transform(number<NumBuffers>{}),
                make_pass_through_transform(number<kN / NLdsLayer>{}),
                make_unmerge_transform(make_tuple(number<kK / kKPack>{}, number<NLdsLayer>{})),
                make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}));

        // Re-merge to the logical 3D physical view [NumBuffers, kN, kK]:
        //   kN = (kN/NLdsLayer) * NLdsLayer
        //   kK = (kK/kKPack) * kKPack
        return transform_tensor_descriptor(
            desc_split,
            make_tuple(make_pass_through_transform(number<NumBuffers>{}),
                       make_merge_transform_v3_division_mod(
                           make_tuple(number<kN / NLdsLayer>{}, number<NLdsLayer>{})),
                       make_merge_transform_v3_division_mod(
                           make_tuple(number<kK / kKPack>{}, number<kKPack>{}))),
            make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));
    }
    else
    {
        // 4D packed physical layout [NumBuffers, kN, kK/kKPack, kKPack].
        constexpr auto desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<NumBuffers>{}, number<kN>{}, number<kK / kKPack>{}, number<kKPack>{}),
            make_tuple(number<SingleBufferSize>{}, number<kK>{}, number<kKPack>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});

        // XOR-swizzle the (kN, kK-group) dims -> scatter banks.
        constexpr auto desc_permuted = transform_tensor_descriptor(
            desc_0,
            make_tuple(make_pass_through_transform(number<NumBuffers>{}),
                       make_xor_transform(make_tuple(number<kN>{}, number<kK / kKPack>{})),
                       make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}),
            make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

        // Re-merge to the logical 3D physical view [NumBuffers, kN, kK]:
        //   kK = (kK/kKPack) * kKPack
        return transform_tensor_descriptor(
            desc_permuted,
            make_tuple(make_pass_through_transform(number<NumBuffers>{}),
                       make_pass_through_transform(number<kN>{}),
                       make_merge_transform_v3_division_mod(
                           make_tuple(number<kK / kKPack>{}, number<kKPack>{}))),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2, 3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));
    }
}

}; // namespace detail
}; // namespace ck_tile
