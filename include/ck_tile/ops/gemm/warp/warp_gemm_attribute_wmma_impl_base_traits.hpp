// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
namespace ck_tile {
template <typename DataType, index_t K, bool MixPrec>
struct LayoutFromDataType;

struct LayoutFrom8BitMixPrec
{
    static constexpr index_t kK1PerLane = 16;
    static constexpr index_t kK0PerLane = 4;
};

struct LayoutFromNon8BitMixPrec
{
    static constexpr index_t kK1PerLane = 32;
    static constexpr index_t kK0PerLane = 2;
};

template <>
struct LayoutFromDataType<fp8_t, 128, true> : LayoutFrom8BitMixPrec
{
};

template <>
struct LayoutFromDataType<bf8_t, 128, true> : LayoutFrom8BitMixPrec
{
};

template <>
struct LayoutFromDataType<pk_fp4_t, 128, true> : LayoutFromNon8BitMixPrec
{
};

// fp4_t is the only format that has the same data layout for MixPrec and non-MixPrec
template <>
struct LayoutFromDataType<pk_fp4_t, 128, false> : LayoutFromNon8BitMixPrec
{
};

template <typename DataType, index_t K>
struct LayoutFromDataType<DataType, K, false>
{
    static constexpr index_t kKLane = 2;
    static constexpr index_t kK1PerLane =
        (std::is_same_v<DataType, fp32_t> || std::is_same_v<DataType, fp64_t>) ? 2 : 8;
    static constexpr index_t kK0PerLane = K / (kK1PerLane * kKLane);
};

template <typename Arch,
          typename ADType,
          typename BDType,
          typename CDType,
          index_t K,
          bool MixPrec = false,
          index_t M    = 16,
          index_t N    = 16>
struct WmmaTraitsBase;

// GFX11 specialization
template <typename ADType,
          typename BDType,
          typename CDType,
          index_t K,
          bool MixPrec,
          index_t M,
          index_t N>
struct WmmaTraitsBase<gfx11_t, ADType, BDType, CDType, K, MixPrec, M, N>
{
    using ArchType = gfx11_t;

    using ADataType = ADType;
    using BDataType = BDType;
    using CDataType = CDType;

    static_assert(M % 16 == 0 && N % 16 == 0, "M and N must be multiples of 16");

    static constexpr index_t kAMBlock = M / 16;
    static constexpr index_t kBNBlock = N / 16;

    static constexpr index_t kCMBlock = M / 16;
    static constexpr index_t kCNBlock = N / 16;

    using AVecType = ext_vector_t<ADataType, kAMBlock * 16>;
    using BVecType = ext_vector_t<BDataType, kBNBlock * 16>;
    using CVecType = ext_vector_t<CDataType, 8 * kCMBlock * kCNBlock>;

    static constexpr index_t kM = M;
    static constexpr index_t kN = N;
    static constexpr index_t kK = K;

    static constexpr index_t kRepeat     = 2;
    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 1;
    static constexpr index_t kAK0PerLane = 1;
    static constexpr index_t kAK1PerLane = K / (kAK0PerLane * kABKLane);
    static constexpr index_t kBK0PerLane = 1;
    static constexpr index_t kBK1PerLane = K / (kBK0PerLane * kABKLane);

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 8;
    static constexpr index_t kCM1PerLane = 1;

    using kABPs2RHssMajor = sequence<0, 2, 1>;
    using kABPs2RHssMinor = sequence<0, 1, 1>;
    using kABYs2RHsMajor  = sequence<1, 2, 2>;
    using kABYs2RHsMinor  = sequence<0, 0, 2>;

    using kCPs2RHssMajor = sequence<1, 2>;
    using kCPs2RHssMinor = sequence<2, 1>;

    using kCYs2RHsMajor =
        std::conditional_t<(kCMBlock == 1 && kCNBlock == 1), sequence<1, 1>, sequence<1, 2, 1, 1>>;
    using kCYs2RHsMinor =
        std::conditional_t<(kCMBlock == 1 && kCNBlock == 1), sequence<1, 3>, sequence<0, 0, 1, 3>>;

    using kCTPs2RHssMajor = sequence<2, 1>;
    using kCTPs2RHssMinor = sequence<2, 1>;
    using kCTYs2RHsMajor =
        std::conditional_t<(kCMBlock == 1 && kCNBlock == 1), sequence<2, 2>, sequence<2, 1, 2, 2>>;
    using kCTYs2RHsMinor =
        std::conditional_t<(kCMBlock == 1 && kCNBlock == 1), sequence<1, 3>, sequence<0, 0, 1, 3>>;
};

// GFX12 specialization
template <typename ADType,
          typename BDType,
          typename CDType,
          index_t K,
          bool MixPrec,
          index_t M,
          index_t N>
struct WmmaTraitsBase<gfx12_t, ADType, BDType, CDType, K, MixPrec, M, N>
{
    using ArchType = gfx12_t;

    using ADataType = ADType;
    using BDataType = BDType;
    using CDataType = CDType;

    static_assert(M % 16 == 0 && N % 16 == 0, "M and N must be multiples of 16");

    static constexpr index_t kAMBlock = M / 16;
    static constexpr index_t kBNBlock = N / 16;

    static constexpr index_t kCMBlock = M / 16;
    static constexpr index_t kCNBlock = N / 16;

    static constexpr index_t kM = M;
    static constexpr index_t kN = N;
    static constexpr index_t kK = K;

    static constexpr index_t kRepeat = 1;
    static constexpr index_t kAMLane = 16;
    static constexpr index_t kBNLane = 16;

    static constexpr index_t kAK1PerLane = LayoutFromDataType<ADType, K, MixPrec>::kK1PerLane;
    static constexpr index_t kAK0PerLane = LayoutFromDataType<ADType, K, MixPrec>::kK0PerLane;
    static constexpr index_t kBK1PerLane = LayoutFromDataType<BDType, K, MixPrec>::kK1PerLane;
    static constexpr index_t kBK0PerLane = LayoutFromDataType<BDType, K, MixPrec>::kK0PerLane;
    static constexpr index_t kABKLane    = 2;

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 8;

    using kABPs2RHssMajor = sequence<2, 1>;
    using kABPs2RHssMinor = sequence<1, 1>;
    using kABYs2RHsMajor  = sequence<1, 2, 2>;
    using kABYs2RHsMinor  = sequence<0, 0, 2>;

    using kCPs2RHssMajor = sequence<1, 2>;
    using kCPs2RHssMinor = sequence<2, 1>;
    using kCYs2RHsMajor =
        std::conditional_t<(kCMBlock == 1 && kCNBlock == 1), sequence<1, 1>, sequence<2, 1, 1, 1>>;
    using kCYs2RHsMinor =
        std::conditional_t<(kCMBlock == 1 && kCNBlock == 1), sequence<1, 3>, sequence<0, 0, 1, 3>>;

    using kCTPs2RHssMajor = sequence<2, 1>;
    using kCTPs2RHssMinor = sequence<2, 1>;
    using kCTYs2RHsMajor =
        std::conditional_t<(kCMBlock == 1 && kCNBlock == 1), sequence<2, 2>, sequence<1, 2, 2, 2>>;
    using kCTYs2RHsMinor =
        std::conditional_t<(kCMBlock == 1 && kCNBlock == 1), sequence<1, 3>, sequence<0, 0, 1, 3>>;

    static constexpr index_t kAPackedSize = ck_tile::numeric_traits<ADataType>::PackedSize;
    static constexpr index_t kBPackedSize = ck_tile::numeric_traits<BDataType>::PackedSize;
    static constexpr index_t kAInputSize  = kK / (kABKLane * kAPackedSize);
    static constexpr index_t kBInputSize  = kK / (kABKLane * kBPackedSize);
    static constexpr index_t kCOutputSize = kM / kCMLane;
    using AVecType                        = ext_vector_t<ADataType, kAInputSize * kAMBlock>;
    using BVecType                        = ext_vector_t<BDataType, kBInputSize * kBNBlock>;
    using CVecType                        = ext_vector_t<CDataType, kCOutputSize * kCNBlock>;
};
} // namespace ck_tile
