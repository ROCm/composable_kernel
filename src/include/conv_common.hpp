#pragma once
#include "ConstantTensorDescriptor.hpp"

// this is ugly, only for 4d
template <class InDesc, class WeiDesc>
constexpr auto get_convolution_output_default_4d_tensor_descriptor(InDesc, WeiDesc)
{
    constexpr auto in_desc  = InDesc{};
    constexpr auto wei_desc = WeiDesc{};

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    static_assert(in_desc.GetNumOfDimension() == 4, "input nDim is not 4");
    static_assert(wei_desc.GetNumOfDimension() == 4, "weight nDim is not 4");
    static_assert(in_desc.GetLength(I1) == wei_desc.GetLength(I1),
                  "input & weight dimension not consistent");

    constexpr auto N  = in_desc.GetLength(I0);
    constexpr auto HI = in_desc.GetLength(I2);
    constexpr auto WI = in_desc.GetLength(I3);

    constexpr auto K = wei_desc.GetLength(I0);
    constexpr auto Y = wei_desc.GetLength(I2);
    constexpr auto X = wei_desc.GetLength(I3);

    constexpr auto HO = HI + 1 - Y;
    constexpr auto WO = WI + 1 - X;

    return make_ConstantTensorDescriptor_packed(Sequence<N, K, HO, WO>{});
}

template <class InDesc, class WeiDesc, class LowerPads, class UpperPads>
constexpr auto get_convolution_with_padding_output_default_4d_tensor_descriptor(InDesc,
                                                                                WeiDesc,
                                                                                LowerPads,
                                                                                UpperPads)
{
    constexpr auto in_desc  = InDesc{};
    constexpr auto wei_desc = WeiDesc{};

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    static_assert(in_desc.GetNumOfDimension() == 4, "input nDim is not 4");
    static_assert(wei_desc.GetNumOfDimension() == 4, "weight nDim is not 4");
    static_assert(in_desc.GetLength(I1) == wei_desc.GetLength(I1),
                  "input & weight dimension not consistent");

    constexpr auto N  = in_desc.GetLength(I0);
    constexpr auto HI = in_desc.GetLength(I2);
    constexpr auto WI = in_desc.GetLength(I3);

    constexpr auto K = wei_desc.GetLength(I0);
    constexpr auto Y = wei_desc.GetLength(I2);
    constexpr auto X = wei_desc.GetLength(I3);

    constexpr auto HPadLow = LowerPads{}.Get(I0);
    constexpr auto WPadLow = LowerPads{}.Get(I1);

    constexpr auto HPadUp = UpperPads{}.Get(I0);
    constexpr auto WPadUp = UpperPads{}.Get(I1);

    constexpr auto HO = HI + HPadLow + HPadUp + 1 - Y;
    constexpr auto WO = WI + WPadLow + WPadUp + 1 - X;

    return make_ConstantTensorDescriptor_packed(Sequence<N, K, HO, WO>{});
}

template <class InDesc, class WeiDesc, class OutDesc>
constexpr std::size_t calculate_convolution_flops(InDesc, WeiDesc, OutDesc)
{
    constexpr auto wei_desc = WeiDesc{};
    constexpr auto out_desc = OutDesc{};

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr index_t N  = out_desc.GetLength(I0);
    constexpr index_t K  = out_desc.GetLength(I1);
    constexpr index_t Ho = out_desc.GetLength(I2);
    constexpr index_t Wo = out_desc.GetLength(I3);

    constexpr index_t C = wei_desc.GetLength(I1);
    constexpr index_t Y = wei_desc.GetLength(I2);
    constexpr index_t X = wei_desc.GetLength(I3);

    return std::size_t(2) * N * K * Ho * Wo * C * Y * X;
}

template <class Float, class InDesc, class WeiDesc, class OutDesc>
constexpr std::size_t calculate_convolution_memory_size(Float, InDesc, WeiDesc, OutDesc)
{
    constexpr auto wei_desc = WeiDesc{};
    constexpr auto out_desc = OutDesc{};

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr index_t N  = out_desc.GetLength(I0);
    constexpr index_t K  = out_desc.GetLength(I1);
    constexpr index_t Ho = out_desc.GetLength(I2);
    constexpr index_t Wo = out_desc.GetLength(I3);

    constexpr index_t C = wei_desc.GetLength(I1);
    constexpr index_t Y = wei_desc.GetLength(I2);
    constexpr index_t X = wei_desc.GetLength(I3);

    return sizeof(Float) *
           (InDesc::GetElementSpace() + WeiDesc::GetElementSpace() + OutDesc::GetElementSpace());
}
