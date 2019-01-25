#pragma once
#include "ConstantTensorDescriptor.cuh"

// this is ugly, only for 4d
template <class InDesc, class WeiDesc>
__host__ __device__ constexpr auto get_convolution_output_default_4d_tensor_descriptor(InDesc,
                                                                                       WeiDesc)
{
    constexpr auto in_desc  = InDesc{};
    constexpr auto wei_desc = WeiDesc{};

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    static_assert(in_desc.GetDimension() == 4, "input nDim is not 4");
    static_assert(wei_desc.GetDimension() == 4, "weight nDim is not 4");
    static_assert(in_desc.GetLength(I1) == wei_desc.GetLength(I1),
                  "input & weight dimension not consistent");

    constexpr auto N  = in_desc.GetLength(I0);
    constexpr auto HI = in_desc.GetLength(I2);
    constexpr auto WI = in_desc.GetLength(I3);

    constexpr auto K = wei_desc.GetLength(I0);
    constexpr auto S = wei_desc.GetLength(I2);
    constexpr auto R = wei_desc.GetLength(I3);

    constexpr auto HO = HI + 1 - S;
    constexpr auto WO = WI + 1 - R;

    return make_ConstantTensorDescriptor(Sequence<N, K, HO, WO>{});
}

template <class InDesc, class WeiDesc, class LowerPads, class UpperPads>
__host__ __device__ constexpr auto get_convolution_with_padding_output_default_4d_tensor_descriptor(
    InDesc, WeiDesc, LowerPads, UpperPads)
{
    constexpr auto in_desc  = InDesc{};
    constexpr auto wei_desc = WeiDesc{};

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    static_assert(in_desc.GetDimension() == 4, "input nDim is not 4");
    static_assert(wei_desc.GetDimension() == 4, "weight nDim is not 4");
    static_assert(in_desc.GetLength(I1) == wei_desc.GetLength(I1),
                  "input & weight dimension not consistent");

    constexpr auto N  = in_desc.GetLength(I0);
    constexpr auto HI = in_desc.GetLength(I2);
    constexpr auto WI = in_desc.GetLength(I3);

    constexpr auto K = wei_desc.GetLength(I0);
    constexpr auto S = wei_desc.GetLength(I2);
    constexpr auto R = wei_desc.GetLength(I3);

    constexpr auto HPadLow = LowerPads{}.Get(I0);
    constexpr auto WPadLow = LowerPads{}.Get(I1);

    constexpr auto HPadUp = UpperPads{}.Get(I0);
    constexpr auto WPadUp = UpperPads{}.Get(I1);

    constexpr auto HO = HI + HPadLow + HPadUp + 1 - S;
    constexpr auto WO = WI + WPadLow + WPadUp + 1 - R;

    return make_ConstantTensorDescriptor(Sequence<N, K, HO, WO>{});
}
