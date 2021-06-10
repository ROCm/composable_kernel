#ifndef CONV_COMMON_HPP
#define CONV_COMMON_HPP

#include "tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor.hpp"

enum ConvTensorLayout
{
    NCHW,
    NHWC,
    CHWN,
    NCHWc,
    NHWCc
};

template <class InDesc,
          class WeiDesc,
          class ConvStrides,
          class ConvDilations,
          class LeftPads,
          class RightPads>
constexpr auto get_convolution_output_default_4d_tensor_descriptor(
    InDesc, WeiDesc, ConvStrides, ConvDilations, LeftPads, RightPads)
{
    using namespace ck;

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

    constexpr index_t N  = in_desc.GetLength(I0);
    constexpr index_t Hi = in_desc.GetLength(I2);
    constexpr index_t Wi = in_desc.GetLength(I3);

    constexpr index_t K = wei_desc.GetLength(I0);
    constexpr index_t Y = wei_desc.GetLength(I2);
    constexpr index_t X = wei_desc.GetLength(I3);

    constexpr index_t LeftPadH = LeftPads{}.Get(I0);
    constexpr index_t LeftPadW = LeftPads{}.Get(I1);

    constexpr index_t RightPadH = RightPads{}.Get(I0);
    constexpr index_t RightPadW = RightPads{}.Get(I1);

    constexpr index_t YEff = (Y - 1) * ConvDilations{}[0] + 1;
    constexpr index_t XEff = (X - 1) * ConvDilations{}[1] + 1;

    constexpr index_t Ho = (Hi + LeftPadH + RightPadH - YEff) / ConvStrides{}[0] + 1;
    constexpr index_t Wo = (Wi + LeftPadW + RightPadW - XEff) / ConvStrides{}[1] + 1;

    return make_native_tensor_descriptor_packed(Sequence<N, K, Ho, Wo>{});
}

template <typename... InDesc,
          typename... WeiDesc,
          typename ConvStrides,
          typename ConvDilations,
          typename LeftPads,
          typename RightPads>
constexpr auto get_convolution_output_default_4d_tensor_descriptor(
    const ck::DynamicTensorDescriptor<InDesc...>& in_desc,
    const ck::DynamicTensorDescriptor<WeiDesc...>& wei_desc,
    const ConvStrides& conv_strides,
    const ConvDilations conv_dilations,
    const LeftPads& left_pads,
    const RightPads& right_pads)
{
    using namespace ck;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    assert(in_desc.GetNumOfDimension() == 4);
    assert(wei_desc.GetNumOfDimension() == 4);
    assert(in_desc.GetLength(I1) == wei_desc.GetLength(I1));

    const auto N  = in_desc.GetLength(I0);
    const auto Hi = in_desc.GetLength(I2);
    const auto Wi = in_desc.GetLength(I3);

    const auto K = wei_desc.GetLength(I0);
    const auto Y = wei_desc.GetLength(I2);
    const auto X = wei_desc.GetLength(I3);

    const auto LeftPadH = left_pads[I0];
    const auto LeftPadW = left_pads[I1];

    const auto RightPadH = right_pads[I0];
    const auto RightPadW = right_pads[I1];

    const auto YEff = (Y - I1) * conv_dilations[I0] + I1;
    const auto XEff = (X - I1) * conv_dilations[I1] + I1;

    const auto Ho = (Hi + LeftPadH + RightPadH - YEff) / conv_strides[I0] + I1;
    const auto Wo = (Wi + LeftPadW + RightPadW - XEff) / conv_strides[I1] + I1;

    return make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(N, K, Ho, Wo));
}

template <class InDesc, class WeiDesc, class OutDesc>
constexpr std::size_t
calculate_convolution_flops(const InDesc& in_desc, const WeiDesc& wei_desc, const OutDesc& out_desc)
{
    using namespace ck;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    const index_t N  = out_desc.GetLength(I0);
    const index_t K  = out_desc.GetLength(I1);
    const index_t Ho = out_desc.GetLength(I2);
    const index_t Wo = out_desc.GetLength(I3);

    const index_t C = wei_desc.GetLength(I1);
    const index_t Y = wei_desc.GetLength(I2);
    const index_t X = wei_desc.GetLength(I3);

    return std::size_t(2) * N * K * Ho * Wo * C * Y * X;
}

template <class Float, class InDesc, class WeiDesc, class OutDesc>
constexpr std::size_t calculate_convolution_memory_size(Float, InDesc, WeiDesc, OutDesc)
{
    using namespace ck;

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

#endif
