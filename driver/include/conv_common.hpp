#ifndef CONV_COMMON_HPP
#define CONV_COMMON_HPP

#include "ConstantTensorDescriptor_deprecated.hpp"
#include "tensor_descriptor.hpp"

template <class InDesc,
          class WeiDesc,
          class ConvStrides,
          class ConvDilations,
          class LowerPads,
          class UpperPads>
constexpr auto get_convolution_output_default_4d_tensor_descriptor_deprecated(
    InDesc, WeiDesc, ConvStrides, ConvDilations, LowerPads, UpperPads)
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

    constexpr index_t HPadLow = LowerPads{}.Get(I0);
    constexpr index_t WPadLow = LowerPads{}.Get(I1);

    constexpr index_t HPadUp = UpperPads{}.Get(I0);
    constexpr index_t WPadUp = UpperPads{}.Get(I1);

    constexpr index_t YEff = (Y - 1) * ConvDilations{}[0] + 1;
    constexpr index_t XEff = (X - 1) * ConvDilations{}[1] + 1;

    constexpr index_t Ho = (Hi + HPadLow + HPadUp - YEff) / ConvStrides{}[0] + 1;
    constexpr index_t Wo = (Wi + WPadLow + WPadUp - XEff) / ConvStrides{}[1] + 1;

    return make_ConstantTensorDescriptor_packed(Sequence<N, K, Ho, Wo>{});
}

template <class InDesc,
          class WeiDesc,
          class ConvStrides,
          class ConvDilations,
          class LowerPads,
          class UpperPads>
constexpr auto get_convolution_output_default_4d_tensor_descriptor(
    InDesc, WeiDesc, ConvStrides, ConvDilations, LowerPads, UpperPads)
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

    constexpr index_t HPadLow = LowerPads{}.Get(I0);
    constexpr index_t WPadLow = LowerPads{}.Get(I1);

    constexpr index_t HPadUp = UpperPads{}.Get(I0);
    constexpr index_t WPadUp = UpperPads{}.Get(I1);

    constexpr index_t YEff = (Y - 1) * ConvDilations{}[0] + 1;
    constexpr index_t XEff = (X - 1) * ConvDilations{}[1] + 1;

    constexpr index_t Ho = (Hi + HPadLow + HPadUp - YEff) / ConvStrides{}[0] + 1;
    constexpr index_t Wo = (Wi + WPadLow + WPadUp - XEff) / ConvStrides{}[1] + 1;

    return make_native_tensor_descriptor_packed(Sequence<N, K, Ho, Wo>{});
}

template <class InDesc, class WeiDesc, class OutDesc>
constexpr std::size_t calculate_convolution_flops(InDesc, WeiDesc, OutDesc)
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
