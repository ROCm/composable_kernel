#pragma once
#include "host_tensor.hpp"
#include "conv_common.hpp"

template <typename TIn,
          typename TWei,
          typename TOut,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
void host_conv_nchw_kcyx_nkhw(const Tensor<TIn>& in,
                              const Tensor<TWei>& wei,
                              Tensor<TOut>& out,
                              const ConvStrides& conv_strides,
                              const ConvDilations& conv_dilations,
                              const InLeftPads& in_left_pads,
                              const InRightPads&)
{
    constexpr auto I0 = ck::Number<0>{};
    constexpr auto I1 = ck::Number<1>{};

    auto f_nchw = [&](auto n, auto k, auto ho, auto wo) {
        double v = 0;
        for(int c = 0; c < wei.mDesc.GetLengths()[1]; ++c)
        {
            for(int y = 0; y < wei.mDesc.GetLengths()[2]; ++y)
            {
                int hi = ho * conv_strides[I0] + y * conv_dilations[I0] - in_left_pads[I0];
                for(int x = 0; x < wei.mDesc.GetLengths()[3]; ++x)
                {
                    int wi = wo * conv_strides[I1] + x * conv_dilations[I1] - in_left_pads[I1];
                    if(hi >= 0 && hi < in.mDesc.GetLengths()[2] && wi >= 0 &&
                       wi < in.mDesc.GetLengths()[3])
                    {
                        v += static_cast<const double>(in(n, c, hi, wi)) *
                             static_cast<const double>(wei(k, c, y, x));
                    }
                }
            }
        }
        out(n, k, ho, wo) = v;
    };

    make_ParallelTensorFunctor(f_nchw,
                               out.mDesc.GetLengths()[0],
                               out.mDesc.GetLengths()[1],
                               out.mDesc.GetLengths()[2],
                               out.mDesc.GetLengths()[3])(std::thread::hardware_concurrency());
}

void host_direct_convolution3d(const Tensor<TIn>& in,
                               const Tensor<TWei>& wei,
                               Tensor<TOut>& out,
                               const ConvStrides& conv_strides,
                               const ConvDilations& conv_dilations,
                               const InLeftPads& in_left_pads,
                               const InRightPads&,
                               const ConvTensorLayout layout = ConvTensorLayout::NDHWC)
{
    using namespace ck;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};

    auto f_ndhwc = [&](auto n, auto do_, auto ho, auto wo, auto k) {
        double v = 0;
        for(int c = 0; c < wei.mDesc.GetLengths()[4]; ++c)
        {
            for(int z = 0; z < wei.mDesc.GetLengths()[1]; ++z)
            {
                int di = do_ * conv_strides[I0] + z * conv_dilations[I0] - in_left_pads[I0];
                for(int y = 0; y < wei.mDesc.GetLengths()[2]; ++y)
                {
                    int hi = ho * conv_strides[I1] + y * conv_dilations[I1] - in_left_pads[I1];
                    for(int x = 0; x < wei.mDesc.GetLengths()[3]; ++x)
                    {
                        int wi = wo * conv_strides[I2] + x * conv_dilations[I2] - in_left_pads[I2];
                        if(di >= 0 && di < in.mDesc.GetLengths()[1] &&  
                           hi >= 0 && hi < in.mDesc.GetLengths()[2] && 
                           wi >= 0 && wi < in.mDesc.GetLengths()[3])
                        {
                            v += static_cast<const double>(in(n, di, hi, wi, c)) *
                                static_cast<const double>(wei(k, z, y, x, c));
                        }
                    }
                }
            }
        }
        out(n, do_, ho, wo, k) = v;
    };

    if(layout == ConvTensorLayout::NDHWC)
    {
        make_ParallelTensorFunctor(f_ndhwc,
                                   out.mDesc.GetLengths()[0],
                                   out.mDesc.GetLengths()[1],
                                   out.mDesc.GetLengths()[2],
                                   out.mDesc.GetLengths()[3],
                                   out.mDesc.GetLengths()[4])(std::thread::hardware_concurrency() - 8);
    }
    else
    {
        throw std::runtime_error("wrong! not supported layout");
    }
}

