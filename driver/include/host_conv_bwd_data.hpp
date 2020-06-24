#pragma once
#include "host_tensor.hpp"

template <typename TIn,
          typename TWei,
          typename TOut,
          typename ConvStrides,
          typename ConvDilations,
          typename LeftPads,
          typename RightPads>
void host_direct_convolution_backward_data(Tensor<TIn>& in_nchw,
                                           const Tensor<TWei>& wei_kcyx,
                                           const Tensor<TOut>& out_nkhw,
                                           ConvStrides,
                                           ConvDilations,
                                           LeftPads,
                                           RightPads)
{
    using namespace ck;

    int N  = in_nchw.mDesc.GetLengths()[0];
    int C  = in_nchw.mDesc.GetLengths()[1];
    int HI = in_nchw.mDesc.GetLengths()[2];
    int WI = in_nchw.mDesc.GetLengths()[3];

    std::size_t K = wei_kcyx.mDesc.GetLengths()[0];
    std::size_t Y = wei_kcyx.mDesc.GetLengths()[2];
    std::size_t X = wei_kcyx.mDesc.GetLengths()[3];

    std::size_t HO = out_nkhw.mDesc.GetLengths()[2];
    std::size_t WO = out_nkhw.mDesc.GetLengths()[3];

    auto f = [&](auto n, auto c, auto hi, auto wi) {
        double v = 0;

        for(int y = 0; y < Y; ++y)
        {
            int h_tmp = hi + LeftPads{}[0] - y * ConvDilations{}[0];

            if(h_tmp % ConvStrides{}[0] == 0)
            {
                int ho = h_tmp / ConvStrides{}[0];

                if(ho >= 0 && ho < HO)
                {
                    for(int x = 0; x < X; ++x)
                    {
                        int w_tmp = wi + LeftPads{}[1] - x * ConvDilations{}[1];

                        if(w_tmp % ConvStrides{}[1] == 0)
                        {
                            int wo = w_tmp / ConvStrides{}[1];

                            if(wo >= 0 && wo < WO)
                            {
                                for(int k = 0; k < K; ++k)
                                {
                                    v += out_nkhw(n, k, ho, wo) * wei_kcyx(k, c, y, x);
                                }
                            }
                        }
                    }
                }
            }
        }

        in_nchw(n, c, hi, wi) = v;
    };

    auto f_par = make_ParallelTensorFunctor(f,
                                            in_nchw.mDesc.GetLengths()[0],
                                            in_nchw.mDesc.GetLengths()[1],
                                            in_nchw.mDesc.GetLengths()[2],
                                            in_nchw.mDesc.GetLengths()[3]);

    f_par(std::thread::hardware_concurrency());
}
