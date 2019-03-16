#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include "config.h"
#include "tensor.hpp"
#include "ConstantTensorDescriptor.hip.hpp"
#include "conv_common.hip.hpp"
#include "device_direct_convolution_1.hpp"
#include "device_direct_convolution_2_nchw_kcyx_nkhw.hpp"
#include "device_implicit_gemm_convolution_1_chwn_cyxk_khwn.hpp"
#include "device_implicit_gemm_convolution_1_chwn_cyxk_khwn_padded.hpp"
#include "device_implicit_gemm_convolution_2_chwn_cyxk_khwn.hpp"

struct GeneratorTensor_1
{
    template <class... Is>
    double operator()(Is... is)
    {
        return 1;
    }
};

struct GeneratorTensor_2
{
    int min_value = 0;
    int max_value = 1;

    template <class... Is>
    double operator()(Is...)
    {
        return (std::rand() % (max_value - min_value)) + min_value;
    }
};

struct GeneratorTensor_3
{
    template <class... Is>
    double operator()(Is... is)
    {
#if 0
        std::initializer_list<std::size_t> ls = {static_cast<std::size_t>(is)...};
        return std::accumulate(ls.begin(), ls.end(), std::size_t(0));
#elif 1
        assert(sizeof...(Is) > 0);
        std::initializer_list<std::size_t> ids = {static_cast<std::size_t>(is)...};
        std::vector<std::size_t> lens(sizeof...(Is), 100);
        std::vector<std::size_t> strides(sizeof...(Is), 1);
        std::partial_sum(lens.rbegin(), lens.rbegin() + (sizeof...(Is) - 1), strides.rbegin() + 1);
        return std::inner_product(ids.begin(), ids.end(), strides.begin(), std::size_t(0)) + 1;
#endif
    }
};

struct GeneratorTensor_Checkboard
{
    template <class... Ts>
    double operator()(Ts... Xs) const
    {
        std::array<unsigned long, sizeof...(Ts)> dims = {{Xs...}};
        return std::accumulate(dims.begin(),
                               dims.end(),
                               true,
                               [](bool init, unsigned long x) -> int { return init != (x % 2); })
                   ? 1
                   : -1;
    }
};

// this is ugly, only for 4d
template <class TConstTensorDesc>
void ostream_ConstantTensorDescriptor(TConstTensorDesc, std::ostream& os = std::cout)
{
    static_assert(TConstTensorDesc::nDim == 4, "nDim is not 4");

    constexpr auto I0   = Number<0>{};
    constexpr auto I1   = Number<1>{};
    constexpr auto I2   = Number<2>{};
    constexpr auto I3   = Number<3>{};
    constexpr auto desc = TConstTensorDesc{};

    os << "Lengths: {" << desc.GetLength(I0) << ", " << desc.GetLength(I1) << ", "
       << desc.GetLength(I2) << ", " << desc.GetLength(I3) << "}, "
       << "Strides: {" << desc.GetStride(I0) << ", " << desc.GetStride(I1) << ", "
       << desc.GetStride(I2) << ", " << desc.GetStride(I3) << "}" << std::endl;
}

// this is ugly, only for 4d
template <class TConstTensorDesc>
auto make_TensorDescriptor(TConstTensorDesc)
{
    static_assert(TConstTensorDesc::nDim == 4, "nDim is not 4");

    constexpr auto I0   = Number<0>{};
    constexpr auto I1   = Number<1>{};
    constexpr auto I2   = Number<2>{};
    constexpr auto I3   = Number<3>{};
    constexpr auto desc = TConstTensorDesc{};

    std::initializer_list<unsigned> lengths = {
        desc.GetLength(I0), desc.GetLength(I1), desc.GetLength(I2), desc.GetLength(I3)};
    std::initializer_list<unsigned> strides = {
        desc.GetStride(I0), desc.GetStride(I1), desc.GetStride(I2), desc.GetStride(I3)};

    return TensorDescriptor(lengths, strides);
}

template <class T, class LowerPads, class UpperPads>
void host_direct_convolution(
    const Tensor<T>& in_nchw, const Tensor<T>& wei_kcyx, Tensor<T>& out, LowerPads, UpperPads)
{
    unsigned h_pad_low = LowerPads{}.Get(Number<0>{});
    unsigned w_pad_low = LowerPads{}.Get(Number<1>{});

    unsigned h_pad_up = UpperPads{}.Get(Number<0>{});
    unsigned w_pad_up = UpperPads{}.Get(Number<1>{});

    auto f = [&](auto n, auto k, auto ho, auto wo) {
        double v = 0;
        for(int c = 0; c < wei_kcyx.mDesc.GetLengths()[1]; ++c)
        {
            for(int y = 0; y < wei_kcyx.mDesc.GetLengths()[2]; ++y)
            {
                int hi = ho + y - h_pad_low;
                for(int x = 0; x < wei_kcyx.mDesc.GetLengths()[3]; ++x)
                {
                    int wi = wo + x - w_pad_low;
                    if(hi >= 0 && hi < in_nchw.mDesc.GetLengths()[2] && wi >= 0 &&
                       wi < in_nchw.mDesc.GetLengths()[3])
                    {
                        v += in_nchw(n, c, hi, wi) * wei_kcyx(k, c, y, x);
                    }
                }
            }
        }
        out(n, k, ho, wo) = v;
    };

    auto f_par = make_ParallelTensorFunctor(f,
                                            out.mDesc.GetLengths()[0],
                                            out.mDesc.GetLengths()[1],
                                            out.mDesc.GetLengths()[2],
                                            out.mDesc.GetLengths()[3]);

    f_par(std::thread::hardware_concurrency());
}

template <class T, class LowerPads, class UpperPads>
void host_winograd_3x3_convolution(
    const Tensor<T>& in_nchw, const Tensor<T>& wei_kcyx, Tensor<T>& out, LowerPads, UpperPads)
{
    constexpr std::size_t HoPerTile = 2;
    constexpr std::size_t WoPerTile = 2;

    std::size_t N  = in_nchw.mDesc.GetLengths()[0];
    std::size_t C  = in_nchw.mDesc.GetLengths()[1];
    std::size_t HI = in_nchw.mDesc.GetLengths()[2];
    std::size_t WI = in_nchw.mDesc.GetLengths()[3];

    std::size_t K = wei_kcyx.mDesc.GetLengths()[0];
    std::size_t Y = wei_kcyx.mDesc.GetLengths()[2];
    std::size_t X = wei_kcyx.mDesc.GetLengths()[3];

    std::size_t HO = out.mDesc.GetLengths()[2];
    std::size_t WO = out.mDesc.GetLengths()[3];

    unsigned h_pad_low = LowerPads{}.Get(Number<0>{});
    unsigned w_pad_low = LowerPads{}.Get(Number<1>{});

    unsigned h_pad_up = UpperPads{}.Get(Number<0>{});
    unsigned w_pad_up = UpperPads{}.Get(Number<1>{});

    std::size_t HiPerTile = HoPerTile + Y - 1;
    std::size_t WiPerTile = WoPerTile + X - 1;

    std::size_t HTile = (HO + HoPerTile - 1) / HoPerTile;
    std::size_t WTile = (WO + WoPerTile - 1) / WoPerTile;

    Tensor<T> in_hold({N, C, HTile, WTile, HiPerTile, WiPerTile});
    Tensor<T> in_transform({N, C, HTile, WTile, HiPerTile, WiPerTile});
    Tensor<T> wei_transform({K, C, HiPerTile, WiPerTile});
    Tensor<T> out_transform({N, K, HTile, WTile, HiPerTile, HiPerTile});
    Tensor<T> out_hold({N, K, HTile, WTile, HoPerTile, WoPerTile});

    auto f_in_hold = [&](auto n, auto c, auto htile, auto wtile) {
        for(int j = 0; j < HiPerTile; ++j)
        {
            int hi = HoPerTile * htile + j - h_pad_low;
            for(int i = 0; i < WiPerTile; ++i)
            {
                int wi = WoPerTile * wtile + i - w_pad_low;

                if(hi >= 0 && hi < in_nchw.mDesc.GetLengths()[2] && wi >= 0 &&
                   wi < in_nchw.mDesc.GetLengths()[3])
                {
                    in_hold(n, c, htile, wtile, j, i) = in_nchw(n, c, hi, wi);
                }
                else
                {
                    in_hold(n, c, htile, wtile, j, i) = T(0);
                }
            }
        }
    };

    auto f_in_transform = [&](auto n, auto c, auto htile, auto wtile) {
        in_transform(n, c, htile, wtile, 0, 0) =
            in_hold(n, c, htile, wtile, 0, 0) - in_hold(n, c, htile, wtile, 0, 2) -
            in_hold(n, c, htile, wtile, 2, 0) + in_hold(n, c, htile, wtile, 2, 2);
        in_transform(n, c, htile, wtile, 0, 1) =
            in_hold(n, c, htile, wtile, 0, 1) + in_hold(n, c, htile, wtile, 0, 2) -
            in_hold(n, c, htile, wtile, 2, 1) - in_hold(n, c, htile, wtile, 2, 2);
        in_transform(n, c, htile, wtile, 0, 2) =
            -in_hold(n, c, htile, wtile, 0, 1) + in_hold(n, c, htile, wtile, 0, 2) +
            in_hold(n, c, htile, wtile, 2, 1) - in_hold(n, c, htile, wtile, 2, 2);
        in_transform(n, c, htile, wtile, 0, 3) =
            in_hold(n, c, htile, wtile, 0, 1) - in_hold(n, c, htile, wtile, 0, 3) -
            in_hold(n, c, htile, wtile, 2, 1) + in_hold(n, c, htile, wtile, 2, 3);

        in_transform(n, c, htile, wtile, 1, 0) =
            in_hold(n, c, htile, wtile, 1, 0) - in_hold(n, c, htile, wtile, 1, 2) +
            in_hold(n, c, htile, wtile, 2, 0) - in_hold(n, c, htile, wtile, 2, 2);
        in_transform(n, c, htile, wtile, 1, 1) =
            in_hold(n, c, htile, wtile, 1, 1) + in_hold(n, c, htile, wtile, 1, 2) +
            in_hold(n, c, htile, wtile, 2, 1) + in_hold(n, c, htile, wtile, 2, 2);
        in_transform(n, c, htile, wtile, 1, 2) =
            -in_hold(n, c, htile, wtile, 1, 1) + in_hold(n, c, htile, wtile, 1, 2) -
            in_hold(n, c, htile, wtile, 2, 1) + in_hold(n, c, htile, wtile, 2, 2);
        in_transform(n, c, htile, wtile, 1, 3) =
            in_hold(n, c, htile, wtile, 1, 1) - in_hold(n, c, htile, wtile, 1, 3) +
            in_hold(n, c, htile, wtile, 2, 1) - in_hold(n, c, htile, wtile, 2, 3);

        in_transform(n, c, htile, wtile, 2, 0) =
            -in_hold(n, c, htile, wtile, 1, 0) + in_hold(n, c, htile, wtile, 1, 2) +
            in_hold(n, c, htile, wtile, 2, 0) - in_hold(n, c, htile, wtile, 2, 2);
        in_transform(n, c, htile, wtile, 2, 1) =
            -in_hold(n, c, htile, wtile, 1, 1) - in_hold(n, c, htile, wtile, 1, 2) +
            in_hold(n, c, htile, wtile, 2, 1) + in_hold(n, c, htile, wtile, 2, 2);
        in_transform(n, c, htile, wtile, 2, 2) =
            in_hold(n, c, htile, wtile, 1, 1) - in_hold(n, c, htile, wtile, 1, 2) -
            in_hold(n, c, htile, wtile, 2, 1) + in_hold(n, c, htile, wtile, 2, 2);
        in_transform(n, c, htile, wtile, 2, 3) =
            -in_hold(n, c, htile, wtile, 1, 1) + in_hold(n, c, htile, wtile, 1, 3) +
            in_hold(n, c, htile, wtile, 2, 1) - in_hold(n, c, htile, wtile, 2, 3);

        in_transform(n, c, htile, wtile, 3, 0) =
            in_hold(n, c, htile, wtile, 1, 0) - in_hold(n, c, htile, wtile, 1, 2) -
            in_hold(n, c, htile, wtile, 3, 0) + in_hold(n, c, htile, wtile, 3, 2);
        in_transform(n, c, htile, wtile, 3, 1) =
            in_hold(n, c, htile, wtile, 1, 1) + in_hold(n, c, htile, wtile, 1, 2) -
            in_hold(n, c, htile, wtile, 3, 1) - in_hold(n, c, htile, wtile, 3, 2);
        in_transform(n, c, htile, wtile, 3, 2) =
            -in_hold(n, c, htile, wtile, 1, 1) + in_hold(n, c, htile, wtile, 1, 2) +
            in_hold(n, c, htile, wtile, 3, 1) - in_hold(n, c, htile, wtile, 3, 2);
        in_transform(n, c, htile, wtile, 3, 3) =
            in_hold(n, c, htile, wtile, 1, 1) - in_hold(n, c, htile, wtile, 1, 3) -
            in_hold(n, c, htile, wtile, 3, 1) + in_hold(n, c, htile, wtile, 3, 3);
    };

    auto f_wei_transform = [&](auto k, auto c) {
        wei_transform(k, c, 0, 0) = wei_kcyx(k, c, 0, 0);
        wei_transform(k, c, 0, 1) =
            0.5 * wei_kcyx(k, c, 0, 0) + 0.5 * wei_kcyx(k, c, 0, 1) + 0.5 * wei_kcyx(k, c, 0, 2);
        wei_transform(k, c, 0, 2) =
            0.5 * wei_kcyx(k, c, 0, 0) - 0.5 * wei_kcyx(k, c, 0, 1) + 0.5 * wei_kcyx(k, c, 0, 2);
        wei_transform(k, c, 0, 3) = wei_kcyx(k, c, 0, 2);

        wei_transform(k, c, 1, 0) =
            0.5 * wei_kcyx(k, c, 0, 0) + 0.5 * wei_kcyx(k, c, 1, 0) + 0.5 * wei_kcyx(k, c, 2, 0);
        wei_transform(k, c, 1, 1) = 0.25 * wei_kcyx(k, c, 0, 0) + 0.25 * wei_kcyx(k, c, 0, 1) +
                                    0.25 * wei_kcyx(k, c, 0, 2) + 0.25 * wei_kcyx(k, c, 1, 0) +
                                    0.25 * wei_kcyx(k, c, 1, 1) + 0.25 * wei_kcyx(k, c, 1, 2) +
                                    0.25 * wei_kcyx(k, c, 2, 0) + 0.25 * wei_kcyx(k, c, 2, 1) +
                                    0.25 * wei_kcyx(k, c, 2, 2);
        wei_transform(k, c, 1, 2) = 0.25 * wei_kcyx(k, c, 0, 0) - 0.25 * wei_kcyx(k, c, 0, 1) +
                                    0.25 * wei_kcyx(k, c, 0, 2) + 0.25 * wei_kcyx(k, c, 1, 0) -
                                    0.25 * wei_kcyx(k, c, 1, 1) + 0.25 * wei_kcyx(k, c, 1, 2) +
                                    0.25 * wei_kcyx(k, c, 2, 0) - 0.25 * wei_kcyx(k, c, 2, 1) +
                                    0.25 * wei_kcyx(k, c, 2, 2);
        wei_transform(k, c, 1, 3) =
            0.5 * wei_kcyx(k, c, 0, 2) + 0.5 * wei_kcyx(k, c, 1, 2) + 0.5 * wei_kcyx(k, c, 2, 2);

        wei_transform(k, c, 2, 0) =
            0.5 * wei_kcyx(k, c, 0, 0) - 0.5 * wei_kcyx(k, c, 1, 0) + 0.5 * wei_kcyx(k, c, 2, 0);
        wei_transform(k, c, 2, 1) = 0.25 * wei_kcyx(k, c, 0, 0) + 0.25 * wei_kcyx(k, c, 0, 1) +
                                    0.25 * wei_kcyx(k, c, 0, 2) - 0.25 * wei_kcyx(k, c, 1, 0) -
                                    0.25 * wei_kcyx(k, c, 1, 1) - 0.25 * wei_kcyx(k, c, 1, 2) +
                                    0.25 * wei_kcyx(k, c, 2, 0) + 0.25 * wei_kcyx(k, c, 2, 1) +
                                    0.25 * wei_kcyx(k, c, 2, 2);
        wei_transform(k, c, 2, 2) = 0.25 * wei_kcyx(k, c, 0, 0) - 0.25 * wei_kcyx(k, c, 0, 1) +
                                    0.25 * wei_kcyx(k, c, 0, 2) - 0.25 * wei_kcyx(k, c, 1, 0) +
                                    0.25 * wei_kcyx(k, c, 1, 1) - 0.25 * wei_kcyx(k, c, 1, 2) +
                                    0.25 * wei_kcyx(k, c, 2, 0) - 0.25 * wei_kcyx(k, c, 2, 1) +
                                    0.25 * wei_kcyx(k, c, 2, 2);
        wei_transform(k, c, 2, 3) =
            0.5 * wei_kcyx(k, c, 0, 2) - 0.5 * wei_kcyx(k, c, 1, 2) + 0.5 * wei_kcyx(k, c, 2, 2);

        wei_transform(k, c, 3, 0) = wei_kcyx(k, c, 2, 0);
        wei_transform(k, c, 3, 1) =
            0.5 * wei_kcyx(k, c, 2, 0) + 0.5 * wei_kcyx(k, c, 2, 1) + 0.5 * wei_kcyx(k, c, 2, 2);
        wei_transform(k, c, 3, 2) =
            0.5 * wei_kcyx(k, c, 2, 0) - 0.5 * wei_kcyx(k, c, 2, 1) + 0.5 * wei_kcyx(k, c, 2, 2);
        wei_transform(k, c, 3, 3) = wei_kcyx(k, c, 2, 2);
    };

    auto f_out_transform = [&](auto n, auto k, auto htile, auto wtile) {
        for(int j = 0; j < HiPerTile; ++j)
        {
            for(int i = 0; i < WiPerTile; ++i)
            {
                double v = 0;
                for(int c = 0; c < C; ++c)
                {
                    v += in_transform(n, c, htile, wtile, j, i) * wei_transform(k, c, j, i);
                }

                out_transform(n, k, htile, wtile, j, i) = v;
            }
        }
    };

    auto f_out_hold = [&](auto n, auto k, auto htile, auto wtile) {
        out_hold(n, k, htile, wtile, 0, 0) =
            out_transform(n, k, htile, wtile, 0, 0) + out_transform(n, k, htile, wtile, 0, 1) +
            out_transform(n, k, htile, wtile, 0, 2) + out_transform(n, k, htile, wtile, 1, 0) +
            out_transform(n, k, htile, wtile, 1, 1) + out_transform(n, k, htile, wtile, 1, 2) +
            out_transform(n, k, htile, wtile, 2, 0) + out_transform(n, k, htile, wtile, 2, 1) +
            out_transform(n, k, htile, wtile, 2, 2);
        out_hold(n, k, htile, wtile, 0, 1) =
            out_transform(n, k, htile, wtile, 0, 1) - out_transform(n, k, htile, wtile, 0, 2) -
            out_transform(n, k, htile, wtile, 0, 3) + out_transform(n, k, htile, wtile, 1, 1) -
            out_transform(n, k, htile, wtile, 1, 2) - out_transform(n, k, htile, wtile, 1, 3) +
            out_transform(n, k, htile, wtile, 2, 1) - out_transform(n, k, htile, wtile, 2, 2) -
            out_transform(n, k, htile, wtile, 2, 3);
        out_hold(n, k, htile, wtile, 1, 0) =
            out_transform(n, k, htile, wtile, 1, 0) + out_transform(n, k, htile, wtile, 1, 1) +
            out_transform(n, k, htile, wtile, 1, 2) - out_transform(n, k, htile, wtile, 2, 0) -
            out_transform(n, k, htile, wtile, 2, 1) - out_transform(n, k, htile, wtile, 2, 2) -
            out_transform(n, k, htile, wtile, 3, 0) - out_transform(n, k, htile, wtile, 3, 1) -
            out_transform(n, k, htile, wtile, 3, 2);
        out_hold(n, k, htile, wtile, 1, 1) =
            out_transform(n, k, htile, wtile, 1, 1) - out_transform(n, k, htile, wtile, 1, 2) -
            out_transform(n, k, htile, wtile, 1, 3) - out_transform(n, k, htile, wtile, 2, 1) +
            out_transform(n, k, htile, wtile, 2, 2) + out_transform(n, k, htile, wtile, 2, 3) -
            out_transform(n, k, htile, wtile, 3, 1) + out_transform(n, k, htile, wtile, 3, 2) +
            out_transform(n, k, htile, wtile, 3, 3);
    };

    auto f_out = [&](auto n, auto k, auto htile, auto wtile) {
        for(int j = 0; j < HoPerTile; ++j)
        {
            std::size_t ho = HoPerTile * htile + j;
            for(int i = 0; i < WoPerTile; ++i)
            {
                std::size_t wo    = WoPerTile * wtile + i;
                out(n, k, ho, wo) = out_hold(n, k, htile, wtile, j, i);
            }
        }
    };

    std::size_t num_thread = std::thread::hardware_concurrency();

    make_ParallelTensorFunctor(f_in_hold, N, C, HTile, WTile)(num_thread);
    make_ParallelTensorFunctor(f_in_transform, N, C, HTile, WTile)(num_thread);
    make_ParallelTensorFunctor(f_wei_transform, K, C)(num_thread);
    make_ParallelTensorFunctor(f_out_transform, N, K, HTile, WTile)(num_thread);
    make_ParallelTensorFunctor(f_out_hold, N, K, HTile, WTile)(num_thread);
    make_ParallelTensorFunctor(f_out, N, K, HTile, WTile)(num_thread);
}

template <class T>
void check_error(const Tensor<T>& ref, const Tensor<T>& result)
{
    float error     = 0;
    float max_diff  = -1;
    float ref_value = 0, result_value = 0;
    for(int i = 0; i < ref.mData.size(); ++i)
    {
        error += std::abs(ref.mData[i] - result.mData[i]);
        float diff = std::abs(ref.mData[i] - result.mData[i]);
        if(max_diff < diff)
        {
            max_diff     = diff;
            ref_value    = ref.mData[i];
            result_value = result.mData[i];
        }
    }

    std::cout << "error: " << error << std::endl;
    std::cout << "max_diff: " << max_diff << ", " << ref_value << ", " << result_value << std::endl;
}

int main(int argc, char* argv[])
{
#if 0
    constexpr unsigned N  = 1;
    constexpr unsigned C  = 1;
    constexpr unsigned HI = 28;
    constexpr unsigned WI = 28;
    constexpr unsigned K  = 1;
    constexpr unsigned Y  = 3;
    constexpr unsigned X  = 3;

    constexpr unsigned HPad = 0;
    constexpr unsigned WPad = 0;
#elif 1
    // 3x3, 34x34
    constexpr unsigned N = 64;
    constexpr unsigned C = 256;
    constexpr unsigned HI = 34;
    constexpr unsigned WI = 34;
    constexpr unsigned K = 64;
    constexpr unsigned Y = 3;
    constexpr unsigned X = 3;

    constexpr unsigned HPad = 0;
    constexpr unsigned WPad = 0;
#elif 0
    // 3x3, 56x56
    constexpr unsigned N  = 64;
    constexpr unsigned C  = 64;
    constexpr unsigned HI = 56;
    constexpr unsigned WI = 56;
    constexpr unsigned K  = 64;
    constexpr unsigned Y  = 3;
    constexpr unsigned X  = 3;
#elif 0
    // 3x3, 58x58
    constexpr unsigned N  = 64;
    constexpr unsigned C  = 64;
    constexpr unsigned HI = 58;
    constexpr unsigned WI = 58;
    constexpr unsigned K  = 64;
    constexpr unsigned Y  = 3;
    constexpr unsigned X  = 3;
#elif 0
    // 5x5, 36x36
    constexpr unsigned N  = 64;
    constexpr unsigned C  = 256;
    constexpr unsigned HI = 36;
    constexpr unsigned WI = 36;
    constexpr unsigned K  = 64;
    constexpr unsigned Y  = 5;
    constexpr unsigned X  = 5;

    constexpr unsigned HPad = 0;
    constexpr unsigned WPad = 0;
#elif 0
    // 7x7, 38x38
    constexpr unsigned N  = 64;
    constexpr unsigned C  = 256;
    constexpr unsigned HI = 38;
    constexpr unsigned WI = 38;
    constexpr unsigned K  = 64;
    constexpr unsigned Y  = 7;
    constexpr unsigned X  = 7;

    constexpr unsigned HPad = 0;
    constexpr unsigned WPad = 0;
#elif 0
    // 3x3, 58x58
    constexpr unsigned N  = 16;
    constexpr unsigned C  = 128;
    constexpr unsigned HI = 58;
    constexpr unsigned WI = 58;
    constexpr unsigned K  = 256;
    constexpr unsigned Y  = 3;
    constexpr unsigned X  = 3;
#elif 0
    // 3x3 filter, 58x58 image, 0x0 padding
    constexpr unsigned N  = 16;
    constexpr unsigned C  = 128;
    constexpr unsigned HI = 58;
    constexpr unsigned WI = 58;
    constexpr unsigned K  = 256;
    constexpr unsigned Y  = 3;
    constexpr unsigned X  = 3;

    constexpr unsigned HPad = 0;
    constexpr unsigned WPad = 0;
#elif 0
    // 3x3 filter, 56x56 image, 1x1 padding
    constexpr unsigned N  = 16;
    constexpr unsigned C  = 128;
    constexpr unsigned HI = 56;
    constexpr unsigned WI = 56;
    constexpr unsigned K  = 256;
    constexpr unsigned Y  = 3;
    constexpr unsigned X  = 3;

    constexpr unsigned HPad = 1;
    constexpr unsigned WPad = 1;
#elif 0
    // 3x3 filter, 28x28 image, 1x1 padding
    constexpr unsigned N  = 16;
    constexpr unsigned C  = 256;
    constexpr unsigned HI = 28;
    constexpr unsigned WI = 28;
    constexpr unsigned K  = 512;
    constexpr unsigned Y  = 3;
    constexpr unsigned X  = 3;

    constexpr unsigned HPad = 1;
    constexpr unsigned WPad = 1;
#elif 1
    // 1x1 filter, 28x28 image
    constexpr unsigned N  = 16;
    constexpr unsigned C  = 256;
    constexpr unsigned HI = 28;
    constexpr unsigned WI = 28;
    constexpr unsigned K  = 512;
    constexpr unsigned Y  = 1;
    constexpr unsigned X  = 1;

    constexpr unsigned HPad = 0;
    constexpr unsigned WPad = 0;
#elif 0
    // 3x3 filter, 20x84 image, 1x1 padding
    constexpr unsigned N  = 16;
    constexpr unsigned C  = 256;
    constexpr unsigned HI = 20;
    constexpr unsigned WI = 84;
    constexpr unsigned K  = 256;
    constexpr unsigned Y  = 3;
    constexpr unsigned X  = 3;

    constexpr unsigned HPad = 1;
    constexpr unsigned WPad = 1;
#elif 0
    // 3x3 filter, 112x112 image, 1x1 padding
    constexpr unsigned N  = 16;
    constexpr unsigned C  = 64;
    constexpr unsigned HI = 112;
    constexpr unsigned WI = 112;
    constexpr unsigned K  = 128;
    constexpr unsigned Y  = 3;
    constexpr unsigned X  = 3;

    constexpr unsigned HPad = 1;
    constexpr unsigned WPad = 1;
#elif 0
    // 5x5 filter, 20x86 image, 1x1 padding
    constexpr unsigned N  = 16;
    constexpr unsigned C  = 256;
    constexpr unsigned HI = 20;
    constexpr unsigned WI = 86;
    constexpr unsigned K  = 512;
    constexpr unsigned Y  = 5;
    constexpr unsigned X  = 5;

    constexpr unsigned HPad = 1;
    constexpr unsigned WPad = 1;
#elif 0
    // 5x5 filter, 28x28 image, 2x2 padding
    constexpr unsigned N  = 16;
    constexpr unsigned C  = 192;
    constexpr unsigned HI = 28;
    constexpr unsigned WI = 28;
    constexpr unsigned K  = 32;
    constexpr unsigned Y  = 5;
    constexpr unsigned X  = 5;

    constexpr unsigned HPad = 2;
    constexpr unsigned WPad = 2;
#endif

    auto lower_pads = Sequence<HPad, WPad>{};
    auto upper_pads = Sequence<HPad, WPad>{};

    auto in_nchw_desc  = make_ConstantTensorDescriptor(Sequence<N, C, HI, WI>{});
    auto wei_kcyx_desc = make_ConstantTensorDescriptor(Sequence<K, C, Y, X>{});
    auto out_nkhw_desc = get_convolution_with_padding_output_default_4d_tensor_descriptor(
        in_nchw_desc, wei_kcyx_desc, lower_pads, upper_pads);

    ostream_ConstantTensorDescriptor(in_nchw_desc, std::cout << "in_nchw_desc: ");
    ostream_ConstantTensorDescriptor(wei_kcyx_desc, std::cout << "wei_kcyx_desc: ");
    ostream_ConstantTensorDescriptor(out_nkhw_desc, std::cout << "out_nkhw_desc: ");

    using Float = float;
    Tensor<Float> in_nchw(make_TensorDescriptor(in_nchw_desc));
    Tensor<Float> wei_kcyx(make_TensorDescriptor(wei_kcyx_desc));
    Tensor<Float> out_nkhw_host(make_TensorDescriptor(out_nkhw_desc));
    Tensor<Float> out_nkhw_device(make_TensorDescriptor(out_nkhw_desc));

    std::size_t num_thread = std::thread::hardware_concurrency();

    if(argc != 3)
    {
        printf("arg1: do_verification, arg2: nrepeat\n");
        exit(1);
    }

    bool do_verification = atoi(argv[1]);
    unsigned nrepeat     = atoi(argv[2]);

    if(do_verification)
    {
#if 0
        in_nchw.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
        wei_kcyx.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
#elif 1
        in_nchw.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
        wei_kcyx.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
#elif 1
        in_nchw.GenerateTensorValue(GeneratorTensor_2{-2, 2}, num_thread);
        wei_kcyx.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
#endif
    }

#if 1
#if 0
    device_direct_convolution_1
#elif 1
    device_direct_convolution_2_nchw_kcyx_nkhw
#elif 0
    device_implicit_gemm_convolution_1_chwn_cyxk_khwn
#elif 0
    device_implicit_gemm_convolution_2_chwn_cyxk_khwn
#endif
    (in_nchw_desc, in_nchw, wei_kcyx_desc, wei_kcyx, out_nkhw_desc, out_nkhw_device, nrepeat);

#elif 1
    device_implicit_gemm_convolution_1_chwn_cyxk_khwn_padded(in_nchw_desc,
                                                             in_nchw,
                                                             wei_kcyx_desc,
                                                             wei_kcyx,
                                                             out_nkhw_desc,
                                                             out_nkhw_device,
                                                             lower_pads,
                                                             upper_pads,
                                                             nrepeat);
#endif

    if(do_verification)
    {
#if 1
        if(Y == 3 && X == 3)
        {
            host_winograd_3x3_convolution(in_nchw, wei_kcyx, out_nkhw_host, lower_pads, upper_pads);
        }
        else
        {
            host_direct_convolution(in_nchw, wei_kcyx, out_nkhw_host, lower_pads, upper_pads);
        }
        check_error(out_nkhw_host, out_nkhw_device);
#endif

#if 0
        LogRange(std::cout << "in_nchw : ", in_nchw.mData, ",") << std::endl;
        LogRange(std::cout << "wei_kcyx: ", wei_kcyx.mData, ",") << std::endl;
        LogRange(std::cout << "out_nkhw_host  : ", out_nkhw_host.mData, ",") << std::endl;
        LogRange(std::cout << "out_nkhw_device: ", out_nkhw_device.mData, ",") << std::endl;
#endif
    }
}
