#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include "nvToolsExt.h"
#include "tensor.hpp"
#include "ConstantTensorDescriptor.cuh"
#include "conv_common.cuh"
#include "device_direct_convolution_1.cuh"
#include "device_direct_convolution_2.cuh"
#include "device_implicit_gemm_convolution.cuh"
//#include "device_winograd_convolution.cuh"

struct GeneratorTensor_1
{
    template <class... Is>
    double operator()(Is... is)
    {
#if 0
        return double(std::rand()) / double(RAND_MAX);
#elif 1
        return 1;
#elif 0
        std::initializer_list<std::size_t> ls = {static_cast<std::size_t>(is)...};
        return std::accumulate(ls.begin(), ls.end(), std::size_t(0));
#else
        assert(sizeof...(Is) > 0);
        std::initializer_list<std::size_t> ids = {static_cast<std::size_t>(is)...};
        std::vector<std::size_t> lens(sizeof...(Is), 100);
        std::vector<std::size_t> strides(sizeof...(Is), 1);
        std::partial_sum(lens.rbegin(), lens.rbegin() + (sizeof...(Is) - 1), strides.rbegin() + 1);
        return std::inner_product(ids.begin(), ids.end(), strides.begin(), std::size_t(0)) + 1;
#endif
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

template <class T>
void host_direct_convolution(const Tensor<T>& in_nchw, const Tensor<T>& wei_kcsr, Tensor<T>& out)
{
    auto f = [&](auto n, auto k, auto ho, auto wo) {
        double v = 0;
        for(int c = 0; c < wei_kcsr.mDesc.GetLengths()[1]; ++c)
        {
            for(int y = 0; y < wei_kcsr.mDesc.GetLengths()[2]; ++y)
            {
                int hi = ho + y;
                for(int x = 0; x < wei_kcsr.mDesc.GetLengths()[3]; ++x)
                {
                    int wi = wo + x;
                    v += in_nchw(n, c, hi, wi) * wei_kcsr(k, c, y, x);
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

template <class T>
void host_winograd_3x3_convolution(const Tensor<T>& in_nchw,
                                   const Tensor<T>& wei_kcsr,
                                   Tensor<T>& out)
{
    constexpr std::size_t OutTileSizeH = 2;
    constexpr std::size_t OutTileSizeW = 2;

    std::size_t N  = in_nchw.mDesc.GetLengths()[0];
    std::size_t C  = in_nchw.mDesc.GetLengths()[1];
    std::size_t HI = in_nchw.mDesc.GetLengths()[2];
    std::size_t WI = in_nchw.mDesc.GetLengths()[3];

    std::size_t K = wei_kcsr.mDesc.GetLengths()[0];
    std::size_t S = wei_kcsr.mDesc.GetLengths()[2];
    std::size_t R = wei_kcsr.mDesc.GetLengths()[3];

    std::size_t HO = out.mDesc.GetLengths()[2];
    std::size_t WO = out.mDesc.GetLengths()[3];

    std::size_t InTileSizeH = OutTileSizeH + S - 1;
    std::size_t InTileSizeW = OutTileSizeW + R - 1;

    std::size_t Y = (HO + OutTileSizeH - 1) / OutTileSizeH;
    std::size_t X = (WO + OutTileSizeW - 1) / OutTileSizeW;

    Tensor<T> in_hold({N, C, Y, X, InTileSizeH, InTileSizeW});
    Tensor<T> in_transform({N, C, Y, X, InTileSizeH, InTileSizeW});
    Tensor<T> wei_transform({K, C, InTileSizeH, InTileSizeW});
    Tensor<T> out_transform({N, K, Y, X, InTileSizeH, InTileSizeH});
    Tensor<T> out_hold({N, K, Y, X, OutTileSizeH, OutTileSizeW});

    auto f_in_hold = [&](auto n, auto c, auto y, auto x) {
        for(int j = 0; j < InTileSizeH; ++j)
        {
            std::size_t hi = OutTileSizeH * y + j;
            for(int i = 0; i < InTileSizeW; ++i)
            {
                std::size_t wi            = OutTileSizeW * x + i;
                in_hold(n, c, y, x, j, i) = in_nchw(n, c, hi, wi);
            }
        }
    };

    auto f_in_transform = [&](auto n, auto c, auto y, auto x) {
        in_transform(n, c, y, x, 0, 0) = in_hold(n, c, y, x, 0, 0) - in_hold(n, c, y, x, 0, 2) -
                                         in_hold(n, c, y, x, 2, 0) + in_hold(n, c, y, x, 2, 2);
        in_transform(n, c, y, x, 0, 1) = in_hold(n, c, y, x, 0, 1) + in_hold(n, c, y, x, 0, 2) -
                                         in_hold(n, c, y, x, 2, 1) - in_hold(n, c, y, x, 2, 2);
        in_transform(n, c, y, x, 0, 2) = -in_hold(n, c, y, x, 0, 1) + in_hold(n, c, y, x, 0, 2) +
                                         in_hold(n, c, y, x, 2, 1) - in_hold(n, c, y, x, 2, 2);
        in_transform(n, c, y, x, 0, 3) = in_hold(n, c, y, x, 0, 1) - in_hold(n, c, y, x, 0, 3) -
                                         in_hold(n, c, y, x, 2, 1) + in_hold(n, c, y, x, 2, 3);

        in_transform(n, c, y, x, 1, 0) = in_hold(n, c, y, x, 1, 0) - in_hold(n, c, y, x, 1, 2) +
                                         in_hold(n, c, y, x, 2, 0) - in_hold(n, c, y, x, 2, 2);
        in_transform(n, c, y, x, 1, 1) = in_hold(n, c, y, x, 1, 1) + in_hold(n, c, y, x, 1, 2) +
                                         in_hold(n, c, y, x, 2, 1) + in_hold(n, c, y, x, 2, 2);
        in_transform(n, c, y, x, 1, 2) = -in_hold(n, c, y, x, 1, 1) + in_hold(n, c, y, x, 1, 2) -
                                         in_hold(n, c, y, x, 2, 1) + in_hold(n, c, y, x, 2, 2);
        in_transform(n, c, y, x, 1, 3) = in_hold(n, c, y, x, 1, 1) - in_hold(n, c, y, x, 1, 3) +
                                         in_hold(n, c, y, x, 2, 1) - in_hold(n, c, y, x, 2, 3);

        in_transform(n, c, y, x, 2, 0) = -in_hold(n, c, y, x, 1, 0) + in_hold(n, c, y, x, 1, 2) +
                                         in_hold(n, c, y, x, 2, 0) - in_hold(n, c, y, x, 2, 2);
        in_transform(n, c, y, x, 2, 1) = -in_hold(n, c, y, x, 1, 1) - in_hold(n, c, y, x, 1, 2) +
                                         in_hold(n, c, y, x, 2, 1) + in_hold(n, c, y, x, 2, 2);
        in_transform(n, c, y, x, 2, 2) = in_hold(n, c, y, x, 1, 1) - in_hold(n, c, y, x, 1, 2) -
                                         in_hold(n, c, y, x, 2, 1) + in_hold(n, c, y, x, 2, 2);
        in_transform(n, c, y, x, 2, 3) = -in_hold(n, c, y, x, 1, 1) + in_hold(n, c, y, x, 1, 3) +
                                         in_hold(n, c, y, x, 2, 1) - in_hold(n, c, y, x, 2, 3);

        in_transform(n, c, y, x, 3, 0) = in_hold(n, c, y, x, 1, 0) - in_hold(n, c, y, x, 1, 2) -
                                         in_hold(n, c, y, x, 3, 0) + in_hold(n, c, y, x, 3, 2);
        in_transform(n, c, y, x, 3, 1) = in_hold(n, c, y, x, 1, 1) + in_hold(n, c, y, x, 1, 2) -
                                         in_hold(n, c, y, x, 3, 1) - in_hold(n, c, y, x, 3, 2);
        in_transform(n, c, y, x, 3, 2) = -in_hold(n, c, y, x, 1, 1) + in_hold(n, c, y, x, 1, 2) +
                                         in_hold(n, c, y, x, 3, 1) - in_hold(n, c, y, x, 3, 2);
        in_transform(n, c, y, x, 3, 3) = in_hold(n, c, y, x, 1, 1) - in_hold(n, c, y, x, 1, 3) -
                                         in_hold(n, c, y, x, 3, 1) + in_hold(n, c, y, x, 3, 3);
    };

    auto f_wei_transform = [&](auto k, auto c) {
        wei_transform(k, c, 0, 0) = wei_kcsr(k, c, 0, 0);
        wei_transform(k, c, 0, 1) =
            0.5 * wei_kcsr(k, c, 0, 0) + 0.5 * wei_kcsr(k, c, 0, 1) + 0.5 * wei_kcsr(k, c, 0, 2);
        wei_transform(k, c, 0, 2) =
            0.5 * wei_kcsr(k, c, 0, 0) - 0.5 * wei_kcsr(k, c, 0, 1) + 0.5 * wei_kcsr(k, c, 0, 2);
        wei_transform(k, c, 0, 3) = wei_kcsr(k, c, 0, 2);

        wei_transform(k, c, 1, 0) =
            0.5 * wei_kcsr(k, c, 0, 0) + 0.5 * wei_kcsr(k, c, 1, 0) + 0.5 * wei_kcsr(k, c, 2, 0);
        wei_transform(k, c, 1, 1) = 0.25 * wei_kcsr(k, c, 0, 0) + 0.25 * wei_kcsr(k, c, 0, 1) +
                                    0.25 * wei_kcsr(k, c, 0, 2) + 0.25 * wei_kcsr(k, c, 1, 0) +
                                    0.25 * wei_kcsr(k, c, 1, 1) + 0.25 * wei_kcsr(k, c, 1, 2) +
                                    0.25 * wei_kcsr(k, c, 2, 0) + 0.25 * wei_kcsr(k, c, 2, 1) +
                                    0.25 * wei_kcsr(k, c, 2, 2);
        wei_transform(k, c, 1, 2) = 0.25 * wei_kcsr(k, c, 0, 0) - 0.25 * wei_kcsr(k, c, 0, 1) +
                                    0.25 * wei_kcsr(k, c, 0, 2) + 0.25 * wei_kcsr(k, c, 1, 0) -
                                    0.25 * wei_kcsr(k, c, 1, 1) + 0.25 * wei_kcsr(k, c, 1, 2) +
                                    0.25 * wei_kcsr(k, c, 2, 0) - 0.25 * wei_kcsr(k, c, 2, 1) +
                                    0.25 * wei_kcsr(k, c, 2, 2);
        wei_transform(k, c, 1, 3) =
            0.5 * wei_kcsr(k, c, 0, 2) + 0.5 * wei_kcsr(k, c, 1, 2) + 0.5 * wei_kcsr(k, c, 2, 2);

        wei_transform(k, c, 2, 0) =
            0.5 * wei_kcsr(k, c, 0, 0) - 0.5 * wei_kcsr(k, c, 1, 0) + 0.5 * wei_kcsr(k, c, 2, 0);
        wei_transform(k, c, 2, 1) = 0.25 * wei_kcsr(k, c, 0, 0) + 0.25 * wei_kcsr(k, c, 0, 1) +
                                    0.25 * wei_kcsr(k, c, 0, 2) - 0.25 * wei_kcsr(k, c, 1, 0) -
                                    0.25 * wei_kcsr(k, c, 1, 1) - 0.25 * wei_kcsr(k, c, 1, 2) +
                                    0.25 * wei_kcsr(k, c, 2, 0) + 0.25 * wei_kcsr(k, c, 2, 1) +
                                    0.25 * wei_kcsr(k, c, 2, 2);
        wei_transform(k, c, 2, 2) = 0.25 * wei_kcsr(k, c, 0, 0) - 0.25 * wei_kcsr(k, c, 0, 1) +
                                    0.25 * wei_kcsr(k, c, 0, 2) - 0.25 * wei_kcsr(k, c, 1, 0) +
                                    0.25 * wei_kcsr(k, c, 1, 1) - 0.25 * wei_kcsr(k, c, 1, 2) +
                                    0.25 * wei_kcsr(k, c, 2, 0) - 0.25 * wei_kcsr(k, c, 2, 1) +
                                    0.25 * wei_kcsr(k, c, 2, 2);
        wei_transform(k, c, 2, 3) =
            0.5 * wei_kcsr(k, c, 0, 2) - 0.5 * wei_kcsr(k, c, 1, 2) + 0.5 * wei_kcsr(k, c, 2, 2);

        wei_transform(k, c, 3, 0) = wei_kcsr(k, c, 2, 0);
        wei_transform(k, c, 3, 1) =
            0.5 * wei_kcsr(k, c, 2, 0) + 0.5 * wei_kcsr(k, c, 2, 1) + 0.5 * wei_kcsr(k, c, 2, 2);
        wei_transform(k, c, 3, 2) =
            0.5 * wei_kcsr(k, c, 2, 0) - 0.5 * wei_kcsr(k, c, 2, 1) + 0.5 * wei_kcsr(k, c, 2, 2);
        wei_transform(k, c, 3, 3) = wei_kcsr(k, c, 2, 2);
    };

    auto f_out_transform = [&](auto n, auto k, auto y, auto x) {
        for(int j = 0; j < InTileSizeH; ++j)
        {
            for(int i = 0; i < InTileSizeW; ++i)
            {
                double v = 0;
                for(int c = 0; c < C; ++c)
                {
                    v += in_transform(n, c, y, x, j, i) * wei_transform(k, c, j, i);
                }

                out_transform(n, k, y, x, j, i) = v;
            }
        }
    };

    auto f_out_hold = [&](auto n, auto k, auto y, auto x) {
        out_hold(n, k, y, x, 0, 0) =
            out_transform(n, k, y, x, 0, 0) + out_transform(n, k, y, x, 0, 1) +
            out_transform(n, k, y, x, 0, 2) + out_transform(n, k, y, x, 1, 0) +
            out_transform(n, k, y, x, 1, 1) + out_transform(n, k, y, x, 1, 2) +
            out_transform(n, k, y, x, 2, 0) + out_transform(n, k, y, x, 2, 1) +
            out_transform(n, k, y, x, 2, 2);
        out_hold(n, k, y, x, 0, 1) =
            out_transform(n, k, y, x, 0, 1) - out_transform(n, k, y, x, 0, 2) -
            out_transform(n, k, y, x, 0, 3) + out_transform(n, k, y, x, 1, 1) -
            out_transform(n, k, y, x, 1, 2) - out_transform(n, k, y, x, 1, 3) +
            out_transform(n, k, y, x, 2, 1) - out_transform(n, k, y, x, 2, 2) -
            out_transform(n, k, y, x, 2, 3);
        out_hold(n, k, y, x, 1, 0) =
            out_transform(n, k, y, x, 1, 0) + out_transform(n, k, y, x, 1, 1) +
            out_transform(n, k, y, x, 1, 2) - out_transform(n, k, y, x, 2, 0) -
            out_transform(n, k, y, x, 2, 1) - out_transform(n, k, y, x, 2, 2) -
            out_transform(n, k, y, x, 3, 0) - out_transform(n, k, y, x, 3, 1) -
            out_transform(n, k, y, x, 3, 2);
        out_hold(n, k, y, x, 1, 1) =
            out_transform(n, k, y, x, 1, 1) - out_transform(n, k, y, x, 1, 2) -
            out_transform(n, k, y, x, 1, 3) - out_transform(n, k, y, x, 2, 1) +
            out_transform(n, k, y, x, 2, 2) + out_transform(n, k, y, x, 2, 3) -
            out_transform(n, k, y, x, 3, 1) + out_transform(n, k, y, x, 3, 2) +
            out_transform(n, k, y, x, 3, 3);
    };

    auto f_out = [&](auto n, auto k, auto y, auto x) {
        for(int j = 0; j < OutTileSizeH; ++j)
        {
            std::size_t ho = OutTileSizeH * y + j;
            for(int i = 0; i < OutTileSizeW; ++i)
            {
                std::size_t wo    = OutTileSizeW * x + i;
                out(n, k, ho, wo) = out_hold(n, k, y, x, j, i);
            }
        }
    };

    std::size_t num_thread = std::thread::hardware_concurrency();

    make_ParallelTensorFunctor(f_in_hold, N, C, Y, X)(num_thread);
    make_ParallelTensorFunctor(f_in_transform, N, C, Y, X)(num_thread);
    make_ParallelTensorFunctor(f_wei_transform, K, C)(num_thread);
    make_ParallelTensorFunctor(f_out_transform, N, K, Y, X)(num_thread);
    make_ParallelTensorFunctor(f_out_hold, N, K, Y, X)(num_thread);
    make_ParallelTensorFunctor(f_out, N, K, Y, X)(num_thread);
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

int main()
{
#if 0
    constexpr unsigned N  = 1;
    constexpr unsigned C  = 1;
    constexpr unsigned HI = 4;
    constexpr unsigned WI = 4;
    constexpr unsigned K  = 1;
    constexpr unsigned S  = 3;
    constexpr unsigned R  = 3;
#elif 0
    constexpr unsigned N = 1;
    constexpr unsigned C = 1;
    constexpr unsigned HI = 34;
    constexpr unsigned WI = 34;
    constexpr unsigned K = 1;
    constexpr unsigned S = 3;
    constexpr unsigned R = 3;
#elif 1
    constexpr unsigned N = 64;
    constexpr unsigned C = 256;
    constexpr unsigned HI = 34;
    constexpr unsigned WI = 34;
    constexpr unsigned K = 64;
    constexpr unsigned S = 3;
    constexpr unsigned R = 3;
#elif 0
    constexpr unsigned N = 64;
    constexpr unsigned C = 64;
    constexpr unsigned HI = 56;
    constexpr unsigned WI = 56;
    constexpr unsigned K = 64;
    constexpr unsigned S = 3;
    constexpr unsigned R = 3;
#elif 0
    constexpr unsigned N  = 64;
    constexpr unsigned C  = 64;
    constexpr unsigned HI = 66;
    constexpr unsigned WI = 66;
    constexpr unsigned K  = 64;
    constexpr unsigned S  = 3;
    constexpr unsigned R  = 3;
#endif

    auto in_nchw_desc  = make_ConstantTensorDescriptor(Sequence<N, C, HI, WI>{});
    auto wei_kcsr_desc = make_ConstantTensorDescriptor(Sequence<K, C, S, R>{});
    auto wei_srck_desc = make_ConstantTensorDescriptor(Sequence<S, R, C, K>{});
    auto out_nkhw_desc =
        get_convolution_output_default_4d_tensor_descriptor(in_nchw_desc, wei_kcsr_desc);

    ostream_ConstantTensorDescriptor(in_nchw_desc, std::cout << "in_nchw_desc: ");
    ostream_ConstantTensorDescriptor(wei_kcsr_desc, std::cout << "wei_kcsr_desc: ");
    ostream_ConstantTensorDescriptor(wei_srck_desc, std::cout << "wei_srck_desc: ");
    ostream_ConstantTensorDescriptor(out_nkhw_desc, std::cout << "out_nkhw_desc: ");

    Tensor<float> in_nchw(make_TensorDescriptor(in_nchw_desc));
    Tensor<float> wei_kcsr(make_TensorDescriptor(wei_kcsr_desc));
    Tensor<float> wei_srck(make_TensorDescriptor(wei_srck_desc));
    Tensor<float> out_nkhw_host(make_TensorDescriptor(out_nkhw_desc));
    Tensor<float> out_nkhw_device(make_TensorDescriptor(out_nkhw_desc));

#if 0
    std::size_t num_thread = std::thread::hardware_concurrency();
    in_nchw.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
    wei_kcsr.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
    wei_srck.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
#elif 1
    std::size_t num_thread = std::thread::hardware_concurrency();
    in_nchw.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
    wei_kcsr.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
    wei_srck.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
#endif

    for(int i = 0; i < 40; ++i)
    {
#if 0
        device_direct_convolution_1(in_nchw_desc, in_nchw, wei_kcsr_desc, wei_kcsr, out_nkhw_desc, out_nkhw_device);
#elif 0
        device_direct_convolution_2(
            in_nchw_desc, in_nchw, wei_kcsr_desc, wei_kcsr, out_nkhw_desc, out_nkhw_device);
#elif 0
        device_implicit_gemm_convolution(
            in_nchw_desc, in_nchw, wei_kcsr_desc, wei_kcsr, out_nkhw_desc, out_nkhw_device);
#elif 1
        device_implicit_gemm_convolution(
            in_nchw_desc, in_nchw, wei_srck_desc, wei_srck, out_nkhw_desc, out_nkhw_device);
#elif 0
        device_winograd_convolution(
            in_nchw_desc, in_nchw, wei_kcsr_desc, wei_kcsr, out_nkhw_desc, out_nkhw_device);
#endif
    }

#if 1
    host_winograd_3x3_convolution(in_nchw, wei_kcsr, out_nkhw_host);
    check_error(out_nkhw_host, out_nkhw_device);
#elif 0
    host_direct_convolution(in_nchw, wei_kcsr, out_nkhw_host);
    check_error(out_nkhw_host, out_nkhw_device);
#endif

#if 0
    LogRange(std::cout << "in_nchw : ", in_nchw.mData, ",") << std::endl;
    LogRange(std::cout << "wei_kcsr: ", wei_kcsr.mData, ",") << std::endl;
    LogRange(std::cout << "out_nkhw_host  : ", out_nkhw_host.mData, ",") << std::endl;
    LogRange(std::cout << "out_nkhw_device: ", out_nkhw_device.mData, ",") << std::endl;
#endif
}
