#include <iostream>
#include <numeric>
#include <initializer_list>
#include "nvToolsExt.h"
#include "tensor.hpp"
#include "constant_tensor_descriptor.cuh"
#include "device_tensor_descriptor.cuh"

#if 0
#include "direct_convolution.cuh"
#else
#include "constant_direct_convolution.cuh"
#endif

template <class T>
struct GeneratorConstant
{
    T value = 0;

    template <class... Is>
    T operator()(Is... is)
    {
        return value;
    }
};

template <class T>
struct GeneratorTensor
{
    template <class... Is>
    T operator()(Is... is)
    {
#if 0
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

// this is ugly, only for 4d
template <class TConstTensorDesc>
void ostream_ConstantTensorDescriptor(TConstTensorDesc, std::ostream& os = std::cout)
{
    static_assert(TConstTensorDesc::nDim == 4, "nDim is not 4");

    constexpr auto I0   = Index<0>{};
    constexpr auto I1   = Index<1>{};
    constexpr auto I2   = Index<2>{};
    constexpr auto I3   = Index<3>{};
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

    constexpr auto I0   = Index<0>{};
    constexpr auto I1   = Index<1>{};
    constexpr auto I2   = Index<2>{};
    constexpr auto I3   = Index<3>{};
    constexpr auto desc = TConstTensorDesc{};

    std::initializer_list<unsigned> lengths = {
        desc.GetLength(I0), desc.GetLength(I1), desc.GetLength(I2), desc.GetLength(I3)};
    std::initializer_list<unsigned> strides = {
        desc.GetStride(I0), desc.GetStride(I1), desc.GetStride(I2), desc.GetStride(I3)};

    return TensorDescriptor(lengths, strides);
}

template <class T>
void host_convolution(const Tensor<T>& in, const Tensor<T>& wei, Tensor<T>& out)
{
    auto f = [&](auto n, auto k, auto ho, auto wo) {
        double v = 0;
        for(int c = 0; c < wei.mDesc.GetLengths()[1]; ++c)
        {
            for(int y = 0; y < wei.mDesc.GetLengths()[2]; ++y)
            {
                int hi = ho + y;
                for(int x = 0; x < wei.mDesc.GetLengths()[3]; ++x)
                {
                    int wi = wo + x;
                    v += in(n, c, hi, wi) * wei(k, c, y, x);
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

#if 0
template <class T>
void device_convolution(
    const Tensor<T>& in, const Tensor<T>& wei, Tensor<T>& out)
{
    DeviceTensorDescriptor<4> in_desc_device(in.mDesc);
    DeviceTensorDescriptor<4> wei_desc_device(wei.mDesc);
    DeviceTensorDescriptor<4> out_desc_device(out.mDesc);

    printf("__func__: in_desc_device: {%u %u %u %u}, {%u %u %u %u}\n",
           in_desc_device.GetLength(0),
           in_desc_device.GetLength(1),
           in_desc_device.GetLength(2),
           in_desc_device.GetLength(3),
           in_desc_device.GetStride(0),
           in_desc_device.GetStride(1),
           in_desc_device.GetStride(2),
           in_desc_device.GetStride(3));

    std::size_t data_sz = sizeof(T);
    DeviceMem in_device_buf(data_sz * in.mDesc.GetElementSpace());
    DeviceMem wei_device_buf(data_sz * wei.mDesc.GetElementSpace());
    DeviceMem out_device_buf(data_sz * out.mDesc.GetElementSpace());

    int num_thread = std::thread::hardware_concurrency();

    out.GenerateTensorValue(GeneratorConstant<float>{0}, num_thread);

    in_device_buf.ToDevice(in.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());
    out_device_buf.ToDevice(out.mData.data());

    dim3 block_dim(64, 1, 1);
    dim3 grid_dim(1, 1, 1);

    gridwise_convolution<T, 3, 3, 4, 4, 2, 2, 1, 1, 8, 8, 1>
        <<<grid_dim, block_dim>>>(in_desc_device,
                                  static_cast<T*>(in_device_buf.GetDeviceBuffer()),
                                  wei_desc_device,
                                  static_cast<T*>(wei_device_buf.GetDeviceBuffer()),
                                  out_desc_device,
                                  static_cast<T*>(out_device_buf.GetDeviceBuffer()));

    checkCudaErrors(cudaGetLastError());
    out_device_buf.FromDevice(out.mData.data());
}
#else
template <class T, class InDesc, class WeiDesc, class OutDesc>
void const_device_convolution(
    InDesc, const Tensor<T>& in, WeiDesc, const Tensor<T>& wei, OutDesc, Tensor<T>& out)
{
    std::size_t data_sz = sizeof(T);
    DeviceMem in_device_buf(data_sz * in.mDesc.GetElementSpace());
    DeviceMem wei_device_buf(data_sz * wei.mDesc.GetElementSpace());
    DeviceMem out_device_buf(data_sz * out.mDesc.GetElementSpace());

    int num_thread = std::thread::hardware_concurrency();

    out.GenerateTensorValue(GeneratorConstant<float>{0}, num_thread);

    in_device_buf.ToDevice(in.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());
    out_device_buf.ToDevice(out.mData.data());

    dim3 block_dim(64, 1, 1);
    dim3 grid_dim(1, 1, 1);

    constexpr auto I0 = Index<0>{};
    constexpr auto I1 = Index<1>{};
    constexpr auto I2 = Index<2>{};
    constexpr auto I3 = Index<3>{};

    constexpr auto in_desc = InDesc{};
    constexpr auto wei_desc = WeiDesc{};
    constexpr auto out_desc = OutDesc{};
    constexpr unsigned NPerBlock = 1;
    constexpr unsigned KPerBlock = 1;
    constexpr unsigned CPerBlockLoop = 1;
    constexpr unsigned OutTileSizeH = 2;
    constexpr unsigned OutTileSizeW = 2;
    constexpr unsigned YPerBlock = (out_desc.GetLength(I2) + OutTileSizeH - 1) / OutTileSizeH;
    constexpr unsigned XPerBlock = (out_desc.GetLength(I3) + OutTileSizeW - 1) / OutTileSizeW;

    constexpr unsigned NBlockCopyLen0 = 1;
    constexpr unsigned NBlockCopyLen1 = 1;
    constexpr unsigned NBlockCopyLen2 = 1;
    constexpr unsigned NBlockCopyLen3 = 64;

    gridwise_convolution<T,
                         InDesc,
                         WeiDesc,
                         OutDesc,
                         NPerBlock,
                         KPerBlock,
                         CPerBlockLoop,
                         OutTileSizeH,
                         OutTileSizeW,
                         YPerBlock,
                         XPerBlock,
                         NBlockCopyLen0,
                         NBlockCopyLen1,
                         NBlockCopyLen2,
                         NBlockCopyLen3>
        <<<grid_dim, block_dim>>>(InDesc{},
                                  static_cast<T*>(in_device_buf.GetDeviceBuffer()),
                                  WeiDesc{},
                                  static_cast<T*>(wei_device_buf.GetDeviceBuffer()),
                                  OutDesc{},
                                  static_cast<T*>(out_device_buf.GetDeviceBuffer()));

    checkCudaErrors(cudaGetLastError());
    out_device_buf.FromDevice(out.mData.data());
}
#endif

int main()
{
#if 1
    constexpr unsigned N  = 1;
    constexpr unsigned C  = 1;
    constexpr unsigned HI = 18;
    constexpr unsigned WI = 18;
    constexpr unsigned K  = 1;
    constexpr unsigned S  = 3;
    constexpr unsigned R  = 3;
#elif 1
    constexpr unsigned N = 1;
    constexpr unsigned C = 1;
    constexpr unsigned HI = 36;
    constexpr unsigned WI = 36;
    constexpr unsigned K = 1;
    constexpr unsigned S = 3;
    constexpr unsigned R = 3;
#elif 0
    constexpr unsigned N  = 1;
    constexpr unsigned C  = 1;
    constexpr unsigned HI = 130;
    constexpr unsigned WI = 130;
    constexpr unsigned K  = 1;
    constexpr unsigned S  = 3;
    constexpr unsigned R  = 3;
#elif 0
    constexpr unsigned N  = 3;
    constexpr unsigned C  = 16;
    constexpr unsigned HI = 130;
    constexpr unsigned WI = 130;
    constexpr unsigned K  = 4;
    constexpr unsigned S  = 3;
    constexpr unsigned R  = 3;
#endif

    auto in_desc  = make_ConstantTensorDescriptor(Sequence<N, C, HI, WI>{});
    auto wei_desc = make_ConstantTensorDescriptor(Sequence<K, C, S, R>{});
    auto out_desc = get_output_4d_tensor_descriptor(in_desc, wei_desc);

    ostream_ConstantTensorDescriptor(in_desc, std::cout << "in_desc: ");
    ostream_ConstantTensorDescriptor(wei_desc, std::cout << "wei_desc: ");
    ostream_ConstantTensorDescriptor(out_desc, std::cout << "out_desc: ");

    Tensor<float> in(make_TensorDescriptor(in_desc));
    Tensor<float> wei(make_TensorDescriptor(wei_desc));
    Tensor<float> out_host(make_TensorDescriptor(out_desc));

    Tensor<float> out_device = out_host;

    int num_thread = std::thread::hardware_concurrency();

    in.GenerateTensorValue(GeneratorTensor<float>{}, num_thread);
    wei.GenerateTensorValue(GeneratorTensor<float>{}, num_thread);

    host_convolution(in, wei, out_host);

#if 0
    device_convolution(in, wei, out_device);
#else
    const_device_convolution(in_desc, in, wei_desc, wei, out_desc, out_device);
#endif

    std::cout << __func__ << ": done" << std::endl;

    LogRange(std::cout << __func__ << "in : ", in.mData, ",") << std::endl;
    LogRange(std::cout << __func__ << "wei: ", wei.mData, ",") << std::endl;
    LogRange(std::cout, out_host.mData, ",") << std::endl;
    LogRange(std::cout, out_device.mData, ",") << std::endl;

    float error = 0;
    for(int i = 0; i < out_host.mData.size(); ++i)
    {
        error += std::abs(out_host.mData[i] - out_device.mData[i]);
    }
    std::cout << "error: " << error << std::endl;
}
