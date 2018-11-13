#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include "nvToolsExt.h"
#include "tensor.hpp"
#include "constant_tensor_descriptor.cuh"
#include "direct_convolution_2.cuh"

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
#if 1
        return std::rand() / RAND_MAX;
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

template <class T, class InDesc, class WeiDesc, class OutDesc>
void device_convolution(
    InDesc, const Tensor<T>& in, WeiDesc, const Tensor<T>& wei, OutDesc, Tensor<T>& out)
{
    std::size_t data_sz = sizeof(T);
    DeviceMem in_device_buf(data_sz * in.mDesc.GetElementSpace());
    DeviceMem wei_device_buf(data_sz * wei.mDesc.GetElementSpace());
    DeviceMem out_device_buf(data_sz * out.mDesc.GetElementSpace());

    int num_thread = std::thread::hardware_concurrency();

    in_device_buf.ToDevice(in.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());
    out_device_buf.ToDevice(out.mData.data());

    constexpr auto I0 = Index<0>{};
    constexpr auto I1 = Index<1>{};
    constexpr auto I2 = Index<2>{};
    constexpr auto I3 = Index<3>{};

    constexpr auto in_desc           = InDesc{};
    constexpr auto wei_desc          = WeiDesc{};
    constexpr auto out_desc          = OutDesc{};
    constexpr unsigned NPerBlock     = 1;
    constexpr unsigned KPerBlock     = 2;
    constexpr unsigned CPerBlockLoop = 4;
    constexpr unsigned OutTileSizeH  = 2;
    constexpr unsigned OutTileSizeW  = 2;
    constexpr unsigned YPerBlock     = 8;
    constexpr unsigned XPerBlock     = 16;

    constexpr unsigned NBlockCopyLen0 = 1;
    constexpr unsigned NBlockCopyLen1 = 1;
    constexpr unsigned NBlockCopyLen2 = 2;
    constexpr unsigned NBlockCopyLen3 = 16;

    constexpr unsigned BlockSize = 128;

    constexpr unsigned GridSize = (out_desc.GetLength(I0) / NPerBlock) *
                                  (out_desc.GetLength(I1) / KPerBlock) *
                                  (out_desc.GetLength(I2) / (OutTileSizeH * YPerBlock)) *
                                  (out_desc.GetLength(I3) / (OutTileSizeW * XPerBlock));

    dim3 block_dim(BlockSize);
    dim3 grid_dim(GridSize);

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventRecord(start, 0);

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
                         NBlockCopyLen3,
                         BlockSize,
                         GridSize>
        <<<grid_dim, block_dim>>>(InDesc{},
                                  static_cast<T*>(in_device_buf.GetDeviceBuffer()),
                                  WeiDesc{},
                                  static_cast<T*>(wei_device_buf.GetDeviceBuffer()),
                                  OutDesc{},
                                  static_cast<T*>(out_device_buf.GetDeviceBuffer()));

    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed time : %f ms\n", elapsedTime);

    checkCudaErrors(cudaGetLastError());
    out_device_buf.FromDevice(out.mData.data());
}

int main()
{
#if 0
    constexpr unsigned N  = 1;
    constexpr unsigned C  = 1;
    constexpr unsigned HI = 18;
    constexpr unsigned WI = 18;
    constexpr unsigned K  = 1;
    constexpr unsigned S  = 3;
    constexpr unsigned R  = 3;
#elif 1
    constexpr unsigned N = 64;
    constexpr unsigned C = 256;
    constexpr unsigned HI = 34;
    constexpr unsigned WI = 34;
    constexpr unsigned K = 64;
    constexpr unsigned S = 3;
    constexpr unsigned R = 3;
#elif 0
    constexpr unsigned N = 2;
    constexpr unsigned C = 3;
    constexpr unsigned HI = 130;
    constexpr unsigned WI = 130;
    constexpr unsigned K = 5;
    constexpr unsigned S = 3;
    constexpr unsigned R = 3;
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

    int num_thread = std::thread::hardware_concurrency();

#if 0
    in.GenerateTensorValue(GeneratorTensor<float>{}, num_thread);
    wei.GenerateTensorValue(GeneratorTensor<float>{}, num_thread);
    out_host.GenerateTensorValue(GeneratorConstant<float>{0}, num_thread);
#endif

    Tensor<float> out_device = out_host;

    device_convolution(in_desc, in, wei_desc, wei, out_desc, out_device);

#if 0
    host_convolution(in, wei, out_host);

    float error      = 0;
    float max_diff   = 0;
    float host_value = 0, device_value = 0;
    for(int i = 0; i < out_host.mData.size(); ++i)
    {
        error += std::abs(out_host.mData[i] - out_device.mData[i]);
        float diff = std::abs(out_host.mData[i] - out_device.mData[i]);
        if(max_diff < diff)
        {
            max_diff     = diff;
            host_value   = out_host.mData[i];
            device_value = out_device.mData[i];
        }
    }
    std::cout << "error: " << error << std::endl;
    std::cout << "max_diff: " << max_diff << ", " << host_value << ", " << device_value
              << std::endl;
#endif

#if 0
    LogRange(std::cout << __func__ << "in : ", in.mData, ",") << std::endl;
    LogRange(std::cout << __func__ << "wei: ", wei.mData, ",") << std::endl;
    LogRange(std::cout, out_host.mData, ",") << std::endl;
    LogRange(std::cout, out_device.mData, ",") << std::endl;
#endif
}
