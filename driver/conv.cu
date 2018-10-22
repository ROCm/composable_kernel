#include <iostream>
#include "nvToolsExt.h"
#include "tensor.hpp"
#include "device_tensor.cuh"
#include "direct_convolution.cuh"

template <typename T>
void host_convolution(const Tensor<T>& in,
                      const Tensor<T>& wei,
                      Tensor<T>& out,
                      std::size_t num_thread)
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

    f_par(num_thread);
}

template <class T>
void device_convolution(const Tensor<T>& in, const Tensor<T>& wei, Tensor<T>& out)

{
    DeviceTensorDescriptor in_desc_device(in.mDesc);
    DeviceTensorDescriptor wei_desc_device(wei.mDesc);
    DeviceTensorDescriptor out_desc_device(out.mDesc);

    std::size_t data_sz = sizeof(T);
    DeviceMem in_device_buf(data_sz * in.mDesc.GetElementSpace());
    DeviceMem wei_device_buf(data_sz * wei.mDesc.GetElementSpace());
    DeviceMem out_device_buf(data_sz * out.mDesc.GetElementSpace());

    in_device_buf.ToDevice(in.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());

    dim3 block_dim(256, 1, 1);
    dim3 grid_dim(1, 1, 1);
    direct_convolution<T, 256>
        <<<grid_dim, block_dim>>>(in_desc_device,
                                  static_cast<T*>(in_device_buf.GetDeviceBuffer()),
                                  wei_desc_device,
                                  static_cast<T*>(wei_device_buf.GetDeviceBuffer()),
                                  out_desc_device,
                                  static_cast<T*>(out_device_buf.GetDeviceBuffer()));

    out_device_buf.FromDevice(out.mData.data());
}

template <class T>
struct Generator
{

    template <class... Is>
    T operator()(Is... is)
    {
        return 1;
    }
};

int main()
{
#if 0
    Tensor<float> in({3, 16, 128, 128});
    Tensor<float> wei({4, 16, 3, 3});
    Tensor<float> out_host({3, 4, 126, 126});
#else
    Tensor<float> in({1, 1, 4, 4});
    Tensor<float> wei({1, 1, 3, 3});
    Tensor<float> out_host({1, 1, 2, 2});
#endif
    Tensor<float> out_device = out_host;

    int num_thread = std::thread::hardware_concurrency();

    std::cout << __func__ << ": num_thread " << num_thread << std::endl;

    in.GenerateTensorValue(Generator<float>{}, num_thread);
    wei.GenerateTensorValue(Generator<float>{}, num_thread);

    host_convolution(in, wei, out_host, num_thread);
    device_convolution(in, wei, out_device);

    std::cout << __func__ << ": done" << std::endl;

    LogRange(std::cout, in.mData, ",") << std::endl;
    LogRange(std::cout, wei.mData, ",") << std::endl;
    LogRange(std::cout, out_host.mData, ",") << std::endl;
    LogRange(std::cout, out_device.mData, ",") << std::endl;
}
