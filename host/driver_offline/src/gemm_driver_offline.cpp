#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>
#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "gemm_common.hpp"
#include "host_gemm.hpp"
#include "device_tensor.hpp"
#include "device_gemm_xdlops_mk_nk_mn.hpp"

#define USE_GEMM_XDL_MK_NK_MN 1

enum GemmAlgo
{
    Xdl_MK_KN_MN, // 0
    Xdl_MK_NK_MN, // 1
};

int main(int argc, char* argv[])
{
    using namespace ck;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};

    // dynamic mode
    if(argc != 10)
    {
        printf("arg1 to 6: layout, algo, do_verification, init_method, do_log, nrepeat\n");
        printf("rest: M, N, K\n");
        exit(1);
    }

    const auto layout          = static_cast<GemmMatrixLayout>(std::stoi(argv[1]));
    const auto algo            = static_cast<GemmAlgo>(std::stoi(argv[2]));
    const bool do_verification = std::stoi(argv[3]);
    const int init_method      = std::stoi(argv[4]);
    const bool do_log          = std::stoi(argv[5]);
    const int nrepeat          = std::stoi(argv[6]);

    const index_t M = std::stoi(argv[7]);
    const index_t N = std::stoi(argv[8]);
    const index_t K = std::stoi(argv[9]);

#if 0
    using ab_data_t  = float;
    using acc_data_t = float;
    using c_data_t   = float;
#elif 1
    using ab_data_t  = half_t;
    using acc_data_t = float;
    using c_data_t   = half_t;
#elif 1
    using ab_data_t  = int8_t;
    using acc_data_t = int32_t;
    using c_data_t   = int8_t;
#endif

    std::vector<std::size_t> a_lengths_host(2), b_lengths_host(2), c_lengths_host(2);
    std::vector<std::size_t> a_strides_host(2), b_strides_host(2), c_strides_host(2);

    if(layout == GemmMatrixLayout::KM_KN_MN)
    {
        a_lengths_host[0] = static_cast<std::size_t>(K);
        a_lengths_host[1] = static_cast<std::size_t>(M);
        a_strides_host[0] = static_cast<std::size_t>(M);
        a_strides_host[1] = static_cast<std::size_t>(1);

        b_lengths_host[0] = static_cast<std::size_t>(K);
        b_lengths_host[1] = static_cast<std::size_t>(N);
        b_strides_host[0] = static_cast<std::size_t>(N);
        b_strides_host[1] = static_cast<std::size_t>(1);

        c_lengths_host[0] = static_cast<std::size_t>(M);
        c_lengths_host[1] = static_cast<std::size_t>(N);
        c_strides_host[0] = static_cast<std::size_t>(N);
        c_strides_host[1] = static_cast<std::size_t>(1);
    }
    else if(layout == GemmMatrixLayout::MK_NK_MN)
    {
        a_lengths_host[0] = static_cast<std::size_t>(M);
        a_lengths_host[1] = static_cast<std::size_t>(K);
        a_strides_host[0] = static_cast<std::size_t>(K);
        a_strides_host[1] = static_cast<std::size_t>(1);

        b_lengths_host[0] = static_cast<std::size_t>(N);
        b_lengths_host[1] = static_cast<std::size_t>(K);
        b_strides_host[0] = static_cast<std::size_t>(K);
        b_strides_host[1] = static_cast<std::size_t>(1);

        c_lengths_host[0] = static_cast<std::size_t>(M);
        c_lengths_host[1] = static_cast<std::size_t>(N);
        c_strides_host[0] = static_cast<std::size_t>(N);
        c_strides_host[1] = static_cast<std::size_t>(1);
    }
    else
    {
        std::runtime_error("wrong! not implemented");
    }

    Tensor<ab_data_t> a(a_lengths_host, a_strides_host);
    Tensor<ab_data_t> b(b_lengths_host, b_strides_host);
    Tensor<c_data_t> c_host(c_lengths_host, c_strides_host);
    Tensor<c_data_t> c_device(c_lengths_host, c_strides_host);

    std::cout << "layout: " << layout << std::endl;
    ostream_HostTensorDescriptor(a.mDesc, std::cout << "a: ");
    ostream_HostTensorDescriptor(b.mDesc, std::cout << "b: ");
    ostream_HostTensorDescriptor(c_host.mDesc, std::cout << "c: ");

    std::size_t num_thread = std::thread::hardware_concurrency();

    switch(init_method)
    {
    case 0:
        // no initialization
        break;
    case 1:
        a.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
        b.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
        break;
    case 2:
        a.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
        b.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
        break;
    case 3:
        a.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
        b.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
        break;
    case 4:
        a.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
        b.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
        break;
    default:
        a.GenerateTensorValue(GeneratorTensor_3<float>{0.0, 1.0}, num_thread);
        b.GenerateTensorValue(GeneratorTensor_3<float>{-0.5, 0.5}, num_thread);
    }

    auto f_make_for_device_mk_nk_mn = [&]() {
        const auto a_desc = make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(K, I1));
        const auto b_desc = make_naive_tensor_descriptor(make_tuple(N, K), make_tuple(K, I1));
        const auto c_desc = make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(N, I1));

        return make_tuple(a_desc, b_desc, c_desc);
    };

#if USE_GEMM_XDL_MK_NK_MN
    if(algo == GemmAlgo::Xdl_MK_NK_MN)
    {
        if(layout != GemmMatrixLayout::MK_NK_MN)
        {
            throw std::runtime_error("wrong! layout");
        }

        const auto descs = f_make_for_device_mk_nk_mn();

        device_gemm_xdlops_mk_nk_mn<ab_data_t, acc_data_t, c_data_t>(
            descs[I0], descs[I1], descs[I2], a, b, c_device, nrepeat);
    }
#endif

    if(do_verification)
    {
        host_gemm(a, b, c_host, layout);

        check_error(c_host, c_device);

        if(do_log)
        {
            LogRangeAsType<float>(std::cout << "a : ", a.mData, ",") << std::endl;
            LogRangeAsType<float>(std::cout << "b: ", b.mData, ",") << std::endl;
            LogRangeAsType<float>(std::cout << "c_host  : ", c_host.mData, ",") << std::endl;
            LogRangeAsType<float>(std::cout << "c_device: ", c_device.mData, ",") << std::endl;
        }
    }
}
