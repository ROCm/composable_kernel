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
#include "host_gemm.hpp"
#include "device_tensor.hpp"
#include "device_base.hpp"
#include "example/3_gemm_xdl_bias_relu_add/include/device_gemm_xdl_two_extra_source_reduce.hpp"

// C[m, n] = Relu(A[m, k] * B[k, n] + C0[m]) + C1[m, n]
// assume C0 is contiguous in memory
//     C0 resides in memory as 1d vector [m], but is represented as 2D matrix [m, n], with stride =
//     0 in the "n" dimension
// assume C1 and C have same layout C

struct BiasReluAdd
{
    template <typename T1, typename T2>
    __host__ constexpr float operator()(float v0, T1 v1, T2 v2) const
    {
        float b = v0 + v1;
        float c = b > 0 ? b : 0;
        float d = c + v2;

        return d;
    }

    template <typename T1, typename T2>
    __device__ constexpr float operator()(float v0, T1 v1, T2 v2) const
    {
#if 0
        float a = v1 + v0;
        float b = max(a, float(0));
        float c = b + v2;

        return c;
#else
        float a = v1 + v2;
        float b = v2;

        float c = (v0 > -v1) ? a + v0 : v2;

        return c;
#endif
    }
};

// v0 is from A * B
// v1 is from C0
// v2 is from C1
struct BiasLeakyReluAdd
{
    template <typename T1, typename T2>
    __host__ constexpr float operator()(float v0, T1 v1, T2 v2) const
    {
        float a = v0 + v1;
        float b = 0.1 * a;
        float c = b > 0 ? b : 0;
        float d = c + v2;

        return d;
    }

    template <typename T1, typename T2>
    __device__ constexpr float operator()(float v0, T1 v1, T2 v2) const
    {
        constexpr float alpha     = 0.1;
        constexpr float alpha_inv = 1.0 / alpha;

        float a = v2 * alpha_inv;
        float b = v1 + v0;
        float c = max(b, float(0));
        float d = alpha * (a + c);

        return d;
    }
};

struct BiasLeakyRelu
{
    template <typename T1, typename T2>
    __host__ constexpr float operator()(float v0, T1 v1, T2) const
    {
        float a = v0 + v1;
        float b = 0.1 * a;
        float c = b > 0 ? b : 0;

        return c;
    }

    template <typename T1, typename T2>
    __device__ constexpr float operator()(float v0, T1 v1, T2) const
    {
        constexpr float alpha = 0.1;

        float b = v1 + v0;
        float c = max(b, float(0));
        float d = alpha * c;

        return d;
    }
};

struct BiasAdd
{
#if 1
    // correct result
    // no scratch memory, good VGPR allocation (59)
    // good perf (101Tflops)
    template <typename T1, typename T2>
    __host__ __device__ constexpr float operator()(float v0, T1 v1, T2 v2) const
    {
        constexpr float alpha = 0.1;
        constexpr float beta  = 0.2;
        constexpr float gamma = 0.3;

        // compiler seems very volatile to the order of these calculation:
        // compiler is very eager to read AccVgpr (v0) out prematurely, resulting in register
        // over-allocation. Therefore, move v0 calculation to the very end
        float a = T1(beta) * v1 + T2(gamma) * v2;
        float b = a + float(alpha) * v0;

        return b;
    }
#elif 0
    float alpha = 0.1;
    float beta = 0.2;
    float gamma = 0.3;

    // wrong result
    // lots of scratch memory
    // huge perf drop
    template <typename T1, typename T2>
    __host__ __device__ constexpr float operator()(float v0, T1 v1, T2 v2) const
    {
        return alpha * v0 + beta * v1 + gamma * v2;
    }
#elif 0
    // correct result
    // some scratch memory (68 dword)
    // some perf drop (94Tflops)
    // fp64 instructions are used
    __host__ __device__ constexpr auto operator()(float v0, ck::half_t v1, ck::half_t v2) const
    {
        return 0.1 * v0 + 0.2 * v1 + 0.3 * v2;
    }
#elif 1
    // wrong result
    // lots of scratch memory
    // huge perf drop
    __host__ __device__ constexpr auto operator()(float v0, ck::half_t v1, ck::half_t v2) const
    {
        return float(0.1) * v0 + float(0.2) * v1 + float(0.3) * v2;
    }
#endif
};

struct PassThrough
{
    template <typename T>
    __host__ __device__ constexpr T operator()(T v) const
    {
        return v;
    }
};

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using ADataType   = ck::half_t;
using BDataType   = ck::half_t;
using CDataType   = ck::half_t;
using AccDataType = float;

using ALayout = ck::tensor_layout::gemm::RowMajor;
using BLayout = ck::tensor_layout::gemm::ColumnMajor;
using CLayout = ck::tensor_layout::gemm::RowMajor;

using AOp = PassThrough;
using BOp = PassThrough;
using COp = BiasReluAdd;

// Compilation parameters for NT problem
// clang-format off
using DeviceGemmInstance =
    //#################################################################|     AData|     BData|     CData|     AccData| ALayout| BLayout| CLayout| AElementwise| BElementwise| CElementwise| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer|  BBlockTransfer|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| CThreadTransfer| CThreadTransfer| ABlockLds| BBlockLds|
    //#################################################################|      Type|      Type|      Type|        Type|        |        |        |    Operation|    Operation|    Operation|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|     ThreadSlice|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|     ThreadSlice|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| SrcDstVectorDim|       DstScalar| AddExtraM| AddExtraN|
    //#################################################################|          |          |          |            |        |        |        |             |             |             |      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_N_K1| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1| Lengths_K0_N_K1| Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|                |       PerVector|          |          |
    //#################################################################|          |          |          |            |        |        |        |             |             |             |      |      |      |      |   |     |     |     |     |                |                |               |               |               |               |               |                |                |               |               |              |               |               |                |                |          |          |
    ck::tensor_operation::device::DeviceGemmXdl_two_extra_source_reduce< ADataType, BDataType, CDataType, AccDataType, ALayout, BLayout, CLayout,          AOp,          BOp,          COp,   256,   256,   128,     4,  8,   32,   32,    4,    2,      S<1, 4, 8>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      S<1, 2, 8>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,               7,               1,      true,      true>;
// clang-format on

template <typename AType,
          typename BType,
          typename CType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
static void host_verify(const Tensor<AType>& a_m_k,
                        const Tensor<BType>& b_k_n,
                        Tensor<CType>& c_m_n,
                        const Tensor<CType>& c0_m_n,
                        const Tensor<CType>& c1_m_n,
                        const AElementwiseOperation& a_element_op,
                        const BElementwiseOperation& b_element_op,
                        const CElementwiseOperation& c_element_op)
{
    auto f_mk_kn_mn = [&](auto m, auto n) {
        const int K = a_m_k.mDesc.GetLengths()[1];

        double v = 0;

        for(int k = 0; k < K; ++k)
        {
            v += static_cast<const double>(a_element_op(a_m_k(m, k))) *
                 static_cast<const double>(b_element_op(b_k_n(k, n)));
        }

        c_m_n(m, n) = c_element_op(
            v, static_cast<const double>(c0_m_n(m, n)), static_cast<const double>(c1_m_n(m, n)));
    };

    make_ParallelTensorFunctor(f_mk_kn_mn,
                               c_m_n.mDesc.GetLengths()[0],
                               c_m_n.mDesc.GetLengths()[1])(std::thread::hardware_concurrency());
}

int main(int argc, char* argv[])
{
    bool do_verification = 0;
    int init_method      = 0;
    int nrepeat          = 5;

    // GEMM shape
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA = 4096;
    ck::index_t StrideB = 4096;
    ck::index_t StrideC = 4096;

    if(argc == 4)
    {
        M = std::stoi(argv[4]);
        N = std::stoi(argv[5]);
        K = std::stoi(argv[6]);
    }
    else if(argc == 10)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        nrepeat         = std::stoi(argv[3]);

        M = std::stoi(argv[4]);
        N = std::stoi(argv[5]);
        K = std::stoi(argv[6]);

        StrideA = std::stoi(argv[7]);
        StrideB = std::stoi(argv[8]);
        StrideC = std::stoi(argv[9]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: run kernel # of times (>1)\n");
        printf("arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC\n");
        exit(0);
    }

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                            std::vector<std::size_t>({stride, 1}));
            }
            else
            {
                return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                            std::vector<std::size_t>({1, stride}));
            }
        };

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
    Tensor<BDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<BDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    // C0[m]
    Tensor<CDataType> c1_m_n(HostTensorDescriptor(
        std::vector<std::size_t>({static_cast<std::size_t>(M), static_cast<std::size_t>(N)}),
        std::vector<std::size_t>({1, 0})));

    // C1[m ,n]
    Tensor<BDataType> c0_m_n(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;
    std::cout << "c0_m_n: " << c0_m_n.mDesc << std::endl;
    std::cout << "c1_m_n: " << c1_m_n.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        c0_m_n.GenerateTensorValue(GeneratorTensor_2<CDataType>{-5, 5});
        c1_m_n.GenerateTensorValue(GeneratorTensor_2<CDataType>{-5, 5});
        break;
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        c0_m_n.GenerateTensorValue(GeneratorTensor_3<CDataType>{0.0, 1.0});
        c1_m_n.GenerateTensorValue(GeneratorTensor_3<CDataType>{0.0, 1.0});
    }

    DeviceMem a_m_k_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpace());
    DeviceMem b_k_n_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpace());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpace());
    DeviceMem c0_m_n_device_buf(sizeof(CDataType) * c0_m_n.mDesc.GetElementSpace());
    DeviceMem c1_m_n_device_buf(sizeof(CDataType) * c1_m_n.mDesc.GetElementSpace());

    a_m_k_device_buf.ToDevice(a_m_k.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n.mData.data());
    c_m_n_device_buf.ToDevice(c_m_n_device_result.mData.data());
    c0_m_n_device_buf.ToDevice(c0_m_n.mData.data());
    c1_m_n_device_buf.ToDevice(c1_m_n.mData.data());

    auto c_element_op = BiasReluAdd{};

    // do GEMM
    auto gemm = DeviceGemmInstance{};

    auto invoker  = gemm.MakeInvoker();
    auto argument = gemm.MakeArgument(static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                                      static_cast<BDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
                                      static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
                                      static_cast<CDataType*>(c0_m_n_device_buf.GetDeviceBuffer()),
                                      static_cast<CDataType*>(c1_m_n_device_buf.GetDeviceBuffer()),
                                      M,
                                      N,
                                      K,
                                      StrideA,
                                      StrideB,
                                      StrideC,
                                      PassThrough{},
                                      PassThrough{},
                                      c_element_op);

    if(!gemm.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    float ave_time = invoker.Run(argument, nrepeat);

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype =
        sizeof(ADataType) * M * K + sizeof(BDataType) * K * M + sizeof(CDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    c_m_n_device_buf.FromDevice(c_m_n_device_result.mData.data());

    if(do_verification)
    {
        host_verify(a_m_k,
                    b_k_n,
                    c_m_n_host_result,
                    c0_m_n,
                    c1_m_n,
                    PassThrough{},
                    PassThrough{},
                    c_element_op);

        check_error(c_m_n_host_result, c_m_n_device_result);
    }
}
