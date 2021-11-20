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
#include "device_gemm_xdl.hpp"

struct PassThrough
{
    template <typename T>
    __host__ __device__ constexpr T operator()(T v) const
    {
        return v;
    }
};

struct Relu
{
    float alpha = 0.1;

    // ReLU
    template <typename T>
    __host__ __device__ constexpr T operator()(T v) const
    {
        T tmp = alpha * v;
        return tmp > 0 ? tmp : 0;
    }
};

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct DeviceGemmInstance;

template <typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct DeviceGemmInstance<ck::half_t,
                          ck::half_t,
                          ck::half_t,
                          ck::tensor_layout::gemm::RowMajor,
                          ck::tensor_layout::gemm::ColumnMajor,
                          ck::tensor_layout::gemm::RowMajor,
                          AElementwiseOperation,
                          BElementwiseOperation,
                          CElementwiseOperation>
{
    using F16 = ck::half_t;
    using F32 = float;

    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    template <ck::index_t... Is>
    using S = ck::Sequence<Is...>;

    using AOp = AElementwiseOperation;
    using BOp = BElementwiseOperation;
    using COp = CElementwiseOperation;

    // Compilation parameters for NT problem
    // clang-format off
    using type =
        //########################################| AData| BData| CData| AccData| ALayout| BLayout| CLayout| AElementwise| BElementwise| CElementwise| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer|  BBlockTransfer|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| CThreadTransfer| CThreadTransfer| ABlockLds| BBlockLds|
        //########################################|  Type|  Type|  Type|    Type|        |        |        |    Operation|    Operation|    Operation|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|     ThreadSlice|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|     ThreadSlice|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| SrcDstVectorDim|       DstScalar| AddExtraM| AddExtraN|
        //########################################|      |      |      |        |        |        |        |             |             |             |      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_N_K1| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1| Lengths_K0_N_K1| Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|                |       PerVector|          |          |
        //########################################|      |      |      |        |        |        |        |             |             |             |      |      |      |      |   |     |     |     |     |                |                |               |               |               |               |               |                |                |               |               |              |               |               |                |                |          |          |
        ck::tensor_operation::device::DeviceGemmXdl<  F16,   F16,   F16,     F32,     Row,     Col,     Row,          AOp,          BOp,          COp,   256,   256,   128,     4,  8,   32,   32,    4,    2,      S<1, 4, 8>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      S<1, 2, 8>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,               7,               1,      true,      true>;
    // clang-format on
};

template <typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct DeviceGemmInstance<float,
                          float,
                          float,
                          ck::tensor_layout::gemm::RowMajor,
                          ck::tensor_layout::gemm::ColumnMajor,
                          ck::tensor_layout::gemm::RowMajor,
                          AElementwiseOperation,
                          BElementwiseOperation,
                          CElementwiseOperation>
{
    using F16 = ck::half_t;
    using F32 = float;

    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    template <ck::index_t... Is>
    using S = ck::Sequence<Is...>;

    using AOp = AElementwiseOperation;
    using BOp = BElementwiseOperation;
    using COp = CElementwiseOperation;

    // Compilation parameters for NT problem
    // clang-format off
    using type =
    //########################################| AData| BData| CData| AccData| ALayout| BLayout| CLayout| AElementwise| BElementwise| CElementwise| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer|  BBlockTransfer|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| CThreadTransfer| CThreadTransfer| ABlockLds| BBlockLds|
    //########################################|  Type|  Type|  Type|    Type|        |        |        |    Operation|    Operation|    Operation|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|     ThreadSlice|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|     ThreadSlice|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| SrcDstVectorDim|       DstScalar| AddExtraM| AddExtraN|
    //########################################|      |      |      |        |        |        |        |             |             |             |      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_N_K1| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1| Lengths_K0_N_K1| Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|                |       PerVector|          |          |
    //########################################|      |      |      |        |        |        |        |             |             |             |      |      |      |      |   |     |     |     |     |                |                |               |               |               |               |               |                |                |               |               |              |               |               |                |                |          |          |
    ck::tensor_operation::device::DeviceGemmXdl<  F32,   F32,   F32,     F32,     Row,     Col,     Row,          AOp,          BOp,          COp,   256,   256,   128,     4,  4,   32,   32,    4,    2,      S<1, 4, 4>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      S<1, 2, 4>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,               7,               1,      true,      true>;
    // clang-format on
};

int main(int argc, char* argv[])
{
    if(argc != 4)
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: run kernel # of times (>1)\n");
        exit(0);
    }

    const bool do_verification = std::stoi(argv[1]);
    const int init_method      = std::stoi(argv[2]);
    const int nrepeat          = std::stoi(argv[3]);

    // GEMM shape
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA = 4096;
    ck::index_t StrideB = 4096;
    ck::index_t StrideC = 4096;

    // matrix data type
    using ADataType = ck::half_t;
    using BDataType = ck::half_t;
    using CDataType = ck::half_t;

    // matrix layout
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

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

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        break;
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
    }

    DeviceMem a_m_k_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpace());
    DeviceMem b_k_n_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpace());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpace());

    a_m_k_device_buf.ToDevice(a_m_k.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n.mData.data());
    c_m_n_device_buf.ToDevice(c_m_n_device_result.mData.data());

    // do GEMM
    auto gemm = typename DeviceGemmInstance<ADataType,
                                            BDataType,
                                            CDataType,
                                            ALayout,
                                            BLayout,
                                            CLayout,
                                            PassThrough,
                                            PassThrough,
                                            Relu>::type{};

    auto invoker  = gemm.MakeInvoker();
    auto argument = gemm.MakeArgument(static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                                      static_cast<BDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
                                      static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
                                      M,
                                      N,
                                      K,
                                      StrideA,
                                      StrideB,
                                      StrideC,
                                      PassThrough{},
                                      PassThrough{},
                                      Relu{});

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
        host_gemm_mk_kn_mn(a_m_k, b_k_n, c_m_n_host_result, PassThrough{}, PassThrough{}, Relu{});

        check_error(c_m_n_host_result, c_m_n_device_result);
    }
}
