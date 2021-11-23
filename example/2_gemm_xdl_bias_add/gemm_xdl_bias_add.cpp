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
#include "example/2_gemm_xdl_bias_add/include/device_gemm_xdl_bias_add.hpp"

struct PassThrough
{
    template <typename T>
    __host__ __device__ constexpr T operator()(T v) const
    {
        return v;
    }
};

// GEMM Bias Add:
// C[m, n] = alpha(A[m, k] * B[k, n]) + beta * C0[m, n] + gamma * C1[m]
// assume C0 has same layout as C
// assume C1 is contiguous in memory
// C1 presents in memory as 1d vector, but is represented as 2D matrix C1[m, n], with stride = 0 in
//    the "n" dimension
//
// alpha * v0 + beta * v1 + gamma * v2
// v0 is from C matrix
// v1 is from residual matrix
// v2 is from bias vector
struct BiasAdd
{
#if 1
    // correct result
    // no scratch memory, good VGPR allocation (59)
    // good perf (101Tflops)
    template <typename T1, typename T2>
    __host__ __device__ constexpr float operator()(float v0, T1 v1, T2 v2) const
    {
        // compiler seems very volatile to the order of these calculation:
        // compiler is very eager to read AccVgpr (v0) out prematurely, resulting in register
        // over-allocation. Therefore, move v0 calculation to the very end
        float a = T1(0.2) * v1 + T2(0.3) * v2;
        float b = a + float(0.1) * v0;

        return b;
    }
#elif 0
    // correct result
    // some scratch memory (68), large VGPR usage (126)
    // very little perf drop (101Tflops)
    __host__ __device__ constexpr auto operator()(float v0, ck::half_t v1, ck::half_t v2) const
    {
        return float(0.1) * v0 + ck::half_t(0.2) * v1 + ck::half_t(0.3) * v2;
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
        ck::tensor_operation::device::DeviceGemmXdl_two_extra_source_reduce<  F16,   F16,   F16,     F32,     Row,     Col,     Row,          AOp,          BOp,          COp,   256,   256,   128,     4,  8,   32,   32,    4,    2,      S<1, 4, 8>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      S<1, 2, 8>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,               7,               1,      true,      true>;
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
    ck::tensor_operation::device::DeviceGemmXdl_two_extra_source_reduce<  F32,   F32,   F32,     F32,     Row,     Col,     Row,          AOp,          BOp,          COp,   256,   256,   128,     4,  4,   32,   32,    4,    2,      S<1, 4, 4>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      S<1, 2, 4>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,               7,               1,      true,      true>;
    // clang-format on
};

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
    if(argc != 10)
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: run kernel # of times (>1)\n");
        printf("arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC\n");
        exit(0);
    }

    const bool do_verification = std::stoi(argv[1]);
    const int init_method      = std::stoi(argv[2]);
    const int nrepeat          = std::stoi(argv[3]);

    // GEMM shape
    ck::index_t M = std::stoi(argv[4]);
    ck::index_t N = std::stoi(argv[5]);
    ck::index_t K = std::stoi(argv[6]);

    ck::index_t StrideA = std::stoi(argv[7]);
    ck::index_t StrideB = std::stoi(argv[8]);
    ck::index_t StrideC = std::stoi(argv[9]);

    // matrix data type
#if 1
    using ADataType = ck::half_t;
    using BDataType = ck::half_t;
    using CDataType = ck::half_t;
#else
    using ADataType = float;
    using BDataType = float;
    using CDataType = float;
#endif

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

    // C0[m ,n]
    Tensor<BDataType> c0_m_n(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    // C1[m]
    Tensor<CDataType> c1_m_n(HostTensorDescriptor(
        std::vector<std::size_t>({static_cast<std::size_t>(M), static_cast<std::size_t>(N)}),
        std::vector<std::size_t>({1, 0})));

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

    auto c_element_op = BiasAdd{};

    // do GEMM
    auto gemm = typename DeviceGemmInstance<ADataType,
                                            BDataType,
                                            CDataType,
                                            ALayout,
                                            BLayout,
                                            CLayout,
                                            PassThrough,
                                            PassThrough,
                                            decltype(c_element_op)>::type{};

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
    }
}
