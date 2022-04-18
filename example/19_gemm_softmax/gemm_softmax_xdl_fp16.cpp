#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>
#include <math.h>
#include "check_err.hpp"
#include "config.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "host_reduce_util.hpp"
#include "host_reduction.hpp"

#include "device_tensor.hpp"
#include "device_gemm_xdl.hpp"
#include "device_gemm_xdl_c_shuffle.hpp"
#include "element_wise_operation.hpp"
#include "reference_gemm.hpp"
#include "gemm_specialization.hpp"

#include "device_reduce_blockwise.hpp"
#include "reduction_enums.hpp"
#include "reduction_operator_mapping.hpp"
#include "device_elementwise_2d.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType   = F16;
using BDataType   = F16;
using CDataType   = F16;
using AccDataType = F32;

// CAUSION - host reduce_max will call numeric_limits<ck::half_t>::lowest()
// However, numeric_limits<ck::half_t>::lowest() will return zero. So, used half_float::half instead
using HostReduceDataType = half_float::half;

using ALayout = ck::tensor_layout::gemm::RowMajor;
using BLayout = ck::tensor_layout::gemm::ColumnMajor;
using CLayout = ck::tensor_layout::gemm::RowMajor;

// clang-format off
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmXdl_C_Shuffle<
    ADataType,              // ADataType
    BDataType,              // BDataType
    CDataType,              // CDataType
    AccDataType,            // AccDataType
    CDataType,              // CShuffleDataType
    ALayout,                // ALayout
    BLayout,                // BLayout
    CLayout,                // CLayout
    PassThrough,            // AElementwiseOperation
    PassThrough,            // BElementwiseOperation
    PassThrough,            // CElementwiseOperation
    256,                    // BlockSize
    256,                    // MPerBlock
    128,                    // NPerBlock
    32,                     // KPerBlock
    8,                      // AK1
    8,                      // BK1
    32,                     // MPerXDL
    32,                     // NPerXDL
    4,                      // MXdlPerWave
    2,                      // NXdlPerWave
    S<4, 64, 1>,            // ABlockTransferThreadClusterLengths_K0_M_K1
    S<1, 0, 2>,             // ABlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,             // ABlockTransferSrcAccessOrder
    2,                      // ABlockTransferSrcVectorDim
    8,                      // ABlockTransferSrcScalarPerVector
    8,                      // ABlockTransferDstScalarPerVector_K1
    true,                   // ABlockLdsAddExtraM
    S<4, 64, 1>,            // BBlockTransferThreadClusterLengths_K0_N_K1
    S<1, 0, 2>,             // BBlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,             // BBlockTransferSrcAccessOrder
    2,                      // BBlockTransferSrcVectorDim
    8,                      // BBlockTransferSrcScalarPerVector
    8,                      // BBlockTransferDstScalarPerVector_K1
    true,                   // BBlockLdsAddExtraN
    1,                      // CShuffleMXdlPerWavePerShuffle
    1,                      // CShuffleNXdlPerWavePerShuffle
    S<1, 1, 32, 1, 1, 8>,   // CBlockTransferClusterLengths_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl
    8>;                     // CBlockTransferScalarPerVector_NWaveNPerXdl
// clang-format on

constexpr int Rank                       = 2;
constexpr int NumReduceDim               = 1;
constexpr ck::ReduceTensorOp ReduceMaxId = ck::ReduceTensorOp::MAX;
constexpr ck::ReduceTensorOp ReduceSumId = ck::ReduceTensorOp::ADD;
constexpr bool ReducePropagateNan        = false;
using ReduceMaxOp = typename ck::reduce_binary_operator<CDataType, ReduceMaxId>::opType;
using ReduceSumOp = typename ck::reduce_binary_operator<CDataType, ReduceSumId>::opType;
using ReduceMaxInElementwiseOperation =
    typename ck::reduce_unary_operator<CDataType, ReduceMaxId, true, true>::InElementwiseOperation;
using ReduceMaxAccElementwiseOperation =
    typename ck::reduce_unary_operator<CDataType, ReduceMaxId, true, true>::AccElementwiseOperation;
using ReduceSumInElementwiseOperation =
    typename ck::reduce_unary_operator<CDataType, ReduceSumId, true, true>::InElementwiseOperation;
using ReduceSumAccElementwiseOperation =
    typename ck::reduce_unary_operator<CDataType, ReduceSumId, true, true>::AccElementwiseOperation;

using DeviceReduceMaxInstance =
    ck::tensor_operation::device::DeviceReduceBlockWise<CDataType,
                                                        CDataType,
                                                        CDataType,
                                                        Rank,
                                                        NumReduceDim,
                                                        ReduceMaxOp,
                                                        ReduceMaxInElementwiseOperation,
                                                        ReduceMaxAccElementwiseOperation,
                                                        ReducePropagateNan,
                                                        false,
                                                        256,
                                                        4,
                                                        64,
                                                        1,
                                                        1,
                                                        0,
                                                        1,
                                                        1>;

using DeviceReduceSumInstance =
    ck::tensor_operation::device::DeviceReduceBlockWise<CDataType,
                                                        CDataType,
                                                        CDataType,
                                                        Rank,
                                                        NumReduceDim,
                                                        ReduceSumOp,
                                                        ReduceSumInElementwiseOperation,
                                                        ReduceSumAccElementwiseOperation,
                                                        ReducePropagateNan,
                                                        false,
                                                        256,
                                                        4,
                                                        64,
                                                        1,
                                                        1,
                                                        0,
                                                        1,
                                                        1>;

struct Sub_Exp
{
    __host__ __device__ constexpr void
    operator()(CDataType& dst, const CDataType& src1, const CDataType& src2) const
    {
        dst = src1 - src2;
        // FIXME - use float16 exponential
        float dst_f32 = static_cast<float>(dst);
        dst           = static_cast<CDataType>(exp(dst_f32));
    }
};

struct Div
{
    __host__ __device__ constexpr void
    operator()(CDataType& dst, const CDataType& src1, const CDataType& src2) const
    {
        dst = src1 / src2;
    }
};

using DeviceElementwiseSubExpInstance = ck::tensor_operation::device::
    DeviceElementwise_2D<CDataType, CDataType, CDataType, Sub_Exp, 256, 32, 8>;

using DeviceElementwiseDivInstance = ck::tensor_operation::device::
    DeviceElementwise_2D<CDataType, CDataType, CDataType, Div, 256, 32, 8>;

using HostGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, CDataType, PassThrough, PassThrough, PassThrough>;

using HostReduceMaxInstance = ReductionHost<HostReduceDataType,
                                            HostReduceDataType,
                                            HostReduceDataType,
                                            ReduceMaxId,
                                            Rank,
                                            NumReduceDim,
                                            ReducePropagateNan,
                                            false>;

using HostReduceSumInstance = ReductionHost<HostReduceDataType,
                                            HostReduceDataType,
                                            HostReduceDataType,
                                            ReduceSumId,
                                            Rank,
                                            NumReduceDim,
                                            ReducePropagateNan,
                                            false>;

template <typename HostTensorA,
          typename HostTensorB,
          typename HostTensorC,
          typename Functor,
          int broadcastDim>
void host_broadcast2D(
    HostTensorC& C, const HostTensorA& A, const HostTensorB& B, int M, int N, Functor functor)
{
    for(int m = 0; m < M; ++m)
    {
        for(int n = 0; n < N; ++n)
        {
            if constexpr(broadcastDim == 1)
                functor(C(m, n), A(m, n), B(n));
            else
                functor(C(m, n), A(m, n), B(m));
        }
    }
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

    const std::vector<int> reduceDims{0};

    if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        nrepeat         = std::stoi(argv[3]);
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
    Tensor<CDataType> c_m_n(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<CDataType> c_n_max(std::vector<std::size_t>({static_cast<std::size_t>(N)}),
                              std::vector<std::size_t>({1}));
    Tensor<CDataType> exp_m_n(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<CDataType> exp_n_sum(std::vector<std::size_t>({static_cast<std::size_t>(N)}),
                                std::vector<std::size_t>({1}));
    Tensor<CDataType> softmax_m_n(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    const auto c_m_n_shape     = ck::to_int_vector(c_m_n.mDesc.GetLengths());
    const auto c_m_n_stride    = ck::to_int_vector(c_m_n.mDesc.GetStrides());
    const auto reduce_n_shape  = ck::to_int_vector(c_n_max.mDesc.GetLengths());
    const auto reduce_n_stride = ck::to_int_vector(c_n_max.mDesc.GetStrides());

    size_t reduce_total_length = c_m_n.mDesc.GetElementSize() / c_n_max.mDesc.GetElementSize();

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n.mDesc << std::endl;
    std::cout << "c_n_max: " << c_n_max.mDesc << std::endl;
    std::cout << "exp_m_n: " << exp_m_n.mDesc << std::endl;
    std::cout << "exp_n_sum: " << exp_n_sum.mDesc << std::endl;
    std::cout << "softmax_m_n: " << softmax_m_n.mDesc << std::endl;

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
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * c_m_n.mDesc.GetElementSpace());
    DeviceMem c_n_max_device_buf(sizeof(CDataType) * c_n_max.mDesc.GetElementSpace());
    DeviceMem indices_device_buf(0);
    DeviceMem exp_m_n_device_buf(sizeof(CDataType) * exp_m_n.mDesc.GetElementSpace());
    DeviceMem exp_n_sum_device_buf(sizeof(CDataType) * exp_n_sum.mDesc.GetElementSpace());
    DeviceMem softmax_m_n_device_buf(sizeof(CDataType) * softmax_m_n.mDesc.GetElementSpace());

    a_m_k_device_buf.ToDevice(a_m_k.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n.mData.data());

    // do GEMM
    auto gemm         = DeviceGemmInstance{};
    auto gemm_invoker = gemm.MakeInvoker();
    auto gemm_argument =
        gemm.MakeArgument(static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
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
                          PassThrough{});

    if(!gemm.IsSupportedArgument(gemm_argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    gemm_invoker.Run(gemm_argument, nrepeat);

    // do reduce max
    auto reduce_max                 = DeviceReduceMaxInstance{};
    auto reduce_max_workaspace_size = reduce_max.GetWorkspaceSizeInBytes(c_m_n_shape, reduceDims);
    DeviceMem reduce_max_workaspace_device_buf(reduce_max_workaspace_size);

    auto reduce_max_argument_ptr = reduce_max.MakeArgumentPointer(
        c_m_n_shape,
        c_m_n_stride,
        reduce_n_shape,
        reduce_n_stride,
        reduceDims,
        1,
        0,
        c_m_n_device_buf.GetDeviceBuffer(),
        c_n_max_device_buf.GetDeviceBuffer(),
        indices_device_buf.GetDeviceBuffer(),
        reduce_max_workaspace_device_buf.GetDeviceBuffer(),
        ReduceMaxInElementwiseOperation{static_cast<int>(reduce_total_length)},
        ReduceMaxAccElementwiseOperation{static_cast<int>(reduce_total_length)});

    if(!reduce_max.IsSupportedArgument(reduce_max_argument_ptr.get()))
    {
        throw std::runtime_error(
            "The runtime parameters seems not supported by the DeviceReduce instance, exiting!");
    };

    auto reduce_max_invoker_ptr = reduce_max.MakeInvokerPointer();
    reduce_max_invoker_ptr->Run(reduce_max_argument_ptr.get(), nrepeat);

    // do broadcast sub and exp
    auto broadcastSubExp = DeviceElementwiseSubExpInstance{};
    auto broadcastSubExp_argument_ptr =
        broadcastSubExp.MakeArgumentPointer(c_m_n_device_buf.GetDeviceBuffer(),
                                            c_n_max_device_buf.GetDeviceBuffer(),
                                            exp_m_n_device_buf.GetDeviceBuffer(),
                                            {M, N},
                                            {StrideC, 1},
                                            {0, 1},
                                            {StrideC, 1},
                                            Sub_Exp{});

    if(!broadcastSubExp.IsSupportedArgument(broadcastSubExp_argument_ptr.get()))
    {
        throw std::runtime_error("The runtime parameters seems not supported by the "
                                 "DeviceElementwise_2D instance, exiting!");
    };

    auto broadcastSubExp_invoker_ptr = broadcastSubExp.MakeInvokerPointer();
    broadcastSubExp_invoker_ptr->Run(broadcastSubExp_argument_ptr.get(), nrepeat);

    // do reduce sum - denominator of softmax
    auto reduce_sum                 = DeviceReduceSumInstance{};
    auto reduce_sum_workaspace_size = reduce_sum.GetWorkspaceSizeInBytes(c_m_n_shape, reduceDims);
    DeviceMem reduce_sum_workaspace_device_buf(reduce_sum_workaspace_size);

    auto reduce_sum_argument_ptr = reduce_sum.MakeArgumentPointer(
        c_m_n_shape,
        c_m_n_stride,
        reduce_n_shape,
        reduce_n_stride,
        reduceDims,
        1, // alpha
        0, // beta
        exp_m_n_device_buf.GetDeviceBuffer(),
        exp_n_sum_device_buf.GetDeviceBuffer(),
        indices_device_buf.GetDeviceBuffer(),
        reduce_sum_workaspace_device_buf.GetDeviceBuffer(),
        ReduceSumInElementwiseOperation{static_cast<int>(reduce_total_length)},
        ReduceSumAccElementwiseOperation{static_cast<int>(reduce_total_length)});

    if(!reduce_sum.IsSupportedArgument(reduce_sum_argument_ptr.get()))
    {
        throw std::runtime_error(
            "The runtime parameters seems not supported by the DeviceReduce instance, exiting!");
    };

    auto reduce_sum_invoker_ptr = reduce_sum.MakeInvokerPointer();
    reduce_sum_invoker_ptr->Run(reduce_sum_argument_ptr.get(), nrepeat);

    // do broadcast div
    auto broadcastDiv = DeviceElementwiseDivInstance{};
    auto broadcastDiv_argument_ptr =
        broadcastDiv.MakeArgumentPointer(exp_m_n_device_buf.GetDeviceBuffer(),
                                         exp_n_sum_device_buf.GetDeviceBuffer(),
                                         softmax_m_n_device_buf.GetDeviceBuffer(),
                                         {M, N},
                                         {StrideC, 1},
                                         {0, 1},
                                         {StrideC, 1},
                                         Div{});

    if(!broadcastDiv.IsSupportedArgument(broadcastDiv_argument_ptr.get()))
    {
        throw std::runtime_error("The runtime parameters seems not supported by the "
                                 "DeviceElementwise_2D instance, exiting!");
    };

    auto broadcastDiv_invoker_ptr = broadcastDiv.MakeInvokerPointer();
    broadcastDiv_invoker_ptr->Run(broadcastDiv_argument_ptr.get(), nrepeat);

    // TODO = do_verification
    if(do_verification)
    {
        std::cout << "verification..." << std::endl;
        const std::vector<int> reduceInvariantDims{1};
        Tensor<CDataType> host_c_m_n(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
        Tensor<CDataType> host_c_n_max(std::vector<std::size_t>({static_cast<std::size_t>(N)}),
                                       std::vector<std::size_t>({1}));
        Tensor<int> host_indices(host_c_n_max.mDesc.GetLengths());
        Tensor<CDataType> host_exp_m_n(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
        Tensor<CDataType> host_exp_n_sum(std::vector<std::size_t>({static_cast<std::size_t>(N)}),
                                         std::vector<std::size_t>({1}));
        Tensor<CDataType> host_softmax_m_n(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

        auto host_gemm          = HostGemmInstance{};
        auto host_gemm_invoker  = host_gemm.MakeInvoker();
        auto host_gemm_argument = host_gemm.MakeArgument(
            a_m_k, b_k_n, host_c_m_n, PassThrough{}, PassThrough{}, PassThrough{});

        auto host_reduce_max = HostReduceMaxInstance{
            host_c_m_n.mDesc, host_c_n_max.mDesc, reduceInvariantDims, reduceDims};

        auto host_reduce_sum = HostReduceSumInstance{
            host_exp_m_n.mDesc, host_exp_n_sum.mDesc, reduceInvariantDims, reduceDims};

        host_gemm_invoker.Run(host_gemm_argument);
        host_reduce_max.Run(1, // alpha
                            reinterpret_cast<const HostReduceDataType*>(host_c_m_n.mData.data()),
                            0, // beta
                            reinterpret_cast<HostReduceDataType*>(host_c_n_max.mData.data()),
                            host_indices.mData.data());

        host_broadcast2D<Tensor<CDataType>, Tensor<CDataType>, Tensor<CDataType>, Sub_Exp, 1>(
            host_exp_m_n, host_c_m_n, host_c_n_max, M, N, Sub_Exp{});

        host_reduce_sum.Run(1, // alpha
                            reinterpret_cast<const HostReduceDataType*>(host_exp_m_n.mData.data()),
                            0, // beta
                            reinterpret_cast<HostReduceDataType*>(host_exp_n_sum.mData.data()),
                            host_indices.mData.data());

        host_broadcast2D<Tensor<CDataType>, Tensor<CDataType>, Tensor<CDataType>, Div, 1>(
            host_softmax_m_n, host_exp_m_n, host_exp_n_sum, M, N, Div{});

        c_m_n_device_buf.FromDevice(c_m_n.mData.data());
        c_n_max_device_buf.FromDevice(c_n_max.mData.data());
        exp_m_n_device_buf.FromDevice(exp_m_n.mData.data());
        exp_n_sum_device_buf.FromDevice(exp_n_sum.mData.data());
        softmax_m_n_device_buf.FromDevice(softmax_m_n.mData.data());

        bool result = true;
        if(result &= ck::utils::check_err(c_m_n.mData, host_c_m_n.mData))
            std::cout << "[PASS] - c_m_n" << std::endl;
        if(result &= ck::utils::check_err(c_n_max.mData, host_c_n_max.mData))
            std::cout << "[PASS] - c_n_max" << std::endl;
        if(result &= ck::utils::check_err(exp_m_n.mData, host_exp_m_n.mData))
            std::cout << "[PASS] - exp_m_n" << std::endl;
        if(result &= ck::utils::check_err(exp_n_sum.mData, host_exp_n_sum.mData))
            std::cout << "[PASS] - exp_n_sum" << std::endl;
        if(result &= ck::utils::check_err(softmax_m_n.mData, host_softmax_m_n.mData))
            std::cout << "[PASS] - softmax_m_n" << std::endl;
    }
    return 0;
}
