#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>
#include "config.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "host_reduce_util.hpp"
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

constexpr int Rank                      = 2;
constexpr int NumReduceDim              = 1;
constexpr ck::ReduceTensorOp ReduceOpId = ck::ReduceTensorOp::MAX;
constexpr ck::NanPropagation NanOpt     = ck::NanPropagation::PROPAGATE_NAN;
constexpr bool PropagateNan = (NanOpt == ck::NanPropagation::NOT_PROPAGATE_NAN) ? false : true;
// constexpr ck::ReduceTensorIndices_t IndicesOpt = ck::ReduceTensorIndices_t::NO_INDICES;
using ReduceOperation = typename ck::reduce_binary_operator<CDataType, ReduceOpId>::opType;
using InElementwiseOperation =
    typename ck::reduce_unary_operator<CDataType, ReduceOpId, true, true>::InElementwiseOperation;
using AccElementwiseOperation =
    typename ck::reduce_unary_operator<CDataType, ReduceOpId, true, true>::AccElementwiseOperation;

using DeviceReduceInstance =
    ck::tensor_operation::device::DeviceReduceBlockWise<CDataType,
                                                        CDataType,
                                                        CDataType,
                                                        Rank,
                                                        NumReduceDim,
                                                        ReduceOperation,
                                                        InElementwiseOperation,
                                                        AccElementwiseOperation,
                                                        PropagateNan,
                                                        false,
                                                        256,
                                                        4,
                                                        64,
                                                        1,
                                                        1,
                                                        0,
                                                        1,
                                                        1>;

struct Sub
{
    __host__ __device__ constexpr void operator()(CDataType& dst, const CDataType& src1, const CDataType& src2) const
    {
        dst = src1 - src2;
    }
};

using DeviceElementwiseInstance = ck::tensor_operation::device::
    DeviceElementwise_2D<CDataType, CDataType, CDataType, Sub, 16, 16, 8, 8, 1, 1, 1, 1, 1>;

using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, CDataType, PassThrough, PassThrough, PassThrough>;

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
    const std::vector<int> reduceInvariantDims{1};

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
    Tensor<int> c_m_n_max(std::vector<std::size_t>({static_cast<std::size_t>(N)}),
                          std::vector<std::size_t>({1}));
    Tensor<CDataType> d_m_n(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    const auto i_inLengths  = ck::to_int_vector(c_m_n.mDesc.GetLengths());
    const auto i_inStrides  = ck::to_int_vector(c_m_n.mDesc.GetStrides());
    const auto i_outLengths = ck::to_int_vector(c_m_n_max.mDesc.GetLengths());
    const auto i_outStrides = ck::to_int_vector(c_m_n_max.mDesc.GetStrides());

    size_t reduce_total_length = c_m_n.mDesc.GetElementSize() / c_m_n_max.mDesc.GetElementSize();

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n.mDesc << std::endl;
    std::cout << "c_m_n_max: " << c_m_n_max.mDesc << std::endl;
    std::cout << "d_m_n: " << d_m_n.mDesc << std::endl;

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
    DeviceMem c_m_n_max_device_buf(sizeof(CDataType) * c_m_n_max.mDesc.GetElementSpace());
    DeviceMem c_m_n_max_indices_dev(0);
    DeviceMem d_m_n_device_buf(sizeof(CDataType) * d_m_n.mDesc.GetElementSpace());

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
    auto reduce_max    = DeviceReduceInstance{};
    auto wsSizeInBytes = reduce_max.GetWorkspaceSizeInBytes(i_inLengths, reduceDims);
    DeviceMem ws_dev(wsSizeInBytes);

    auto reduce_max_argument_ptr = reduce_max.MakeArgumentPointer(
        i_inLengths,
        i_inStrides,
        i_outLengths,
        i_outStrides,
        reduceDims,
        1,
        0,
        c_m_n_device_buf.GetDeviceBuffer(),
        c_m_n_max_device_buf.GetDeviceBuffer(),
        c_m_n_max_indices_dev.GetDeviceBuffer(),
        ws_dev.GetDeviceBuffer(),
        InElementwiseOperation{static_cast<int>(reduce_total_length)},
        AccElementwiseOperation{static_cast<int>(reduce_total_length)});

    if(!reduce_max.IsSupportedArgument(reduce_max_argument_ptr.get()))
    {
        throw std::runtime_error(
            "The runtime parameters seems not supported by the DeviceReduce instance, exiting!");
    };

    auto reduce_max_invoker_ptr = reduce_max.MakeInvokerPointer();
    reduce_max_invoker_ptr->Run(reduce_max_argument_ptr.get(), nrepeat);

    // do broadcast sub
    auto broadcastSub = DeviceElementwiseInstance{};
    auto broadcastSub_argument_ptr =
        broadcastSub.MakeArgumentPointer(c_m_n_device_buf.GetDeviceBuffer(),
                                         c_m_n_max_device_buf.GetDeviceBuffer(),
                                         d_m_n_device_buf.GetDeviceBuffer(),
                                         {M, N},
                                         {StrideC, 1},
                                         {0, 1},
                                         {StrideC, 1},
                                         Sub{});

    if(!broadcastSub.IsSupportedArgument(broadcastSub_argument_ptr.get()))
    {
        throw std::runtime_error("The runtime parameters seems not supported by the "
                                 "DeviceElementwise_2D instance, exiting!");
    };

    auto broadcastSub_invoker_ptr = broadcastSub.MakeInvokerPointer();
    broadcastSub_invoker_ptr->Run(broadcastSub_argument_ptr.get(), nrepeat);

    // TODO - Need BroadcastSub + exponential + ReduceSum + BroadcastDiv
    // TODO = do_verification
    (void)do_verification;
    return 0;
}
