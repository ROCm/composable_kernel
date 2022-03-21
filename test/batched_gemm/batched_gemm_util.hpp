#ifndef BATCHED_GEMM_UTILS_HPP
#define BATCHED_GEMM_UTILS_HPP

#include "config.hpp"
#include "device.hpp"
#include "host_tensor.hpp"

namespace ck {
namespace batched_gemm_util {

struct GemmParams
{
    GemmParams()
        : M(1024), N(1024), K(1024), StrideA(1024), StrideB(1024), StrideC(1024), alpha(1), beta(0)
    {
    }

    ck::index_t M;
    ck::index_t N;
    ck::index_t K;

    ck::index_t StrideA;
    ck::index_t StrideB;
    ck::index_t StrideC;

    float alpha;
    float beta;
};

template <typename BatchedGemmInstance,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
void RunHostBatchedGemm(const Tensor<ADataType>& A,
                        const Tensor<BDataType>& B,
                        Tensor<CDataType>& C,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CElementwiseOperation c_element_op)
{
    auto ref_batched_gemm = BatchedGemmInstance{};
    auto ref_invoker      = ref_batched_gemm.MakeInvoker();

    auto ref_argument =
        ref_batched_gemm.MakeArgument(A, B, C, a_element_op, b_element_op, c_element_op);

    ref_invoker.Run(ref_argument);
}

template <typename DeviceGemmPtr,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
void RunDeviceBatchedGemm(DeviceGemmPtr& batched_gemm_ptr,
                          const ck::batched_gemm_util::GemmParams& params,
                          const Tensor<ADataType>& A,
                          const Tensor<BDataType>& B,
                          Tensor<CDataType>& C,
                          AElementwiseOperation a_element_op,
                          BElementwiseOperation b_element_op,
                          CElementwiseOperation c_element_op)
{
    DeviceMem a_g_m_k_device_buf(sizeof(ADataType) * A.mDesc.GetElementSpace());
    DeviceMem b_g_k_n_device_buf(sizeof(BDataType) * B.mDesc.GetElementSpace());
    DeviceMem c_g_m_n_device_buf(sizeof(CDataType) * C.mDesc.GetElementSpace());

    a_g_m_k_device_buf.ToDevice(A.mData.data());
    b_g_k_n_device_buf.ToDevice(B.mData.data());

    const auto batch_count = A.mDesc.GetLengths()[0];
    auto invoker_ptr       = batched_gemm_ptr->MakeInvokerPointer();
    auto argument_ptr      = batched_gemm_ptr->MakeArgumentPointer(
        static_cast<ADataType*>(a_g_m_k_device_buf.GetDeviceBuffer()),
        static_cast<BDataType*>(b_g_k_n_device_buf.GetDeviceBuffer()),
        static_cast<CDataType*>(c_g_m_n_device_buf.GetDeviceBuffer()),
        params.M,
        params.N,
        params.K,
        params.StrideA,
        params.StrideB,
        params.StrideC,
        a_element_op,
        b_element_op,
        c_element_op,
        batch_count);

    if(!batched_gemm_ptr->IsSupportedArgument(argument_ptr.get()))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    invoker_ptr->Run(argument_ptr.get());
    c_g_m_n_device_buf.FromDevice(C.mData.data());
}

} // namespace batched_gemm_util
} // namespace ck
#endif
