#ifndef GEMM_UTILS_HPP
#define GEMM_UTILS_HPP

#include "check_err.hpp"
#include "config.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "reference_gemm.hpp"
#include "tensor_layout.hpp"

namespace ck {
namespace gemm_util {

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

template <typename GemmInstance,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
void RunHostGEMM(const Tensor<ADataType>& A,
                 const Tensor<BDataType>& B,
                 Tensor<CDataType>& C,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
{
    auto ref_gemm    = GemmInstance{};
    auto ref_invoker = ref_gemm.MakeInvoker();

    auto ref_argument = ref_gemm.MakeArgument(A, B, C, a_element_op, b_element_op, c_element_op);

    ref_invoker.Run(ref_argument);
}

template <typename DeviceGemmPtr_,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
void RunDeviceGEMM(DeviceGemmPtr_& gemmPtr,
                   const ck::gemm_util::GemmParams& params,
                   const Tensor<ADataType>& A,
                   const Tensor<BDataType>& B,
                   Tensor<CDataType>& C,
                   AElementwiseOperation a_element_op,
                   BElementwiseOperation b_element_op,
                   CElementwiseOperation c_element_op)
{
    DeviceMem a_m_k_device_buf(sizeof(ADataType) * A.mDesc.GetElementSpace());
    DeviceMem b_k_n_device_buf(sizeof(BDataType) * B.mDesc.GetElementSpace());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * C.mDesc.GetElementSpace());

    a_m_k_device_buf.ToDevice(A.mData.data());
    b_k_n_device_buf.ToDevice(B.mData.data());

    auto invoker_ptr = gemmPtr->MakeInvokerPointer();
    auto argument_ptr =
        gemmPtr->MakeArgumentPointer(static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                                     static_cast<BDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
                                     static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
                                     params.M,
                                     params.N,
                                     params.K,
                                     params.StrideA,
                                     params.StrideB,
                                     params.StrideC,
                                     a_element_op,
                                     b_element_op,
                                     c_element_op);

    if(!gemmPtr->IsSupportedArgument(argument_ptr.get()))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    invoker_ptr->Run(argument_ptr.get());
    c_m_n_device_buf.FromDevice(C.mData.data());
}

template <typename DeviceGemmPtr_,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct TestGemm
{
    auto PrepareGemmTensor(const ck::gemm_util::GemmParams& params)
    {
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

        Tensor<ADataType> a_m_k(
            f_host_tensor_descriptor(params.M, params.K, params.StrideA, ALayout{}));
        Tensor<BDataType> b_k_n(
            f_host_tensor_descriptor(params.K, params.N, params.StrideB, BLayout{}));
        Tensor<CDataType> c_m_n_host_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));
        Tensor<CDataType> c_m_n_device_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));

        auto f_generate_tensor_value = [](auto desc, auto type) {
            using dataType = decltype(type);

            if(std::is_same<dataType, int8_t>::value)
            {
                desc.GenerateTensorValue(GeneratorTensor_2<int8_t>{-5, 5});
            }
            else
            {
                desc.GenerateTensorValue(GeneratorTensor_3<dataType>{-0.5, 0.5});
            }
        };

        f_generate_tensor_value(a_m_k, ADataType{});
        f_generate_tensor_value(b_k_n, BDataType{});

        return std::make_tuple(a_m_k, b_k_n, c_m_n_host_result, c_m_n_device_result);
    }

    auto operator()(DeviceGemmPtr_& gemmPtr)
    {
        std::cout << "ALayout = " << ALayout{}.name << ", BLayout = " << BLayout{}.name
                  << ", CLayout = " << CLayout{}.name << std::endl;
        std::cout << gemmPtr->GetTypeString() << std::endl;

        // Arrange
        ck::gemm_util::GemmParams params;
        params.M       = 1024;
        params.N       = 1024;
        params.K       = 1024;
        params.StrideA = 1024;
        params.StrideB = 1024;
        params.StrideC = 1024;

        auto host_tensors = PrepareGemmTensor(params);

        const Tensor<ADataType>& a  = std::get<0>(host_tensors);
        const Tensor<BDataType>& b  = std::get<1>(host_tensors);
        Tensor<CDataType>& c_host   = std::get<2>(host_tensors);
        Tensor<CDataType>& c_device = std::get<3>(host_tensors);

        auto a_element_op = AElementwiseOperation{};
        auto b_element_op = BElementwiseOperation{};
        auto c_element_op = CElementwiseOperation{};

        using ReferenceGemmInstance =
            ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                      BDataType,
                                                      CDataType,
                                                      AElementwiseOperation,
                                                      BElementwiseOperation,
                                                      CElementwiseOperation>;
        ck::gemm_util::RunHostGEMM<ReferenceGemmInstance>(
            a, b, c_host, a_element_op, b_element_op, c_element_op);

        // Act
        ck::gemm_util::RunDeviceGEMM(
            gemmPtr, params, a, b, c_device, a_element_op, b_element_op, c_element_op);

        // Assert
        bool res = false;
        if(std::is_same<CDataType, float>::value)
        {
            res = ck::utils::check_err(c_device.mData, c_host.mData);
            std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
        }
        else if(std::is_same<CDataType, ck::half_t>::value)
        {
            res = ck::utils::check_err(c_device.mData, c_host.mData);
            std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
        }
        else if(std::is_same<CDataType, int8_t>::value)
        {
            res = ck::utils::check_err(c_device.mData, c_host.mData);
            std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
        }

        return res;
    }
};

template <typename DeviceGemmPtr_,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct TestGemmBF16
{
    using BF16 = ck::bhalf_t;

    auto PrepareGemmTensorBF16(const ck::gemm_util::GemmParams& params)
    {
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

        // use fp32 host kernel to verify bf16 device kernel
        Tensor<BF16> a_m_k_bf16(
            f_host_tensor_descriptor(params.M, params.K, params.StrideA, ALayout{}));
        Tensor<BF16> b_k_n_bf16(
            f_host_tensor_descriptor(params.K, params.N, params.StrideB, BLayout{}));
        Tensor<BF16> c_m_n_device_bf16(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));

        Tensor<float> a_m_k_fp32(
            f_host_tensor_descriptor(params.M, params.K, params.StrideA, ALayout{}));
        Tensor<float> b_k_n_fp32(
            f_host_tensor_descriptor(params.K, params.N, params.StrideB, BLayout{}));
        Tensor<float> c_m_n_host_fp32(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));
        Tensor<float> c_m_n_device_fp32(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));

        a_m_k_bf16.GenerateTensorValue(GeneratorTensor_3<BF16>{-0.5, 0.5});
        b_k_n_bf16.GenerateTensorValue(GeneratorTensor_3<BF16>{-0.5, 0.5});

        bf16_to_f32_(a_m_k_bf16, a_m_k_fp32);
        bf16_to_f32_(b_k_n_bf16, b_k_n_fp32);

        return std::make_tuple(a_m_k_bf16,
                               b_k_n_bf16,
                               c_m_n_device_bf16,
                               a_m_k_fp32,
                               b_k_n_fp32,
                               c_m_n_host_fp32,
                               c_m_n_device_fp32);
    }

    auto operator()(DeviceGemmPtr_& gemmPtr)
    {
        // Arrange
        ck::gemm_util::GemmParams params;
        params.M       = 1024;
        params.N       = 1024;
        params.K       = 1024;
        params.StrideA = 1024;
        params.StrideB = 1024;
        params.StrideC = 1024;

        auto host_tensors            = PrepareGemmTensorBF16(params);
        const Tensor<BF16>& a_bf16   = std::get<0>(host_tensors);
        const Tensor<BF16>& b_bf16   = std::get<1>(host_tensors);
        Tensor<BF16>& c_device_bf16  = std::get<2>(host_tensors);
        Tensor<float>& a_fp32        = std::get<3>(host_tensors);
        Tensor<float>& b_fp32        = std::get<4>(host_tensors);
        Tensor<float>& c_host_fp32   = std::get<5>(host_tensors);
        Tensor<float>& c_device_fp32 = std::get<6>(host_tensors);

        auto a_element_op = AElementwiseOperation{};
        auto b_element_op = BElementwiseOperation{};
        auto c_element_op = CElementwiseOperation{};

        // use fp32 host kernel to verify bf16 device kernel
        using ReferenceGemmInstance =
            ck::tensor_operation::host::ReferenceGemm<float,
                                                      float,
                                                      float,
                                                      AElementwiseOperation,
                                                      BElementwiseOperation,
                                                      CElementwiseOperation>;
        ck::gemm_util::RunHostGEMM<ReferenceGemmInstance>(
            a_fp32, b_fp32, c_host_fp32, a_element_op, b_element_op, c_element_op);

        // Act
        ck::gemm_util::RunDeviceGEMM(gemmPtr,
                                     params,
                                     a_bf16,
                                     b_bf16,
                                     c_device_bf16,
                                     a_element_op,
                                     b_element_op,
                                     c_element_op);

        bf16_to_f32_(c_device_bf16, c_device_fp32);

        // Assert
        bool res = ck::utils::check_err(
            c_device_fp32.mData, c_host_fp32.mData, "Error: incorrect results!", 1e-2f, 1e-3f);
        std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;

        return res;
    };
};

} // namespace gemm_util
} // namespace ck
#endif
