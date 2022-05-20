#ifndef GEMM_UTILS_HPP
#define GEMM_UTILS_HPP

#include "check_err.hpp"
#include "config.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "reference_cgemm.hpp"
#include "tensor_layout.hpp"

namespace ck {
namespace cgemm_util {

struct CGemmParams
{
    CGemmParams()
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

template <typename CGemmInstance,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
void RunHostCGEMM(const Tensor<ADataType>& A_real,
                  const Tensor<ADataType>& A_imag,
                  const Tensor<BDataType>& B_real,
                  const Tensor<BDataType>& B_imag,
                  Tensor<CDataType>& C_real,
                  Tensor<CDataType>& C_imag,
                  AElementwiseOperation a_element_op,
                  BElementwiseOperation b_element_op,
                  CElementwiseOperation c_element_op)
{
    auto ref_cgemm   = CGemmInstance{};
    auto ref_invoker = ref_cgemm.MakeInvoker();

    auto ref_argument = ref_cgemm.MakeArgument(
        A_real, A_imag, B_real, B_imag, C_real, C_imag, a_element_op, b_element_op, c_element_op);

    ref_invoker.Run(ref_argument);
}

template <typename DeviceCGemmPtr_,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
void RunDeviceCGEMM(DeviceCGemmPtr_& cgemmPtr,
                    const ck::cgemm_util::CGemmParams& params,
                    const Tensor<ADataType>& A_real,
                    const Tensor<ADataType>& A_imag,
                    const Tensor<BDataType>& B_real,
                    const Tensor<BDataType>& B_imag,
                    Tensor<CDataType>& C_real,
                    Tensor<CDataType>& C_imag,
                    Tensor<CDataType>& Aux,
                    Tensor<CDataType>& Aux_2,
                    AElementwiseOperation a_element_op,
                    BElementwiseOperation b_element_op,
                    CElementwiseOperation c_element_op)
{
    DeviceMem a_m_k_real_device_buf(sizeof(ADataType) * A_real.mDesc.GetElementSpace());
    DeviceMem a_m_k_imag_device_buf(sizeof(ADataType) * A_imag.mDesc.GetElementSpace());
    DeviceMem b_k_n_real_device_buf(sizeof(BDataType) * B_real.mDesc.GetElementSpace());
    DeviceMem b_k_n_imag_device_buf(sizeof(BDataType) * B_imag.mDesc.GetElementSpace());
    DeviceMem c_m_n_real_device_buf(sizeof(CDataType) * C_real.mDesc.GetElementSpace());
    DeviceMem c_m_n_imag_device_buf(sizeof(CDataType) * C_imag.mDesc.GetElementSpace());
    DeviceMem aux_device_buf(sizeof(CDataType) * Aux.mDesc.GetElementSpace());
    DeviceMem aux_2_device_buf(sizeof(CDataType) * Aux_2.mDesc.GetElementSpace());

    a_m_k_real_device_buf.ToDevice(A_real.mData.data());
    a_m_k_imag_device_buf.ToDevice(A_imag.mData.data());
    b_k_n_real_device_buf.ToDevice(B_real.mData.data());
    b_k_n_imag_device_buf.ToDevice(B_imag.mData.data());

    auto invoker_ptr  = cgemmPtr->MakeInvokerPointer();
    auto argument_ptr = cgemmPtr->MakeArgumentPointer(
        static_cast<ADataType*>(a_m_k_real_device_buf.GetDeviceBuffer()),
        static_cast<ADataType*>(a_m_k_imag_device_buf.GetDeviceBuffer()),
        static_cast<BDataType*>(b_k_n_real_device_buf.GetDeviceBuffer()),
        static_cast<BDataType*>(b_k_n_imag_device_buf.GetDeviceBuffer()),
        static_cast<CDataType*>(c_m_n_real_device_buf.GetDeviceBuffer()),
        static_cast<CDataType*>(c_m_n_imag_device_buf.GetDeviceBuffer()),
        static_cast<CDataType*>(aux_device_buf.GetDeviceBuffer()),
        static_cast<CDataType*>(aux_2_device_buf.GetDeviceBuffer()),
        params.M,
        params.N,
        params.K,
        params.StrideA,
        params.StrideB,
        params.StrideC,
        a_element_op,
        b_element_op,
        c_element_op);

    if(!cgemmPtr->IsSupportedArgument(argument_ptr.get()))
    {
        throw std::runtime_error(
            "wrong! device_cgemm with the specified compilation parameters does "
            "not support this CGEMM problem");
    }

    invoker_ptr->Run(argument_ptr.get());
    c_m_n_real_device_buf.FromDevice(C_real.mData.data());
    c_m_n_imag_device_buf.FromDevice(C_imag.mData.data());
}

template <typename DeviceCGemmPtr_,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct TestCGemm
{
    auto PrepareCGemmTensor(const ck::cgemm_util::CGemmParams& params)
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

        Tensor<ADataType> a_m_k_real(
            f_host_tensor_descriptor(params.M, params.K, params.StrideA, ALayout{}));
        Tensor<ADataType> a_m_k_imag(
            f_host_tensor_descriptor(params.M, params.K, params.StrideA, ALayout{}));
        Tensor<BDataType> b_k_n_real(
            f_host_tensor_descriptor(params.K, params.N, params.StrideB, BLayout{}));
        Tensor<BDataType> b_k_n_imag(
            f_host_tensor_descriptor(params.K, params.N, params.StrideB, BLayout{}));
        Tensor<CDataType> c_m_n_real_host_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));
        Tensor<CDataType> c_m_n_imag_host_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));
        Tensor<CDataType> c_m_n_real_device_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));
        Tensor<CDataType> c_m_n_imag_device_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));
        Tensor<CDataType> aux(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));
        Tensor<CDataType> aux_2(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));

        auto f_generate_tensor_value = [](auto& tensor, auto type) {
            using dataType = decltype(type);

            tensor.GenerateTensorValue(GeneratorTensor_2<dataType>{-5, 5});
        };

        f_generate_tensor_value(a_m_k_real, ADataType{});
        f_generate_tensor_value(a_m_k_imag, ADataType{});
        f_generate_tensor_value(b_k_n_real, BDataType{});
        f_generate_tensor_value(b_k_n_imag, BDataType{});

        return std::make_tuple(a_m_k_real,
                               a_m_k_imag,
                               b_k_n_real,
                               b_k_n_imag,
                               c_m_n_real_host_result,
                               c_m_n_imag_host_result,
                               c_m_n_real_device_result,
                               c_m_n_imag_device_result,
                               aux,
                               aux_2);
    }

    auto operator()(DeviceCGemmPtr_& cgemmPtr)
    {
        std::cout << "ALayout = " << ALayout{}.name << ", BLayout = " << BLayout{}.name
                  << ", CLayout = " << CLayout{}.name << std::endl;
        std::cout << cgemmPtr->GetTypeString() << std::endl;

        // Arrange
        ck::cgemm_util::CGemmParams params;
        params.M       = 1024;
        params.N       = 1024;
        params.K       = 1024;
        params.StrideA = 1024;
        params.StrideB = 1024;
        params.StrideC = 1024;

        auto host_tensors = PrepareCGemmTensor(params);

        const Tensor<ADataType>& a_real  = std::get<0>(host_tensors);
        const Tensor<ADataType>& a_imag  = std::get<1>(host_tensors);
        const Tensor<BDataType>& b_real  = std::get<2>(host_tensors);
        const Tensor<BDataType>& b_imag  = std::get<3>(host_tensors);
        Tensor<CDataType>& c_host_real   = std::get<4>(host_tensors);
        Tensor<CDataType>& c_host_imag   = std::get<5>(host_tensors);
        Tensor<CDataType>& c_device_real = std::get<6>(host_tensors);
        Tensor<CDataType>& c_device_imag = std::get<7>(host_tensors);
        Tensor<CDataType>& aux           = std::get<8>(host_tensors);
        Tensor<CDataType>& aux_2         = std::get<9>(host_tensors);

        auto a_element_op = AElementwiseOperation{};
        auto b_element_op = BElementwiseOperation{};
        auto c_element_op = CElementwiseOperation{};

        using ReferenceGemmInstance =
            ck::tensor_operation::host::ReferenceCGemm<ADataType,
                                                       BDataType,
                                                       CDataType,
                                                       AElementwiseOperation,
                                                       BElementwiseOperation,
                                                       CElementwiseOperation>;
        ck::cgemm_util::RunHostCGEMM<ReferenceGemmInstance>(a_real,
                                                            a_imag,
                                                            b_real,
                                                            b_imag,
                                                            c_host_real,
                                                            c_host_imag,
                                                            a_element_op,
                                                            b_element_op,
                                                            c_element_op);

        // Act
        ck::cgemm_util::RunDeviceCGEMM(cgemmPtr,
                                       params,
                                       a_real,
                                       a_imag,
                                       b_real,
                                       b_imag,
                                       c_device_real,
                                       c_device_imag,
                                       aux,
                                       aux_2,
                                       a_element_op,
                                       b_element_op,
                                       c_element_op);

        // Assert
        bool res = false;
        if(std::is_same<CDataType, float>::value)
        {
            const bool res_real = ck::utils::check_err(
                c_device_real.mData, c_host_real.mData, "Error: incorrect results in real part!");
            const bool res_imag =
                ck::utils::check_err(c_device_imag.mData,
                                     c_host_imag.mData,
                                     "Error: incorrect results in imaginary part!");
            res = res_real && res_imag;
            std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
        }
        else if(std::is_same<CDataType, ck::half_t>::value)
        {
            const bool res_real = ck::utils::check_err(
                c_device_real.mData, c_host_real.mData, "Error: incorrect results in real part!");
            const bool res_imag =
                ck::utils::check_err(c_device_imag.mData,
                                     c_host_imag.mData,
                                     "Error: incorrect results in imaginary part!");
            res = res_real && res_imag;
            std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
        }
        else if(std::is_same<CDataType, int8_t>::value)
        {
            const bool res_real = ck::utils::check_err(
                c_device_real.mData, c_host_real.mData, "Error: incorrect results in real part!");
            const bool res_imag =
                ck::utils::check_err(c_device_imag.mData,
                                     c_host_imag.mData,
                                     "Error: incorrect results in imaginary part!");
            res = res_real && res_imag;
            std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
        }

        return res;
    }
};

template <typename DeviceCGemmPtr_,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct TestCGemmBF16
{
    using BF16 = ck::bhalf_t;

    auto PrepareCGemmTensorBF16(const ck::cgemm_util::CGemmParams& params)
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
        Tensor<BF16> a_m_k_real_bf16(
            f_host_tensor_descriptor(params.M, params.K, params.StrideA, ALayout{}));
        Tensor<BF16> a_m_k_imag_bf16(
            f_host_tensor_descriptor(params.M, params.K, params.StrideA, ALayout{}));
        Tensor<BF16> b_k_n_real_bf16(
            f_host_tensor_descriptor(params.K, params.N, params.StrideB, BLayout{}));
        Tensor<BF16> b_k_n_imag_bf16(
            f_host_tensor_descriptor(params.K, params.N, params.StrideB, BLayout{}));
        Tensor<BF16> c_m_n_real_device_bf16(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));
        Tensor<BF16> c_m_n_imag_device_bf16(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));
        Tensor<BF16> aux_bf16(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));
        Tensor<BF16> aux_2_bf16(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));

        Tensor<float> a_m_k_real_fp32(
            f_host_tensor_descriptor(params.M, params.K, params.StrideA, ALayout{}));
        Tensor<float> a_m_k_imag_fp32(
            f_host_tensor_descriptor(params.M, params.K, params.StrideA, ALayout{}));
        Tensor<float> b_k_n_real_fp32(
            f_host_tensor_descriptor(params.K, params.N, params.StrideB, BLayout{}));
        Tensor<float> b_k_n_imag_fp32(
            f_host_tensor_descriptor(params.K, params.N, params.StrideB, BLayout{}));
        Tensor<float> c_m_n_real_host_fp32(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));
        Tensor<float> c_m_n_imag_host_fp32(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));
        Tensor<float> c_m_n_real_device_fp32(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));
        Tensor<float> c_m_n_imag_device_fp32(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));

        a_m_k_real_bf16.GenerateTensorValue(GeneratorTensor_3<BF16>{-0.5, 0.5});
        a_m_k_imag_bf16.GenerateTensorValue(GeneratorTensor_3<BF16>{-0.5, 0.5});
        b_k_n_real_bf16.GenerateTensorValue(GeneratorTensor_3<BF16>{-0.5, 0.5});
        b_k_n_imag_bf16.GenerateTensorValue(GeneratorTensor_3<BF16>{-0.5, 0.5});

        bf16_to_f32_(a_m_k_real_bf16, a_m_k_real_fp32);
        bf16_to_f32_(a_m_k_imag_bf16, a_m_k_imag_fp32);
        bf16_to_f32_(b_k_n_real_bf16, b_k_n_real_fp32);
        bf16_to_f32_(b_k_n_imag_bf16, b_k_n_imag_fp32);

        return std::make_tuple(a_m_k_real_bf16,
                               a_m_k_imag_bf16,
                               b_k_n_real_bf16,
                               b_k_n_imag_bf16,
                               c_m_n_real_device_bf16,
                               c_m_n_imag_device_bf16,
                               aux_bf16,
                               aux_2_bf16,
                               a_m_k_real_fp32,
                               a_m_k_imag_fp32,
                               b_k_n_real_fp32,
                               b_k_n_imag_fp32,
                               c_m_n_real_host_fp32,
                               c_m_n_imag_host_fp32,
                               c_m_n_real_device_fp32,
                               c_m_n_imag_device_fp32);
    }

    auto operator()(DeviceCGemmPtr_& cgemmPtr)
    {
        // Arrange
        ck::cgemm_util::CGemmParams params;
        params.M       = 1024;
        params.N       = 1024;
        params.K       = 1024;
        params.StrideA = 1024;
        params.StrideB = 1024;
        params.StrideC = 1024;

        auto host_tensors                 = PrepareCGemmTensorBF16(params);
        const Tensor<BF16>& a_real_bf16   = std::get<0>(host_tensors);
        const Tensor<BF16>& a_imag_bf16   = std::get<1>(host_tensors);
        const Tensor<BF16>& b_real_bf16   = std::get<2>(host_tensors);
        const Tensor<BF16>& b_imag_bf16   = std::get<3>(host_tensors);
        Tensor<BF16>& c_real_device_bf16  = std::get<4>(host_tensors);
        Tensor<BF16>& c_imag_device_bf16  = std::get<5>(host_tensors);
        Tensor<BF16>& aux_bf16            = std::get<6>(host_tensors);
        Tensor<BF16>& aux_2_bf16          = std::get<7>(host_tensors);
        Tensor<float>& a_real_fp32        = std::get<8>(host_tensors);
        Tensor<float>& a_imag_fp32        = std::get<9>(host_tensors);
        Tensor<float>& b_real_fp32        = std::get<10>(host_tensors);
        Tensor<float>& b_imag_fp32        = std::get<11>(host_tensors);
        Tensor<float>& c_real_host_fp32   = std::get<12>(host_tensors);
        Tensor<float>& c_imag_host_fp32   = std::get<13>(host_tensors);
        Tensor<float>& c_real_device_fp32 = std::get<14>(host_tensors);
        Tensor<float>& c_imag_device_fp32 = std::get<15>(host_tensors);

        auto a_element_op = AElementwiseOperation{};
        auto b_element_op = BElementwiseOperation{};
        auto c_element_op = CElementwiseOperation{};

        // use fp32 host kernel to verify bf16 device kernel
        using ReferenceCGemmInstance =
            ck::tensor_operation::host::ReferenceCGemm<float,
                                                       float,
                                                       float,
                                                       AElementwiseOperation,
                                                       BElementwiseOperation,
                                                       CElementwiseOperation>;
        ck::cgemm_util::RunHostCGEMM<ReferenceCGemmInstance>(a_real_fp32,
                                                             a_imag_fp32,
                                                             b_real_fp32,
                                                             b_imag_fp32,
                                                             c_real_host_fp32,
                                                             c_imag_host_fp32,
                                                             a_element_op,
                                                             b_element_op,
                                                             c_element_op);

        // Act
        ck::cgemm_util::RunDeviceCGEMM(cgemmPtr,
                                       params,
                                       a_real_bf16,
                                       a_imag_bf16,
                                       b_real_bf16,
                                       b_imag_bf16,
                                       c_real_device_bf16,
                                       c_imag_device_bf16,
                                       aux_bf16,
                                       aux_2_bf16,
                                       a_element_op,
                                       b_element_op,
                                       c_element_op);

        bf16_to_f32_(c_real_device_bf16, c_real_device_fp32);
        bf16_to_f32_(c_imag_device_bf16, c_imag_device_fp32);

        // Assert
        const bool res_real = ck::utils::check_err(c_real_device_fp32.mData,
                                                   c_real_host_fp32.mData,
                                                   "Error: incorrect results in real part!",
                                                   1e-2f,
                                                   1e-1f);
        const bool res_imag = ck::utils::check_err(c_imag_device_fp32.mData,
                                                   c_imag_host_fp32.mData,
                                                   "Error: incorrect results in imaginary part!",
                                                   1e-2f,
                                                   1e-1f);
        const bool res      = res_real && res_imag;

        std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;

        return res;
    };
};

} // namespace cgemm_util
} // namespace ck
#endif
