// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_abd_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"

template <typename T>
constexpr bool always_false = false;

template <typename Y, typename X>
inline __host__ __device__ Y fast_type_convert(X x)
{
    static_assert(always_false<X>, "not implemented");
    (void)x;
}

// template <>
// inline __host__ __device__ ck::half_t fast_type_convert<ck::half_t, ck::f8_t>(ck::f8_t x)
// {
//     constexpr const uint16_t mask      = 0x7fff;
//     constexpr const uint16_t sign_mask = 0x8000;
//     // constexpr const uint16_t exp_compensate = 0x2000;  // for float8_e4m3fn
//     constexpr const uint16_t exp_compensate = 0x1c00; // for float8_e4m3fnuz

//     uint8_t x_u8   = reinterpret_cast<uint8_t&>(x);
//     uint16_t x_u16 = static_cast<uint16_t>(x_u8) << 8;
//     uint16_t exp   = (x_u16 & mask) >> 1;
//     uint16_t y     = (x_u16 & sign_mask) | (exp + exp_compensate);
//     return reinterpret_cast<ck::half_t&>(y);
// }

template <>
inline __host__ __device__ ck::half2_t
fast_type_convert<ck::half2_t, ck::packed_f4x2_t>(ck::packed_f4x2_t x)
{
    uint8_t x_u8 = ck::bit_cast<uint8_t>(x);
    uint8_t x_l  = (x_u8 & 0x0f) >> 0;
    uint8_t x_h  = (x_u8 & 0xf0) >> 4;

    // FIXME:
    return {ck::type_convert<ck::half_t>(0.06125f * x_l),
            ck::type_convert<ck::half_t>(0.06125f * x_h)};

    uint8_t l_s  = x_l & 0x8;
    uint8_t l_em = x_l & 0x7;
    uint8_t l_u8 = (l_s << 4) | (l_em << 2);
    l_u8 += 0x38;
    // l_u8 = 0;

    uint8_t h_s  = x_h & 0x8;
    uint8_t h_em = x_h & 0x7;
    uint8_t h_u8 = (h_s << 4) | (h_em << 2);
    h_u8 += 0x38;
    // h_u8= 0;

    auto l_f16 = ck::type_convert<ck::half_t>(ck::bit_cast<ck::f8_t>(l_u8));
    auto h_f16 = ck::type_convert<ck::half_t>(ck::bit_cast<ck::f8_t>(h_u8));

    half2 result;
    result.data[0] = l_f16;
    result.data[1] = h_f16;
    return ck::bit_cast<ck::half2_t>(result);
}

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PF4 = ck::packed_f4_t;
using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType        = F16;
using B0DataType       = PF4;
using B1DataType       = F16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using EDataType        = F16;

using ALayout = Row;
using BLayout = Col;
using DLayout = Row;
using ELayout = Row;

struct ElementwiseScale
{
    __host__ __device__ void
    operator()(ck::half2_t& y, const ck::packed_f4x2_t& x0, const ck::half2_t& x1) const
    {
        auto scale = fast_type_convert<ck::half2_t>(x0);
        y          = scale * x1;
    }

    constexpr const static bool is_pack2_invocable = true;
};

// struct AlphaBetaAdd
// {
//     AlphaBetaAdd(float alpha, float beta) : alpha_(alpha), beta_(beta){};

//     template <typename E, typename C, typename D>
//     __host__ __device__ constexpr void operator()(E& e, const C& c, const D& d) const;

//     template <>
//     __host__ __device__ constexpr void operator()<ck::half_t, float, ck::half_t>(
//         ck::half_t& e, const float& c, const ck::half_t& d) const
//     {
//         e = ck::type_convert<ck::half_t>(alpha_ * c + beta_ * ck::type_convert<float>(d));
//     };

//     float alpha_;
//     float beta_;
// };

using AElementOp   = PassThrough;
using BElementOp   = ElementwiseScale;
// using CDEElementOp = AlphaBetaAdd;
using CDEElementOp = PassThrough;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

using DeviceOpInstance = ck::tensor_operation::device::DeviceGemmMultipleABD_Xdl_CShuffle<
    ck::Tuple<ALayout>,
    ck::Tuple<BLayout, BLayout>,
    ck::Tuple<DLayout>,
    ELayout,
    ck::Tuple<ADataType>,
    ck::Tuple<B0DataType, B1DataType>,
    AccDataType,
    CShuffleDataType,
    ck::Tuple</*DDataType*/>,
    EDataType,
    AElementOp,
    BElementOp,
    CDEElementOp,
    GemmSpec,
    1,
    256,
    256,
    128,
    32,
    8,
    8,
    32,
    32,
    4,
    2,
    S<4, 64, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    8,
    8,
    1,
    S<4, 64, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    8,
    8,
    1,
    1,
    1,
    S<1, 32, 1, 8>,
    8>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // GEMM shape
    ck::index_t M = 1;
    ck::index_t N = 128;
    ck::index_t K = 128;

    ck::index_t StrideA = 128;
    ck::index_t StrideB = 128;
    ck::index_t StrideD = 128;
    ck::index_t StrideE = 128;

    float alpha = 1.0f;
    float beta  = 1.0f;

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 6)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        alpha = std::stof(argv[4]);
        beta  = std::stof(argv[5]);
    }
    else if(argc == 13)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M = std::stoi(argv[4]);
        N = std::stoi(argv[5]);
        K = std::stoi(argv[6]);

        StrideA = std::stoi(argv[7]);
        StrideB = std::stoi(argv[8]);
        StrideD = std::stoi(argv[9]);
        StrideE = std::stoi(argv[10]);

        alpha = std::stof(argv[11]);
        beta  = std::stof(argv[12]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideD, StrideE, alpha, "
               "beta\n");
        exit(0);
    }

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            using namespace ck::literals;

            if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<B0DataType> b0_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
    Tensor<B1DataType> b1_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
    Tensor<EDataType> e_m_n_host_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));
    Tensor<EDataType> e_m_n_device_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b0_k_n: " << b0_k_n.mDesc << std::endl;
    std::cout << "b1_k_n: " << b1_k_n.mDesc << std::endl;
    std::cout << "e_m_n: " << e_m_n_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});   // input, random
        // a_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{});        // input, all 1
        b0_k_n.GenerateTensorValue(GeneratorTensor_2<B0DataType>{0, 256}); // weight. random
        // b0_k_n.GenerateTensorValue(GeneratorTensor_1<B0DataType>{0});     // weight. all 0
        // b0_k_n.GenerateTensorValue(GeneratorTensor_1<B0DataType>{34});     // weight. all 1, 0b0010_0010, high and low 4 bits are both 1.0
        // b0_k_n.GenerateTensorValue(GeneratorTensor_1<B0DataType>{162});    // weight. 1, -1, 1, -1, ...  0b1010_0010, high and low 4 bits are both 1.0
        // b1_k_n.GenerateTensorValue(GeneratorTensor_2<B1DataType>{56, 57}); // weight scale
        b1_k_n.GenerateTensorValue(GeneratorTensor_1<B1DataType>{});          // weight scale, all 1
        break;
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        // b0_k_n.GenerateTensorValue(GeneratorTensor_3<B0DataType>{-0.5, 0.5});  // FIXME:
        b1_k_n.GenerateTensorValue(GeneratorTensor_3<B1DataType>{-0.5, 0.5});
    }

    // b0_k_n has doubled size, fill the tail half with 0
    // memset(&b0_k_n(0, 0) + sizeof(B0DataType) * b0_k_n.mDesc.GetElementSpaceSize() / 2,
    //        0,
    //        sizeof(B0DataType) * b0_k_n.mDesc.GetElementSpaceSize() / 2);

    // b0_k_n(0, 0) = (2 < 4) | 1;
    // b0_k_n(1, 0) = (4 < 4) | 3;
    // b0_k_n(2, 0) = (6 < 4) | 5;

    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b0_device_buf(sizeof(B0DataType) * b0_k_n.mDesc.GetElementSpaceSize());
    DeviceMem b1_device_buf(sizeof(B1DataType) * b1_k_n.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) * e_m_n_device_result.mDesc.GetElementSpaceSize());

    printf("b0_k_n number of bytes: %d\n", sizeof(B0DataType) * b0_k_n.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b0_device_buf.ToDevice(b0_k_n.mData.data());
    b1_device_buf.ToDevice(b1_k_n.mData.data());
    e_device_buf.ToDevice(e_m_n_device_result.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};
    // auto cde_element_op = CDEElementOp{alpha, beta};

    // do GEMM
    auto device_op = DeviceOpInstance{};
    auto invoker   = device_op.MakeInvoker();
    auto argument =
        device_op.MakeArgument(std::array<const void*, 1>{a_device_buf.GetDeviceBuffer()},
                               std::array<const void*, 2>{b0_device_buf.GetDeviceBuffer(),
                                                          b1_device_buf.GetDeviceBuffer()},
                               std::array<const void*, 0>{},
                               e_device_buf.GetDeviceBuffer(),
                               M,
                               N,
                               K,
                               std::array<ck::index_t, 1>{StrideA},
                               std::array<ck::index_t, 2>{StrideB, StrideB},
                               std::array<ck::index_t, 0>{},
                               StrideE,
                               a_element_op,
                               b_element_op,
                               cde_element_op);

    if(!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop      = std::size_t(2) * M * N * K;
    std::size_t num_btype = sizeof(ADataType) * M * K + sizeof(B0DataType) * K / 2 * N +
                            sizeof(B1DataType) * K * N + sizeof(EDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    e_device_buf.FromDevice(e_m_n_device_result.mData.data());

    if(do_verification)
    {
        Tensor<CShuffleDataType> c_m_n({M, N});

        Tensor<ADataType> b_k_n({K, N});
        ck::vector_type_maker_t<ADataType, 2> tmp_out;


        for(int n = 0; n < N; ++n)
        {
            for(int k = 0; k < K; k += 2)
            {
                b_element_op(
                    reinterpret_cast<ck::vector_type_maker_t<ADataType, 2>::type&>(tmp_out),
                    *reinterpret_cast<ck::vector_type_maker_t<B0DataType, 2>::type*>(
                        &b0_k_n(k / 2, n)),
                    *reinterpret_cast<ck::vector_type_maker_t<B1DataType, 2>::type*>(
                        &b1_k_n(k, n)));
                b_k_n(k, n)     = tmp_out.AsType<ADataType>()(ck::Number<0>{});
                b_k_n(k + 1, n) = tmp_out.AsType<ADataType>()(ck::Number<1>{});
            }
        }

        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<F16,
                                                                                F16,
                                                                                CShuffleDataType,
                                                                                AccDataType,
                                                                                PassThrough,
                                                                                PassThrough,
                                                                                PassThrough>;
        auto ref_gemm               = ReferenceGemmInstance{};
        auto ref_invoker            = ref_gemm.MakeInvoker();

        auto ref_argument =
            ref_gemm.MakeArgument(a_m_k, b_k_n, c_m_n, PassThrough{}, PassThrough{}, PassThrough{});

        ref_invoker.Run(ref_argument);

        for(int m = 0; m < M; ++m)
        {
            for(int n = 0; n < N; ++n)
            {
                cde_element_op(e_m_n_host_result(m, n), c_m_n(m, n));
            }
        }

        e_device_buf.FromDevice(e_m_n_device_result.mData.data());

        if(ck::utils::check_err(
               e_m_n_device_result, e_m_n_host_result, "Error: Incorrect results! ", 0, -1))
        {
            printf("verification passed!");
            return 0;
        }
        return 1;
    }

    return 0;
}
