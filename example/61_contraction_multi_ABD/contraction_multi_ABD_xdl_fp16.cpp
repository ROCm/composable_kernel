// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_contraction_multiple_abd_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType        = F16;
using BDataType        = F16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using DDataType        = F16;
using EDataType        = F16;

static constexpr ck::index_t NumDimM = 2;
static constexpr ck::index_t NumDimN = 2;
static constexpr ck::index_t NumDimK = 2;

//struct AddScale
//{
    //static constexpr auto I0 = ck::Number<0>{};
    //static constexpr auto I1 = ck::Number<1>{};
    //static constexpr auto I2 = ck::Number<2>{};
    //static constexpr auto I3 = ck::Number<3>{};

    //__host__ __device__ constexpr void
    //operator()(ck::half4_t& a, const ck::half4_t& a0, const ck::half4_t& a1) const
    //{
        //const auto a0_v_t = ck::vector_type<ck::half_t, 4>{a0};
        //const auto a1_v_t = ck::vector_type<ck::half_t, 4>{a1};

        //auto r_v_t = ck::vector_type<ck::half_t, 4>{};

        //r_v_t.AsType<ck::half_t>()(I0) =
            //scale * (a0_v_t.AsType<ck::half_t>()[I0] + a1_v_t.AsType<ck::half_t>()[I0]);
        //r_v_t.AsType<ck::half_t>()(I1) =
            //scale * (a0_v_t.AsType<ck::half_t>()[I1] + a1_v_t.AsType<ck::half_t>()[I1]);
        //r_v_t.AsType<ck::half_t>()(I2) =
            //scale * (a0_v_t.AsType<ck::half_t>()[I2] + a1_v_t.AsType<ck::half_t>()[I2]);
        //r_v_t.AsType<ck::half_t>()(I3) =
            //scale * (a0_v_t.AsType<ck::half_t>()[I3] + a1_v_t.AsType<ck::half_t>()[I3]);

        //a = r_v_t.AsType<ck::half4_t>()[I0];
    //}

    //__host__ __device__ constexpr void
    //operator()(ck::half_t& a, const ck::half_t& a0, const ck::half_t& a1) const
    //{
        //a = scale * (a0 + a1);
    //}

    //static constexpr ck::index_t vec_len = 4;

    //float scale = 1.0;
//};

struct AlphaBetaAdd
{
    AlphaBetaAdd(float alpha, float beta) : alpha_(alpha), beta_(beta){};

    template <typename E, typename C, typename D>
    __host__ __device__ constexpr void operator()(E& e, const C& c, const D& d) const;

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, float, ck::half_t>(
        ck::half_t& e, const float& c, const ck::half_t& d) const
    {
        e = ck::type_convert<ck::half_t>(alpha_ * c + beta_ * ck::type_convert<float>(d));
    };

    float alpha_;
    float beta_;
};

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = AlphaBetaAdd;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

using DeviceOpInstance = ck::tensor_operation::device::DeviceContractionMultipleABD_Xdl_CShuffle<
NumDimM,
    NumDimN,
    NumDimK,
    ck::Tuple<ADataType>,
    ck::Tuple<BDataType>,
    AccDataType,
    CShuffleDataType,
    ck::Tuple<DDataType>,
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

    //// GEMM shape
    //ck::index_t M = 3840;
    //ck::index_t N = 4096;
    //ck::index_t K = 4096;

    //ck::index_t StrideA = 4096;
    //ck::index_t StrideB = 4096;
    //ck::index_t StrideD = 4096;
    //ck::index_t StrideE = 4096;

    float alpha = 1.0f;
    float beta  = 1.0f;

    // A[M0, M1, K0, K1]
    std::vector<ck::index_t> a_ms_ks_lengths{30, 128, 32, 64};
    std::vector<ck::index_t> a_ms_ks_strides{524288, 4096, 128, 1};
    // B[N0, N1, K0, K1]
    std::vector<ck::index_t> b_ns_ks_lengths{32, 64, 32, 64};
    std::vector<ck::index_t> b_ns_ks_strides{524288, 4096, 128, 1};
    // D[M0, M1, N0, N1]
    std::vector<ck::index_t> d_ms_ns_lengths{30, 128, 32, 64};
    std::vector<ck::index_t> d_ms_ns_strides{524288, 4096, 128, 1};
    // E[M0, M1, N0, N1]
    std::vector<ck::index_t> e_ms_ns_lengths{30, 128, 32, 64};
    std::vector<ck::index_t> e_ms_ns_strides{524288, 4096, 128, 1};

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
    //else if(argc == 6)
    //{
        //do_verification = std::stoi(argv[1]);
        //init_method     = std::stoi(argv[2]);
        //time_kernel     = std::stoi(argv[3]);

        //alpha = std::stof(argv[4]);
        //beta  = std::stof(argv[5]);
    //}
    //else if(argc == 13)
    //{
        //do_verification = std::stoi(argv[1]);
        //init_method     = std::stoi(argv[2]);
        //time_kernel     = std::stoi(argv[3]);

        //M = std::stoi(argv[4]);
        //N = std::stoi(argv[5]);
        //K = std::stoi(argv[6]);

        //StrideA = std::stoi(argv[7]);
        //StrideB = std::stoi(argv[8]);
        //StrideD = std::stoi(argv[9]);
        //StrideE = std::stoi(argv[10]);

        //alpha = std::stof(argv[11]);
        //beta  = std::stof(argv[12]);
    //}
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideD, StrideE, alpha, "
               "beta\n");
        exit(0);
    }

    //auto f_host_tensor_descriptor =
        //[](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            //using namespace ck::literals;

            //if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
            //{
                //return HostTensorDescriptor({row, col}, {stride, 1_uz});
            //}
            //else
            //{
                //return HostTensorDescriptor({row, col}, {1_uz, stride});
            //}
        //};

    Tensor<ADataType> a_ms_ks(a_ms_ks_lengths, a_ms_ks_strides);
    Tensor<BDataType> b_ns_ks(b_ns_ks_lengths, b_ns_ks_strides);
    Tensor<EDataType> d_ms_ns(d_ms_ns_lengths, d_ms_ns_strides);
    Tensor<EDataType> e_ms_ns_host_result(e_ms_ns_lengths, e_ms_ns_strides);
    Tensor<EDataType> e_ms_ns_device_result(e_ms_ns_lengths, e_ms_ns_strides);

    std::cout << "a_ms_ks: " << a_ms_ks.mDesc << std::endl;
    std::cout << "b_ns_ks: " << b_ns_ks.mDesc << std::endl;
    std::cout << "d_ms_ns: " << d_ms_ns.mDesc << std::endl;
    std::cout << "e_ms_ns: " << e_ms_ns_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b_ns_ks.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        d_ms_ns.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        break;
    default:
        a_ms_ks.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_ns_ks.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        d_ms_ns.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        break;
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_ms_ks.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_ns_ks.mDesc.GetElementSpaceSize());
    DeviceMem d_device_buf(sizeof(DDataType) * d_ms_ns.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) * e_ms_ns_device_result.mDesc.GetElementSpaceSize());
    //Tensor<ADataType> a0_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    //Tensor<ADataType> a1_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    //Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
    //Tensor<DDataType> d_m_n(f_host_tensor_descriptor(M, N, StrideD, DLayout{}));
    //Tensor<EDataType> e_m_n_host_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));
    //Tensor<EDataType> e_m_n_device_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));

    //std::cout << "a0_m_k: " << a0_m_k.mDesc << std::endl;
    //std::cout << "a1_m_k: " << a1_m_k.mDesc << std::endl;
    //std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    //std::cout << "d_m_n: " << d_m_n.mDesc << std::endl;
    //std::cout << "e_m_n: " << e_m_n_host_result.mDesc << std::endl;

    a_device_buf.ToDevice(a_ms_ks.mData.data());
    b_device_buf.ToDevice(b_ns_ks.mData.data());
    d_device_buf.ToDevice(d_ms_ns.mData.data());

    // set zero
    e_device_buf.SetZero();

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{alpha, beta};

    // do GEMM
    auto device_op = DeviceOpInstance{};
    auto invoker   = device_op.MakeInvoker();
    auto argument =
        device_op.MakeArgument(std::array<const void*, 1>{a_device_buf.GetDeviceBuffer()},
                               std::array<const void*, 1>{b_device_buf.GetDeviceBuffer()},
                               std::array<const void*, 1>{d_device_buf.GetDeviceBuffer()},
                               e_device_buf.GetDeviceBuffer(),
                               std::array<std::vector<ck::index_t>, 1>{a_ms_ks_lengths},
                               std::array<std::vector<ck::index_t>, 1>{a_ms_ks_strides},
                               std::array<std::vector<ck::index_t>, 1>{b_ns_ks_lengths},
                               std::array<std::vector<ck::index_t>, 1>{b_ns_ks_strides},
                               std::array<std::vector<ck::index_t>, 1>{d_ms_ns_lengths},
                               std::array<std::vector<ck::index_t>, 1>{d_ms_ns_strides},
                               e_ms_ns_lengths,
                               e_ms_ns_strides,
                               a_element_op,
                               b_element_op,
                               cde_element_op);

    if(!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    invoker.Run(argument, StreamConfig{nullptr, time_kernel});
    //float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    //std::size_t flop = std::size_t(2) * M * N * K;
    //std::size_t num_btype =
        //sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(EDataType) * M * N;

    //float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    //float gb_per_sec = num_btype / 1.E6 / ave_time;

    //std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              //<< std::endl;

              //e_device_buf.FromDevice(e_m_n_device_result.mData.data());

    if(do_verification)
    {
#if 0
        Tensor<CShuffleDataType> c_m_n({M, N});

        Tensor<ADataType> a_m_k({M, K});

        for(int m = 0; m < M; ++m)
        {
            for(int k = 0; k < K; ++k)
            {
                a_element_op(a_m_k(m, k), a0_m_k(m, k), a1_m_k(m, k));
            }
        }

        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                                BDataType,
                                                                                CShuffleDataType,
                                                                                AccDataType,
                                                                                PassThrough,
                                                                                BElementOp,
                                                                                PassThrough>;
        auto ref_gemm               = ReferenceGemmInstance{};
        auto ref_invoker            = ref_gemm.MakeInvoker();

        auto ref_argument =
            ref_gemm.MakeArgument(a_m_k, b_k_n, c_m_n, PassThrough{}, b_element_op, PassThrough{});

        ref_invoker.Run(ref_argument);

        for(int m = 0; m < M; ++m)
        {
            for(int n = 0; n < N; ++n)
            {
                cde_element_op(e_m_n_host_result(m, n), c_m_n(m, n), d_m_n(m, n));
            }
        }

        e_device_buf.FromDevice(e_m_n_device_result.mData.data());

        return ck::utils::check_err(e_m_n_device_result, e_m_n_host_result) ? 0 : 1;
#endif
    }

    return 0;
}
