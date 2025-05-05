
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_b_scale.hpp"

#include <vector>
#include <string>
#include <iostream>

std::vector<float> LoadTxtTensor(const std::string& path)
{
    std::ifstream file(path);
    std::vector<float> data;
    float val;
    while (file >> val)
        data.push_back(val);
    return data;
}


std::vector<ck::half_t> LoadFp16Binary(const std::string& path, size_t count) {
    std::vector<ck::half_t> data(count);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    in.read(reinterpret_cast<char*>(data.data()), count * sizeof(ck::half_t));
    return data;
}



std::vector<int8_t> LoadInt8Binary(const std::string& path, size_t count) {
    std::vector<int8_t> data(count);
    std::ifstream in(path, std::ios::binary);
    in.read(reinterpret_cast<char*>(data.data()), count * sizeof(int8_t));
    return data;
}


// std::vector<float> LoadTensorFromPt(const std::string& path) {
//     torch::Tensor tensor = torch::load(path).to(torch::kCPU);
//     tensor = tensor.contiguous();
//     auto* ptr = tensor.data_ptr<float>();
//     return std::vector<float>(ptr, ptr + tensor.numel());
// }


void PrintGemmParams(const void* A, const void* B, const void* C,
    int M, int N, int K,
    int StrideA, int StrideB, int StrideC,
    int ScaleStrideBN, const void* BScale, int KBatch)
{
std::cout << "[W8Only_Gemm_Debug] A_input = " << A << std::endl;
std::cout << "[W8Only_Gemm_Debug] B_input = " << B << std::endl;
std::cout << "[W8Only_Gemm_Debug] C_output = " << C << std::endl;
std::cout << "[W8Only_Gemm_Debug] M = " << M << std::endl;
std::cout << "[W8Only_Gemm_Debug] N = " << N << std::endl;
std::cout << "[W8Only_Gemm_Debug] K = " << K << std::endl;
std::cout << "[W8Only_Gemm_Debug] StrideA = " << StrideA << std::endl;
std::cout << "[W8Only_Gemm_Debug] StrideB = " << StrideB << std::endl;
std::cout << "[W8Only_Gemm_Debug] StrideC = " << StrideC << std::endl;
std::cout << "[W8Only_Gemm_Debug] Scale_Stride_BN = " << ScaleStrideBN << std::endl;
std::cout << "[W8Only_Gemm_Debug] B_scale = " << BScale << std::endl;
std::cout << "[W8Only_Gemm_Debug] KBatch = " << KBatch << std::endl;
}

using ADataType        = ck::half_t;
using BDataType        = int8_t;
using BScaleDataType   = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
using CDataType        = ck::half_t;

using ALayout = Row;
using BLayout = Row;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

static constexpr bool PermuteA = false;
static constexpr bool PermuteB = false;

static constexpr ck::index_t Scale_Block_N = 1;
static constexpr ck::index_t Scale_Block_K = 1024;

 

// clang-format off
using DeviceGemmV2Instance = 
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV3<
        ALayout,   BLayout,  CLayout,   
        ADataType, BDataType, BScaleDataType, CDataType, AccDataType, CShuffleDataType, 
        AElementOp, BElementOp, CElementOp, GemmDefault, 
        256, Scale_Block_N, Scale_Block_K,
        128, 128, 64, 
        8, 4,
        32,  32,
        2,   2,
        S<8, 32, 1>,  S<1, 0, 2>,  S<1, 0, 2>,
        2,   8,  8,  0,
        S<16, 16, 1>, S<0, 2, 1>,  S<0, 2, 1>,
        1,   8,  4,  0,
        1,   1,  S<1, 32, 1, 8>, 8,
        ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v4, CDataType, CDataType, PermuteA, PermuteB>;

// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                        AccDataType,
                                                                        CDataType,
                                                                        AccDataType,
                                                                        PassThrough,
                                                                        PassThrough,
                                                                        PassThrough>;
template <typename ProblemType>
bool run_gemm(const ProblemType& problem_size, const ExecutionConfig& config)
{
    using namespace ck::literals;

    auto M       = problem_size.M;
    auto N       = problem_size.N;
    auto K       = problem_size.K;
    auto StrideA = problem_size.StrideA;
    auto StrideB = problem_size.StrideB;
    auto StrideC = problem_size.StrideC;
    auto KBatch  = problem_size.KBatch;

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    auto f_get_default_stride =
        [](std::size_t row, std::size_t col, ck::index_t stride, auto layout) {
            if(stride == -1)
            {
                // give a chance if stride is -1, return a default packed stride
                if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
                {
                    return static_cast<std::size_t>(col);
                }
                else
                {
                    return static_cast<std::size_t>(row);
                }
            }
            else
                return static_cast<std::size_t>(stride);
        };

    // ck::index_t Scale_Stride_BN = (K + Scale_Block_K - 1) / Scale_Block_K;
    ck::index_t Scale_Stride_BN = 1;

    StrideA = f_get_default_stride(M, K, StrideA, ALayout{});
    StrideB = f_get_default_stride(K, N, StrideB, BLayout{});
    StrideC = f_get_default_stride(M, N, StrideC, CLayout{});

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
    Tensor<BDataType> b_k_n_permute(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
    Tensor<BScaleDataType> b1_k_n(f_host_tensor_descriptor((K + Scale_Block_K - 1) / Scale_Block_K,
                                                           (N + Scale_Block_N - 1) / Scale_Block_N,
                                                           Scale_Stride_BN,
                                                           BLayout{}));
    // Tensor<BScaleDataType> b1_k_n(f_host_tensor_descriptor((K + Scale_Block_K - 1) / Scale_Block_K,
    //                                                        (N + Scale_Block_N - 1) / Scale_Block_N,
    //                                                        Scale_Stride_BN,
    //                                                        ck::tensor_layout::gemm::RowMajor{}));
    // Tensor<BScaleDataType> b1_k_n(f_host_tensor_descriptor(1, 4096, 1, ck::tensor_layout::gemm::RowMajor{}));

    switch(config.init_method)
    {
    case 0:
        a_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        b_k_n.GenerateTensorValue(GeneratorTensor_1<BDataType>{1});
        b1_k_n.GenerateTensorValue(GeneratorTensor_1<BScaleDataType>{1});
        break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        b1_k_n.GenerateTensorValue(GeneratorTensor_3<BScaleDataType>{0, 1.0});
        break;
    case 2:
        a_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        b1_k_n.GenerateTensorValue(GeneratorTensor_1<BScaleDataType>{1});
        break;
    case 3:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_k_n.GenerateTensorValue(GeneratorTensor_1<BDataType>{1});
        b1_k_n.GenerateTensorValue(GeneratorTensor_1<BScaleDataType>{1});
        break;
    case 4:
        a_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        b_k_n.GenerateTensorValue(GeneratorTensor_1<BDataType>{1});
        b1_k_n.GenerateTensorValue(GeneratorTensor_3<BScaleDataType>{0, 1.0});
        break;
    case 5:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        b1_k_n.GenerateTensorValue(GeneratorTensor_1<BScaleDataType>{1});
        break;
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.5, 0.5});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        b1_k_n.GenerateTensorValue(GeneratorTensor_3<BScaleDataType>{0, 1.0});
    }

    Tensor<CDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "b1_k_n: " << b1_k_n.mDesc << std::endl;
     
    // std::string pt_dir = "/mnt/raid0/zhaoan12/letao_gemm_pt/";

    // std::vector<float> A_input_actual  = LoadTxtTensor(pt_dir + "A.txt");
    // std::vector<float> B_input_actual  = LoadTxtTensor(pt_dir + "B.txt");
    // std::vector<float> D0_input_actual = LoadTxtTensor(pt_dir + "scale.txt");
    

    // // 拷贝 A_input_actual (float → ck::half_t)
    // a_m_k.mData.resize(A_input_actual.size());
    // for (size_t i = 0; i < A_input_actual.size(); ++i) {
    //     a_m_k.mData[i] = ck::type_convert<ck::half_t>(A_input_actual[i]);
    // }

    // // 拷贝 B_input_actual (int → int8_t)
    // b_k_n.mData.resize(B_input_actual.size());
    // for (size_t i = 0; i < B_input_actual.size(); ++i) {
    //     b_k_n.mData[i] = static_cast<int8_t>(B_input_actual[i]);  
    // }

    // // 拷贝 D0_input_actual (float → ck::half_t)
    // b1_k_n.mData.resize(D0_input_actual.size());
    // for (size_t i = 0; i < D0_input_actual.size(); ++i) {
    //     b1_k_n.mData[i] = ck::type_convert<ck::half_t>(D0_input_actual[i]);
    // }


    // std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;

    // std::cout << "D0_input_actual.size:" << D0_input_actual.size() << std::endl;


    std::string pt_dir = "/mnt/raid0/zhaoan12/gemm_save_rocm/";

    std::vector<ck::half_t> A_input_actual  = LoadFp16Binary(pt_dir + "A_fp16.bin", M * K);
    std::vector<int8_t>     B_input_actual  = LoadInt8Binary(pt_dir + "B_int8.bin", K * N);
    std::vector<ck::half_t> D0_input_actual = LoadFp16Binary(pt_dir + "scale_fp16.bin", N);
        
    std::cout << "== Loaded A_fp16[0:10] ==" << std::endl;
    for (int i = 0; i < 10; ++i)
        std::cout << "A_input_actual[" << i << "] = " << ck::type_convert<float>(A_input_actual[i]) << std::endl;

    std::cout << "== Loaded B_int8[0:10] ==" << std::endl;
    for (int i = 0; i < 10; ++i)
        std::cout << "B_input_actual[" << i << "] = " << static_cast<int>(B_input_actual[i]) << std::endl;

    std::cout << "== Loaded scale_fp16[0:10] ==" << std::endl;
    for (int i = 0; i < 10; ++i)
        std::cout << "D0_input_actual[" << i << "] = " << ck::type_convert<float>(D0_input_actual[i]) << std::endl;


    // A: float → half
    a_m_k.mData.resize(A_input_actual.size());
    for (size_t i = 0; i < A_input_actual.size(); ++i)
        a_m_k.mData[i] = ck::type_convert<ck::half_t>(A_input_actual[i]);

    // B: float → int8
    b_k_n.mData.resize(B_input_actual.size());
    for (size_t i = 0; i < B_input_actual.size(); ++i)
        b_k_n.mData[i] = static_cast<int8_t>(std::round(B_input_actual[i]));

    // Scale: float → half
    b1_k_n.mData.resize(D0_input_actual.size());
    for (size_t i = 0; i < D0_input_actual.size(); ++i)
        b1_k_n.mData[i] = ck::type_convert<ck::half_t>(D0_input_actual[i]);



    std::cout << "== Converted A_fp16 (a_m_k.mData)[0:10] ==" << std::endl;
    for (size_t i = 0; i < 10; ++i)
        std::cout << "a_m_k.mData[" << i << "] = " << ck::type_convert<float>(a_m_k.mData[i]) << std::endl;

    std::cout << "== Converted B_int8 (b_k_n.mData)[0:10] ==" << std::endl;
    for (size_t i = 0; i < 10; ++i)
        std::cout << "b_k_n.mData[" << i << "] = " << static_cast<int>(b_k_n.mData[i]) << std::endl;

    std::cout << "== Converted scale_fp16 (b1_k_n.mData)[0:10] ==" << std::endl;
    for (size_t i = 0; i < 10; ++i)
        std::cout << "b1_k_n.mData[" << i << "] = " << ck::type_convert<float>(b1_k_n.mData[i]) << std::endl;

            
    DeviceMem a_m_k_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_device_buf(sizeof(BDataType) * b_k_n_permute.mDesc.GetElementSpaceSize());
    DeviceMem b1_scale_device_buf(sizeof(BScaleDataType) * b1_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpaceSize());

    // // weight permute
    // if constexpr(PermuteB)
    // {
    //     int K1 = KPerBlock;
    //     int K0 = K / KPerBlock;

    //     // int K0, N, K1
    //     for(int j = 0; j < K0; j++)
    //     {
    //         for(int i = 0; i < N; i++)
    //         {
    //             for(int jj = 0; jj < K1; jj++)
    //             {
    //                 b_k_n_permute(j * N * K1 + i * K1 + jj) = b_k_n(i * K + (j * K1 + jj));
    //             }
    //         }
    //     }
    // }
    // else
    // {
         for(int i = 0; i < N; i++)
         {
             for(int j = 0; j < K; j++)
             {
                 b_k_n_permute(i * K + j) = b_k_n(i * K + j);
             }
         }
    // } 

    a_m_k_device_buf.ToDevice(a_m_k.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n_permute.mData.data());
    b1_scale_device_buf.ToDevice(b1_k_n.mData.data());
    DeviceMem workspace;

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    // do GEMM
    auto gemm      = DeviceGemmV2Instance{};
    auto invoker   = gemm.MakeInvoker();
    float ave_time = 0;

#if !defined(__HIP_DEVICE_COMPILE__)
    std::cout << "[W8Only_Gemm_Debug] A_input = " << static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()) << std::endl;
    std::cout << "[W8Only_Gemm_Debug] B_input = " << static_cast<BDataType*>(b_k_n_device_buf.GetDeviceBuffer()) << std::endl;
    std::cout << "[W8Only_Gemm_Debug] C_output = " << static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()) << std::endl;
    std::cout << "[W8Only_Gemm_Debug] M = " << M << std::endl;
    std::cout << "[W8Only_Gemm_Debug] N = " << N << std::endl;
    std::cout << "[W8Only_Gemm_Debug] K = " << K << std::endl;
    std::cout << "[W8Only_Gemm_Debug] StrideA = " << StrideA << std::endl;
    std::cout << "[W8Only_Gemm_Debug] StrideB = " << StrideB << std::endl;
    std::cout << "[W8Only_Gemm_Debug] StrideC = " << StrideC << std::endl;
    std::cout << "[W8Only_Gemm_Debug] Scale_Stride_BN = " << Scale_Stride_BN << std::endl;
    std::cout << "[W8Only_Gemm_Debug] B_scale = " << static_cast<BScaleDataType*>(b1_scale_device_buf.GetDeviceBuffer()) << std::endl;
    std::cout << "[W8Only_Gemm_Debug] KBatch = " << KBatch << std::endl;
#endif



PrintGemmParams(
    static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
    static_cast<BDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
    static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
    M, N, K,
    StrideA, StrideB, StrideC,
    Scale_Stride_BN,
    static_cast<BScaleDataType*>(b1_scale_device_buf.GetDeviceBuffer()),
    KBatch);

    auto argument =
        gemm.MakeArgument(static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                          static_cast<BDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
                          static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
                          M,
                          N,
                          K,
                          StrideA,
                          StrideB,
                          StrideC,
                          Scale_Stride_BN,
                          static_cast<BScaleDataType*>(b1_scale_device_buf.GetDeviceBuffer()),
                          KBatch,
                          a_element_op,
                          b_element_op,
                          c_element_op);

    if(!gemm.IsSupportedArgument(argument))
    {
        std::cerr << gemm.GetTypeString() << " does not support this problem" << std::endl;

        return true;
    }

    bool pass = true;
    if(config.do_verification)
    {
        Tensor<float> b_k_n_dequant({K, N});

        float v_b = 0;
        for(int n = 0; n < N; n++)
        {
            for(int k = 0; k < K; k++)
            {
                v_b = ck::type_convert<float>(b_k_n(k, n));

                b_k_n_dequant(k, n) =
                    ck::type_convert<float>(v_b) *
                    ck::type_convert<float>(b1_k_n(k / Scale_Block_K, n / Scale_Block_N));
            }
        }

        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n_dequant, c_m_n_host_result, PassThrough{}, PassThrough{}, PassThrough{});

        ref_invoker.Run(ref_argument);

        ave_time = invoker.Run(argument, StreamConfig{nullptr, false, 0});
        c_m_n_device_buf.FromDevice(c_m_n_device_result.mData.data());

        pass &= ck::utils::check_err(c_m_n_device_result,
                                     c_m_n_host_result,
                                     "Error: Incorrect results!",
                                     get_rtol<CDataType>(),
                                     get_atol<CDataType>());

        std::cout << "\n== First 10 values of b_k_n_dequant ==" << std::endl;
        for(int i = 0; i < 15; ++i)
        {
            std::cout << "b_k_n_dequant[" << i << "] = " <<  ck::type_convert<float>(b_k_n_dequant.mData[i]) << std::endl;
        }
                            
        std::cout << "\n== First 10 values of loaded scale ==\n";
        for (int i = 0; i < 10; ++i) {
            std::cout << "b1_k_n(0, " << i << ") = " << ck::type_convert<float>(b1_k_n(0, i)) << std::endl;
        }


        for(int i = 0; i < 10; i++)
        {
            std::cout << "data[" << i << "]: " << ck::type_convert<float>(c_m_n_device_result.mData[i]) << std::endl;
        }


        std::cout << "\n== CK: Checking b_k_n(k,n), scale, and dequant ==" << std::endl;
        std::vector<std::pair<int, int>> check_indices = {
            {0, 0}, {1, 0}, {0, 1}, {1, 1}, {1023, 4095}, {100, 2000}
        };

        for (const auto& [k, n] : check_indices) {
            v_b = ck::type_convert<float>(b_k_n(k, n));
            float v_scale = ck::type_convert<float>(b1_k_n(k / 1024, n));
            float v_dequant = ck::type_convert<float>(b_k_n_dequant(k, n));
            std::cout << "b_k_n(" << k << "," << n << ") = " << v_b
                    << ", scale = " << v_scale
                    << ", dequant = " << v_dequant << std::endl;
        }

    }

    if(config.time_kernel)
    {
        ave_time =
            invoker.Run(argument, StreamConfig{nullptr, config.time_kernel, 0, 20, 50, true, 50});

        std::size_t flop = 2_uz * M * N * K;
        std::size_t num_btype =
            sizeof(ADataType) * M * K +
            sizeof(BDataType) * K * N /
                (ck::is_same_v<ck::remove_cvref_t<BDataType>, ck::pk_i4_t> ? 2 : 1) +
            sizeof(CDataType) * M * N;

        float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

        float gb_per_sec = num_btype / 1.E6 / ave_time;

        std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s, " << gemm.GetTypeString() << std::endl;
    }
    return pass;
}

bool run_gemm_splitk_example(int argc, char* argv[])
{
    ProblemSizeSplitK problem_size;
    ExecutionConfig config;

    // problem_size.M = 8;
    // problem_size.N = 3072;
    // problem_size.K = 1024;

    problem_size.M = 8;
    problem_size.N = 1024;
    problem_size.K = 4096;

    config.do_verification = true;
    config.init_method     = 1;
    config.time_kernel     = true;

    return !parse_cmd_args(argc, argv, problem_size, config) || run_gemm(problem_size, config);
}

int main(int argc, char* argv[]) { return !run_gemm_splitk_example(argc, argv); }
