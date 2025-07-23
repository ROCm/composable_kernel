// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include <type_traits>

template <typename T>
constexpr const char* DataTypeToString()
{
    if constexpr(std::is_same_v<T, ck_tile::half_t>)
    {
        return "fp16";
    }
    else if constexpr(std::is_same_v<T, ck_tile::fp8_t>)
    {
        return "fp8";
    }
    else if constexpr(std::is_same_v<T, ck_tile::bf8_t>)
    {
        return "bf8";
    }
    else if constexpr(std::is_same_v<T, ck_tile::bf16_t>)
    {
        return "bf16";
    }
    else
    {
        return "unknown";
    }
}

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

// mfma_type, 0:32x32, 1:16x16
template <typename FlatmmConfig, typename T>
auto shuffle_b(const ck_tile::HostTensor<T>& t)
{
    assert(t.get_lengths().size() == 2);
    int n_                = t.get_lengths()[1];
    int k_                = t.get_lengths()[0];
    constexpr int divisor = FlatmmConfig::N_Warp_Tile == 32 ? 2 : 4;
    ck_tile::HostTensor<T> t_view({n_ / FlatmmConfig::N_Warp_Tile,
                                   FlatmmConfig::N_Warp_Tile,
                                   k_ / FlatmmConfig::K_Warp_Tile,
                                   divisor,
                                   FlatmmConfig::K_Warp_Tile / divisor});
    std::copy(t.begin(), t.end(), t_view.begin());
    return ck_tile::reference_permute(t_view, {0, 2, 3, 1, 4});
}

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    // Calculate thresholds
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
    // Calculate error due to split_k accumulation
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
        max_accumulated_value, kbatch);
    // Use higher threshold
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

template <typename FlatmmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDatatype,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          bool persistent,
          typename CDEElementWise>
float flatmm_calc(const ck_tile::FlatmmHostArgs<>& args, const ck_tile::stream_config& s);

template <typename FlatmmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDatatype,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename CDEElementWise = ck_tile::element_wise::PassThrough>
float invoke_flatmm(ck_tile::DeviceMem& a_dev_buf,
                    ck_tile::DeviceMem& b_shuffle_dev_buf,
                    ck_tile::DeviceMem& c_dev_buf,
                    ck_tile::index_t M,
                    ck_tile::index_t N,
                    ck_tile::index_t K,
                    ck_tile::index_t stride_A,
                    ck_tile::index_t stride_B,
                    ck_tile::index_t stride_C,
                    ck_tile::index_t kbatch,
                    int n_warmup,
                    int n_repeat)
{
    ck_tile::FlatmmHostArgs<> args = {a_dev_buf.GetDeviceBuffer(),
                                      b_shuffle_dev_buf.GetDeviceBuffer(),
                                      {},
                                      c_dev_buf.GetDeviceBuffer(),
                                      kbatch,
                                      M,
                                      N,
                                      K,
                                      stride_A,
                                      stride_B,
                                      {},
                                      stride_C};

    float ave_time = flatmm_calc<FlatmmConfig,
                                 ADataType,
                                 BDataType,
                                 DsDatatype,
                                 AccDataType,
                                 CDataType,
                                 ALayout,
                                 BLayout,
                                 DsLayout,
                                 CLayout,
                                 false,
                                 CDEElementWise>(
        args, ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat, true, true, 50});

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_byte =
        sizeof(ADataType) * M * K + sizeof(BDataType) * N * K + sizeof(CDataType) * M * N;
    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Run Flatmm kernel with DataType = " << DataTypeToString<ADataType>()
              << " M =" << M << " N =" << N << " K =" << K << " StrideA =" << stride_A
              << " StrideB =" << stride_B << " StrideC =" << stride_C << " : " << ave_time
              << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, " << std::endl;

    return ave_time;
}

template <typename PrecType,
          typename FlatmmConfig,
          typename ALayout,
          typename BLayout,
          typename CLayout>
int run_flatmm_example_with_layouts(int argc,
                                    char* argv[],
                                    const ALayout a_layout                  = ALayout{},
                                    const BLayout b_layout                  = BLayout{},
                                    [[maybe_unused]] const CLayout c_layout = CLayout{})
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    using ADataType   = typename GemmBasicTypeConfig<PrecType>::ADataType;
    using BDataType   = typename GemmBasicTypeConfig<PrecType>::BDataType;
    using CDataType   = typename GemmBasicTypeConfig<PrecType>::CDataType;
    using AccDataType = typename GemmBasicTypeConfig<PrecType>::AccDataType;

    ck_tile::index_t M = arg_parser.get_int("m");
    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t K = arg_parser.get_int("k");

    ck_tile::index_t stride_A = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_B = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_C = arg_parser.get_int("stride_c");

    ck_tile::index_t kbatch      = arg_parser.get_int("split_k");
    int n_warmup                 = arg_parser.get_int("warmup");
    int n_repeat                 = arg_parser.get_int("repeat");
    ck_tile::index_t init_method = arg_parser.get_int("init");
    // persistent not added

    stride_A = ck_tile::get_default_stride(M, K, stride_A, is_row_major(a_layout));
    stride_B = ck_tile::get_default_stride(K, N, stride_B, is_row_major(b_layout));
    stride_C = ck_tile::get_default_stride(M, N, stride_C, is_row_major(CLayout{}));

    ck_tile::HostTensor<ADataType> a_host(
        ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(a_layout)));
    ck_tile::HostTensor<BDataType> b_origin_host(
        ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(b_layout)));
    ck_tile::HostTensor<CDataType> c_rslt_host(
        ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

    // TODO: add different init types
    if(init_method == 0)
    {
        ck_tile::FillUniformDistribution<ADataType>{-.5f, .5f}(a_host);
        ck_tile::FillUniformDistribution<BDataType>{-.5f, .5f}(b_origin_host);
    }
    else if(init_method == 1)
    {
        ck_tile::FillMonotonicSeq<ADataType>{}(a_host);
        ck_tile::FillMonotonicSeq<BDataType>{}(b_origin_host);
    }
    else if(init_method == 2)
    {
        ck_tile::FillUniformDistribution<ADataType>{1.f, 1.f}(a_host);
        ck_tile::FillUniformDistribution<BDataType>{1.f, 1.f}(b_origin_host);
    }
    else
    {
        a_host.SetZero();
        b_origin_host.SetZero();
    }

    ck_tile::DeviceMem a_dev_buf(a_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_dev_buf(c_rslt_host.get_element_space_size_in_bytes());

    a_dev_buf.ToDevice(a_host.data());
    c_rslt_host.SetZero();

    // do pre-shuffle
    ck_tile::HostTensor<BDataType> b_shuffle_host = shuffle_b<FlatmmConfig>(b_origin_host);
    ck_tile::DeviceMem b_shuffle_dev_buf(b_shuffle_host.get_element_space_size_in_bytes());
    b_shuffle_dev_buf.ToDevice(b_shuffle_host.data());

    invoke_flatmm<FlatmmConfig,
                  ADataType,
                  BDataType,
                  ck_tile::tuple<>,
                  AccDataType,
                  CDataType,
                  ALayout,
                  BLayout,
                  ck_tile::tuple<>,
                  CLayout>(a_dev_buf,
                           b_shuffle_dev_buf,
                           c_dev_buf,
                           M,
                           N,
                           K,
                           stride_A,
                           stride_B,
                           stride_C,
                           kbatch,
                           n_warmup,
                           n_repeat);

    c_dev_buf.FromDevice(c_rslt_host.data());
    bool pass = true;

    if(arg_parser.get_int("v") == 1)
    {
        ck_tile::HostTensor<CDataType> c_ref_host(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        c_ref_host.SetZero();

        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_host, b_origin_host, c_ref_host);
        const float max_accumulated_value =
            *std::max_element(c_ref_host.mData.begin(), c_ref_host.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, kbatch, max_accumulated_value);
        pass = ck_tile::check_err(c_rslt_host,
                                  c_ref_host,
                                  "Error: Incorrect results!",
                                  rtol_atol.at(ck_tile::number<0>{}),
                                  rtol_atol.at(ck_tile::number<1>{}));

        std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                  << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                  << std::endl;
        std::cout << "The CPU veification result is:" << (pass ? "correct" : "fail") << std::endl;
    }
    else if(arg_parser.get_int("v") == 2)
    {
        ck_tile::DeviceMem b_origin_dev_buf(b_origin_host.get_element_space_size_in_bytes());
        b_origin_dev_buf.ToDevice(b_origin_host.data());

        ck_tile::HostTensor<CDataType> c_gpu_ref_host(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        ck_tile::DeviceMem c_gpu_ref_dev_buf(c_gpu_ref_host.get_element_space_size_in_bytes());
        c_gpu_ref_host.SetZero();
        c_gpu_ref_dev_buf.SetZero();

        ADataType* d_A;
        BDataType* d_B;
        CDataType* d_C;

        ck_tile::hip_check_error(hipMalloc(&d_A, M * K * sizeof(ADataType)));
        ck_tile::hip_check_error(hipMalloc(&d_B, N * K * sizeof(BDataType)));
        ck_tile::hip_check_error(hipMalloc(&d_C, M * N * sizeof(CDataType)));

        ck_tile::hip_check_error(hipMemcpy(
            d_A, a_dev_buf.GetDeviceBuffer(), M * K * sizeof(ADataType), hipMemcpyHostToDevice));
        ck_tile::hip_check_error(hipMemcpy(d_B,
                                           b_origin_dev_buf.GetDeviceBuffer(),
                                           N * K * sizeof(BDataType),
                                           hipMemcpyHostToDevice));

        ck_tile::reference_gemm_gpu<ADataType,
                                    BDataType,
                                    AccDataType,
                                    CDataType,
                                    ALayout,
                                    BLayout,
                                    CLayout>(d_A, d_B, d_C, M, N, K, stride_A, stride_B, stride_C);

        ck_tile::hip_check_error(hipMemcpy(c_gpu_ref_dev_buf.GetDeviceBuffer(),
                                           d_C,
                                           M * N * sizeof(CDataType),
                                           hipMemcpyDeviceToHost));

        ck_tile::hip_check_error(hipFree(d_A));
        ck_tile::hip_check_error(hipFree(d_B));
        ck_tile::hip_check_error(hipFree(d_C));

        c_gpu_ref_dev_buf.FromDevice(c_gpu_ref_host.data());
        const float max_accumulated_value =
            *std::max_element(c_gpu_ref_host.mData.begin(), c_gpu_ref_host.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, kbatch, max_accumulated_value);
        pass = ck_tile::check_err(c_rslt_host,
                                  c_gpu_ref_host,
                                  "Error: Incorrect results!",
                                  rtol_atol.at(ck_tile::number<0>{}),
                                  rtol_atol.at(ck_tile::number<1>{}));

        std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                  << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                  << std::endl;
        std::cout << "The GPU veification result is: " << (pass ? "correct" : "fail") << std::endl;
    }

    return pass;
}
