// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include <random>
#include <stdexcept>
#include "ck_tile/host/permute_pk_int4.hpp"

template <typename T>
auto shuffle_aq(const ck_tile::HostTensor<T>* t, int block_aq_k)
{
    if(t->get_lengths().size() != 2)
    {
        throw std::runtime_error("Host tensor is not rank 2 tensor.");
    }
    int m_   = t->get_lengths()[0];
    int aqk_ = t->get_lengths()[1];
    if(aqk_ % block_aq_k != 0)
    {
        throw std::runtime_error("shuffle_aq needs a aqk of multiple times of block_aq_k.");
    }
    ck_tile::HostTensor<T> t_view({m_, aqk_ / block_aq_k, block_aq_k});
    std::copy(t->begin(), t->end(), t_view.begin());
    return ck_tile::reference_permute(t_view, {1, 0, 2});
}

template <typename GemmConfig, typename T>
auto shuffle_b(const ck_tile::HostTensor<T>& t)
{
    assert(t.get_lengths().size() == 2);
    int n_                = t.get_lengths()[1];
    int k_                = t.get_lengths()[0];
    constexpr int divisor = GemmConfig::N_Warp_Tile == 32 ? 2 : 4;
    ck_tile::HostTensor<T> t_view({n_ / GemmConfig::N_Warp_Tile,
                                   GemmConfig::N_Warp_Tile,
                                   k_ / GemmConfig::K_Warp_Tile,
                                   divisor,
                                   GemmConfig::K_Warp_Tile / divisor});
    std::copy(t.begin(), t.end(), t_view.begin());
    return ck_tile::reference_permute(t_view, {0, 2, 3, 1, 4});
}

template <typename GemmConfig,
          typename TypeConfig,
          typename ALayout,
          typename AQLayout,
          typename BLayout,
          typename BQLayout,
          typename CLayout,
          uint32_t QuantGroupSize,
          ck_tile::QuantType QuantMode,
          typename CDEElementWise = ck_tile::element_wise::PassThrough>
float invoke_gemm(ck_tile::DeviceMem& a_m_k_dev_buf,
                  ck_tile::DeviceMem* aq_dev_buf,
                  ck_tile::DeviceMem& b_k_n_dev_buf,
                  ck_tile::DeviceMem* bq_dev_buf,
                  ck_tile::DeviceMem& c_m_n_dev_buf,
                  ck_tile::index_t M,
                  ck_tile::index_t N,
                  ck_tile::index_t K,
                  ck_tile::index_t AQK,
                  ck_tile::index_t BQK,
                  ck_tile::index_t stride_A,
                  ck_tile::index_t stride_AQ,
                  ck_tile::index_t stride_B,
                  ck_tile::index_t stride_BQ,
                  ck_tile::index_t stride_C,
                  ck_tile::index_t kbatch,
                  int n_warmup,
                  int n_repeat,
                  bool flush_cache,
                  int rotating_count)
{
    ck_tile::QuantGemmHostArgs args;
    args.a_ptr     = a_m_k_dev_buf.GetDeviceBuffer();
    args.aq_ptr    = (aq_dev_buf != nullptr) ? aq_dev_buf->GetDeviceBuffer() : nullptr;
    args.b_ptr     = b_k_n_dev_buf.GetDeviceBuffer();
    args.bq_ptr    = (bq_dev_buf != nullptr) ? bq_dev_buf->GetDeviceBuffer() : nullptr;
    args.c_ptr     = c_m_n_dev_buf.GetDeviceBuffer();
    args.k_batch   = kbatch;
    args.M         = M;
    args.N         = N;
    args.K         = K;
    args.QK_A      = AQK;
    args.QK_B      = BQK;
    args.stride_A  = stride_A;
    args.stride_B  = stride_B;
    args.stride_C  = stride_C;
    args.stride_AQ = stride_AQ;
    args.stride_BQ = stride_BQ;

    float ave_time = gemm_calc_quant<GemmConfig,
                                     TypeConfig,
                                     ALayout,
                                     BLayout,
                                     CLayout,
                                     QuantGroupSize,
                                     QuantMode,
                                     CDEElementWise>(
        args,
        ck_tile::stream_config{
            nullptr, true, 1, n_warmup, n_repeat, true, flush_cache, rotating_count});

    std::size_t flop     = std::size_t(2) * M * N * K;
    std::size_t num_byte = sizeof(typename TypeConfig::ADataType) * M * K +
                           sizeof(typename TypeConfig::BDataType) * N * K +
                           sizeof(typename TypeConfig::CDataType) * M * N;
    if(aq_dev_buf != nullptr)
    {
        num_byte += sizeof(typename TypeConfig::QDataType) * M * AQK;
    }
    if(bq_dev_buf != nullptr)
    {
        num_byte += sizeof(typename TypeConfig::QDataType) * N * BQK;
    }

    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Run Gemm kernel with M =" << M << " N =" << N << " K =" << K
              << " StrideA =" << stride_A << " StrideAQ =" << stride_AQ << " StrideB =" << stride_B
              << " StrideC =" << stride_C << " A_Layout =" << ALayout::name
              << " B_Layout =" << BLayout::name << " C_Layout =" << CLayout::name;
    if constexpr(QuantMode == ck_tile::QuantType::BQuantGrouped ||
                 QuantMode == ck_tile::QuantType::RowColQuant)
    {
        std::cout << " StrideBQ =" << stride_BQ;
    }
    std::cout << " A_Type = " << DataTypeTraits<typename TypeConfig::ADataType>::name
              << " AQ_Type = " << DataTypeTraits<typename TypeConfig::QDataType>::name
              << " B_Type = " << DataTypeTraits<typename TypeConfig::BDataType>::name;
    if constexpr(!std::is_same_v<typename TypeConfig::QDataType, void>)
    {
        std::cout << " BQ_Type = " << DataTypeTraits<typename TypeConfig::QDataType>::name;
    }
    std::cout << " Acc_Type = " << DataTypeTraits<typename TypeConfig::AccDataType>::name
              << " C_Type = " << DataTypeTraits<typename TypeConfig::CDataType>::name
              << " QuantMode = " << quant_type_to_string(QuantMode)
              << " PreshuffleQuant = " << (GemmConfig::PreshuffleQuant ? "true" : "false") << " : "
              << " PreshuffleB = " << (GemmConfig::PreshuffleB ? "true" : "false") << " : "
              << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << std::endl;

    return ave_time;
}

template <typename GemmConfig,
          typename TypeConfig,
          uint32_t QuantGroupSize,
          ck_tile::QuantType QuantMode,
          typename ALayout,
          typename AQLayout,
          typename BLayout,
          typename BQLayout,
          typename CLayout>
int run_gemm_example_with_layouts(int argc,
                                  char* argv[],
                                  const ALayout a_layout                  = ALayout{},
                                  const AQLayout aq_layout                = AQLayout{},
                                  const BLayout b_layout                  = BLayout{},
                                  const BQLayout bq_layout                = BQLayout{},
                                  [[maybe_unused]] const CLayout c_layout = CLayout{})
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    using ADataType   = typename TypeConfig::ADataType;
    using AQDataType  = typename TypeConfig::QDataType;
    using BDataType   = typename TypeConfig::BDataType;
    using BQDataType  = typename TypeConfig::QDataType;
    using AccDataType = typename TypeConfig::AccDataType;
    using CDataType   = typename TypeConfig::CDataType;

    ck_tile::index_t M = arg_parser.get_int("m");
    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t K = arg_parser.get_int("k");

    if constexpr(QuantMode == ck_tile::QuantType::AQuantGrouped ||
                 QuantMode == ck_tile::QuantType::BQuantGrouped)
    {
        if(K % QuantGroupSize != 0)
        {
            throw std::runtime_error(
                "K must be aligned with QuantGroupSize for AQuantGrouped/BQuantGrouped mode");
        }
    }
    ck_tile::index_t AQK, BQK;
    if constexpr(QuantMode == ck_tile::QuantType::AQuantGrouped)
    {
        AQK = K / QuantGroupSize; // Group quantization: AQK = K / GroupSize
        BQK = 0;                  // No B quantization
    }
    else if constexpr(QuantMode == ck_tile::QuantType::BQuantGrouped)
    {
        AQK = 0;                  // No A quantization
        BQK = K / QuantGroupSize; // Group quantization: BQK = K / GroupSize
    }
    else if constexpr(QuantMode == ck_tile::QuantType::RowColQuant ||
                      QuantMode == ck_tile::QuantType::TensorQuant)
    {
        AQK = 1; // Row quantization: tensor shape [M, 1] or [1]
        BQK = 1; // Column quantization: tensor shape [1, N] or [1]
    }
    else
    {
        throw std::runtime_error("Unsupported QuantMode");
    }

    ck_tile::index_t stride_A  = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_AQ = arg_parser.get_int("stride_q");
    ck_tile::index_t stride_B  = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_C  = arg_parser.get_int("stride_c");

    ck_tile::index_t stride_BQ = arg_parser.get_int("stride_q");

    ck_tile::index_t kbatch      = arg_parser.get_int("split_k");
    int n_warmup                 = arg_parser.get_int("warmup");
    int n_repeat                 = arg_parser.get_int("repeat");
    ck_tile::index_t init_method = arg_parser.get_int("init");
    bool flush_cache             = arg_parser.get_bool("flush_cache");
    int rotating_count           = arg_parser.get_int("rotating_count");

    stride_A = ck_tile::get_default_stride(M, K, stride_A, is_row_major(a_layout));
    stride_B = ck_tile::get_default_stride(K, N, stride_B, is_row_major(b_layout));
    stride_C = ck_tile::get_default_stride(M, N, stride_C, is_row_major(CLayout{}));

    // Conditional stride calculation based on QuantMode
    if constexpr(QuantMode == ck_tile::QuantType::AQuantGrouped)
    {
        stride_AQ = ck_tile::get_default_stride(M, AQK, stride_AQ, is_row_major(aq_layout));
        stride_BQ = 0; // No B quantization
    }
    else if constexpr(QuantMode == ck_tile::QuantType::BQuantGrouped)
    {
        stride_AQ = 0; // No A quantization
        stride_BQ = ck_tile::get_default_stride(BQK, N, stride_BQ, is_row_major(bq_layout));
    }
    else if constexpr(QuantMode == ck_tile::QuantType::RowColQuant)
    {
        stride_AQ = ck_tile::get_default_stride(M, 1, stride_AQ, is_row_major(aq_layout));
        stride_BQ = ck_tile::get_default_stride(1, N, stride_BQ, is_row_major(bq_layout));
    }
    else if constexpr(QuantMode == ck_tile::QuantType::TensorQuant)
    {
        stride_AQ = 1; // Tensor quantization: tensor shape [1]
        stride_BQ = 1; // Tensor quantization: tensor shape [1]
    }

    ck_tile::HostTensor<ADataType> a_m_k(
        ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(a_layout)));
    ck_tile::HostTensor<BDataType> b_k_n(
        ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(b_layout)));
    ck_tile::HostTensor<CDataType> c_m_n_dev_result(
        ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

    // Create AQ tensor with appropriate shape
    std::unique_ptr<ck_tile::HostTensor<AQDataType>> aq_tensor_ptr = nullptr;
    if constexpr(QuantMode == ck_tile::QuantType::AQuantGrouped ||
                 QuantMode == ck_tile::QuantType::RowColQuant)
    {
        aq_tensor_ptr = std::make_unique<ck_tile::HostTensor<AQDataType>>(
            ck_tile::host_tensor_descriptor(M, AQK, stride_AQ, is_row_major(aq_layout)));
    }
    else if constexpr(QuantMode == ck_tile::QuantType::TensorQuant)
    {
        aq_tensor_ptr = std::make_unique<ck_tile::HostTensor<AQDataType>>(
            ck_tile::host_tensor_descriptor(1, 1, stride_AQ, is_row_major(aq_layout)));
    }

    // Create BQ tensor with appropriate shape
    std::unique_ptr<ck_tile::HostTensor<BQDataType>> bq_tensor_ptr = nullptr;
    if constexpr(QuantMode == ck_tile::QuantType::BQuantGrouped ||
                 QuantMode == ck_tile::QuantType::RowColQuant)
    {
        bq_tensor_ptr = std::make_unique<ck_tile::HostTensor<BQDataType>>(
            ck_tile::host_tensor_descriptor(BQK, N, stride_BQ, is_row_major(bq_layout)));
    }
    else if constexpr(QuantMode == ck_tile::QuantType::TensorQuant)
    {
        bq_tensor_ptr = std::make_unique<ck_tile::HostTensor<BQDataType>>(
            ck_tile::host_tensor_descriptor(1, 1, stride_BQ, is_row_major(bq_layout)));
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<std::uint32_t> fill_seed(0, 500);

    if(init_method == 0)
    {
        if constexpr(QuantMode == ck_tile::QuantType::BQuantGrouped)
        {
            if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
            {
                ck_tile::FillUniformDistribution<ck_tile::pk_int4_t>{-5.0f, 5.0f, fill_seed(gen)}(
                    b_k_n);
            }
            else
            {
                ck_tile::FillUniformDistribution<BDataType>{-2.0f, 3.0f, fill_seed(gen)}(b_k_n);
            }
            ck_tile::FillUniformDistribution<BQDataType>{-2.0f, 2.0f, fill_seed(gen)}(
                *bq_tensor_ptr);
            ck_tile::FillUniformDistribution<ADataType>{-5.0f, 5.0f, fill_seed(gen)}(a_m_k);
        }
        else if constexpr(QuantMode == ck_tile::QuantType::AQuantGrouped)
        {
            if constexpr(std::is_same_v<ADataType, ck_tile::pk_int4_t>)
            {
                ck_tile::FillUniformDistribution<ck_tile::pk_int4_t>{-5.0f, 5.0f, fill_seed(gen)}(
                    a_m_k);
            }
            else
            {
                ck_tile::FillUniformDistribution<ADataType>{-2.0f, 3.0f, fill_seed(gen)}(a_m_k);
            }
            ck_tile::FillUniformDistribution<AQDataType>{-2.0f, 2.0f, fill_seed(gen)}(
                *aq_tensor_ptr);
            ck_tile::FillUniformDistribution<BDataType>{-5.0f, 5.0f, fill_seed(gen)}(b_k_n);
        }
        else
        {
            ck_tile::FillUniformDistribution<ADataType>{-2.0f, 2.0f, fill_seed(gen)}(a_m_k);
            ck_tile::FillUniformDistribution<BDataType>{-2.0f, 2.0f, fill_seed(gen)}(b_k_n);
            ck_tile::FillUniformDistribution<AQDataType>{-2.0f, 2.0f, fill_seed(gen)}(
                *aq_tensor_ptr);
            ck_tile::FillUniformDistribution<BQDataType>{-2.0f, 2.0f, fill_seed(gen)}(
                *bq_tensor_ptr);
        }
    }
    else if(init_method == 1)
    {
        std::cout << "Monotonic initialization is not supported." << std::endl;
        return 0;
    }
    else if(init_method == 2)
    {
        if constexpr(QuantMode == ck_tile::QuantType::BQuantGrouped)
        {
            ck_tile::FillConstant<ADataType>{static_cast<ADataType>(0x38)}(a_m_k);
            ck_tile::FillConstant<BDataType>{static_cast<BDataType>(0x22)}(b_k_n);
            ck_tile::FillConstant<BQDataType>{static_cast<BQDataType>(0.5f)}(*bq_tensor_ptr);
        }
        else
        {
            ck_tile::FillConstant<ADataType>{static_cast<ADataType>(0x22)}(a_m_k);
            ck_tile::FillConstant<AQDataType>{static_cast<AQDataType>(0.5f)}(*aq_tensor_ptr);
            ck_tile::FillConstant<BDataType>{static_cast<BDataType>(0x38)}(b_k_n);

            if constexpr(QuantMode == ck_tile::QuantType::RowColQuant)
            {
                ck_tile::FillConstant<BQDataType>{static_cast<BQDataType>(0.5f)}(*bq_tensor_ptr);
            }
        }
    }
    else
    {
        a_m_k.SetZero();
        aq_tensor_ptr->SetZero();
        b_k_n.SetZero();
        bq_tensor_ptr->SetZero();
    }

    ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

    std::unique_ptr<ck_tile::DeviceMem> aq_dev_buf_ptr = nullptr;
    if constexpr(QuantMode == ck_tile::QuantType::AQuantGrouped ||
                 QuantMode == ck_tile::QuantType::RowColQuant ||
                 QuantMode == ck_tile::QuantType::TensorQuant)
    {
        aq_dev_buf_ptr =
            std::make_unique<ck_tile::DeviceMem>(aq_tensor_ptr->get_element_space_size_in_bytes());
    }

    std::unique_ptr<ck_tile::DeviceMem> bq_dev_buf_ptr = nullptr;
    if constexpr(QuantMode == ck_tile::QuantType::BQuantGrouped ||
                 QuantMode == ck_tile::QuantType::RowColQuant ||
                 QuantMode == ck_tile::QuantType::TensorQuant)
    {
        bq_dev_buf_ptr =
            std::make_unique<ck_tile::DeviceMem>(bq_tensor_ptr->get_element_space_size_in_bytes());
    }

    if constexpr(QuantMode == ck_tile::QuantType::AQuantGrouped ||
                 QuantMode == ck_tile::QuantType::RowColQuant ||
                 QuantMode == ck_tile::QuantType::TensorQuant)
    {
        if constexpr(GemmConfig::PreshuffleQuant)
        {
            ck_tile::HostTensor<AQDataType> aq_shuffle_host =
                shuffle_aq(aq_tensor_ptr.get(), GemmConfig::K_Tile / QuantGroupSize);
            aq_dev_buf_ptr->ToDevice(aq_shuffle_host.data());
        }
        else
        {
            aq_dev_buf_ptr->ToDevice(aq_tensor_ptr->data());
        }
    }

    if constexpr(std::is_same_v<ADataType, ck_tile::pk_int4_t>)
    {
        // Permute vector pk_i4x4 data for device implementation
        ck_tile::HostTensor<ADataType> a_m_k_dev = a_m_k;
        ck_tile::permute_vectors_i4x4_b(a_m_k_dev);
        a_m_k_dev_buf.ToDevice(a_m_k_dev.data());
    }
    else
    {
        a_m_k_dev_buf.ToDevice(a_m_k.data());
    }

    ck_tile::HostTensor<BDataType> b_k_n_dev = b_k_n;
    if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
    {

        if constexpr(GemmConfig::PreshuffleB)
        {
            b_k_n_dev = shuffle_b<GemmConfig>(b_k_n);
        }
        ck_tile::permute_vectors_i4x4_b(b_k_n_dev);
        b_k_n_dev_buf.ToDevice(b_k_n_dev.data());
    }
    else
    {
        if constexpr(GemmConfig::PreshuffleB)
        {
            b_k_n_dev = shuffle_b<GemmConfig>(b_k_n);
        }
        b_k_n_dev_buf.ToDevice(b_k_n_dev.data());
    }

    c_m_n_dev_buf.SetZero();
    c_m_n_dev_result.SetZero();

    if constexpr(QuantMode == ck_tile::QuantType::BQuantGrouped ||
                 QuantMode == ck_tile::QuantType::RowColQuant ||
                 QuantMode == ck_tile::QuantType::TensorQuant)
    {
        bq_dev_buf_ptr->ToDevice(bq_tensor_ptr->data());
    }

    invoke_gemm<GemmConfig,
                TypeConfig,
                ALayout,
                AQLayout,
                BLayout,
                BQLayout,
                CLayout,
                QuantGroupSize,
                QuantMode>(a_m_k_dev_buf,
                           aq_dev_buf_ptr.get(),
                           b_k_n_dev_buf,
                           bq_dev_buf_ptr.get(),
                           c_m_n_dev_buf,
                           M,
                           N,
                           K,
                           AQK,
                           BQK,
                           stride_A,
                           stride_AQ,
                           stride_B,
                           stride_BQ,
                           stride_C,
                           kbatch,
                           n_warmup,
                           n_repeat,
                           flush_cache,
                           rotating_count);

    c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());
    bool pass = true;

    if(arg_parser.get_int("v") == 1)
    {
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        if constexpr(QuantMode == ck_tile::QuantType::AQuantGrouped)
        {
            ck_tile::reference_gemm_quant<ADataType,
                                          AQDataType,
                                          BDataType,
                                          AccDataType,
                                          CDataType,
                                          QuantGroupSize,
                                          true>(a_m_k, *aq_tensor_ptr, b_k_n, c_m_n_host_ref);
        }
        else if constexpr(QuantMode == ck_tile::QuantType::BQuantGrouped)
        {
            ck_tile::reference_gemm_quant<ADataType,
                                          AQDataType,
                                          BDataType,
                                          AccDataType,
                                          CDataType,
                                          QuantGroupSize,
                                          false>(a_m_k, *bq_tensor_ptr, b_k_n, c_m_n_host_ref);
        }
        else if constexpr(QuantMode == ck_tile::QuantType::RowColQuant)
        {
            ck_tile::reference_gemm_rowcol_quant<ADataType,
                                                 AQDataType,
                                                 BDataType,
                                                 BQDataType,
                                                 AccDataType,
                                                 CDataType>(
                a_m_k, *aq_tensor_ptr, b_k_n, *bq_tensor_ptr, c_m_n_host_ref);
        }
        else if constexpr(QuantMode == ck_tile::QuantType::TensorQuant)
        {
            ck_tile::reference_gemm_tensor_quant<ADataType,
                                                 AQDataType,
                                                 BDataType,
                                                 BQDataType,
                                                 AccDataType,
                                                 CDataType>(
                a_m_k, *aq_tensor_ptr, b_k_n, *bq_tensor_ptr, c_m_n_host_ref);
        }

        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, kbatch, max_accumulated_value);
        pass = ck_tile::check_err(c_m_n_dev_result,
                                  c_m_n_host_ref,
                                  "Error: Incorrect results!",
                                  rtol_atol.at(ck_tile::number<0>{}),
                                  rtol_atol.at(ck_tile::number<1>{}));

        if(!pass)
        {
            std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                      << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                      << std::endl;
        }
        std::cout << "The CPU verification result is:" << (pass ? "correct" : "fail") << std::endl;
    }
    else if(arg_parser.get_int("v") == 2)
    {
        std::cout << "GPU verification is not implemented yet. Re-run with -v=1" << std::endl;
        return false;
    }

    return pass;
}
