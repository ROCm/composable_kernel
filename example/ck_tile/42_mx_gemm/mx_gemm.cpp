// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>
#include <type_traits>

#include "ck_tile/host.hpp"
#include "mx_gemm.hpp"
#include "mx_gemm_arch_traits.hpp"
#include "mx_gemm_instance.hpp"

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ScaleM,
          typename ScaleN,
          bool UsePersistentKernel = false>
float invoke_mx_gemm(ck_tile::DeviceMem& a_dev_buf,
                     ck_tile::DeviceMem& b_dev_buf,
                     ck_tile::DeviceMem& c_dev_buf,
                     ck_tile::index_t M,
                     ck_tile::index_t N,
                     ck_tile::index_t K,
                     ck_tile::index_t stride_A,
                     ck_tile::index_t stride_B,
                     ck_tile::index_t stride_C,
                     ck_tile::index_t kbatch,
                     ScaleM scale_m,
                     ScaleN scale_n,
                     int n_warmup,
                     int n_repeat)
{
    MXGemmHostArgs<ScaleM, ScaleN> args(a_dev_buf.GetDeviceBuffer(),
                                        b_dev_buf.GetDeviceBuffer(),
                                        c_dev_buf.GetDeviceBuffer(),
                                        kbatch,
                                        M,
                                        N,
                                        K,
                                        stride_A,
                                        stride_B,
                                        stride_C,
                                        scale_m,
                                        scale_n);

    // Simplified invocation - comp_async handles hot loop and tail internally
    auto invoke_splitk_path = [&](auto split_k_) {
        return mx_gemm_calc<GemmConfig,
                            ADataType,
                            BDataType,
                            AccDataType,
                            CDataType,
                            ALayout,
                            BLayout,
                            CLayout,
                            ScaleM,
                            ScaleN,
                            UsePersistentKernel,
                            split_k_.value>(
            args, ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat, true, true, 50});
    };

    float ave_time = (args.k_batch == 1) ? invoke_splitk_path(std::false_type{})
                                         : invoke_splitk_path(std::true_type{});

    constexpr int APackedSize = ck_tile::numeric_traits<ADataType>::PackedSize;
    constexpr int BPackedSize = ck_tile::numeric_traits<BDataType>::PackedSize;

    std::size_t flop     = std::size_t(2) * M * N * K + std::size_t(2) * M * N * K / 32;
    std::size_t num_byte = sizeof(ADataType) * M * K / APackedSize +
                           sizeof(BDataType) * N * K / BPackedSize + sizeof(CDataType) * M * N +
                           sizeof(ck_tile::e8m0_t) * M * K / 32 +
                           sizeof(ck_tile::e8m0_t) * N * K / 32;
    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Run " << ck_tile::gemm_prec_str<ADataType, BDataType>() << " MX GEMM kernel " //
              << " M = " << M << " N = " << N << " K = " << K << " StrideA = " << stride_A
              << " StrideB = " << stride_B << " StrideC = " << stride_C
              << " Preshuffle = " << GemmConfig::Preshuffle << " : " << ave_time << " ms, "
              << tflops << " TFlops, " << gb_per_sec << " GB/s, " << std::endl;

    return ave_time;
}

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "1024", "m dimension")
        .insert("n", "1024", "n dimension")
        .insert("k", "2048", "k dimension")
        .insert("a_layout", "R", "A tensor data layout - Row by default")
        .insert("b_layout", "C", "B tensor data layout - Row by default")
        .insert("c_layout", "R", "C tensor data layout - Row by default")
        .insert("stride_a", "0", "Tensor A stride")
        .insert("stride_b", "0", "Tensor B stride")
        .insert("stride_c", "0", "Tensor C stride")
        .insert("v", "1", "0. No validation, 1. Validation on CPU, 2. Validation on GPU")
        .insert(
            "mx_prec", "fp4xfp4", "data type for activation and weight, support: fp4xfp4, fp8xfp8")
        .insert("warmup", "50", "number of iterations before benchmark the kernel")
        .insert("repeat", "100", "number of iterations to benchmark the kernel")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
        .insert("preshuffle", "0", "0: regular path, 1: preshuffled-B path")
        .insert("split_k", "1", "splitK value")
        .insert("init", "0", "0:random, 1:constant(1)");
    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

#include "run_mx_gemm.inc"

int main(int argc, char* argv[]) { return run_mx_gemm_example<MX_GemmConfig16>(argc, argv); }
