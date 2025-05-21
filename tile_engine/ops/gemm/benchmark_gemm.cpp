// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <functional>
#include <tuple>
#include <exception>

#include "ck_tile/host.hpp"
#include "gemm_profiler.hpp"
#include "gemm_host_api.hpp"
#include "benchmark_gemm.hpp"

void benchmark_gemm(const ck_tile::ArgParser& arg_parser,
                    const std::vector<std::function<std::tuple<std::string, float>(
                        ck_tile::GemmHostArgs&, const ck_tile::stream_config&)>>& callables)
{
    GemmProblem gemm_problem{arg_parser.get_int("split_k"),
                             arg_parser.get_int("m"),
                             arg_parser.get_int("n"),
                             arg_parser.get_int("k"),
                             arg_parser.get_int("stride_a"),
                             arg_parser.get_int("stride_b"),
                             arg_parser.get_int("stride_c"),
                             DataTypeTraits<ADataType>::name,
                             DataTypeTraits<BDataType>::name,
                             DataTypeTraits<AccDataType>::name,
                             DataTypeTraits<CDataType>::name,
                             ALayout::name,
                             BLayout::name,
                             CLayout::name,
                             arg_parser.get_bool("structured_sparsity")};

    Setting setting{
        arg_parser.get_int("warmup"),
        arg_parser.get_int("repeat"),
        arg_parser.get_bool("timer"),
        arg_parser.get_int("verify"),
        arg_parser.get_int("init"),
        arg_parser.get_bool("log"),
        arg_parser.get_str("csv_filename"),
    };

    auto& profiler = GemmProfiler::instance(setting);

    const ALayout layout_a = ALayout{};
    const BLayout layout_b = BLayout{};
    const CLayout layout_c = CLayout{};

    gemm_problem.stride_a_ = ck_tile::get_default_stride(
        gemm_problem.m_, gemm_problem.k_, gemm_problem.stride_a_, is_row_major(layout_a));
    gemm_problem.stride_b_ = ck_tile::get_default_stride(
        gemm_problem.k_, gemm_problem.n_, gemm_problem.stride_b_, is_row_major(layout_b));
    gemm_problem.stride_c_ = ck_tile::get_default_stride(
        gemm_problem.m_, gemm_problem.n_, gemm_problem.stride_c_, is_row_major(layout_c));

    ck_tile::HostTensor<ADataType> a_m_k(ck_tile::host_tensor_descriptor(
        gemm_problem.m_, gemm_problem.k_, gemm_problem.stride_a_, is_row_major(layout_a)));
    ck_tile::HostTensor<BDataType> b_k_n(ck_tile::host_tensor_descriptor(
        gemm_problem.k_, gemm_problem.n_, gemm_problem.stride_b_, is_row_major(layout_b)));
    ck_tile::HostTensor<CDataType> c_m_n_dev_result(ck_tile::host_tensor_descriptor(
        gemm_problem.m_, gemm_problem.n_, gemm_problem.stride_c_, is_row_major(layout_c)));

    if(setting.init_method_ == 0)
    {
        ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k);
        ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n);
    }
    else if(setting.init_method_ == 1)
    {
        ck_tile::FillMonotonicSeq<ADataType>{}(a_m_k);
        ck_tile::FillMonotonicSeq<BDataType>{}(b_k_n);
    }
    else if(setting.init_method_ == 2)
    {
        ck_tile::FillConstant<ADataType>{static_cast<ADataType>(1)}(a_m_k);
        ck_tile::FillConstant<BDataType>{static_cast<BDataType>(1)}(b_k_n);
    }
    else
    {
        a_m_k.SetZero();
        b_k_n.SetZero();
    }

    if(gemm_problem.structured_sparsity_)
    {
        ck_tile::AdjustToStructuredSparsity<ADataType>{}(a_m_k);
    }

    ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

    if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
    {
        // Permute vector pk_i4x4 data for device implementation
        ck_tile::HostTensor<BDataType> b_k_n_dev = b_k_n;
        // permute_tensor_b<decltype(b_k_n_dev)>(b_k_n_dev);
        permute_vectors_i4x4_b(b_k_n_dev);
        b_k_n_dev_buf.ToDevice(b_k_n_dev.data());
    }
    else
    {
        b_k_n_dev_buf.ToDevice(b_k_n.data());
    }

    a_m_k_dev_buf.ToDevice(a_m_k.data());
    c_m_n_dev_buf.SetZero();
    c_m_n_dev_result.SetZero();

    ck_tile::GemmHostArgs gemm_args;
    gemm_args.a_ptr    = a_m_k_dev_buf.GetDeviceBuffer();
    gemm_args.b_ptr    = b_k_n_dev_buf.GetDeviceBuffer();
    gemm_args.c_ptr    = c_m_n_dev_buf.GetDeviceBuffer();
    gemm_args.k_batch  = gemm_problem.split_k_;
    gemm_args.M        = gemm_problem.m_;
    gemm_args.N        = gemm_problem.n_;
    gemm_args.K        = gemm_problem.k_;
    gemm_args.stride_A = gemm_problem.stride_a_;
    gemm_args.stride_B = gemm_problem.stride_b_;
    gemm_args.stride_C = gemm_problem.stride_c_;

    ck_tile::HostTensor<CDataType> c_m_n_host_result(ck_tile::host_tensor_descriptor(
        gemm_problem.m_, gemm_problem.n_, gemm_problem.stride_c_, is_row_major(layout_c)));

    if(setting.verify_)
    {
        gemm_host_reference<ADataType,
                            BDataType,
                            AccDataType,
                            CDataType,
                            ALayout,
                            BLayout,
                            CLayout>(setting.verify_,
                                     a_m_k,
                                     b_k_n,
                                     c_m_n_host_result,
                                     a_m_k_dev_buf,
                                     b_k_n_dev_buf,
                                     gemm_problem.m_,
                                     gemm_problem.n_,
                                     gemm_problem.k_,
                                     gemm_problem.stride_a_,
                                     gemm_problem.stride_b_,
                                     gemm_problem.stride_c_);
    }

    try
    {
        for(auto& callable : callables)
        {
            profiler.benchmark(gemm_problem,
                               c_m_n_dev_buf,
                               c_m_n_host_result,
                               c_m_n_dev_result,
                               callable(gemm_args,
                                        ck_tile::stream_config{nullptr,
                                                               true,
                                                               setting.log_,
                                                               setting.n_warmup_,
                                                               setting.n_repeat_,
                                                               setting.is_gpu_timer_}));
        }
        profiler.select_best_instance(static_cast<Metric>(arg_parser.get_int("metric")));
    }
    catch(const std::exception& e)
    {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[])
{
    try
    {
        auto [result, parser] = create_args(argc, argv);
        if(!result)
            return EXIT_FAILURE;
        benchmark_gemm(parser, get_kernel_func_by_trait(parser));
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
