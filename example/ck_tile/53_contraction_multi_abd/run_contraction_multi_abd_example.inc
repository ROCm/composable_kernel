// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <memory>
#include <optional>
#include "contraction_multi_abd_utils.hpp"
#include "ck_tile/host/reference/reference_batched_contraction.hpp"

template <typename ADataType_, typename BDataType_, typename EDataType_, typename AccDataType_>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType_) < sizeof(BDataType_), ADataType_, BDataType_>;

    const auto rtol = ck_tile::get_relative_threshold<ComputeType, EDataType_, AccDataType_>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, EDataType_, AccDataType_>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));

    return ck_tile::make_tuple(rtol, atol);
}

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

// C++17-compatible helper to create array of HostTensors
namespace {
template <typename DataType, std::size_t N, std::size_t... Is>
std::array<ck_tile::HostTensor<DataType>, N>
make_host_tensors_impl(const std::array<ck_tile::HostTensorDescriptor, N>& descs,
                       std::index_sequence<Is...>)
{
    return {ck_tile::HostTensor<DataType>(descs[Is])...};
}

template <typename DataType, std::size_t N>
std::array<ck_tile::HostTensor<DataType>, N>
make_host_tensors(const std::array<ck_tile::HostTensorDescriptor, N>& descs)
{
    return make_host_tensors_impl<DataType, N>(descs, std::make_index_sequence<N>{});
}
} // anonymous namespace

int run_contraction_multi_abd_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    std::vector<ck_tile::index_t> G_dims = parse_dimensions(arg_parser.get_str("g_dims"));
    std::vector<ck_tile::index_t> M_dims = parse_dimensions(arg_parser.get_str("m_dims"));
    std::vector<ck_tile::index_t> N_dims = parse_dimensions(arg_parser.get_str("n_dims"));
    std::vector<ck_tile::index_t> K_dims = parse_dimensions(arg_parser.get_str("k_dims"));

    ck_tile::index_t G_total = calculate_total_elements(G_dims);
    ck_tile::index_t M_total = calculate_total_elements(M_dims);
    ck_tile::index_t N_total = calculate_total_elements(N_dims);
    ck_tile::index_t K_total = calculate_total_elements(K_dims);

    // A[i] = [G, M, K], B[j] = [G, N, K], D[d]/E = [G, M, N]
    std::vector<ck_tile::index_t> A_dims = concatenate_dim_components({G_dims, M_dims, K_dims});
    std::vector<ck_tile::index_t> B_dims = concatenate_dim_components({G_dims, N_dims, K_dims});
    std::vector<ck_tile::index_t> E_dims = concatenate_dim_components({G_dims, M_dims, N_dims});

    print_dims("G_dims", G_dims, G_total);
    print_dims("M_dims", M_dims, M_total);
    print_dims("N_dims", N_dims, N_total);
    print_dims("K_dims", K_dims, K_total);

    // Per-tensor layouts (must stay in sync with AsLayout / BsLayout / DsLayout / ELayout below)
    using ALayout = Row;
    using BLayout = Col;
    using DLayout = Row;
    using ELayout = Row;

    // Compute strides honoring each tensor's layout so that the host buffers used for
    // verification have the same memory layout the kernel expects.
    const auto a_strides_v = compute_strides_for_layout<ALayout>(G_dims, M_dims, K_dims);
    const auto b_strides_v = compute_strides_for_layout<BLayout>(G_dims, N_dims, K_dims);
    const auto d_strides_v = compute_strides_for_layout<DLayout>(G_dims, M_dims, N_dims);
    const auto e_strides_v = compute_strides_for_layout<ELayout>(G_dims, M_dims, N_dims);

    auto to_size_t_vec = [](const std::vector<ck_tile::index_t>& v) {
        return std::vector<std::size_t>(v.begin(), v.end());
    };

    // Create host tensors for each A, B, D, E with explicit layout-aware strides.
    const auto a_desc =
        ck_tile::HostTensorDescriptor(std::vector<std::size_t>(A_dims.begin(), A_dims.end()),
                                      to_size_t_vec(a_strides_v));
    const auto b_desc =
        ck_tile::HostTensorDescriptor(std::vector<std::size_t>(B_dims.begin(), B_dims.end()),
                                      to_size_t_vec(b_strides_v));
    const auto d_desc =
        ck_tile::HostTensorDescriptor(std::vector<std::size_t>(E_dims.begin(), E_dims.end()),
                                      to_size_t_vec(d_strides_v));
    const auto e_desc =
        ck_tile::HostTensorDescriptor(std::vector<std::size_t>(E_dims.begin(), E_dims.end()),
                                      to_size_t_vec(e_strides_v));

    ck_tile::HostTensor<A0DataType> a0_host(a_desc);
    ck_tile::HostTensor<A1DataType> a1_host(a_desc);
    ck_tile::HostTensor<B0DataType> b0_host(b_desc);
    ck_tile::HostTensor<D0DataType> d0_host(d_desc);
    ck_tile::HostTensor<EDataType> e_host(e_desc);

    ck_tile::FillUniformDistribution<A0DataType>{-5.f, 5.f, std::nullopt}(a0_host);
    ck_tile::FillUniformDistribution<A1DataType>{-5.f, 5.f, std::nullopt}(a1_host);
    ck_tile::FillUniformDistribution<B0DataType>{-5.f, 5.f, std::nullopt}(b0_host);
    ck_tile::FillUniformDistribution<D0DataType>{-2.f, 2.f, std::nullopt}(d0_host);

    ck_tile::DeviceMem a0_dev(a0_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem a1_dev(a1_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b0_dev(b0_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d0_dev(d0_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem e_dev(e_host.get_element_space_size_in_bytes());

    a0_dev.ToDevice(a0_host.data());
    a1_dev.ToDevice(a1_host.data());
    b0_dev.ToDevice(b0_host.data());
    d0_dev.ToDevice(d0_host.data());
    e_dev.SetZero();
    e_host.SetZero();

    auto convert_strides = [](const std::vector<std::size_t>& strides) {
        std::vector<ck_tile::index_t> converted(strides.size());
        std::copy(strides.begin(), strides.end(), converted.begin());
        return converted;
    };

    std::vector<ck_tile::index_t> A0_strides = convert_strides(a0_host.get_strides());
    std::vector<ck_tile::index_t> A1_strides = convert_strides(a1_host.get_strides());
    std::vector<ck_tile::index_t> B0_strides = convert_strides(b0_host.get_strides());
    std::vector<ck_tile::index_t> D0_strides = convert_strides(d0_host.get_strides());
    std::vector<ck_tile::index_t> E_strides  = convert_strides(e_host.get_strides());

    const std::array<const void*, NumATensor> as_ptr = {a0_dev.GetDeviceBuffer(),
                                                        a1_dev.GetDeviceBuffer()};
    const std::array<const void*, NumBTensor> bs_ptr = {b0_dev.GetDeviceBuffer()};
    const std::array<const void*, NumDTensor> ds_ptr = {d0_dev.GetDeviceBuffer()};
    const std::array<std::vector<ck_tile::index_t>, NumATensor> As_dims = {A_dims, A_dims};
    const std::array<std::vector<ck_tile::index_t>, NumBTensor> Bs_dims = {B_dims};
    const std::array<std::vector<ck_tile::index_t>, NumDTensor> Ds_dims = {E_dims};
    const std::array<std::vector<ck_tile::index_t>, NumATensor> As_strides = {A0_strides,
                                                                              A1_strides};
    const std::array<std::vector<ck_tile::index_t>, NumBTensor> Bs_strides = {B0_strides};
    const std::array<std::vector<ck_tile::index_t>, NumDTensor> Ds_strides = {D0_strides};

    int n_warmup = arg_parser.get_int("warmup");
    int n_repeat = arg_parser.get_int("repeat");

    using AsLayout = ck_tile::tuple<ALayout, ALayout>;
    using BsLayout = ck_tile::tuple<BLayout>;
    using DsLayout = ck_tile::tuple<DLayout>;

    using AElementWise   = ck_tile::element_wise::AddScale;
    using BElementWise   = ck_tile::element_wise::PassThrough;
    using CDEElementWise = AddDs;

    std::cout << "\n=== Running GPU Kernel ===" << std::endl;

    float ave_time = contraction_multi_abd<AsDataType,
                                           BsDataType,
                                           DsDataType,
                                           AccDataType,
                                           EDataType,
                                           AsLayout,
                                           BsLayout,
                                           DsLayout,
                                           ELayout,
                                           AElementWise,
                                           BElementWise,
                                           CDEElementWise>(
        as_ptr,
        bs_ptr,
        ds_ptr,
        e_dev.GetDeviceBuffer(),
        As_dims,
        Bs_dims,
        Ds_dims,
        E_dims,
        As_strides,
        Bs_strides,
        Ds_strides,
        E_strides,
        ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat},
        G_dims.size(),
        M_dims.size(),
        N_dims.size(),
        K_dims.size());

    std::size_t flop     = std::size_t(2) * G_total * M_total * N_total * K_total;
    std::size_t num_byte = sizeof(A0DataType) * NumATensor * G_total * M_total * K_total +
                           sizeof(B0DataType) * NumBTensor * G_total * N_total * K_total +
                           sizeof(D0DataType) * NumDTensor * G_total * M_total * N_total +
                           sizeof(EDataType) * G_total * M_total * N_total;

    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    e_dev.FromDevice(e_host.data());

    bool pass = true;
    if(arg_parser.get_int("v") == 1)
    {
        std::cout << "Computing CPU reference..." << std::endl;

        const auto e_ref_desc = ck_tile::HostTensorDescriptor(E_dims);
        ck_tile::HostTensor<EDataType> e_ref_host(e_ref_desc);
        e_ref_host.SetZero();

        ck_tile::compute_reference_batched_contraction_multi_abd<AsDataType,
                                                                 BsDataType,
                                                                 DsDataType,
                                                                 AccDataType,
                                                                 EDataType,
                                                                 AElementWise,
                                                                 BElementWise,
                                                                 CDEElementWise>({a0_host, a1_host},
                                                                                 {b0_host},
                                                                                 {d0_host},
                                                                                 e_ref_host,
                                                                                 G_total,
                                                                                 M_total,
                                                                                 N_total,
                                                                                 K_total,
                                                                                 AElementWise{},
                                                                                 BElementWise{},
                                                                                 CDEElementWise{},
                                                                                 G_dims,
                                                                                 M_dims,
                                                                                 N_dims,
                                                                                 K_dims);

        const float max_accumulated_value =
            *std::max_element(e_ref_host.mData.begin(), e_ref_host.mData.end());

        const auto rtol_atol = calculate_rtol_atol<A0DataType, B0DataType, EDataType, AccDataType>(
            K_total, 1, max_accumulated_value);

        pass = ck_tile::check_err(e_host,
                                  e_ref_host,
                                  "Error: Incorrect results!",
                                  rtol_atol.at(ck_tile::number<0>{}),
                                  rtol_atol.at(ck_tile::number<1>{}));

        std::cout << "The CPU verification result is: " << (pass ? "correct" : "fail") << std::endl;
    }

    return pass;
}
