// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/batched_contraction_multi_abd.hpp"

#include "contraction_multi_abd_common.hpp"

// The kernel header is force-included via -include.
// It defines: SelectedKernel, KERNEL_NAME, AsDataType, BsDataType, DsDataType, EDataType,
//             AccDataType, ALayout, BLayout, ELayout, DsLayout,
//             NumATensors, NumBTensors, NumDTensors, NumDimsG, NumDimsM, NumDimsN, NumDimsK

// Helper to build a BatchedContractionMultiABDHostArgs from problem + device buffers.
// Template params are baked in from the force-included kernel header.
template <typename AsDataType_,
          typename BsDataType_,
          typename DsDataType_,
          typename EDataType_,
          ck_tile::index_t NG,
          ck_tile::index_t NM,
          ck_tile::index_t NN,
          ck_tile::index_t NK,
          ck_tile::index_t NA,
          ck_tile::index_t NB,
          ck_tile::index_t ND>
auto make_contraction_multi_abd_host_args(const std::array<const void*, NA>& as_dev,
                                          const std::array<const void*, NB>& bs_dev,
                                          const std::array<const void*, ND>& ds_dev,
                                          void* e_dev,
                                          const std::vector<int>& g_dims,
                                          const std::vector<int>& m_dims,
                                          const std::vector<int>& n_dims,
                                          const std::vector<int>& k_dims)
{
    using HostArgs = ck_tile::BatchedContractionMultiABDHostArgs<NG, NM, NN, NK, NA, NB, ND>;
    using ADims    = typename HostArgs::ADims;
    using BDims    = typename HostArgs::BDims;
    using EDims    = typename HostArgs::EDims;

    // Build dim arrays
    auto make_a_dims = [&]() -> ADims {
        ADims d;
        int pos = 0;
        for(int g : g_dims)
            d[pos++] = g;
        for(int m : m_dims)
            d[pos++] = m;
        for(int k : k_dims)
            d[pos++] = k;
        return d;
    };
    auto make_b_dims = [&]() -> BDims {
        BDims d;
        int pos = 0;
        for(int g : g_dims)
            d[pos++] = g;
        for(int n : n_dims)
            d[pos++] = n;
        for(int k : k_dims)
            d[pos++] = k;
        return d;
    };
    auto make_e_dims = [&]() -> EDims {
        EDims d;
        int pos = 0;
        for(int g : g_dims)
            d[pos++] = g;
        for(int m : m_dims)
            d[pos++] = m;
        for(int n : n_dims)
            d[pos++] = n;
        return d;
    };

    ADims a_dims_arr = make_a_dims();
    BDims b_dims_arr = make_b_dims();
    EDims e_dims_arr = make_e_dims();

    // Build row-major (C-contiguous) strides for each tensor
    auto row_major_strides = [](const std::vector<int>& dims) {
        std::vector<ck_tile::index_t> strides(dims.size());
        ck_tile::index_t stride = 1;
        for(int i = static_cast<int>(dims.size()) - 1; i >= 0; --i)
        {
            strides[i] = stride;
            stride *= dims[i];
        }
        return strides;
    };

    std::vector<int> a_dim_vec;
    a_dim_vec.insert(a_dim_vec.end(), g_dims.begin(), g_dims.end());
    a_dim_vec.insert(a_dim_vec.end(), m_dims.begin(), m_dims.end());
    a_dim_vec.insert(a_dim_vec.end(), k_dims.begin(), k_dims.end());

    std::vector<int> b_dim_vec;
    b_dim_vec.insert(b_dim_vec.end(), g_dims.begin(), g_dims.end());
    b_dim_vec.insert(b_dim_vec.end(), n_dims.begin(), n_dims.end());
    b_dim_vec.insert(b_dim_vec.end(), k_dims.begin(), k_dims.end());

    std::vector<int> e_dim_vec;
    e_dim_vec.insert(e_dim_vec.end(), g_dims.begin(), g_dims.end());
    e_dim_vec.insert(e_dim_vec.end(), m_dims.begin(), m_dims.end());
    e_dim_vec.insert(e_dim_vec.end(), n_dims.begin(), n_dims.end());

    auto a_strides_vec = row_major_strides(a_dim_vec);
    auto b_strides_vec = row_major_strides(b_dim_vec);
    auto e_strides_vec = row_major_strides(e_dim_vec);

    auto make_a_strides_arr = [&]() -> ADims {
        ADims s;
        for(int i = 0; i < NG + NM + NK; ++i)
            s[i] = a_strides_vec[i];
        return s;
    };
    auto make_b_strides_arr = [&]() -> BDims {
        BDims s;
        for(int i = 0; i < NG + NN + NK; ++i)
            s[i] = b_strides_vec[i];
        return s;
    };
    auto make_e_strides_arr = [&]() -> EDims {
        EDims s;
        for(int i = 0; i < NG + NM + NN; ++i)
            s[i] = e_strides_vec[i];
        return s;
    };

    std::array<ADims, NA> as_dims_arr;
    std::array<BDims, NB> bs_dims_arr;
    std::array<EDims, ND> ds_dims_arr;
    std::array<ADims, NA> as_strides_arr;
    std::array<BDims, NB> bs_strides_arr;
    std::array<EDims, ND> ds_strides_arr;

    for(int i = 0; i < NA; ++i)
    {
        as_dims_arr[i]    = a_dims_arr;
        as_strides_arr[i] = make_a_strides_arr();
    }
    for(int i = 0; i < NB; ++i)
    {
        bs_dims_arr[i]    = b_dims_arr;
        bs_strides_arr[i] = make_b_strides_arr();
    }
    for(int i = 0; i < ND; ++i)
    {
        ds_dims_arr[i]    = e_dims_arr;
        ds_strides_arr[i] = make_e_strides_arr();
    }

    return HostArgs{as_dev,
                    bs_dev,
                    ds_dev,
                    e_dev,
                    as_dims_arr,
                    bs_dims_arr,
                    ds_dims_arr,
                    e_dims_arr,
                    as_strides_arr,
                    bs_strides_arr,
                    ds_strides_arr,
                    make_e_strides_arr()};
}

// Run one benchmark with the force-included SelectedKernel.
inline void run_contraction_multi_abd_benchmark(const ContractionMultiABDProblem& problem,
                                                int n_warmup   = 50,
                                                int n_repeat   = 100,
                                                bool verify    = false,
                                                bool log       = false,
                                                bool gpu_timer = true)
{
    using AElementType = std::tuple_element_t<0, AsDataType>;
    using BElementType = std::tuple_element_t<0, BsDataType>;
    using DElementType = std::tuple_element_t<0, DsDataType>;

    const int G = problem.G_total();
    const int M = problem.M_total();
    const int N = problem.N_total();
    const int K = problem.K_total();

    // Allocate host tensors
    std::vector<AElementType> ha(G * M * K), hb(G * N * K), he(G * M * N);
    std::vector<std::vector<DElementType>> hds(NumDTensors, std::vector<DElementType>(G * M * N));

    // Fill with simple values
    for(auto& v : ha)
        v = static_cast<AElementType>(1.0f / K);
    for(auto& v : hb)
        v = static_cast<BElementType>(1.0f / K);
    for(auto& hd : hds)
        for(auto& v : hd)
            v = static_cast<DElementType>(0.0f);

    // Device buffers
    std::vector<ck_tile::DeviceMem> a_bufs(NumATensors), b_bufs(NumBTensors), d_bufs(NumDTensors);
    ck_tile::DeviceMem e_buf(G * M * N * sizeof(EDataType));

    for(int i = 0; i < NumATensors; ++i)
    {
        a_bufs[i].Realloc(G * M * K * sizeof(AElementType));
    }
    for(int i = 0; i < NumBTensors; ++i)
    {
        b_bufs[i].Realloc(G * N * K * sizeof(BElementType));
    }
    for(int i = 0; i < NumDTensors; ++i)
    {
        d_bufs[i].Realloc(G * M * N * sizeof(DElementType));
    }

    for(int i = 0; i < NumATensors; ++i)
        a_bufs[i].ToDevice(ha.data());
    for(int i = 0; i < NumBTensors; ++i)
        b_bufs[i].ToDevice(hb.data());
    for(int i = 0; i < NumDTensors; ++i)
        d_bufs[i].ToDevice(hds[i].data());
    e_buf.SetZero();

    std::array<const void*, NumATensors> as_dev;
    std::array<const void*, NumBTensors> bs_dev;
    std::array<const void*, NumDTensors> ds_dev;
    for(int i = 0; i < NumATensors; ++i)
        as_dev[i] = a_bufs[i].GetDeviceBuffer();
    for(int i = 0; i < NumBTensors; ++i)
        bs_dev[i] = b_bufs[i].GetDeviceBuffer();
    for(int i = 0; i < NumDTensors; ++i)
        ds_dev[i] = d_bufs[i].GetDeviceBuffer();
    void* e_dev = e_buf.GetDeviceBuffer();

    auto args = make_contraction_multi_abd_host_args<AsDataType,
                                                     BsDataType,
                                                     DsDataType,
                                                     EDataType,
                                                     NumDimsG,
                                                     NumDimsM,
                                                     NumDimsN,
                                                     NumDimsK,
                                                     NumATensors,
                                                     NumBTensors,
                                                     NumDTensors>(as_dev,
                                                                  bs_dev,
                                                                  ds_dev,
                                                                  e_dev,
                                                                  problem.g_dims,
                                                                  problem.m_dims,
                                                                  problem.n_dims,
                                                                  problem.k_dims);

    // Verify: CPU reference (simple passthrough -- sum over K, uniform input)
    std::vector<EDataType> he_ref;
    if(verify)
    {
        // Reference: E[g,m,n] = sum over (A, B) pairs of sum_k A[g,m,k] * B[g,n,k].
        // With ha = hb = 1/K, one pair contributes sum_k (1/K)*(1/K) = 1/K, so the
        // total over all pairs is NumA * NumB / K. D tensors are zero, so the
        // epilogue adds nothing.
        //
        // Note: for NumATensors > 1 or NumBTensors > 1 this comparison currently
        // fails, because the kernel's (A, B) loop stores rather than accumulates and
        // only the last pair survives. That is a real kernel defect, not a reference
        // error -- verify is doing its job by reporting it.
        const float expected = static_cast<float>(NumATensors) * static_cast<float>(NumBTensors) /
                               static_cast<float>(K);
        he_ref.resize(G * M * N, static_cast<EDataType>(expected));
    }

    ck_tile::stream_config stream{nullptr, gpu_timer, log ? 1 : 0, n_warmup, n_repeat};

    float avg_time = SelectedKernel::launch(args, stream);

    // Guard against unsupported-arguments return (-1) before computing throughput.
    if(avg_time < 0.0f)
    {
        std::cerr << "error: kernel " << KERNEL_NAME
                  << " returned unsupported-arguments signal (avg_time=" << avg_time
                  << "); aborting benchmark.\n";
        std::exit(1);
    }

    if(verify)
    {
        // Copy E back to host and compare
        e_buf.FromDevice(he.data());
        bool pass = true;
        for(int i = 0; i < G * M * N; ++i)
        {
            const float got  = static_cast<float>(he[i]);
            const float ref  = static_cast<float>(he_ref[i]);
            const float diff = std::abs(got - ref);
            if(diff > 1e-3f * std::abs(ref) + 1e-5f)
            {
                std::cerr << "verify FAILED at index " << i << ": got=" << got << " ref=" << ref
                          << "\n";
                pass = false;
                break;
            }
        }
        if(pass)
            std::cout << "verify PASSED\n";
        else
            std::exit(1);
    }

    size_t flop = 2ULL * G * M * N * K;
    size_t num_byte =
        static_cast<size_t>(G) *
        (NumATensors * sizeof(AElementType) * M * K + NumBTensors * sizeof(BElementType) * N * K +
         NumDTensors * sizeof(DElementType) * M * N + sizeof(EDataType) * M * N);

    float tflops    = static_cast<float>(flop) / 1e9f / avg_time;
    float bandwidth = static_cast<float>(num_byte) / 1e6f / avg_time;

    std::cout << std::fixed << std::setprecision(4) << "kernel: " << KERNEL_NAME
              << "  latency(ms): " << avg_time << "  tflops: " << tflops
              << "  bandwidth(GB/s): " << bandwidth << "\n";
}
