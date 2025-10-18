// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <functional>
#include <numeric>
#include <thread>

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile {

template <typename ADataType,
          typename BDataType,
          typename DDataType,
          typename EDataType,
          typename AccDataType,
          typename CDEElementWise>

void calculate_reference_flat_indexing(
    const ck_tile::HostTensor<ADataType>& a_full_dims,
    const ck_tile::HostTensor<BDataType>& b_full_dims,
    const std::vector<ck_tile::HostTensor<DDataType>>& ds_full_dims_host,
    ck_tile::HostTensor<EDataType>& e_full_dims_host_ref,
    ck_tile::index_t G_total,
    ck_tile::index_t M_total,
    ck_tile::index_t N_total,
    ck_tile::index_t K_total,
    const CDEElementWise& cde_elementwise)
{
    std::cout << "Calculating reference using optimized flat indexing with parallel processing..."
              << std::endl;

    // Parallel computation over G and M dimensions using pattern from reference_batched_gemm.hpp
    auto f_gm = [&](auto g_flat, auto m_flat) {
        for(ck_tile::index_t n_flat = 0; n_flat < N_total; ++n_flat)
        {
            AccDataType sum = 0;

            // Compute dot product over K dimension
            for(ck_tile::index_t k_flat = 0; k_flat < K_total; ++k_flat)
            {
                auto a_val =
                    a_full_dims.mData[g_flat * M_total * K_total + m_flat * K_total + k_flat];
                auto b_val =
                    b_full_dims.mData[g_flat * N_total * K_total + n_flat * K_total + k_flat];
                sum += static_cast<AccDataType>(a_val) * static_cast<AccDataType>(b_val);
            }

            // Apply elementwise operation with D tensors
            EDataType result = static_cast<EDataType>(sum);
            if(ds_full_dims_host.size() == 0)
            {
                ;
            }
            else if(ds_full_dims_host.size() == 1)
            {
                cde_elementwise(result,
                                ck_tile::type_convert<float>(sum),
                                ck_tile::type_convert<float>(
                                    ds_full_dims_host[0].mData[g_flat * M_total * N_total +
                                                               m_flat * N_total + n_flat]));
            }
            else if(ds_full_dims_host.size() == 2)
            {
                cde_elementwise(
                    result,
                    ck_tile::type_convert<float>(sum),
                    ck_tile::type_convert<float>(
                        ds_full_dims_host[0]
                            .mData[g_flat * M_total * N_total + m_flat * N_total + n_flat]),
                    ck_tile::type_convert<float>(
                        ds_full_dims_host[1]
                            .mData[g_flat * M_total * N_total + m_flat * N_total + n_flat]));
            }
            else if(ds_full_dims_host.size() == 3)
            {
                cde_elementwise(
                    result,
                    ck_tile::type_convert<float>(sum),
                    ck_tile::type_convert<float>(
                        ds_full_dims_host[0]
                            .mData[g_flat * M_total * N_total + m_flat * N_total + n_flat]),
                    ck_tile::type_convert<float>(
                        ds_full_dims_host[1]
                            .mData[g_flat * M_total * N_total + m_flat * N_total + n_flat]),
                    ck_tile::type_convert<float>(
                        ds_full_dims_host[2]
                            .mData[g_flat * M_total * N_total + m_flat * N_total + n_flat]));
            }
            else if(ds_full_dims_host.size() == 4)
            {
                cde_elementwise(
                    result,
                    ck_tile::type_convert<float>(sum),
                    ck_tile::type_convert<float>(
                        ds_full_dims_host[0]
                            .mData[g_flat * M_total * N_total + m_flat * N_total + n_flat]),
                    ck_tile::type_convert<float>(
                        ds_full_dims_host[1]
                            .mData[g_flat * M_total * N_total + m_flat * N_total + n_flat]),
                    ck_tile::type_convert<float>(
                        ds_full_dims_host[2]
                            .mData[g_flat * M_total * N_total + m_flat * N_total + n_flat]),
                    ck_tile::type_convert<float>(
                        ds_full_dims_host[3]
                            .mData[g_flat * M_total * N_total + m_flat * N_total + n_flat]));
            }
            else
            {
                throw std::runtime_error("Unsupported NumDTensor for reference calculation");
            }

            // Store result
            e_full_dims_host_ref.mData[g_flat * M_total * N_total + m_flat * N_total + n_flat] =
                static_cast<EDataType>(result);
        }
    };

    // Execute parallel computation using hardware concurrency
    // Parallelize over G_total and M_total dimensions for optimal CPU utilization
    make_ParallelTensorFunctor(f_gm, G_total, M_total)(std::thread::hardware_concurrency());
}

template <typename ADataType,
          typename BDataType,
          typename DDataType,
          typename EDataType,
          typename AccDataType,
          typename CDEElementWise>
void calculate_reference_multi_dimensional(
    const HostTensor<ADataType>& a_full_dims,
    const HostTensor<BDataType>& b_full_dims,
    const std::vector<HostTensor<DDataType>>& ds_full_dims_host,
    HostTensor<EDataType>& e_full_dims_host_ref,
    const std::vector<index_t>& G_dims,
    const std::vector<index_t>& M_dims,
    const std::vector<index_t>& N_dims,
    const std::vector<index_t>& K_dims,
    const std::vector<index_t>& A_dims,
    const std::vector<index_t>& B_dims,
    const std::vector<index_t>& E_dims,
    const CDEElementWise& cde_elementwise)
{
    std::cout << "Calculating reference using multi-dimensional indexing..." << std::endl;

    std::vector<std::size_t> g_idx(G_dims.size());
    std::vector<std::size_t> m_idx(M_dims.size());
    std::vector<std::size_t> n_idx(N_dims.size());
    std::vector<std::size_t> k_idx(K_dims.size());
    std::vector<std::size_t> a_idx, b_idx, e_idx;

    a_idx.reserve(A_dims.size());
    b_idx.reserve(B_dims.size());
    e_idx.reserve(E_dims.size());

    auto calculate_total_elements = [](const std::vector<ck_tile::index_t>& dims) {
        return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<ck_tile::index_t>());
    };

    for(ck_tile::index_t g_flat = 0; g_flat < calculate_total_elements(G_dims); ++g_flat)
    {
        ck_tile::index_t temp = g_flat;
        for(int i = G_dims.size() - 1; i >= 0; --i)
        {
            g_idx[i] = temp % G_dims[i];
            temp /= G_dims[i];
        }

        for(ck_tile::index_t m_flat = 0; m_flat < calculate_total_elements(M_dims); ++m_flat)
        {
            temp = m_flat;
            for(int i = M_dims.size() - 1; i >= 0; --i)
            {
                m_idx[i] = temp % M_dims[i];
                temp /= M_dims[i];
            }

            for(ck_tile::index_t n_flat = 0; n_flat < calculate_total_elements(N_dims); ++n_flat)
            {
                temp = n_flat;
                for(int i = N_dims.size() - 1; i >= 0; --i)
                {
                    n_idx[i] = temp % N_dims[i];
                    temp /= N_dims[i];
                }

                AccDataType sum = 0;

                for(ck_tile::index_t k_flat = 0; k_flat < calculate_total_elements(K_dims);
                    ++k_flat)
                {
                    temp = k_flat;
                    for(int i = K_dims.size() - 1; i >= 0; --i)
                    {
                        k_idx[i] = temp % K_dims[i];
                        temp /= K_dims[i];
                    }

                    a_idx.clear();
                    b_idx.clear();

                    a_idx.insert(a_idx.end(), g_idx.begin(), g_idx.end());
                    a_idx.insert(a_idx.end(), m_idx.begin(), m_idx.end());
                    a_idx.insert(a_idx.end(), k_idx.begin(), k_idx.end());

                    b_idx.insert(b_idx.end(), g_idx.begin(), g_idx.end());
                    b_idx.insert(b_idx.end(), n_idx.begin(), n_idx.end());
                    b_idx.insert(b_idx.end(), k_idx.begin(), k_idx.end());

                    auto a_val = a_full_dims(a_idx);
                    auto b_val = b_full_dims(b_idx);

                    sum += static_cast<AccDataType>(a_val) * static_cast<AccDataType>(b_val);
                }

                e_idx.clear();
                e_idx.insert(e_idx.end(), g_idx.begin(), g_idx.end());
                e_idx.insert(e_idx.end(), m_idx.begin(), m_idx.end());
                e_idx.insert(e_idx.end(), n_idx.begin(), n_idx.end());

                EDataType result = static_cast<EDataType>(sum);
                if(ds_full_dims_host.size() == 0)
                {
                    ;
                }
                else if(ds_full_dims_host.size() == 1)
                {
                    cde_elementwise(result,
                                    ck_tile::type_convert<float>(sum),
                                    ck_tile::type_convert<float>(ds_full_dims_host[0](e_idx)));
                }
                else if(ds_full_dims_host.size() == 2)
                {
                    cde_elementwise(result,
                                    ck_tile::type_convert<float>(sum),
                                    ck_tile::type_convert<float>(ds_full_dims_host[0](e_idx)),
                                    ck_tile::type_convert<float>(ds_full_dims_host[1](e_idx)));
                }
                else if(ds_full_dims_host.size() == 3)
                {
                    cde_elementwise(result,
                                    ck_tile::type_convert<float>(sum),
                                    ck_tile::type_convert<float>(ds_full_dims_host[0](e_idx)),
                                    ck_tile::type_convert<float>(ds_full_dims_host[1](e_idx)),
                                    ck_tile::type_convert<float>(ds_full_dims_host[2](e_idx)));
                }
                else if(ds_full_dims_host.size() == 4)
                {
                    cde_elementwise(result,
                                    ck_tile::type_convert<float>(sum),
                                    ck_tile::type_convert<float>(ds_full_dims_host[0](e_idx)),
                                    ck_tile::type_convert<float>(ds_full_dims_host[1](e_idx)),
                                    ck_tile::type_convert<float>(ds_full_dims_host[2](e_idx)),
                                    ck_tile::type_convert<float>(ds_full_dims_host[3](e_idx)));
                }
                else
                {
                    throw std::runtime_error("Unsupported NumDTensor for reference calculation");
                }

                e_full_dims_host_ref(e_idx) = static_cast<EDataType>(result);
            }
        }
    }
}

} // namespace ck_tile
