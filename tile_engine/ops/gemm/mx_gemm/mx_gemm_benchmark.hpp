// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/reference/reference_gemm.hpp"
#include "gemm/gemm_benchmark.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"

template <ck_tile::index_t MNPack      = 2,
          ck_tile::index_t KPack       = 2,
          ck_tile::index_t XdlMNThread = 16,
          ck_tile::index_t XdlKThread  = 4>
auto pack_mx_scales_mn_x_k(const ck_tile::HostTensor<ck_tile::e8m0_t>& src, bool k_last)
{
    auto src_lengths               = src.get_lengths();
    const ck_tile::index_t mn      = k_last ? src_lengths[0] : src_lengths[1];
    const ck_tile::index_t k_scale = k_last ? src_lengths[1] : src_lengths[0];

    if(mn % MNPack != 0 || k_scale % KPack != 0)
        throw std::runtime_error(
            "MX scale packing requires mn and k_scale divisible by MNPack/KPack");

    const ck_tile::index_t mn_packed = mn / MNPack;
    const ck_tile::index_t k_packed  = k_scale / KPack;

    if(mn_packed % XdlMNThread != 0 || k_packed % XdlKThread != 0)
        throw std::runtime_error(
            "MX scale packing requires mn_packed and k_packed divisible by XdlMNThread/XdlKThread");

    const ck_tile::index_t total_packed = mn_packed * k_packed;

    std::vector<int32_t> packed(total_packed);

    for(ck_tile::index_t packed_mn = 0; packed_mn < mn_packed; packed_mn++)
    {
        for(ck_tile::index_t packed_k = 0; packed_k < k_packed; packed_k++)
        {
            int32_t val               = 0;
            ck_tile::index_t mn_lane  = packed_mn % XdlMNThread;
            ck_tile::index_t mn_group = packed_mn / XdlMNThread;
            ck_tile::index_t k_lane   = packed_k % XdlKThread;
            ck_tile::index_t k_group  = packed_k / XdlKThread;

            for(ck_tile::index_t ik = 0; ik < KPack; ik++)
            {
                for(ck_tile::index_t imn = 0; imn < MNPack; imn++)
                {
                    const ck_tile::index_t byte_idx = ik * MNPack + imn;
                    const ck_tile::index_t orig_mn =
                        mn_group * XdlMNThread * MNPack + imn * XdlMNThread + mn_lane;
                    const ck_tile::index_t orig_k =
                        k_group * XdlKThread * KPack + ik * XdlKThread + k_lane;

                    ck_tile::e8m0_t v = k_last ? src(orig_mn, orig_k) : src(orig_k, orig_mn);
                    val |= static_cast<int32_t>(v.get()) << (byte_idx * 8);
                }
            }

            packed[packed_mn * k_packed + packed_k] = val;
        }
    }

    return packed;
}

void mx_gemm_host_reference(int verify,
                            ck_tile::HostTensor<ADataType>& a_m_k,
                            ck_tile::HostTensor<BDataType>& b_k_n,
                            ck_tile::HostTensor<CDataType>& c_m_n_host_result,
                            ck_tile::HostTensor<ScaleType>& scale_a_host,
                            ck_tile::HostTensor<ScaleType>& scale_b_host)
{
    if(verify > 0)
    {
        c_m_n_host_result.SetZero();

        ck_tile::
            reference_mx_gemm<ADataType, BDataType, ScaleType, ScaleType, AccDataType, CDataType>(
                a_m_k, b_k_n, c_m_n_host_result, scale_a_host, scale_b_host);
    }
}

#pragma clang diagnostic pop
