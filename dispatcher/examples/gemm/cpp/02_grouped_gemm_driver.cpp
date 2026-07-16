// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Minimal standalone grouped-GEMM driver (dispatcher way).
 *
 * Grouped GEMM cannot ride the standard dispatcher.run(A,B,C,problem) path:
 * that backend hardcodes a single GemmHostArgs. Instead, this driver includes a
 * single generated grouped kernel header (CK_TILE_SINGLE_KERNEL_INCLUDE) and
 * calls SelectedKernel::launch(descs, stream) directly with a vector of
 * descriptors -- the same 2-arg signature the dispatcher generates (workspace is
 * allocated INSIDE launch()). It builds per-group tensors, runs, and verifies
 * each group against ck_tile::reference_gemm.
 *
 * Build (single-kernel include style):
 *   hipcc -std=c++17 --offload-arch=gfx942 \
 *     -DCK_TILE_SINGLE_KERNEL_INCLUDE \
 *     -I <ck>/include -I <generated_dir> \
 *     -include <generated_dir>/<one>_grouped.hpp \
 *     02_grouped_gemm_driver.cpp -o grouped_gemm_driver
 */

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"

// The generated grouped kernel header is injected on the command line with
// -include and -DCK_TILE_SINGLE_KERNEL_INCLUDE. It exports into the global
// namespace: SelectedKernel, ADataType, BDataType, CDataType, AccDataType,
// ALayout, BLayout, CLayout, and KERNEL_NAME.

template <typename Layout>
static constexpr inline auto is_row_major(Layout)
{
    return ck_tile::bool_constant<
        std::is_same_v<ck_tile::remove_cvref_t<Layout>, ck_tile::tensor_layout::gemm::RowMajor>>{};
}

static std::vector<int> parse_csv_ints(const std::string& s)
{
    std::vector<int> out;
    std::string cur;
    for(char c : s)
    {
        if(c == ',')
        {
            if(!cur.empty())
            {
                out.push_back(std::stoi(cur));
                cur.clear();
            }
        }
        else
            cur.push_back(c);
    }
    if(!cur.empty())
        out.push_back(std::stoi(cur));
    return out;
}

static std::string get_opt(int argc, char** argv, const std::string& key, const std::string& def)
{
    for(int i = 1; i < argc - 1; ++i)
        if(key == argv[i])
            return argv[i + 1];
    return def;
}

int main(int argc, char** argv)
{
    const int group_count = std::stoi(get_opt(argc, argv, "--groups", "8"));
    const int kbatch      = std::stoi(get_opt(argc, argv, "--kbatch", "1"));
    const int warmup      = std::stoi(get_opt(argc, argv, "--warmup", "10"));
    const int repeat      = std::stoi(get_opt(argc, argv, "--repeat", "50"));
    const bool validate   = get_opt(argc, argv, "--validate", "1") != "0";

    std::vector<int> Ms = parse_csv_ints(get_opt(argc, argv, "--Ms", ""));
    std::vector<int> Ns = parse_csv_ints(get_opt(argc, argv, "--Ns", ""));
    std::vector<int> Ks = parse_csv_ints(get_opt(argc, argv, "--Ks", ""));

    const int dm = std::stoi(get_opt(argc, argv, "--m", "256"));
    const int dn = std::stoi(get_opt(argc, argv, "--n", "256"));
    const int dk = std::stoi(get_opt(argc, argv, "--k", "512"));

    auto sz = static_cast<std::size_t>(group_count);
    if(Ms.size() != sz || Ns.size() != sz || Ks.size() != sz)
    {
        Ms.assign(group_count, dm);
        Ns.assign(group_count, dn);
        Ks.assign(group_count, dk);
    }

    std::cout << "Kernel: " << KERNEL_NAME << "\n";
    std::cout << "groups=" << group_count << " kbatch=" << kbatch << "\n";

    std::vector<ck_tile::HostTensor<ADataType>> a_host, b_host;
    std::vector<ck_tile::HostTensor<CDataType>> c_host;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> a_dev, b_dev, c_dev;
    std::vector<ck_tile::index_t> sA(group_count), sB(group_count), sC(group_count);

    std::vector<ck_tile::GroupedGemmHostArgs<>> descs;
    descs.reserve(group_count);

    for(int i = 0; i < group_count; ++i)
    {
        const ck_tile::index_t M = Ms[i], N = Ns[i], K = Ks[i];
        sA[i] = ck_tile::get_default_stride(M, K, 0, is_row_major(ALayout{}));
        sB[i] = ck_tile::get_default_stride(K, N, 0, is_row_major(BLayout{}));
        sC[i] = ck_tile::get_default_stride(M, N, 0, is_row_major(CLayout{}));

        a_host.push_back(ck_tile::HostTensor<ADataType>(
            ck_tile::host_tensor_descriptor(M, K, sA[i], is_row_major(ALayout{}))));
        b_host.push_back(ck_tile::HostTensor<BDataType>(
            ck_tile::host_tensor_descriptor(K, N, sB[i], is_row_major(BLayout{}))));
        c_host.push_back(ck_tile::HostTensor<CDataType>(
            ck_tile::host_tensor_descriptor(M, N, sC[i], is_row_major(CLayout{}))));

        ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_host[i]);
        ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_host[i]);
        c_host[i].SetZero();

        a_dev.push_back(std::make_unique<ck_tile::DeviceMem>(a_host[i]));
        b_dev.push_back(std::make_unique<ck_tile::DeviceMem>(b_host[i]));
        c_dev.push_back(std::make_unique<ck_tile::DeviceMem>(c_host[i]));
        c_dev[i]->SetZero();

        descs.push_back(ck_tile::GroupedGemmHostArgs<>{a_dev[i]->GetDeviceBuffer(),
                                                       b_dev[i]->GetDeviceBuffer(),
                                                       {},
                                                       c_dev[i]->GetDeviceBuffer(),
                                                       kbatch,
                                                       M,
                                                       N,
                                                       K,
                                                       sA[i],
                                                       sB[i],
                                                       {},
                                                       sC[i]});
    }

    const ck_tile::stream_config s{nullptr, true, /*log=*/0, warmup, repeat};
    float ave_time = SelectedKernel::launch(descs, s);

    std::size_t flop = 0, bytes = 0;
    for(int i = 0; i < group_count; ++i)
    {
        flop += std::size_t(2) * Ms[i] * Ns[i] * Ks[i];
        bytes += sizeof(ADataType) * Ms[i] * Ks[i] + sizeof(BDataType) * Ks[i] * Ns[i] +
                 sizeof(CDataType) * Ms[i] * Ns[i];
    }
    const float tflops = static_cast<float>(flop) / 1.E9 / ave_time;
    const float gbps   = static_cast<float>(bytes) / 1.E6 / ave_time;
    std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, " << gbps
              << " GB/s\n";

    for(int i = 0; i < group_count; ++i)
        c_dev[i]->FromDevice(c_host[i].data());

    bool pass = true;
    if(validate)
    {
        for(int i = 0; i < group_count; ++i)
        {
            ck_tile::HostTensor<CDataType> ref(
                ck_tile::host_tensor_descriptor(Ms[i], Ns[i], sC[i], is_row_major(CLayout{})));
            ref.SetZero();
            ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
                a_host[i], b_host[i], ref);
            const float maxv = *std::max_element(ref.mData.begin(), ref.mData.end());
            const auto rtol  = ck_tile::get_relative_threshold<ADataType, CDataType, AccDataType>(
                ck_tile::integer_divide_ceil(Ks[i], kbatch));
            const auto atol = ck_tile::get_absolute_threshold<ADataType, CDataType, AccDataType>(
                maxv / kbatch, ck_tile::integer_divide_ceil(Ks[i], kbatch));
            bool ok =
                ck_tile::check_err(c_host[i], ref, "group[" + std::to_string(i) + "]", rtol, atol);
            pass &= ok;
        }
        std::cout << "Verification: " << (pass ? "PASS" : "FAIL") << "\n";
    }

    return pass ? 0 : 1;
}
