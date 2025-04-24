// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <cstring>

#include "ck_tile/host.hpp"

#include "reference_batched_gemm.hpp"
#include "reference_batched_softmax.hpp"
#include "flash_attention_fwd.hpp"

/*
 * Toy code of flash attention forward pass
 * Assume simplest case.
 * Q [Batch, HeadNum, SeqenceLengthQ, HeadDim]
 * K [Batch, HeadNum, SeqenceLengthK, HeadDim]
 * V [Batch, HeadNum, HeadDim, SeqenceLengthK]
 * O [Batch, HeadNum, SeqenceLengthQ, HeadDim]
 */

int main(int argc, char* argv[])
{
    using QDataType           = ck_tile::half_t;
    using KDataType           = ck_tile::half_t;
    using VDataType           = ck_tile::half_t;
    using SaccDataType        = float;
    using SMPLComputeDataType = float;
    using PDataType           = ck_tile::half_t;
    using OaccDataType        = float;
    using ODataType           = ck_tile::half_t;

    ck_tile::index_t Batch        = 64;   // Batch Number * Head Number
    ck_tile::index_t M0           = 4096; // SequenceLengthQ
    ck_tile::index_t N0           = 4096; // SequencelengthK
    ck_tile::index_t K0           = 128;  // HeadDim
    ck_tile::index_t N1           = 128;  // HeadDim
    ck_tile::index_t verification = 0;
    ck_tile::index_t init_method  = 1;

    if(argc == 3)
    {
        init_method  = std::stoi(argv[1]);
        verification = std::stoi(argv[2]);
    }
    else if(argc == 8)
    {
        init_method  = std::stoi(argv[1]);
        verification = std::stoi(argv[2]);
        Batch        = std::stoi(argv[3]);
        M0           = std::stoi(argv[4]);
        N0           = std::stoi(argv[5]);
        K0           = std::stoi(argv[6]);
        N1           = std::stoi(argv[7]);
    }

    std::array<ck_tile::index_t, 3> q_lengths{Batch, M0, K0};
    std::array<ck_tile::index_t, 3> q_strides{M0 * K0, K0, 1};

    std::array<ck_tile::index_t, 3> k_lengths{Batch, N0, K0};
    std::array<ck_tile::index_t, 3> k_strides{N0 * K0, K0, 1};

    std::array<ck_tile::index_t, 3> v_lengths{Batch, N1, N0};
    std::array<ck_tile::index_t, 3> v_strides{N1 * N0, N0, 1};

    std::array<ck_tile::index_t, 3> s_lengths{Batch, M0, N0};
    std::array<ck_tile::index_t, 3> s_strides{M0 * N0, N0, 1};

    std::array<ck_tile::index_t, 3> p_lengths{Batch, M0, N0};
    std::array<ck_tile::index_t, 3> p_strides{M0 * N0, N0, 1};

    std::array<ck_tile::index_t, 3> o_lengths{Batch, M0, N1};
    std::array<ck_tile::index_t, 3> o_strides{M0 * N1, N1, 1};

    // host verify
    ck_tile::HostTensor<QDataType> q_host(q_lengths, q_strides);
    ck_tile::HostTensor<KDataType> k_host(k_lengths, k_strides);
    ck_tile::HostTensor<VDataType> v_host(v_lengths, v_strides);
    ck_tile::HostTensor<ODataType> o_host_dev(o_lengths, o_strides);

    switch(init_method)
    {
    case 0: break;
    case 1:
        ck_tile::FillUniformDistributionIntegerValue<QDataType>{-3.f, 3.f}(q_host);
        ck_tile::FillUniformDistributionIntegerValue<KDataType>{-3.f, 3.f}(k_host);
        ck_tile::FillUniformDistributionIntegerValue<VDataType>{-3.f, 3.f}(v_host);
        break;
    case 2:
        ck_tile::FillUniformDistribution<QDataType>{-3.f, 3.f}(q_host);
        ck_tile::FillUniformDistribution<KDataType>{-3.f, 3.f}(k_host);
        ck_tile::FillUniformDistribution<VDataType>{-3.f, 3.f}(v_host);
        break;
    default:
        ck_tile::FillUniformDistributionIntegerValue<QDataType>{-2.f, 2.f}(q_host);
        ck_tile::FillUniformDistributionIntegerValue<KDataType>{-2.f, 2.f}(k_host);
        ck_tile::FillUniformDistributionIntegerValue<VDataType>{-2.f, 2.f}(v_host);
    }
    ck_tile::DeviceMem q_buf(q_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem k_buf(k_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem v_buf(v_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem o_buf(o_host_dev.get_element_space_size_in_bytes());

    q_buf.ToDevice(q_host.mData.data());
    k_buf.ToDevice(k_host.mData.data());
    v_buf.ToDevice(v_host.mData.data());

    constexpr ck_tile::index_t kM0PerBlock = 128;
    constexpr ck_tile::index_t kN0PerBlock = 128;
    constexpr ck_tile::index_t kK0PerBlock = 32;
    constexpr ck_tile::index_t kN1PerBlock = 128;
    constexpr ck_tile::index_t kK1PerBlock = 32;

    constexpr ck_tile::index_t kBlockSize = 256;
    constexpr ck_tile::index_t kHeadDim   = 128;

    ck_tile::index_t kGridSize = Batch * (M0 / kM0PerBlock) * (N1 / kN1PerBlock);

    std::cout << "grid size " << kGridSize << std::endl;

    constexpr ck_tile::index_t kWarpPerCu    = 8; // 2 warps per SIMD
    constexpr ck_tile::index_t kWarpPerBlock = kBlockSize / warpSize;
    constexpr ck_tile::index_t kBlockPerCu   = kWarpPerCu / kWarpPerBlock;

    float ave_time = ck_tile::launch_kernel(ck_tile::stream_config{nullptr, true},
                                            ck_tile::make_kernel<kBlockSize, kBlockPerCu>(
                                                ck_tile::FlashAttentionFwd<QDataType,
                                                                           KDataType,
                                                                           VDataType,
                                                                           SaccDataType,
                                                                           SMPLComputeDataType,
                                                                           PDataType,
                                                                           OaccDataType,
                                                                           ODataType,
                                                                           kBlockSize,
                                                                           kHeadDim,
                                                                           kM0PerBlock,
                                                                           kN0PerBlock,
                                                                           kK0PerBlock,
                                                                           kN1PerBlock,
                                                                           kK1PerBlock>{},
                                                kGridSize,
                                                kBlockSize,
                                                0,
                                                static_cast<QDataType*>(q_buf.GetDeviceBuffer()),
                                                static_cast<KDataType*>(k_buf.GetDeviceBuffer()),
                                                static_cast<VDataType*>(v_buf.GetDeviceBuffer()),
                                                static_cast<ODataType*>(o_buf.GetDeviceBuffer()),
                                                M0,
                                                N0,
                                                K0,
                                                N1,
                                                Batch,
                                                K0,        // StrideQ
                                                K0,        // StrideK
                                                N0,        // StrideV
                                                N1,        // StrideO
                                                M0 * K0,   // BatchStrideQ
                                                N0 * K0,   // BatchStrideK
                                                N1 * N0,   // BatchStrideV
                                                M0 * N1)); // BatchStrideO

    // reference
    auto pass = true;
    if(verification)
    {
        o_buf.FromDevice(o_host_dev.mData.data());

        ck_tile::HostTensor<SMPLComputeDataType> s_host_ref(s_lengths, s_strides);
        ck_tile::HostTensor<PDataType> p_host_ref(p_lengths, p_strides);
        ck_tile::HostTensor<ODataType> o_host_ref(o_lengths, o_strides);

        ck_tile::reference_batched_gemm<QDataType, KDataType, SaccDataType, SMPLComputeDataType>(
            q_host, k_host, s_host_ref);
        ck_tile::reference_batched_softmax<SMPLComputeDataType, SMPLComputeDataType, PDataType>(
            s_host_ref, p_host_ref);
        ck_tile::reference_batched_gemm<PDataType, VDataType, OaccDataType, ODataType>(
            p_host_ref, v_host, o_host_ref);

        pass &= ck_tile::check_err(o_host_dev, o_host_ref);
        std::cout << "valid:" << (pass ? "y" : "n") << std::endl;
    }

    std::size_t flop =
        std::size_t(2) * Batch * M0 * N0 * K0 + std::size_t(2) * Batch * M0 * N1 * N0;
    std::size_t num_btype =
        sizeof(QDataType) * Batch * M0 * K0 + sizeof(KDataType) * Batch * N0 * K0 +
        sizeof(VDataType) * Batch * N1 * N0 + sizeof(ODataType) * Batch * M0 * N1;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    return !pass;
}
