// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Unified test for Sparge pipeline: blockmap generation + sparse attention (Jenga/VSA).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "ck_tile/host.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host/reference/reference_blocked_attention.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"

#include "fmha_fwd_trek.hpp"
#include "sparge_blockmap_trek.hpp"
#include "sparge_tool.hpp"

// ============================================================================
// Helpers
// ============================================================================

template <typename T>
ck_tile::HostTensor<T>
make_qkv_tensor(ck_tile::index_t batch, ck_tile::index_t nhead, ck_tile::index_t seqlen, ck_tile::index_t hdim, bool i_perm)
{
    if(i_perm)
        return ck_tile::HostTensor<T>({batch, nhead, seqlen, hdim});
    return ck_tile::HostTensor<T>({batch, seqlen, nhead, hdim});
}

template <typename T>
ck_tile::HostTensor<T> to_bhsd(const ck_tile::HostTensor<T>& tensor, bool is_bhsd)
{
    auto lens               = tensor.get_lengths();
    ck_tile::index_t batch  = lens[0];
    ck_tile::index_t seqlen = is_bhsd ? lens[2] : lens[1];
    ck_tile::index_t nhead  = is_bhsd ? lens[1] : lens[2];
    ck_tile::index_t hdim   = lens[3];

    ck_tile::HostTensor<T> out({batch, nhead, seqlen, hdim});
    for(ck_tile::index_t b = 0; b < batch; ++b)
        for(ck_tile::index_t h = 0; h < nhead; ++h)
            for(ck_tile::index_t s = 0; s < seqlen; ++s)
                for(ck_tile::index_t d = 0; d < hdim; ++d)
                    out(b, h, s, d) = is_bhsd ? tensor(b, h, s, d) : tensor(b, s, h, d);
    return out;
}

template <typename T>
auto get_error_tolerance()
{
    double rtol = 1e-2;
    double atol = 4e-2;
    if constexpr(std::is_same_v<T, ck_tile::bf16_t>)
    {
        atol = 2e-1;
        rtol = 2e-1;
    }
    return ck_tile::make_tuple(rtol, atol);
}

template <typename T>
float to_float_for_compare(T value)
{
    return static_cast<float>(value);
}

template <>
float to_float_for_compare<ck_tile::bf16_t>(ck_tile::bf16_t value)
{
#if CK_TILE_USE_CUSTOM_DATA_TYPE
    return static_cast<float>(value);
#else
    return ck_tile::bf16_to_float_raw(ck_tile::bit_cast<ck_tile::bf16_raw_t>(value));
#endif
}

// ============================================================================
// Arg parser
// ============================================================================
auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser
        .insert("v", "1", "0:no validation, 1:cpu validation")
        .insert("pipeline", "jenga", "attention pipeline: jenga / vsa")
        .insert("b", "1", "batch size")
        .insert("h", "4", "num of head for q")
        .insert("h_k", "-1", "num of head for k/v, -1 means equal to h")
        .insert("s", "4096", "seqlen_q")
        .insert("s_k", "-1", "seqlen_k, -1 means equal to s")
        .insert("d", "128", "head dim for q, k")
        .insert("d_v", "-1", "head dim for v, -1 means equal to d")
        .insert("topk", "0.3", "topk ratio for blockmap (fraction of K-blocks to keep)")
        .insert("cdfthreshd", "-1", "CDF threshold for blockmap (overrides topk if >= 0)")
        .insert("simthreshd1", "0.6", "similarity threshold for blockmap")
        .insert("prec", "fp16", "data type: fp16/bf16")
        .insert("iperm", "1", "permute input, 1: b*h*s*d, 0: b*s*h*d")
        .insert("operm", "1", "permute output")
        .insert("seed", "42", "random seed")
        .insert("warmup", "5", "warmup iterations")
        .insert("repeat", "20", "benchmark iterations")
        .insert("kname", "0", "print kernel name");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

// ============================================================================
// Main test
// ============================================================================
template <typename T>
bool run_test(const ck_tile::ArgParser& arg_parser)
{
    int do_validation           = arg_parser.get_int("v");
    std::string pipeline        = arg_parser.get_str("pipeline");
    ck_tile::index_t batch      = arg_parser.get_int("b");
    ck_tile::index_t nhead      = arg_parser.get_int("h");
    ck_tile::index_t nhead_k    = arg_parser.get_int("h_k");
    ck_tile::index_t seqlen_q   = arg_parser.get_int("s");
    ck_tile::index_t seqlen_k   = arg_parser.get_int("s_k");
    ck_tile::index_t hdim_q     = arg_parser.get_int("d");
    ck_tile::index_t hdim_v     = arg_parser.get_int("d_v");
    float topk                  = arg_parser.get_float("topk");
    float cdfthreshd            = arg_parser.get_float("cdfthreshd");
    float simthreshd1           = arg_parser.get_float("simthreshd1");
    bool i_perm                 = arg_parser.get_bool("iperm");
    bool o_perm                 = arg_parser.get_bool("operm");
    uint32_t seed               = arg_parser.get_uint32("seed");
    int warmup                  = arg_parser.get_int("warmup");
    int repeat                  = arg_parser.get_int("repeat");
    int kname                   = arg_parser.get_int("kname");

    if(nhead_k < 0) nhead_k = nhead;
    if(seqlen_k < 0) seqlen_k = seqlen_q;
    if(hdim_v < 0) hdim_v = hdim_q;

    // If cdfthreshd >= 0, use CDF mode; otherwise use topk mode
    if(cdfthreshd >= 0.0f)
        topk = -1.0f;

    constexpr ck_tile::index_t BLKQ = 64;
    constexpr ck_tile::index_t BLKK = 128;

    if(hdim_q != 128 || hdim_v != 128)
    {
        std::cout << "\n>>> TEST SKIPPED <<<\n"
                  << "Kernel instances are generated for hdim=128 only.\n";
        return true;
    }

    ck_tile::index_t num_q_blocks = (seqlen_q + BLKQ - 1) / BLKQ;
    ck_tile::index_t num_k_blocks = (seqlen_k + BLKK - 1) / BLKK;

    std::string prec_str = std::is_same_v<T, ck_tile::half_t> ? "fp16" : "bf16";
    std::cout << "[" << pipeline << "|" << prec_str
              << "] b=" << batch << " h=" << nhead << " s=" << seqlen_q
              << " d=" << hdim_q << " topk=" << topk
              << " sim1=" << simthreshd1 << std::flush;

    // ---- allocate host tensors ----
    auto q_host = make_qkv_tensor<T>(batch, nhead, seqlen_q, hdim_q, i_perm);
    auto k_host = make_qkv_tensor<T>(batch, nhead_k, seqlen_k, hdim_q, i_perm);
    auto v_host = make_qkv_tensor<T>(batch, nhead_k, seqlen_k, hdim_v, i_perm);
    auto output_host = o_perm ? ck_tile::HostTensor<T>({batch, nhead, seqlen_q, hdim_v})
                              : ck_tile::HostTensor<T>({batch, seqlen_q, nhead, hdim_v});

    ck_tile::HostTensor<uint8_t> block_map_host({batch, nhead, num_q_blocks, num_k_blocks});
    ck_tile::HostTensor<int32_t> lut_host({batch, nhead, num_q_blocks, num_k_blocks});
    ck_tile::HostTensor<int32_t> valid_block_num_host({batch, nhead, num_q_blocks});

    ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed}(q_host);
    ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed + 1}(k_host);
    ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed + 2}(v_host);

    // ---- device tensors ----
    ck_tile::DeviceMem q_dev(q_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem k_dev(k_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem v_dev(v_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem o_dev(output_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem block_map_dev(block_map_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem lut_dev(lut_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem valid_bn_dev(valid_block_num_host.get_element_space_size_in_bytes());

    q_dev.ToDevice(q_host.data());
    k_dev.ToDevice(k_host.data());
    v_dev.ToDevice(v_host.data());
    o_dev.SetZero();
    block_map_dev.SetZero();
    lut_dev.SetZero();
    valid_bn_dev.SetZero();

    // ---- strides (BHSD when i_perm=true) ----
    auto q_strides = q_host.get_strides();
    auto k_strides = k_host.get_strides();
    auto v_strides = v_host.get_strides();
    auto o_strides = output_host.get_strides();

    float scale_s = 1.0f / std::sqrt(static_cast<float>(hdim_q));

    // ---- build blockmap args ----
    sparge_blockmap_traits bmap_traits;
    bmap_traits.data_type = std::is_same_v<T, ck_tile::half_t> ? "fp16" : "bf16";
    bmap_traits.hdim_q    = hdim_q;

    sparge_blockmap_args bmap_args;
    bmap_args.q_ptr              = q_dev.GetDeviceBuffer();
    bmap_args.k_ptr              = k_dev.GetDeviceBuffer();
    bmap_args.batch              = batch;
    bmap_args.seqlen_q           = seqlen_q;
    bmap_args.seqlen_k           = seqlen_k;
    bmap_args.hdim_q             = hdim_q;
    bmap_args.nhead_q            = nhead;
    bmap_args.nhead_k            = nhead_k;
    bmap_args.stride_q           = q_strides[i_perm ? 2 : 1];
    bmap_args.stride_k           = k_strides[i_perm ? 2 : 1];
    bmap_args.nhead_stride_q     = q_strides[i_perm ? 1 : 2];
    bmap_args.nhead_stride_k     = k_strides[i_perm ? 1 : 2];
    bmap_args.batch_stride_q     = q_strides[0];
    bmap_args.batch_stride_k     = k_strides[0];
    bmap_args.simthreshd1        = simthreshd1;
    bmap_args.cdfthreshd         = (topk < 0.0f) ? cdfthreshd : -1.0f;
    bmap_args.topk               = topk;
    bmap_args.scale              = scale_s;
    bmap_args.block_map_ptr      = block_map_dev.GetDeviceBuffer();
    bmap_args.lut_ptr            = (pipeline == "vsa") ? lut_dev.GetDeviceBuffer() : nullptr;
    bmap_args.valid_block_num_ptr = (pipeline == "vsa") ? valid_bn_dev.GetDeviceBuffer() : nullptr;

    // ---- build attention args ----
    ck_tile::stream_config stream_cfg;
    stream_cfg.stream_id_  = nullptr;
    stream_cfg.time_kernel_ = true;
    stream_cfg.log_level_  = kname;
    stream_cfg.cold_niters_ = warmup;
    stream_cfg.nrepeat_    = repeat;

    float avg_ms = -1.0f;

    if(pipeline == "jenga")
    {
        fmha_jenga_fwd_traits attn_traits;
        attn_traits.hdim_q        = hdim_q;
        attn_traits.hdim_v        = hdim_v;
        attn_traits.data_type     = std::is_same_v<T, ck_tile::half_t> ? "fp16" : "bf16";
        attn_traits.is_v_rowmajor = true;
        attn_traits.mask_type     = mask_enum::no_mask;
        attn_traits.bm0           = BLKQ;

        fmha_jenga_fwd_args attn_args;
        attn_args.q_ptr                    = q_dev.GetDeviceBuffer();
        attn_args.k_ptr                    = k_dev.GetDeviceBuffer();
        attn_args.v_ptr                    = v_dev.GetDeviceBuffer();
        attn_args.block_relation_onehot_ptr = block_map_dev.GetDeviceBuffer();
        attn_args.o_ptr                    = o_dev.GetDeviceBuffer();
        attn_args.seqlen_q     = seqlen_q;
        attn_args.seqlen_k     = seqlen_k;
        attn_args.batch        = batch;
        attn_args.max_seqlen_q = seqlen_q;
        attn_args.hdim_q       = hdim_q;
        attn_args.hdim_v       = hdim_v;
        attn_args.nhead_q      = nhead;
        attn_args.nhead_k      = nhead_k;
        attn_args.scale_s      = scale_s;
        attn_args.stride_q       = q_strides[i_perm ? 2 : 1];
        attn_args.stride_k       = k_strides[i_perm ? 2 : 1];
        attn_args.stride_v       = v_strides[i_perm ? 2 : 1];
        attn_args.stride_o       = o_strides[o_perm ? 2 : 1];
        attn_args.nhead_stride_q = q_strides[i_perm ? 1 : 2];
        attn_args.nhead_stride_k = k_strides[i_perm ? 1 : 2];
        attn_args.nhead_stride_v = v_strides[i_perm ? 1 : 2];
        attn_args.nhead_stride_o = o_strides[o_perm ? 1 : 2];
        attn_args.batch_stride_q = q_strides[0];
        attn_args.batch_stride_k = k_strides[0];
        attn_args.batch_stride_v = v_strides[0];
        attn_args.batch_stride_o = o_strides[0];
        attn_args.window_size_left  = -1;
        attn_args.window_size_right = -1;
        attn_args.mask_type         = 0;

        avg_ms = sparge_jenga_fwd(bmap_traits, bmap_args, attn_traits, attn_args, stream_cfg);
    }
    else if(pipeline == "vsa")
    {
        fmha_vsa_fwd_traits attn_traits;
        attn_traits.hdim_q        = hdim_q;
        attn_traits.hdim_v        = hdim_v;
        attn_traits.data_type     = std::is_same_v<T, ck_tile::half_t> ? "fp16" : "bf16";
        attn_traits.is_v_rowmajor = true;
        attn_traits.mask_type     = mask_enum::no_mask;
        attn_traits.bm0           = BLKQ;

        fmha_vsa_fwd_args attn_args;
        attn_args.q_ptr              = q_dev.GetDeviceBuffer();
        attn_args.k_ptr              = k_dev.GetDeviceBuffer();
        attn_args.v_ptr              = v_dev.GetDeviceBuffer();
        attn_args.lut_ptr            = lut_dev.GetDeviceBuffer();
        attn_args.valid_block_num_ptr = valid_bn_dev.GetDeviceBuffer();
        attn_args.o_ptr              = o_dev.GetDeviceBuffer();
        attn_args.seqlen_q     = seqlen_q;
        attn_args.seqlen_k     = seqlen_k;
        attn_args.batch        = batch;
        attn_args.max_seqlen_q = seqlen_q;
        attn_args.hdim_q       = hdim_q;
        attn_args.hdim_v       = hdim_v;
        attn_args.nhead_q      = nhead;
        attn_args.nhead_k      = nhead_k;
        attn_args.scale_s      = scale_s;
        attn_args.stride_q       = q_strides[i_perm ? 2 : 1];
        attn_args.stride_k       = k_strides[i_perm ? 2 : 1];
        attn_args.stride_v       = v_strides[i_perm ? 2 : 1];
        attn_args.stride_o       = o_strides[o_perm ? 2 : 1];
        attn_args.nhead_stride_q = q_strides[i_perm ? 1 : 2];
        attn_args.nhead_stride_k = k_strides[i_perm ? 1 : 2];
        attn_args.nhead_stride_v = v_strides[i_perm ? 1 : 2];
        attn_args.nhead_stride_o = o_strides[o_perm ? 1 : 2];
        attn_args.batch_stride_q = q_strides[0];
        attn_args.batch_stride_k = k_strides[0];
        attn_args.batch_stride_v = v_strides[0];
        attn_args.batch_stride_o = o_strides[0];
        attn_args.window_size_left  = -1;
        attn_args.window_size_right = -1;
        attn_args.mask_type         = 0;

        avg_ms = sparge_vsa_fwd_combined(bmap_traits, bmap_args, attn_traits, attn_args, stream_cfg);
    }
    else
    {
        std::cerr << "Unknown pipeline: " << pipeline << " (use jenga or vsa)\n";
        return false;
    }

    // ---- TFLOPS calculation (dense FMHA formula, so sparsity gains show as higher TFLOPS) ----
    std::size_t flop = static_cast<std::size_t>(batch) * nhead *
        (static_cast<std::size_t>(2) * seqlen_q * seqlen_k * hdim_q +
         static_cast<std::size_t>(2) * seqlen_q * seqlen_k * hdim_v);
    float tflops = (avg_ms > 0.f) ? static_cast<float>(flop) / 1.E9f / avg_ms : 0.f;

    if(avg_ms > 0.f)
    {
        std::cout << std::fixed << ", " << std::setprecision(3) << avg_ms << " ms, "
                  << std::setprecision(2) << tflops << " TFlops" << std::flush;
    }

    // ---- copy results back ----
    o_dev.FromDevice(output_host.data());
    block_map_dev.FromDevice(block_map_host.data());

    // ---- count active blocks ----
    ck_tile::index_t total_blocks = batch * nhead * num_q_blocks * num_k_blocks;
    ck_tile::index_t active_blocks = 0;
    for(size_t i = 0; i < block_map_host.mData.size(); ++i)
        if(block_map_host.mData[i])
            active_blocks++;
    float actual_sparsity = 1.0f - static_cast<float>(active_blocks) / static_cast<float>(total_blocks);
    std::cout << ", sparsity=" << std::setprecision(2) << actual_sparsity
              << "(" << active_blocks << "/" << total_blocks << ")" << std::flush;

    // ---- validation ----
    bool pass = true;
    if(do_validation)
    {
        auto q_ref = to_bhsd(q_host, i_perm);
        auto k_ref = to_bhsd(k_host, i_perm);
        auto v_ref = to_bhsd(v_host, i_perm);

        ck_tile::HostTensor<T> output_ref({batch, nhead, seqlen_q, hdim_v});
        ck_tile::reference_blocked_attention<T, uint8_t>(
            q_ref, k_ref, v_ref, block_map_host, output_ref, BLKQ, BLKK, scale_s);

        auto [rtol, atol] = get_error_tolerance<T>();

        float max_diff     = 0.0f;
        size_t num_errors  = 0;

        auto output_host_bhsd = to_bhsd(output_host, o_perm);
        for(size_t i = 0; i < output_host_bhsd.mData.size(); ++i)
        {
            float gpu_val  = to_float_for_compare(output_host_bhsd.mData[i]);
            float ref_val  = to_float_for_compare(output_ref.mData[i]);
            float diff     = std::abs(gpu_val - ref_val);
            float rel_diff = (std::abs(ref_val) > 1e-6f) ? diff / std::abs(ref_val) : diff;

            max_diff = std::max(max_diff, diff);

            if(diff > atol && rel_diff > rtol)
                num_errors++;
        }

        pass = (num_errors == 0);
        std::cout << ", " << (pass ? "PASS" : "FAIL")
                  << "(err=" << num_errors << "/" << output_host_bhsd.mData.size()
                  << " maxdiff=" << max_diff << ")";
    }

    std::cout << std::endl;
    return pass;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
    {
        std::cerr << "Failed to parse arguments\n";
        return -1;
    }

    std::string prec = arg_parser.get_str("prec");

    bool test_result = false;
    if(prec == "fp16")
    {
        test_result = run_test<ck_tile::half_t>(arg_parser);
    }
    else if(prec == "bf16")
    {
        test_result = run_test<ck_tile::bf16_t>(arg_parser);
    }
    else
    {
        std::cerr << "Unsupported precision: " << prec << "\n";
        return -1;
    }

    return test_result ? 0 : -1;
}
