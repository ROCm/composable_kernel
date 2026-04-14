// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Demo: Sparge block-map -> (delta LUT) -> VSA sparse attention (all-in-device)

#include <iostream>
#include <cmath>
#include <string>
#include "ck_tile/host.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host/reference/reference_blocked_attention.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"

#include "sparge_blockmap_trek.hpp"
#include "fmha_fwd_trek.hpp"
#include "sparge_tool.hpp"

// ============================================================================
// Helper Functions
// ============================================================================

template <typename T>
ck_tile::HostTensor<T> make_qkv_tensor(ck_tile::index_t batch,
                                       ck_tile::index_t nhead,
                                       ck_tile::index_t seqlen,
                                       ck_tile::index_t hdim,
                                       bool i_perm)
{
    if(i_perm)
    {
        return ck_tile::HostTensor<T>({batch, nhead, seqlen, hdim});
    }
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
    {
        for(ck_tile::index_t h = 0; h < nhead; ++h)
        {
            for(ck_tile::index_t s = 0; s < seqlen; ++s)
            {
                for(ck_tile::index_t d = 0; d < hdim; ++d)
                {
                    out(b, h, s, d) = is_bhsd ? tensor(b, h, s, d) : tensor(b, s, h, d);
                }
            }
        }
    }
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
// Command line argument parser
// ============================================================================

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("v", "1", "0:no validation, 1:cpu validation")
        .insert("b", "1", "batch size")
        .insert("h", "4", "num of head for q")
        .insert("h_k", "-1", "num of head for k/v, -1 means equal to h")
        .insert("s", "4096", "seqlen_q")
        .insert("s_k", "-1", "seqlen_k, -1 means equal to s")
        .insert("d", "128", "head dim for q, k")
        .insert("d_v", "-1", "head dim for v, -1 means equal to d")
        .insert("prec", "fp16", "data type: fp16/bf16")
        .insert("iperm", "1", "permute input, 1: b*h*s*d, 0: b*s*h*d")
        .insert("operm", "1", "permute output")
        .insert("seed", "42", "random seed")
        .insert("warmup", "5", "warmup iterations")
        .insert("repeat", "20", "benchmark iterations")
        .insert("kname", "0", "print kernel name")
        // Sparge-specific
        .insert("blkq", "64", "Sparge BLKQ")
        .insert("blkk", "128", "Sparge BLKK")
        .insert("simthreshd1", "0.6", "Sparge sim threshold")
        .insert("cdfthreshd", "0.98", "Sparge CDF threshold (used when topk < 0)")
        .insert("topk", "-1.0", "Sparge topk ratio in (0,1]; if > 0, overrides cdfthreshd");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

// ============================================================================
// Main Test Function
// ============================================================================

template <typename T>
bool run_test(const ck_tile::ArgParser& arg_parser)
{
    int do_validation         = arg_parser.get_int("v");
    ck_tile::index_t batch    = arg_parser.get_int("b");
    ck_tile::index_t nhead    = arg_parser.get_int("h");
    ck_tile::index_t nhead_k  = arg_parser.get_int("h_k");
    ck_tile::index_t seqlen_q = arg_parser.get_int("s");
    ck_tile::index_t seqlen_k = arg_parser.get_int("s_k");
    ck_tile::index_t hdim_q   = arg_parser.get_int("d");
    ck_tile::index_t hdim_v   = arg_parser.get_int("d_v");
    bool i_perm               = arg_parser.get_bool("iperm");
    bool o_perm               = arg_parser.get_bool("operm");
    uint32_t seed             = arg_parser.get_uint32("seed");
    int warmup                = arg_parser.get_int("warmup");
    int repeat                = arg_parser.get_int("repeat");
    int kname                 = arg_parser.get_int("kname");

    // Sparge params
    ck_tile::index_t blkq = arg_parser.get_int("blkq");
    ck_tile::index_t blkk = arg_parser.get_int("blkk");
    float simthreshd1     = arg_parser.get_float("simthreshd1");
    float cdfthreshd      = arg_parser.get_float("cdfthreshd");
    float topk            = arg_parser.get_float("topk");

    if(nhead_k < 0)
        nhead_k = nhead;
    if(seqlen_k < 0)
        seqlen_k = seqlen_q;
    if(hdim_v < 0)
        hdim_v = hdim_q;

    if(blkq != 64 || blkk != 128 || hdim_q != 128 || hdim_v != 128)
    {
        std::cout << "\n>>> TEST SKIPPED <<<" << std::endl;
        std::cout << "Sparge VSA kernel instances are generated for BLKQ=64, BLKK=128, "
                     "hdim_q=128, hdim_v=128 only."
                  << std::endl;
        std::cout << "TEST SKIPPED" << std::endl;
        return true;
    }

    ck_tile::index_t BLKQ = blkq;
    ck_tile::index_t BLKK = blkk;

    ck_tile::index_t num_q_blocks = (seqlen_q + BLKQ - 1) / BLKQ;
    ck_tile::index_t num_k_blocks = (seqlen_k + BLKK - 1) / BLKK;

    std::cout << "============================================================" << std::endl;
    std::cout << "[Sparge -> VSA Sparse Attention Demo]" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "  Batch: " << batch << ", nhead_q: " << nhead << ", nhead_k: " << nhead_k
              << std::endl;
    std::cout << "  seqlen_q: " << seqlen_q << ", seqlen_k: " << seqlen_k << std::endl;
    std::cout << "  hdim_q: " << hdim_q << ", hdim_v: " << hdim_v << std::endl;
    std::cout << "  BLKQ=" << BLKQ << ", BLKK=" << BLKK << std::endl;
    std::cout << "  num_q_blocks: " << num_q_blocks << ", num_k_blocks: " << num_k_blocks
              << std::endl;
    std::cout << "  Sparge(simthreshd1=" << simthreshd1 << ", cdfthreshd=" << cdfthreshd
              << ", topk=" << topk << ")" << std::endl;
    std::cout << "  i_perm: " << i_perm << ", o_perm: " << o_perm << std::endl;

    // Create host tensors and fill with random data
    ck_tile::HostTensor<T> q_host = make_qkv_tensor<T>(batch, nhead, seqlen_q, hdim_q, i_perm);
    ck_tile::HostTensor<T> k_host = make_qkv_tensor<T>(batch, nhead_k, seqlen_k, hdim_q, i_perm);
    ck_tile::HostTensor<T> v_host = make_qkv_tensor<T>(batch, nhead_k, seqlen_k, hdim_v, i_perm);
    ck_tile::HostTensor<T> output_host =
        o_perm ? ck_tile::HostTensor<T>({batch, nhead, seqlen_q, hdim_v})
               : ck_tile::HostTensor<T>({batch, seqlen_q, nhead, hdim_v});

    std::cout << "\nInitializing tensors..." << std::endl;
    ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed}(q_host);
    ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed + 1}(k_host);
    ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f, seed + 2}(v_host);

    // ==================================================================
    // Allocate device memory once, HtoD once
    // ==================================================================
    ck_tile::DeviceMem q_buf(q_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem k_buf(k_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem v_buf(v_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem o_buf(output_host.get_element_space_size_in_bytes());

    q_buf.ToDevice(q_host.data());
    k_buf.ToDevice(k_host.data());
    v_buf.ToDevice(v_host.data());

    const std::size_t bmap_bytes =
        static_cast<std::size_t>(batch) * nhead * num_q_blocks * num_k_blocks * sizeof(uint8_t);
    const std::size_t lut_bytes =
        static_cast<std::size_t>(batch) * nhead * num_q_blocks * num_k_blocks * sizeof(int32_t);
    const std::size_t valid_bytes =
        static_cast<std::size_t>(batch) * nhead * num_q_blocks * sizeof(int32_t);

    ck_tile::DeviceMem bmap_buf(bmap_bytes);
    ck_tile::DeviceMem lut_buf(lut_bytes);
    ck_tile::DeviceMem valid_buf(valid_bytes);
    bmap_buf.SetZero();
    lut_buf.SetZero();
    valid_buf.SetZero();

    // ==================================================================
    // Common stride calculations
    // ==================================================================
    assert(nhead % nhead_k == 0);
    const float scale_s = 1.0f / std::sqrt(static_cast<float>(hdim_q));

    const ck_tile::index_t stride_q       = i_perm ? hdim_q : nhead * hdim_q;
    const ck_tile::index_t stride_k       = i_perm ? hdim_q : nhead_k * hdim_q;
    const ck_tile::index_t stride_v       = i_perm ? hdim_v : nhead_k * hdim_v;
    const ck_tile::index_t stride_o       = o_perm ? hdim_v : nhead * hdim_v;
    const ck_tile::index_t nhead_stride_q = i_perm ? seqlen_q * hdim_q : hdim_q;
    const ck_tile::index_t nhead_stride_k = i_perm ? seqlen_k * hdim_q : hdim_q;
    const ck_tile::index_t nhead_stride_v = i_perm ? seqlen_k * hdim_v : hdim_v;
    const ck_tile::index_t nhead_stride_o = o_perm ? seqlen_q * hdim_v : hdim_v;
    const ck_tile::index_t batch_stride_q = nhead * seqlen_q * hdim_q;
    const ck_tile::index_t batch_stride_k = nhead_k * seqlen_k * hdim_q;
    const ck_tile::index_t batch_stride_v = nhead_k * hdim_v * seqlen_k;
    const ck_tile::index_t batch_stride_o = nhead * seqlen_q * hdim_v;

    std::string data_type = "fp16";
    if constexpr(std::is_same_v<T, ck_tile::bf16_t>)
        data_type = "bf16";

    std::string msk_str = "0";
    mask_info mask      = mask_info::decode(msk_str, seqlen_q, seqlen_k);

    // ==================================================================
    // GPU: Build block map + VSA LUT (always run, device-only)
    // ==================================================================
    std::cout << "Building Sparge block map + VSA LUT (GPU)..." << std::endl;
    {
        sparge_blockmap_args args;
        args.q_ptr               = q_buf.GetDeviceBuffer();
        args.k_ptr               = k_buf.GetDeviceBuffer();
        args.batch               = batch;
        args.seqlen_q            = seqlen_q;
        args.seqlen_k            = seqlen_k;
        args.hdim_q              = hdim_q;
        args.nhead_q             = nhead;
        args.nhead_k             = nhead_k;
        args.stride_q            = stride_q;
        args.stride_k            = stride_k;
        args.nhead_stride_q      = nhead_stride_q;
        args.nhead_stride_k      = nhead_stride_k;
        args.batch_stride_q      = batch_stride_q;
        args.batch_stride_k      = batch_stride_k;
        args.simthreshd1         = simthreshd1;
        args.cdfthreshd          = cdfthreshd;
        args.topk                = topk;
        args.scale               = scale_s;
        args.block_map_ptr       = bmap_buf.GetDeviceBuffer();
        args.lut_ptr             = lut_buf.GetDeviceBuffer();
        args.valid_block_num_ptr = valid_buf.GetDeviceBuffer();

        sparge_blockmap_traits traits;
        traits.data_type = data_type;
        traits.hdim_q    = hdim_q;

        sparge_blockmap_fwd(traits, args, ck_tile::stream_config{});
    }

    // ==================================================================
    // VSA sparse attention kernel (always run, LUT stays on device)
    // ==================================================================
    std::cout << "\n--- Running VSA sparse attention kernel ---" << std::endl;

    fmha_vsa_fwd_args fmha_args;
    fmha_args.q_ptr               = q_buf.GetDeviceBuffer();
    fmha_args.k_ptr               = k_buf.GetDeviceBuffer();
    fmha_args.v_ptr               = v_buf.GetDeviceBuffer();
    fmha_args.lut_ptr             = lut_buf.GetDeviceBuffer();
    fmha_args.valid_block_num_ptr = valid_buf.GetDeviceBuffer();
    fmha_args.o_ptr               = o_buf.GetDeviceBuffer();
    fmha_args.batch               = batch;
    fmha_args.seqlen_q            = seqlen_q;
    fmha_args.seqlen_k            = seqlen_k;
    fmha_args.max_seqlen_q        = seqlen_q;
    fmha_args.hdim_q              = hdim_q;
    fmha_args.hdim_v              = hdim_v;
    fmha_args.nhead_q             = nhead;
    fmha_args.nhead_k             = nhead_k;
    fmha_args.scale_s             = scale_s;
    fmha_args.stride_q            = stride_q;
    fmha_args.stride_k            = stride_k;
    fmha_args.stride_v            = stride_v;
    fmha_args.stride_o            = stride_o;
    fmha_args.nhead_stride_q      = nhead_stride_q;
    fmha_args.nhead_stride_k      = nhead_stride_k;
    fmha_args.nhead_stride_v      = nhead_stride_v;
    fmha_args.nhead_stride_o      = nhead_stride_o;
    fmha_args.batch_stride_q      = batch_stride_q;
    fmha_args.batch_stride_k      = batch_stride_k;
    fmha_args.batch_stride_v      = batch_stride_v;
    fmha_args.batch_stride_o      = batch_stride_o;
    fmha_args.window_size_left    = mask.left;
    fmha_args.window_size_right   = mask.right;
    fmha_args.mask_type           = static_cast<ck_tile::index_t>(mask.type);

    fmha_vsa_fwd_traits fmha_traits;
    fmha_traits.hdim_q        = hdim_q;
    fmha_traits.hdim_v        = hdim_v;
    fmha_traits.data_type     = data_type;
    fmha_traits.is_v_rowmajor = true;
    fmha_traits.mask_type     = mask.type;

    ck_tile::stream_config stream_config{nullptr,
                                         true,
                                         /* log_level = */ kname ? 1 : 0,
                                         warmup,
                                         repeat,
                                         false};

    float avg_time_ms = sparge_vsa_fwd(fmha_traits, fmha_args, stream_config);

    std::cout << "\n>>>> VSA sparse attention average time: " << avg_time_ms << " ms <<<<"
              << std::endl;

    // DtoH: attention output (always needed)
    o_buf.FromDevice(output_host.data(), output_host.get_element_space_size_in_bytes());

    // DtoH: block_map (needed for sparsity stats and validation)
    ck_tile::HostTensor<uint8_t> block_map_gpu({batch, nhead, num_q_blocks, num_k_blocks});
    bmap_buf.FromDevice(block_map_gpu.data(), bmap_bytes);

    // ==================================================================
    // Sparsity statistics (pure CPU, reads block_map HostTensor)
    // ==================================================================
    std::size_t total_blocks  = 0;
    std::size_t active_blocks = 0;
    for(ck_tile::index_t b = 0; b < batch; ++b)
    {
        for(ck_tile::index_t h = 0; h < nhead; ++h)
        {
            for(ck_tile::index_t qb = 0; qb < num_q_blocks; ++qb)
            {
                for(ck_tile::index_t kb = 0; kb < num_k_blocks; ++kb)
                {
                    total_blocks++;
                    if(block_map_gpu(b, h, qb, kb) != 0)
                        active_blocks++;
                }
            }
        }
    }
    float actual_sparsity =
        1.0f - static_cast<float>(active_blocks) / static_cast<float>(total_blocks);
    std::cout << "\n  Actual sparsity: " << actual_sparsity << " (" << active_blocks << "/"
              << total_blocks << " blocks active)" << std::endl;

    // ==================================================================
    // Validation (only when -v=1)
    // ==================================================================
    bool pass = true;
    if(do_validation)
    {
        std::cout << "\n--- Performing CPU validation ---" << std::endl;

        // CPU golden: block map + VSA LUT
        std::cout << "Building Sparge block map (CPU golden)..." << std::endl;
        sparge::SpargeParams p;
        p.BLKQ        = static_cast<int>(BLKQ);
        p.BLKK        = static_cast<int>(BLKK);
        p.simthreshd1 = simthreshd1;
        p.cdfthreshd  = cdfthreshd;
        p.topk        = topk;
        p.i_perm      = i_perm;

        ck_tile::HostTensor<uint8_t> block_relation_onehot =
            sparge::build_block_map_meansim(q_host, k_host, p);

        std::cout << "Converting block map to VSA LUT (delta, CPU)..." << std::endl;
        auto vsa_lut_cpu = sparge::block_map_to_vsa_lut_delta(block_relation_onehot);

        // DtoH: LUT + valid_block_num (only for validation)
        sparge::VSALut vsa_lut_gpu{
            ck_tile::HostTensor<int32_t>({batch, nhead, num_q_blocks, num_k_blocks}),
            ck_tile::HostTensor<int32_t>({batch, nhead, num_q_blocks}),
        };
        lut_buf.FromDevice(vsa_lut_gpu.lut.data(), lut_bytes);
        valid_buf.FromDevice(vsa_lut_gpu.valid_block_num.data(), valid_bytes);

        // Validate block map
        std::cout << "\n--- Validating GPU block map vs CPU golden ---" << std::endl;
        {
            std::size_t bmap_mismatches = 0;
            for(ck_tile::index_t b = 0; b < batch; ++b)
            {
                for(ck_tile::index_t h = 0; h < nhead; ++h)
                {
                    for(ck_tile::index_t qb = 0; qb < num_q_blocks; ++qb)
                    {
                        for(ck_tile::index_t kb = 0; kb < num_k_blocks; ++kb)
                        {
                            if(block_map_gpu(b, h, qb, kb) != block_relation_onehot(b, h, qb, kb))
                            {
                                bmap_mismatches++;
                                if(bmap_mismatches <= 10)
                                {
                                    std::cout
                                        << "  block_map mismatch at [" << b << "," << h << "," << qb
                                        << "," << kb << "]: GPU="
                                        << static_cast<int>(block_map_gpu(b, h, qb, kb)) << " CPU="
                                        << static_cast<int>(block_relation_onehot(b, h, qb, kb))
                                        << std::endl;
                                }
                            }
                        }
                    }
                }
            }
            std::cout << "  Block map mismatches: " << bmap_mismatches << " / "
                      << (batch * nhead * num_q_blocks * num_k_blocks) << std::endl;
            if(bmap_mismatches > 0)
            {
                std::cout << ">>> GPU BLOCK MAP VALIDATION FAILED <<<" << std::endl;
                pass = false;
            }
            else
            {
                std::cout << ">>> GPU BLOCK MAP VALIDATION PASSED <<<" << std::endl;
            }
        }

        // Validate VSA LUT
        std::cout << "\n--- Validating GPU VSA LUT vs CPU golden ---" << std::endl;
        {
            std::size_t lut_mismatches   = 0;
            std::size_t valid_mismatches = 0;
            for(ck_tile::index_t b = 0; b < batch; ++b)
            {
                for(ck_tile::index_t h = 0; h < nhead; ++h)
                {
                    for(ck_tile::index_t qb = 0; qb < num_q_blocks; ++qb)
                    {
                        if(vsa_lut_gpu.valid_block_num(b, h, qb) !=
                           vsa_lut_cpu.valid_block_num(b, h, qb))
                        {
                            valid_mismatches++;
                            if(valid_mismatches <= 5)
                            {
                                std::cout << "  valid_block_num mismatch at [" << b << "," << h
                                          << "," << qb
                                          << "]: GPU=" << vsa_lut_gpu.valid_block_num(b, h, qb)
                                          << " CPU=" << vsa_lut_cpu.valid_block_num(b, h, qb)
                                          << std::endl;
                            }
                        }
                        for(ck_tile::index_t kb = 0; kb < num_k_blocks; ++kb)
                        {
                            if(vsa_lut_gpu.lut(b, h, qb, kb) != vsa_lut_cpu.lut(b, h, qb, kb))
                            {
                                lut_mismatches++;
                                if(lut_mismatches <= 10)
                                {
                                    std::cout
                                        << "  LUT mismatch at [" << b << "," << h << "," << qb
                                        << "," << kb << "]: GPU=" << vsa_lut_gpu.lut(b, h, qb, kb)
                                        << " CPU=" << vsa_lut_cpu.lut(b, h, qb, kb) << std::endl;
                                }
                            }
                        }
                    }
                }
            }
            std::cout << "  LUT mismatches: " << lut_mismatches << std::endl;
            std::cout << "  valid_block_num mismatches: " << valid_mismatches << std::endl;
            if(lut_mismatches == 0 && valid_mismatches == 0)
            {
                std::cout << ">>> GPU VSA LUT VALIDATION PASSED <<<" << std::endl;
            }
            else
            {
                std::cout << ">>> GPU VSA LUT VALIDATION FAILED <<<" << std::endl;
                pass = false;
            }
        }

        // Validate attention output
        float scale = 1.0f / std::sqrt(static_cast<float>(hdim_q));

        std::cout << "\nComputing reference attention output..." << std::endl;
        auto q_ref = to_bhsd(q_host, i_perm);
        auto k_ref = to_bhsd(k_host, i_perm);
        auto v_ref = to_bhsd(v_host, i_perm);

        ck_tile::HostTensor<T> output_ref({batch, nhead, seqlen_q, hdim_v});
        ck_tile::reference_blocked_attention<T, uint8_t>(
            q_ref, k_ref, v_ref, block_relation_onehot, output_ref, BLKQ, BLKK, scale);

        auto [rtol, atol] = get_error_tolerance<T>();

        float max_diff         = 0.0f;
        float max_rel_diff     = 0.0f;
        std::size_t num_errors = 0;

        auto output_host_bhsd = to_bhsd(output_host, o_perm);
        for(std::size_t i = 0; i < output_host_bhsd.mData.size(); ++i)
        {
            float gpu_val  = to_float_for_compare(output_host_bhsd.mData[i]);
            float ref_val  = to_float_for_compare(output_ref.mData[i]);
            float diff     = std::abs(gpu_val - ref_val);
            float rel_diff = (std::abs(ref_val) > 1e-6f) ? diff / std::abs(ref_val) : diff;

            max_diff     = std::max(max_diff, diff);
            max_rel_diff = std::max(max_rel_diff, rel_diff);

            if(diff > atol && rel_diff > rtol)
            {
                num_errors++;
                if(num_errors <= 5)
                {
                    std::cout << "  Mismatch at index " << i << ": GPU=" << gpu_val
                              << ", Ref=" << ref_val << ", Diff=" << diff << std::endl;
                }
            }
        }

        std::cout << "\nAttention validation results:" << std::endl;
        std::cout << "  Max absolute difference: " << max_diff << std::endl;
        std::cout << "  Max relative difference: " << max_rel_diff << std::endl;
        std::cout << "  Number of mismatches: " << num_errors << " / "
                  << output_host_bhsd.mData.size() << std::endl;

        if(num_errors == 0)
        {
            std::cout << "\n>>> VALIDATION PASSED <<<" << std::endl;
        }
        else
        {
            std::cout << "\n>>> VALIDATION FAILED <<<" << std::endl;
            pass = false;
        }
    }

    std::cout << "\n" << (pass ? "TEST PASSED" : "TEST FAILED") << std::endl;
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
        std::cerr << "Failed to parse arguments" << std::endl;
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
        std::cerr << "Unsupported precision: " << prec << std::endl;
        return -1;
    }

    return test_result ? 0 : -1;
}
