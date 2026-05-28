// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <ck_tile/core/numeric/bfloat16.hpp>
#include <ck_tile/core/numeric/half.hpp>
#include <ck_tile/core/numeric/math.hpp>
#include <ck_tile/core/utility/functional.hpp>
#include <ck_tile/host/arg_parser.hpp>
#include <ck_tile/host/device_memory.hpp>
#include <ck_tile/host/fill.hpp>
#include <ck_tile/host/check_err.hpp>
#include <ck_tile/host/host_tensor.hpp>
#include <ck_tile/host/reference/reference_batched_gemm.hpp>
#include <ck_tile/host/reference/reference_batched_masking.hpp>
#include <ck_tile/host/reference/reference_batched_softmax.hpp>

#include "unified_attention.hpp"
#include "mask.hpp"

// const ck_tile::index_t page_blk_size         = 32;
// num_queries_per_kv is now a runtime arg (see parse_cmd_args)

auto parse_cmd_args(int argc, char* argv[]) -> std::pair<bool, ck_tile::ArgParser>
{
    ck_tile::ArgParser arg_parser;
    arg_parser
        .insert("prec", "bf16", "data type. fp16/bf16")
        // .insert("b", "3", "batch size")
        .insert("nqpkv", "1", "num queries per kv head (GQA ratio, e.g. 1 for MHA, 8 for GQA-8)")
        .insert("h_k", "8", "num head for k/v. num head for q is nqpkv times this")
        .insert("s", "3328", "max seqlen_q")
        .insert("s_k", "-1", "max seqlen_k, -1 means equal to s")
        .insert("nb", "1024", "num_blks")
        .insert("b", "3", "batch")
        .insert("d", "128", "head dim for q & k")
        .insert("scale_s", "0", "scale factor of S. 0 means equal to 1/sqrt(hdim)")
        // TODO scale factors
        .insert("scale", "1", "")
        .insert("scale_k", "1", "")
        .insert("scale_v", "1", "")
        .insert("scale_out", "1", "")
        .insert("iperm",
                "0",
                "permute input\n"
                "if true, will be b*h*s*d, else b*s*h*d")
        .insert("operm", "0", "permute output")
        .insert("mask",
                "b",
                "attention mask. accepts the same syntax as 01_fmha:\n"
                "  '0'             : no mask\n"
                "  '1' or 't'      : causal mask from top-left\n"
                "  '2' or 'b'      : causal mask from bottom-right (default)\n"
                "  'xt:N'/'xb:N'   : xformer-style window_size N from top-left/bottom-right\n"
                "                    N<0 means causal, N>0 means sliding-window attention\n"
                "  't:l,r'/'b:l,r' : FA-style left/right window from top-left/bottom-right\n"
                "  'g:y,x'         : generic mask coordinate")
        .insert("verify", "1", "0:no verify, 1:verify")
        .insert("varlen", "1", "0: fixed length, 1: variable length")
        // Debug switch for analytical bug isolation.
        //   0 — normal random fill (default).
        //   1 — Q=K=V=1. Uniform softmax × V=1 → o[m,d] = 1 for every valid
        //       (token, head, dim). Catches NaN/Inf in the accumulator but
        //       NOT mask/indexing bugs (sum-of-uniform-weights stays 1
        //       regardless of which cells are valid).
        //   2 — Q=K=1, V[n_kv, h, d] = n_kv. Q*K^T is constant → softmax is
        //       uniform over the *valid* KV range per row, so
        //       o[m, h, d] = mean(n in valid_range(m)). Reads the actual SWA
        //       window centre off the output, so any mask/Step-D/page-index
        //       bug shows up immediately as a deviation from the analytical
        //       mean.
        .insert("debug_probe",
                "0",
                "0:random fill (default), 1:Q=K=V=1 (NaN check), "
                "2:Q=K=1 V=position (mask/index check)")
        .insert("seed",
                "11939",
                "random seed used for initializing input tensors. 0 for "
                "non-deterministic seed")
        .insert("warmup", "5", "number of iterations before benchmark the kernel")
        .insert("repeat", "30", "number of iterations to benchmark the kernel")
        .insert("page_blk_size", "128", "page block size of kv cache")
        // Optional effective seqlen override (exclude PAD) for batch mode
        .insert("query_lens",
                "",
                "Batch-mode only: per-batch effective seqlen for Q (exclude PAD).\n"
                "Comma-separated list of length 'b'. If empty, no override.")
        .insert("kv_lens",
                "",
                "Batch-mode only: per-batch effective seqlen for KV (exclude PAD).\n"
                "Comma-separated list of length 'b'. If empty, no override.")
        // Per-query-head learnable sink tensor (GPT-OSS / vLLM convention;
        // one scalar per Q head, the "virtual key" that participates in the
        // softmax denominator but contributes nothing to the V accumulator).
        // The flag is parsed into `Problem::sinks` as a host-side
        // `std::vector<float>` of length `nhead_q` and threaded through to
        // the host reference. The kernel does not yet consume it (no
        // `kHasSink` device-side branch); until that lands, a non-empty
        // sinks vector makes the reference diverge from the kernel and
        // verification is expected to fail.
        // Accepted syntaxes:
        //   ''            : no sink (default — host reference is the
        //                   classic no-sink softmax).
        //   'none'        : explicit no sink (same as empty).
        //   'random'      : sample N(0, 0.5) scalars; deterministic on
        //                   `-seed=`.
        //   'random:S'    : same, but with an explicit seed S (overrides
        //                   `-seed=` for the sink draw only).
        //   'const:F'     : broadcast the single float F across all heads
        //                   (e.g. 'const:0.0', 'const:-1e4', 'const:1.5').
        //   'F1,F2,...'   : explicit per-head CSV of length `nhead_q`.
        .insert("sink",
                "",
                "attention sinks (one scalar per Q head). Empty / 'none' = no sink.\n"
                "  'random[:seed]' : per-head N(0, 0.5) draw\n"
                "  'const:F'       : broadcast F across all heads\n"
                "  'F1,F2,...'     : explicit per-head CSV (length == h_k*nqpkv)\n"
                "The host reference applies this immediately; the kernel does\n"
                "not yet consume it.");

    bool result = arg_parser.parse(argc, argv);
    return std::make_pair(result, arg_parser);
}

// Parse the `-sink=` argument into a per-Q-head float vector. Returns
// an empty vector when the flag is absent or set to "none"; both cases
// reduce to the no-sink reference path in `host::fmha_fwd`.
//
// Accepted syntaxes (see the CLI help in parse_cmd_args for the canonical
// list):
//   ""                    -> no sink
//   "none"                -> no sink
//   "random"              -> per-head N(0, 0.5), seeded by `default_seed`
//   "random:N"            -> per-head N(0, 0.5), seeded by N
//   "const:F"             -> broadcast F across all `nhead_q` heads
//   "F1,F2,...,Fn"        -> explicit CSV; size must equal `nhead_q`
//
// Errors abort the program with std::exit(2) — this is a CLI parser
// failure, not a verification failure, so a non-zero exit code that
// differs from the verification "RED" code keeps the smoke test honest.
inline std::vector<float> parse_sinks(const std::string& spec,
                                      ck_tile::index_t nhead_q,
                                      uint32_t default_seed)
{
    if(spec.empty() || spec == "none")
    {
        return {};
    }

    auto fail = [&](const std::string& msg) {
        std::cerr << "ERROR: -sink= parse failed: " << msg
                  << " (got '" << spec << "', expected 'none', 'random[:N]', "
                  << "'const:F', or a CSV of " << nhead_q << " floats)" << std::endl;
        std::exit(2);
    };

    // 'random' / 'random:N'
    if(spec.rfind("random", 0) == 0)
    {
        uint32_t seed = default_seed;
        if(spec.size() > std::string("random").size())
        {
            if(spec[std::string("random").size()] != ':')
                fail("'random' must be followed by ':N' or nothing");
            try
            {
                seed = static_cast<uint32_t>(
                    std::stoul(spec.substr(std::string("random:").size())));
            }
            catch(...)
            {
                fail("invalid seed after 'random:'");
            }
        }
        std::mt19937                  gen(seed ? seed : 12345u);
        std::normal_distribution<float> dist(0.f, 0.5f);
        std::vector<float>            out(static_cast<size_t>(nhead_q));
        for(auto& v : out)
            v = dist(gen);
        return out;
    }

    // 'const:F'
    {
        const std::string prefix = "const:";
        if(spec.rfind(prefix, 0) == 0)
        {
            float f;
            try
            {
                f = std::stof(spec.substr(prefix.size()));
            }
            catch(...)
            {
                fail("invalid float after 'const:'");
                return {};
            }
            return std::vector<float>(static_cast<size_t>(nhead_q), f);
        }
    }

    // CSV of nhead_q floats
    {
        std::vector<float> out;
        out.reserve(static_cast<size_t>(nhead_q));
        std::stringstream ss(spec);
        std::string token;
        while(std::getline(ss, token, ','))
        {
            if(token.empty())
                continue;
            try
            {
                out.push_back(std::stof(token));
            }
            catch(...)
            {
                fail("invalid float in CSV element '" + token + "'");
                return {};
            }
        }
        if(static_cast<ck_tile::index_t>(out.size()) != nhead_q)
        {
            std::stringstream msg;
            msg << "CSV length " << out.size() << " != nhead_q (" << nhead_q << ")";
            fail(msg.str());
        }
        return out;
    }
}

auto seqlen_preprocess(ck_tile::index_t batch,
                       ck_tile::index_t max_seqlen_q,
                       ck_tile::index_t max_seqlen_kv,
                       const std::vector<int>& query_lens_input,
                       const std::vector<int>& kv_lens_input,
                       bool varlen) -> std::pair<std::vector<int>, std::vector<int>>
{
    // If both query_lens and kv_lens are provided, return them directly
    if(!query_lens_input.empty() && !kv_lens_input.empty())
    {
        return std::make_pair(query_lens_input, kv_lens_input);
    }

    std::vector<int> query_lens;
    std::vector<int> kv_lens;

    if(!varlen)
    {
        // Fixed length mode: fill with max seqlen
        query_lens.assign(batch, max_seqlen_q);
        kv_lens.assign(batch, max_seqlen_kv);
    }
    else
    {
        // Variable length mode: generate random lengths up to max
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> q_dist(1, max_seqlen_q);
        std::uniform_int_distribution<int> kv_dist(1, max_seqlen_kv);

        query_lens.resize(batch);
        kv_lens.resize(batch);

        for(ck_tile::index_t i = 0; i < batch; ++i)
        {
            query_lens[i] = q_dist(gen);
            kv_lens[i]    = kv_dist(gen);
        }
    }

    return std::make_pair(query_lens, kv_lens);
}

struct Problem
{
    explicit Problem(const ck_tile::ArgParser& args)
    {
        data_type = args.get_str("prec") == "fp16"
                        ? ck_tile::unified_attention_args::data_type_enum::fp16
                        : ck_tile::unified_attention_args::data_type_enum::bf16;
        num_blks  = args.get_int("nb");
        nhead_kv  = args.get_int("h_k");
        num_queries_per_kv = args.get_int("nqpkv");
        nhead_q = nhead_kv * num_queries_per_kv;

        ck_tile::index_t max_seqlen_q  = args.get_int("s");
        ck_tile::index_t max_seqlen_kv = args.get_int("s_k");

        if(max_seqlen_kv == -1)
        {
            max_seqlen_kv = max_seqlen_q;
        }

        hdim       = args.get_int("d");
        query_lens = args.get_int_vec("query_lens");
        kv_lens    = args.get_int_vec("kv_lens");
        assert(query_lens.size() == kv_lens.size() &&
               "query_lens and kv_lens must have the same length b");
        batch         = args.get_int("b");
        page_blk_size = args.get_int("page_blk_size");

        bool varlen = args.get_bool("varlen");
        auto [query_lens_, kv_lens_] =
            seqlen_preprocess(batch, max_seqlen_q, max_seqlen_kv, query_lens, kv_lens, varlen);

        query_lens = query_lens_;
        kv_lens    = kv_lens_;
        batch      = query_lens.size();

        // Calculate scale_s
        scale_s = args.get_float("scale_s");
        if(scale_s == 0.0f)
            scale_s = 1.0f / ck_tile::sqrt(static_cast<float>(hdim));

        // Initialize other scales
        scale      = args.get_float("scale");
        scale_k    = args.get_float("scale_k");
        scale_v    = args.get_float("scale_v");
        num_tokens = 0;
        for(const auto& len : query_lens)
        {
            num_tokens += len;
        }

        mask_str = args.get_str("mask");
        // Decode once with the maximum batch shape for top-level reporting and
        // for the kernel-side mask_type. The host reference re-decodes per-batch
        // with each batch's effective seqlens (varlen-aware) inside run_impl.
        const ck_tile::index_t report_seqlen_q =
            query_lens.empty()
                ? max_seqlen_q
                : *std::max_element(query_lens.begin(), query_lens.end());
        const ck_tile::index_t report_seqlen_kv =
            kv_lens.empty()
                ? max_seqlen_kv
                : *std::max_element(kv_lens.begin(), kv_lens.end());
        mask = mask_info::decode(mask_str, report_seqlen_q, report_seqlen_kv);

        // Sink plumbing. `sinks` is per-Q-head; empty vector means "no
        // sink" (matches Triton's `sinks=None` convention). The host
        // reference path consumes it immediately, but `args.sink_ptr` is
        // not set on the device side yet — the kargs wiring and the
        // kernel-side `kHasSink` branch are still TODO. Until both
        // arrive, a non-empty `sinks` makes the reference diverge from
        // the kernel output (intentional: smoke tests use this gap to
        // detect when the kernel-side path comes online).
        sink_str = args.get_str("sink");
        const uint32_t sink_default_seed = static_cast<uint32_t>(args.get_uint32("seed"));
        sinks                            = parse_sinks(sink_str, nhead_q, sink_default_seed);
    }

    std::vector<ck_tile::index_t> get_query_shape() const { return {num_tokens, nhead_q, hdim}; }

    std::vector<ck_tile::index_t> get_key_shape() const
    {
        return {num_blks, page_blk_size, nhead_kv, hdim};
    }

    std::vector<ck_tile::index_t> get_value_shape() const
    {
        return {num_blks, page_blk_size, nhead_kv, hdim};
    }

    std::vector<ck_tile::index_t> get_output_shape() const { return {num_tokens, nhead_q, hdim}; }

    ck_tile::unified_attention_args::data_type_enum data_type;
    ck_tile::index_t batch;
    ck_tile::index_t num_blks;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_kv;
    ck_tile::index_t num_queries_per_kv;
    ck_tile::index_t hdim;
    ck_tile::index_t page_blk_size;
    ck_tile::index_t num_tokens;
    float scale_s;
    float scale;
    float scale_k;
    float scale_v;
    std::string mask_str;
    mask_info mask;
    std::vector<int> query_lens;
    std::vector<int> kv_lens;
    // Per-Q-head sink scalars. Empty == no sink.
    std::string        sink_str;
    std::vector<float> sinks;
};

struct RunConfig
{
    explicit RunConfig(const ck_tile::ArgParser& args)
    {
        seed = args.get_uint32("seed");
        if(*seed == 0)
        {
            seed.reset();
        }

        kernel_warmup = args.get_int("warmup");
        kernel_repeat = args.get_int("repeat");
        verify        = args.get_bool("verify");
        debug_probe   = args.get_int("debug_probe");
    }

    std::optional<uint32_t> seed;
    int kernel_warmup;
    int kernel_repeat;
    bool verify;
    int debug_probe;
};

template <typename DataType>
auto generate_qkv(const Problem& problem,
                  [[maybe_unused]] std::optional<uint32_t> seed        = std::nullopt,
                  int                                     debug_probe = 0)
    -> std::tuple<ck_tile::HostTensor<DataType>,
                  ck_tile::HostTensor<DataType>,
                  ck_tile::HostTensor<DataType>>
{
    ck_tile::HostTensor<DataType> q(problem.get_query_shape());
    ck_tile::HostTensor<DataType> k(problem.get_key_shape());
    ck_tile::HostTensor<DataType> v(problem.get_value_shape());

    if(debug_probe == 1)
    {
        std::fill(q.begin(), q.end(), DataType{1});
        std::fill(k.begin(), k.end(), DataType{1});
        std::fill(v.begin(), v.end(), DataType{1});
    }
    else if(debug_probe == 2)
    {
        // Q = K = 1 → Q*K^T is constant → softmax is uniform over the valid
        // KV range for each row. V is filled with the *logical* token index
        // assuming an identity block_tables (main() overrides block_tables to
        // identity when debug_probe == 2). The expected output is then
        //   o[m, h, d] = mean( n_kv  for  n_kv in valid_range(m) )
        // which is a per-row analytical constant that depends on the SWA
        // mask and the V index resolution. Any deviation pinpoints the
        // offending stage (mask coords, Step D, page-table lookup, within-
        // page offset).
        std::fill(q.begin(), q.end(), DataType{1});
        std::fill(k.begin(), k.end(), DataType{1});
        // V[phys_blk, within, head_kv, d] = phys_blk * page_blk_size + within
        const auto vshape = v.mDesc.get_lengths(); // {num_blks, page_blk_size, nhead_kv, hdim}
        const auto nb     = static_cast<ck_tile::index_t>(vshape[0]);
        const auto pb_sz  = static_cast<ck_tile::index_t>(vshape[1]);
        const auto nh     = static_cast<ck_tile::index_t>(vshape[2]);
        const auto hd     = static_cast<ck_tile::index_t>(vshape[3]);
        for(ck_tile::index_t pb = 0; pb < nb; ++pb)
            for(ck_tile::index_t wp = 0; wp < pb_sz; ++wp)
            {
                const float val = static_cast<float>(pb * problem.page_blk_size + wp);
                for(ck_tile::index_t h = 0; h < nh; ++h)
                    for(ck_tile::index_t d = 0; d < hd; ++d)
                        v(pb, wp, h, d) = static_cast<DataType>(val);
            }
    }
    else
    {
        ck_tile::FillNormalDistribution<DataType>{0.f, 3.f, seed}(q);
        ck_tile::FillNormalDistribution<DataType>{0.f, 3.f, seed}(k);
        ck_tile::FillNormalDistribution<DataType>{0.f, 3.f, seed}(v);
    }

    return std::make_tuple(q, k, v);
}

namespace host {
template <typename AccDataType,
          typename PDataType,
          typename QDataType,
          typename KDataType,
          typename VDataType,
          typename ODataType,
          typename QElementOp,
          typename KElementOp,
          typename VElementOp,
          typename SAccElementOp>
CK_TILE_HOST void fmha_fwd(const ck_tile::HostTensor<QDataType>& q_bshd,
                           const ck_tile::HostTensor<KDataType>& k_bshd,
                           const ck_tile::HostTensor<VDataType>& v_bshd,
                           const mask_info& mask,
                           ck_tile::HostTensor<ODataType>& o_bshd,
                           const QElementOp& q_element_op        = {},
                           const KElementOp& k_element_op        = {},
                           const VElementOp& v_element_op        = {},
                           const SAccElementOp& s_acc_element_op = {},
                           // Per-Q-head sink scalars (length `nhead_q`).
                           // Empty span == no sink (classic softmax). When set,
                           // each (head_q, seqlen_q) row gets one virtual key
                           // with raw logit `sinks[head_q]` (same scale as the
                           // already-scaled `s_host_ref` entries — matches
                           // Triton's `attn = cat([attn, sinks_aux])` pattern in
                           // op_tests/triton_tests/attention/test_unified_attention.py).
                           const std::vector<float>& sinks = {})
{
    const int batch_size = q_bshd.mDesc.get_lengths()[0];
    const int seqlen_q   = q_bshd.mDesc.get_lengths()[1];
    const int seqlen_kv  = k_bshd.mDesc.get_lengths()[1];
    const int nhead_q    = q_bshd.mDesc.get_lengths()[2];
    const int nhead_kv   = k_bshd.mDesc.get_lengths()[2];
    const int hdim_qk    = q_bshd.mDesc.get_lengths()[3];
    const int hdim_v     = v_bshd.mDesc.get_lengths()[3];

    const int nr = nhead_q / nhead_kv;

    const bool has_sinks = !sinks.empty();
    if(has_sinks)
    {
        assert(static_cast<int>(sinks.size()) == nhead_q &&
               "sinks vector must have length nhead_q");
    }

    ck_tile::HostTensor<QDataType> q_host_ref({nhead_q, seqlen_q, hdim_qk});
    ck_tile::HostTensor<KDataType> k_host_ref({nhead_q, seqlen_kv, hdim_qk});
    ck_tile::HostTensor<VDataType> v_host_ref({nhead_q, hdim_v, seqlen_kv});
    ck_tile::HostTensor<ODataType> o_host_ref({nhead_q, seqlen_q, hdim_v});

    ck_tile::HostTensor<AccDataType> s_host_ref({nhead_q, seqlen_q, seqlen_kv});
    ck_tile::HostTensor<PDataType> p_host_ref({nhead_q, seqlen_q, seqlen_kv});
    // do computation for each batch
    for(int b = 0; b < batch_size; ++b)
    {
        // copy per-batch data from input tensors
        // clang-format off
        q_host_ref.ForEach([&](auto& self, auto idx) { self(idx) = q_bshd(b, idx[1], idx[0]     ,
        idx[2]); }); 
        k_host_ref.ForEach([&](auto& self, auto idx) { self(idx) = k_bshd(b, idx[1],
        idx[0] / nr, idx[2]); });
        v_host_ref.ForEach([&](auto& self, auto idx) { self(idx) =
        v_bshd(b, idx[2], idx[0] / nr, idx[1]); });
        // clang-format on
        ck_tile::reference_batched_gemm<QDataType, KDataType, AccDataType>(
            q_host_ref, k_host_ref, s_host_ref, q_element_op, k_element_op, s_acc_element_op);

        if(mask.type != mask_enum::no_mask)
        {
            // Always use the GenericMask (IsLocal=true) path so both classical causal
            // (left=-1, right=0) and sliding-window (left>=0) flow through the same
            // codepath. The helper translates left/right into y/x mask coordinates
            // and is_top_left selects the corner.
            const bool is_top_left = (mask.type == mask_enum::mask_top_left);
            ck_tile::reference_batched_masking(
                s_host_ref,
                ck_tile::make_generic_attention_mask_from_lr_window<
                    UnifiedAttentionMasks::GenericMask>(
                    mask.left, mask.right, seqlen_q, seqlen_kv, /*repeat_idx=*/1, is_top_left));
        }
        if(has_sinks)
        {
            // Sink-aware softmax (the "virtual key" trick, inlined). For each
            // (head_q, seqlen_q) row, treat `sinks[head_q]` as an additional
            // raw logit in the same scale as the already-scaled S row, then:
            //   m       = max(max(S[h,q,:]), sinks[h])
            //   denom   = sum_n exp(S[h,q,n] - m) + exp(sinks[h] - m)
            //   P[h,q,n] = exp(S[h,q,n] - m) / denom
            // The sink contributes nothing to the V accumulator below (no
            // V row for the virtual key), matching the GPT-OSS / Triton UA
            // convention. The "if mass == 0" guard from
            // reference_batched_softmax is unnecessary here: with a finite
            // sink the denominator is always >= exp(sink - m) > 0.
            using Acc = AccDataType;
            for(int h = 0; h < nhead_q; ++h)
            {
                const Acc sink_v = ck_tile::type_convert<Acc>(sinks[h]);
                for(int q = 0; q < seqlen_q; ++q)
                {
                    Acc m_val = sink_v;
                    for(int n = 0; n < seqlen_kv; ++n)
                    {
                        const Acc s_val = ck_tile::type_convert<Acc>(s_host_ref(h, q, n));
                        if(s_val > m_val)
                            m_val = s_val;
                    }
                    Acc denom = ck_tile::exp(sink_v - m_val);
                    for(int n = 0; n < seqlen_kv; ++n)
                    {
                        const Acc s_val = ck_tile::type_convert<Acc>(s_host_ref(h, q, n));
                        denom += ck_tile::exp(s_val - m_val);
                    }
                    // The "denom == 0 -> inv = 1" guard from reference_batched_softmax
                    // is unnecessary here: with any finite sink, `denom >= exp(sink_v
                    // - m_val) > 0`. Skipping it sidesteps the example tree's
                    // -Wfloat-equal -Werror discipline.
                    const Acc inv_denom = Acc{1.f} / denom;
                    for(int n = 0; n < seqlen_kv; ++n)
                    {
                        const Acc s_val = ck_tile::type_convert<Acc>(s_host_ref(h, q, n));
                        p_host_ref(h, q, n) =
                            ck_tile::type_convert<PDataType>(ck_tile::exp(s_val - m_val) * inv_denom);
                    }
                }
            }
        }
        else
        {
            ck_tile::reference_batched_softmax<AccDataType, AccDataType>(
                s_host_ref, p_host_ref, ck_tile::identity{});
        }
        ck_tile::reference_batched_gemm<PDataType, VDataType, AccDataType>(
            p_host_ref, v_host_ref, o_host_ref, ck_tile::identity{}, v_element_op);

        // copy resulting per-batch data to the output tensor
        o_host_ref.ForEach(
            [&](auto& self, auto idx) { o_bshd(b, idx[1], idx[0], idx[2]) = self(idx); });
    }
}
} // namespace host

template <typename DataType>
bool run_impl(const Problem& problem, const RunConfig& run_config)
{
    auto [q, k, v] = generate_qkv<DataType>(problem, run_config.seed, run_config.debug_probe);

    ck_tile::DeviceMem q_buf(q.get_element_space_size_in_bytes());
    ck_tile::DeviceMem k_buf(k.get_element_space_size_in_bytes());
    ck_tile::DeviceMem v_buf(v.get_element_space_size_in_bytes());
    /// FIXME: use correct size for output tensor. just use q size for now since hidm_qk = hdim_v
    ck_tile::DeviceMem o_buf(q.get_element_space_size_in_bytes());

    q_buf.ToDevice(q.data());
    k_buf.ToDevice(k.data());
    v_buf.ToDevice(v.data());
    // Ensure output buffer is zero-initialized so padded regions compare cleanly
    o_buf.SetZero();

    ck_tile::unified_attention_args args{};

    args.scale_s            = problem.scale_s;
    args.data_type          = problem.data_type;
    args.num_seqs           = problem.batch;
    args.num_head_q         = problem.nhead_q;
    args.num_queries_per_kv = problem.num_queries_per_kv;
    args.page_blk_size      = problem.page_blk_size;
    args.mask_type          = static_cast<int>(problem.mask.type);
    args.hdim               = problem.hdim;

    // SWA window parameters from the parsed mask_info. These fields land
    // in kargs but the kernel still uses the hard-coded `(-1, 0, false)`
    // mask until the device-side SWA path is wired up.
    args.window_size_left  = problem.mask.left;
    args.window_size_right = problem.mask.right;
    args.is_top_left       = (problem.mask.type == mask_enum::mask_top_left);

    // Mirror the per-Q-head sink vector to device memory and hand the
    // pointer to kargs. Empty `problem.sinks` leaves `args.sink_ptr` as
    // its default `nullptr` — matches the classic no-sink convention.
    // The buffer must outlive `ck_tile::unified_attention(args, ...)`
    // below, so declare it in `run_impl`'s scope alongside seq_lens_buf
    // / block_tables_buf. We use the `DeviceMem` default-ctor + Realloc
    // idiom (same as the grouped-gemm examples) because `DeviceMem`
    // owns a HIP allocation but defines no copy/move, so `sink_buf =
    // DeviceMem(size)` would shallow-copy and double-free on scope
    // exit. The device-side kernel does not dereference `sink_ptr` yet;
    // this commit only verifies the pointer survives the round-trip
    // through kargs without observable behaviour change.
    ck_tile::DeviceMem sink_buf;
    if(!problem.sinks.empty())
    {
        sink_buf.Realloc(problem.sinks.size() * sizeof(float));
        sink_buf.ToDevice(problem.sinks.data());
        args.sink_ptr = sink_buf.GetDeviceBuffer();
    }

    args.num_blks = problem.num_blks;

    args.q_ptr          = q_buf.GetDeviceBuffer();
    args.query_stride_0 = problem.hdim * problem.nhead_q;
    args.query_stride_1 = problem.hdim;

    args.k_ptr = k_buf.GetDeviceBuffer();

    args.stride_k_cache_0 = problem.hdim * problem.nhead_kv * problem.page_blk_size;
    args.stride_k_cache_1 = problem.hdim * problem.nhead_kv;
    args.stride_k_cache_2 = problem.hdim;
    args.stride_k_cache_3 = 1;

    args.v_ptr            = v_buf.GetDeviceBuffer();
    args.stride_v_cache_0 = args.stride_k_cache_0;
    args.stride_v_cache_1 = args.stride_k_cache_1;
    args.stride_v_cache_2 = args.stride_k_cache_2;
    args.stride_v_cache_3 = args.stride_k_cache_3;

    args.o_ptr           = o_buf.GetDeviceBuffer();
    args.output_stride_0 = args.query_stride_0;
    args.output_stride_1 = args.query_stride_1;

    // Optional cumulative seqlen overrides (exclude PAD)
    auto make_effective_vec = [&](const std::vector<int>& opt_vec, ck_tile::index_t fallback) {
        std::vector<ck_tile::index_t> eff;
        if(!opt_vec.empty() && opt_vec[0] != -1)
        {
            eff.assign(opt_vec.begin(), opt_vec.end());
            if(eff.size() < static_cast<size_t>(problem.batch))
            {
                eff.resize(problem.batch, eff.back());
            }
        }
        else
        {
            eff.assign(problem.batch, fallback);
        }
        return eff;
    };

    const auto eff_query_lens = make_effective_vec(problem.query_lens, 1024);
    const auto eff_kv_lens    = make_effective_vec(problem.kv_lens, 1024);

    args.num_tokens = std::accumulate(eff_query_lens.begin(), eff_query_lens.end(), 0);

    // Calculate cumulative sums for kernel arguments if varlen is used
    std::vector<ck_tile::index_t> cu_query_lens;

    auto calculate_cumulative = [&](const std::vector<ck_tile::index_t>& per_batch_vec,
                                    std::vector<ck_tile::index_t>& cum_vec) {
        cum_vec.resize(per_batch_vec.size() + 1);
        cum_vec[0] = 0;
        for(std::size_t i = 0; i < per_batch_vec.size(); ++i)
            cum_vec[i + 1] = cum_vec[i] + per_batch_vec[i];
    };
    calculate_cumulative(eff_query_lens, cu_query_lens);

    ck_tile::DeviceMem seq_lens_buf(eff_kv_lens.size() * sizeof(ck_tile::index_t));
    ck_tile::DeviceMem query_start_len_buf(cu_query_lens.size() * sizeof(ck_tile::index_t));

    seq_lens_buf.ToDevice(eff_kv_lens.data());
    query_start_len_buf.ToDevice(cu_query_lens.data());

    args.seq_lens_ptr = reinterpret_cast<const ck_tile::index_t*>(seq_lens_buf.GetDeviceBuffer());
    args.query_start_len_ptr =
        reinterpret_cast<const ck_tile::index_t*>(query_start_len_buf.GetDeviceBuffer());

    auto max_element = [&](const std::vector<ck_tile::index_t>& opt_vec) {
        ck_tile::index_t max = opt_vec[0];
        for(ck_tile::index_t i : opt_vec)
        {
            if(i > max)
            {
                max = i;
            }
        }
        return max;
    };

    ck_tile::index_t max_kv_len = max_element(eff_kv_lens);

    ck_tile::index_t max_num_blocks_per_seq =
        (max_kv_len + problem.page_blk_size - 1) / problem.page_blk_size;

    // Create block_tables
    ck_tile::DeviceMem block_tables_buf(problem.batch * max_num_blocks_per_seq *
                                        sizeof(ck_tile::index_t));

    // Allocate host memory for block_tables
    std::vector<ck_tile::index_t> block_tables_host(problem.batch * max_num_blocks_per_seq);

    // Fill block_tables. For debug_probe==2 we pin an *identity* table so the
    // V-position probe's analytical expectation holds: with V_phys[pb, wp] =
    // pb*PB + wp and identity table, V[logical_n] = logical_n exactly.
    std::mt19937 rng(run_config.seed ? *run_config.seed : std::random_device{}());
    if(run_config.debug_probe == 2)
    {
        for(size_t i = 0; i < block_tables_host.size(); ++i)
        {
            block_tables_host[i] = static_cast<ck_tile::index_t>(i);
        }
    }
    else
    {
        std::uniform_int_distribution<ck_tile::index_t> dist(0, problem.num_blks - 1);
        for(size_t i = 0; i < block_tables_host.size(); ++i)
        {
            block_tables_host[i] = dist(rng);
        }
    }

    // Copy to device
    block_tables_buf.ToDevice(block_tables_host.data());

    // Set pointer in args
    args.block_tables_ptr =
        reinterpret_cast<const ck_tile::index_t*>(block_tables_buf.GetDeviceBuffer());
    args.block_table_stride = max_num_blocks_per_seq;

    ck_tile::stream_config stream_config{nullptr,
                                         true,
                                         /*log_level=*/0,
                                         run_config.kernel_warmup,
                                         run_config.kernel_repeat};

    auto [result, time] = ck_tile::unified_attention(args, stream_config);

    if(!result)
    {
        std::cerr << "faild to run unified_attention()" << std::endl;
        return false;
    }

    std::size_t flop = [&] {
        long flop_result = 0;

        for(size_t b = 0; b < eff_query_lens.size(); ++b)
        {
            long query_lens         = eff_query_lens[b];
            long kv_lens            = eff_kv_lens[b];
            long valid_out_elements = 0;

            // Causal logic for valid output elements
            if(query_lens > kv_lens)
            {
                valid_out_elements = (kv_lens * kv_lens + kv_lens) / 2;
            }
            else
            {
                valid_out_elements =
                    query_lens * kv_lens - ((query_lens * query_lens - query_lens) / 2);
            }

            flop_result += 2 * problem.nhead_q * valid_out_elements * (problem.hdim + problem.hdim);
        }
        return flop_result;
    }();
    // TODO fix this
    // std::size_t flop = 1;
    float tflops = static_cast<float>(flop) / 1.e9 / time;
    long mem     = 0;

    mem += problem.num_tokens * problem.nhead_q * problem.hdim * 2 * 2; // q and o, fp16
    // Count unique block indices used in block_tables_host
    std::unordered_set<ck_tile::index_t> unique_blocks(block_tables_host.begin(),
                                                       block_tables_host.end());
    mem += unique_blocks.size() * problem.nhead_kv * problem.hdim * 2 * 2; // k and v, fp16
    mem += problem.batch * max_num_blocks_per_seq * 4;                     // int32 block table
    mem += problem.batch * 4;                                              // int32 seq_lens_ptr

    std::cout << "[" << problem.data_type << "|";
    std::cout << "] b:" << problem.batch << ", h:" << problem.nhead_q << "/" << problem.nhead_kv
              << ", d:" << problem.hdim << ", scale_s:" << problem.scale_s << ", query_lens:[";
    for(size_t i = 0; i < problem.query_lens.size(); ++i)
    {
        std::cout << problem.query_lens[i];
        if(i < problem.query_lens.size() - 1)
            std::cout << ",";
    }
    std::cout << "], kv_lens:[";
    for(size_t i = 0; i < problem.kv_lens.size(); ++i)
    {
        std::cout << problem.kv_lens[i];
        if(i < problem.kv_lens.size() - 1)
            std::cout << ",";
    }
    std::cout << "], mask:" << problem.mask;
    // Surface the sink spec when non-empty so smoke-test logs make it
    // obvious which configurations are exercising the (host-only) sink path.
    if(!problem.sinks.empty())
    {
        std::cout << ", sink:" << problem.sink_str;
    }
    std::cout << std::fixed << ", " << std::setprecision(8) << time
              << " ms, " << std::setprecision(2) << tflops << " TFlops, " << std::setprecision(2)
              << (static_cast<double>(mem) / 1e12 / (time / 1e3)) << " TB/s" << std::endl;

    if(!run_config.verify)
    {
        return true;
    }

    // variable lengths are provided -> compute per-batch references
    // with the effective lengths; else compute a single full reference.
    // Variable-length aware verification: zero-fill padded region and only compute valid part.
    ck_tile::HostTensor<DataType> o_ref(problem.get_output_shape());
    o_ref.SetZero();

    for(int b = 0; b < problem.batch; ++b)
    {
        const ck_tile::index_t seqlen_q_eff  = eff_query_lens[b];
        const ck_tile::index_t seqlen_kv_eff = eff_kv_lens[b];

        if(seqlen_q_eff <= 0 || seqlen_kv_eff <= 0)
            continue;

        // Slice current batch from inputs (bshd) and build single-batch tensors
        ck_tile::HostTensor<DataType> q_b({1, seqlen_q_eff, problem.nhead_q, problem.hdim});
        ck_tile::HostTensor<DataType> k_b({1, seqlen_kv_eff, problem.nhead_kv, problem.hdim});
        ck_tile::HostTensor<DataType> v_b({1, seqlen_kv_eff, problem.nhead_kv, problem.hdim});
        ck_tile::HostTensor<DataType> o_b({1, seqlen_q_eff, problem.nhead_q, problem.hdim});
        ck_tile::index_t seq_q_off = cu_query_lens[b];

        // Copy effective region
        q_b.ForEach([&](auto& self, auto idx) {
            // idx: [0, s, h, d]
            self(idx) = q(seq_q_off + idx[1], idx[2], idx[3]);
        });
        k_b.ForEach([&](auto& self, auto idx) {
            // kv cache is paged
            ck_tile::index_t table_col          = int(idx[1] / problem.page_blk_size);
            ck_tile::index_t block_table_offset = b * max_num_blocks_per_seq + table_col;
            ck_tile::index_t block_idx          = block_tables_host[block_table_offset];

            self(idx) = k(block_idx, idx[1] % problem.page_blk_size, idx[2], idx[3]);
        });
        v_b.ForEach([&](auto& self, auto idx) {
            ck_tile::index_t table_col          = int(idx[1] / problem.page_blk_size);
            ck_tile::index_t block_table_offset = b * max_num_blocks_per_seq + table_col;
            ck_tile::index_t block_idx          = block_tables_host[block_table_offset];

            self(idx) = v(block_idx, idx[1] % problem.page_blk_size, idx[2], idx[3]);
        });
        // v_b.ForEach([&](auto& self, auto idx) { self(idx) = v(b, idx[1], idx[2], idx[3]); });

        // Decode the mask freshly with this batch's effective seqlens so the host
        // reference matches the per-batch attention shape (varlen-aware).
        const auto batch_mask = mask_info::decode(problem.mask_str, seqlen_q_eff, seqlen_kv_eff);

        // Compute reference for this batch segment (host::fmha_fwd expects bshd tensors).
        // Forward `problem.sinks` (empty vector when no sink) to the sink-aware
        // softmax inside host::fmha_fwd. Note this is in the *post-scale* space
        // — the existing `scales{problem.scale_s}` operator
        // is applied during the QK gemm, after which the sink raw value sits in
        // the same numerical space as the scaled S entries, matching the Triton
        // reference (`attn = cat([attn*scale, sinks_aux])`).
        host::fmha_fwd<float, DataType>(q_b,
                                        k_b,
                                        v_b,
                                        batch_mask,
                                        o_b,
                                        ck_tile::identity{},
                                        ck_tile::identity{},
                                        ck_tile::identity{},
                                        ck_tile::scales{problem.scale_s},
                                        problem.sinks);

        // Scatter into o_ref's bshd descriptor memory
        for(int s = 0; s < seqlen_q_eff; ++s)
        {
            for(int h = 0; h < problem.nhead_q; ++h)
            {
                for(int d = 0; d < problem.hdim; ++d)
                {
                    o_ref(seq_q_off + s, h, d) = o_b(0, s, h, d);
                }
            }
        }
    }

    ck_tile::HostTensor<DataType> o(problem.get_output_shape());
    o_buf.FromDevice(o.data());

    const auto [rtol, atol] = [&] {
        if constexpr(std::is_same_v<DataType, ck_tile::fp16_t>)
            return std::make_tuple(1e-3, 1e-3);
        else
            return std::make_tuple(1e-2, 1e-2);
    }();

    size_t total = static_cast<size_t>(problem.num_tokens) * static_cast<size_t>(problem.nhead_q) *
                   static_cast<size_t>(problem.hdim);

    size_t nonzero = 0;

    for(int tok = 0; tok < problem.num_tokens; ++tok)
    {
        for(int h = 0; h < problem.nhead_q; ++h)
        {
            for(int d = 0; d < problem.hdim; ++d)
            {
                if(static_cast<float>(o(tok, h, d)) != 0.0f)
                {
                    nonzero++;
                }
            }
        }
    }

    float percent =
        (total > 0) ? (100.0f * static_cast<float>(nonzero) / static_cast<float>(total)) : 0.0f;

    std::cout << "\nNon-zero elements in output tensor o: " << nonzero << " / " << total << " ("
              << percent << "%)\n";

    // std::cout << "\n=== Complete Output Tensor (o) ===\n";
    // for (int tok = 0; tok < problem.num_tokens; ++tok) {
    //     std::cout << "Token " << tok << ":\n";
    //     for (int h = 0; h < problem.nhead_q; ++h) {
    //         std::cout << "  Head " << h << ": ";
    //         for (int d = 0; d < problem.hdim; ++d) {
    //             std::cout << static_cast<float>(o(tok, h, d)) << " ";
    //         }
    //         std::cout << "\n";
    //     }
    // }

    // std::cout << "\n=== Complete Reference Tensor (o_ref) ===\n";
    // for (int tok = 0; tok < problem.num_tokens; ++tok) {
    //     std::cout << "Token " << tok << ":\n";
    //     for (int h = 0; h < problem.nhead_q; ++h) {
    //         std::cout << "  Head " << h << ": ";
    //         for (int d = 0; d < problem.hdim; ++d) {
    //             std::cout << static_cast<float>(o_ref(tok, h, d)) << " ";
    //         }
    //         std::cout << "\n";
    //     }
    // }
    return ck_tile::check_err(o, o_ref, std::string("found incorrect results!"), rtol, atol);
}

int main(int argc, char* argv[])
{

    auto [parse_result, args] = parse_cmd_args(argc, argv);

    if(!parse_result)
    {
        std::cerr << "failed to parse command line arguments" << std::endl;
    }

    Problem problem(args);
    RunConfig run_config(args);

    const auto run = [&] {
        if(problem.data_type == ck_tile::unified_attention_args::data_type_enum::fp16)
        {
            return run_impl<ck_tile::fp16_t>(problem, run_config);
        }
        else
        {
            return run_impl<ck_tile::bf16_t>(problem, run_config);
        }
    };

    return !run();
}
