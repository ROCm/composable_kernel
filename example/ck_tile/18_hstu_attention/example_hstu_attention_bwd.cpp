// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <array>
#include <string>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <random>
#include <cassert>

#include <ck_tile/host/host_tensor.hpp>
#include <ck_tile/host/fill.hpp>
#include <ck_tile/host/device_memory.hpp>
#include <ck_tile/host/stream_config.hpp>
#include <ck_tile/host/arg_parser.hpp>
#include <ck_tile/host/hip_check_error.hpp>
#include <ck_tile/host/check_err.hpp>
#include <ck_tile/host/timer.hpp>

#include "hstu_attention_fwd_type_config.hpp"
#include "hstu_attention_bool_switch.hpp"
#include "hstu_attention_params.hpp"
#include "reference_hstu_attention_fwd.hpp"
#include "reference_hstu_attention_bwd.hpp"

#include "hstu_attention_host_util.hpp"
#include "hstu_attention_api.hpp"

#include "example_helper.hpp"

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    using size_type = typename std::vector<T>::size_type;

    os << "[";
    for(size_type idx = 0; idx < v.size(); ++idx)
    {
        if(0 < idx)
        {
            os << ", ";
        }
        os << v[idx];
    }
    return os << "]";
}

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;

    // clang-format off
    arg_parser.insert("v", "1", "weather do CPU validation or not")
        .insert("g", "1", "num of attention group, bigger than 1 indicating group hstu")
        .insert("prec", "fp16", "data type. fp16/bf16")
        .insert("jagged", "0", "q/k/v batched sequence is jagged or not")
        .insert("b", "12", "number of batches")
        .insert("nhead", "4", "number of heads")
        .insert("hdim_qk", "64", "headdim size of Q/K")
        .insert("hdim_v", "64", "headdim size of V/O")
        .insert("seqlens", "400", "uih seqlen of single or all batches for query tensor, actually allocated seqlen will include the target of each batch and context_len")
        .insert("seqlens_kv", "", "uih seqlen of single or all batches for key/value tensor, actually allocated seqlen will include the target of each batch and context_len")
        .insert("max_seqlen", "0", "max uih_seqlen, can be ignored, or else must be equal/bigger than the maximum of all uih seqlens")
        .insert("max_seqlen_kv", "0", "max uih_seqlen_kv, can be ignored, or else must be equal/bigger than the maximum of all uih seqlens")
        .insert("g_max_seqlens", "0", "max uih_seqlen of groups, can be ignored, or else each must be equal/bigger than maximum of all uih seqlens in its group")
        .insert("g_max_seqlens_kv", "0", "max uih_seqlen of groups, can be ignored, or else each must be equal/bigger than maximum of all uih seqlens in its group")
        .insert("targets", "", "sequence length at the end of query/key token sequence that should be excluded from attention") 
        .insert("max_target", "0", "max target, can be ignored, or else must be equal/bigger than the maximum of all targets")
        .insert("softmax", "0", "use softmax or not")
        .insert("p_drop", "0", "probability for dropping out the attention values")
        .insert("causal", "1", "enable causal mask or not")
        .insert("local_len", "5", "length of the diagonal window for enabling masking, value 0 to disable") 
        .insert("g_local_lens", "5,", "list of all group's length of the diagonal window for enabling masking, value 0 to disable") 
        .insert("context_len", "6", "sequence length at the begin of the query sequence the should be included for attention")
        .insert("g_context_lens", "6,", "list of all group's sequence length at the begin of the query sequence that should be included for attention")
        .insert("minfull_len", "6", "sequence length at the end of the query sequence that should be included for attention")
        .insert("g_minfull_lens", "6", "list of all groups's sequence length at the end of the query sequence that should be included for attention")
	.insert("seed", "13579", "seed by the uniform or normal distribution generator")
        .insert("norm_dist", "0", "if true, initialize the data in normal distribution, or else in uniform distribution")
        .insert("alpha", "0", "scale factor of S=Q@K. 0 means equal to 1/sqrt(hdim)")
        .insert("attn_scale", "0", "scale factor of SiLU(Q@K). 0 means using 1/max_seqlen for scaling")
        .insert("g_attn_scales", "1.0,", "list of all groups's scale factors of S=@@K. 0 means using 1/max_seqlen of the group for scaling")
        .insert("init_qkv", "0", "initialize q, k, v tensor from local files q.dat, k.dat and v.data")
        .insert("perf", "0", "weather measure execution time or not")
        .insert("dump_output", "0", "dump both device and reference hstu attention outputs to files, only used when validation is true");
    // clang-format on

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

// threshold for different dtypes
template <typename DataType>
auto get_elimit()
{
    double rtol = 1.6e-2;
    double atol = 1e-5;

    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<ck_tile::bf16_t>()
{
    double rtol = 1.6e-2;
    double atol = 1e-5;
    return ck_tile::make_tuple(rtol, atol);
}

static const uint64_t PHILOX_SEED   = 1UL;
static const uint64_t PHILOX_OFFSET = 0UL;

template <typename InOutDataType>
bool run_no_group_hstu_forward_backward(const ck_tile::ArgParser& arg_parser, bool is_jagged)
{
    using CompDataType = typename HstuAttentionFwdTypeConfig<InOutDataType>::CompDataType;

    bool do_validation = static_cast<bool>(arg_parser.get_int("v"));
    int num_batch      = arg_parser.get_int("b");
    int num_head       = arg_parser.get_int("nhead");
    int hdim_qk        = arg_parser.get_int("hdim_qk");
    int hdim_v         = arg_parser.get_int("hdim_v");
    bool use_softmax   = static_cast<bool>(arg_parser.get_int("softmax"));
    bool use_causal    = static_cast<bool>(arg_parser.get_int("causal"));

    float alpha          = arg_parser.get_float("alpha");
    float attn_scale     = arg_parser.get_float("attn_scale");
    float p_drop         = arg_parser.get_float("p_drop");
    int seed             = arg_parser.get_int("seed");
    bool use_normal_dist = arg_parser.get_int("norm_dist");
    bool measure_perf    = static_cast<bool>(arg_parser.get_int("perf"));
    bool dump_output     = static_cast<bool>(arg_parser.get_int("dump_output"));

    bool initialize_qkv = static_cast<bool>(arg_parser.get_int("init_qkv"));

    std::string str_of_integers;

    str_of_integers              = arg_parser.get_str("targets");
    std::vector<int> num_targets = get_integers_from_string(str_of_integers);

    int window_size = arg_parser.get_int("local_len");

    int contextual_seqlen    = arg_parser.get_int("context_len");
    int min_full_attn_seqlen = arg_parser.get_int("minfull_len");

    std::string str_of_lengths_q   = arg_parser.get_str("seqlens");
    std::vector<int> seq_lengths_q = get_integers_from_string(str_of_lengths_q);

    std::string str_of_lengths_kv   = arg_parser.get_str("seqlens_kv");
    std::vector<int> seq_lengths_kv = get_integers_from_string(str_of_lengths_kv);

    int input_max_uih_seqlen_q  = arg_parser.get_int("max_seqlen");
    int input_max_uih_seqlen_kv = arg_parser.get_int("max_seqlen_kv");
    int input_max_target        = arg_parser.get_int("max_target");

    int max_uih_seqlen_q  = 0;
    int max_uih_seqlen_kv = 0;

    int max_target = 0;

    bool is_cross_attention = false;

    HSTU_CHECK(!seq_lengths_q.empty(), "sequence lengths of q shoud be defined!");

    // assume seq_lengths_kv is same as seq_lengths_q if not defined, or else when
    // seq_lengths_kv is explicitly defined, we think the input case is a cross_attention case
    if(seq_lengths_kv.empty())
        seq_lengths_kv = seq_lengths_q;
    else
        is_cross_attention = true;

    if(!is_cross_attention)
    {
        // assume input_max_uih_seqlen_kv is same as input_max_uih_seqlen_q if not strictly defined
        if(input_max_uih_seqlen_kv <= 0)
            input_max_uih_seqlen_kv = input_max_uih_seqlen_q;
    };

    if(is_jagged)
    {
        // supplement seq_lengths_q using the last input value if user-provided lengths not enough
        supplement_array_by_last_element(seq_lengths_q, num_batch);

        // supplement seq_lengths_kv using the last input value if user-provided lengths not enough
        supplement_array_by_last_element(seq_lengths_kv, num_batch);

        for(int i = 0; i < num_batch; i++)
        {
            max_uih_seqlen_q  = max(max_uih_seqlen_q, seq_lengths_q[i]);
            max_uih_seqlen_kv = max(max_uih_seqlen_kv, seq_lengths_kv[i]);
        };
    }
    else
    {
        HSTU_CHECK(1 == seq_lengths_q.size() && 1 == seq_lengths_kv.size(),
                   "sequence lengths for batched mode shoud have single element!");
        max_uih_seqlen_q  = seq_lengths_q[0];
        max_uih_seqlen_kv = seq_lengths_kv[0];
    };

    if(!num_targets.empty())
    {
        // supplement num_targets using the last input value if user-provided lengths not enough
        supplement_array_by_last_element(num_targets, num_batch);

        // only consider num_batch values even if more values are provided by the user
        for(int i = 0; i < num_batch; i++)
            max_target = max(max_target, num_targets[i]);
    };

    // the user input of max_uih_seqlen can either be ignored or be bigger than all uih_seqlens
    // the user input of max_target can either be ignored or be bigger than all targets
    HSTU_CHECK(input_max_uih_seqlen_q <= 0 || input_max_uih_seqlen_q >= max_uih_seqlen_q,
               "the user input of max_uih_seqlen can either be ignored or be bigger than all "
               "uih_seqlens!");
    HSTU_CHECK(input_max_uih_seqlen_kv <= 0 || input_max_uih_seqlen_kv >= max_uih_seqlen_kv,
               "the user input of max_uih_seqlen can either be ignored or be bigger than all "
               "uih_seqlens!");
    HSTU_CHECK(input_max_target <= 0 || input_max_target >= max_target,
               "the user input of max_target can either be ignored or be bigger than all targets!");

    HSTU_CHECK(contextual_seqlen >= 0, "contextual_seqlen should be non-negative!");

    max_uih_seqlen_q  = (input_max_uih_seqlen_q > 0) ? input_max_uih_seqlen_q : max_uih_seqlen_q;
    max_uih_seqlen_kv = (input_max_uih_seqlen_kv > 0) ? input_max_uih_seqlen_kv : max_uih_seqlen_kv;
    max_target        = (input_max_target > 0) ? input_max_target : max_target;

    int phy_seqlen_q  = 0;
    int phy_seqlen_kv = 0;
    int max_seqlen_q  = max_uih_seqlen_q + max_target + contextual_seqlen;
    int max_seqlen_kv = is_cross_attention ? max_uih_seqlen_kv + contextual_seqlen
                                           : max_uih_seqlen_kv + max_target + contextual_seqlen;

    std::vector<int> seq_offsets_q;
    std::vector<int> seq_offsets_kv;

    if(is_jagged)
    {
        seq_offsets_q.push_back(0);

        for(int i = 0; i < num_batch; i++)
        {
            int batch_seqlen = num_targets.empty()
                                   ? seq_lengths_q[i] + contextual_seqlen
                                   : seq_lengths_q[i] + num_targets[i] + contextual_seqlen;

            phy_seqlen_q += batch_seqlen;
            seq_offsets_q.push_back(phy_seqlen_q);
        };

        seq_offsets_kv.push_back(0);

        for(int i = 0; i < num_batch; i++)
        {
            if(!is_cross_attention)
            {
                int batch_seqlen = num_targets.empty()
                                       ? seq_lengths_kv[i] + contextual_seqlen
                                       : seq_lengths_kv[i] + num_targets[i] + contextual_seqlen;

                phy_seqlen_kv += batch_seqlen;
                seq_offsets_kv.push_back(phy_seqlen_kv);
            }
            else // for cross_attention, assume target_in_kv == false
            {
                int batch_seqlen = seq_lengths_kv[i] + contextual_seqlen;

                phy_seqlen_kv += batch_seqlen;
                seq_offsets_kv.push_back(phy_seqlen_kv);
            }
        };
    }
    else
    {
        phy_seqlen_q  = max_seqlen_q;
        phy_seqlen_kv = max_seqlen_kv;
    };

    int min_seqlen_q  = std::numeric_limits<int>::max();
    int min_seqlen_kv = std::numeric_limits<int>::max();

    if(is_jagged)
    {
        for(int i = 0; i < num_batch; i++)
        {
            min_seqlen_q  = min(min_seqlen_q, seq_offsets_q[i + 1] - seq_offsets_q[i]);
            min_seqlen_kv = min(min_seqlen_kv, seq_offsets_kv[i + 1] - seq_offsets_kv[i]);
        };
    };

    long total_flops = 0;

    // estimate the total flops occurred, ignoring the scaling and SiLu
    if(is_jagged)
    {
        for(int i = 0; i < num_batch; i++)
        {
            int len_q  = seq_offsets_q[i + 1] - seq_offsets_q[i];
            int len_kv = seq_offsets_kv[i + 1] - seq_offsets_kv[i];
            total_flops += (static_cast<long>(len_q) * len_kv * hdim_qk +
                            static_cast<long>(len_q) * hdim_v * len_kv) *
                           2;
        };

        total_flops *= num_head;
    }
    else
    {
        total_flops = static_cast<long>(num_batch) * num_head *
                      (static_cast<long>(phy_seqlen_q) * phy_seqlen_kv * hdim_qk +
                       static_cast<long>(phy_seqlen_q) * hdim_v * phy_seqlen_kv) *
                      2;
    };

    int batches_for_alloc = is_jagged ? 1 : num_batch;

    bool store_lse = use_softmax;

    ck_tile::HostTensor<InOutDataType> q_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_q, num_head, hdim_qk});
    ck_tile::HostTensor<InOutDataType> k_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_kv, num_head, hdim_qk});
    ck_tile::HostTensor<InOutDataType> v_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_kv, num_head, hdim_v});
    ck_tile::HostTensor<InOutDataType> o_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_q, num_head, hdim_v});
    ck_tile::HostTensor<CompDataType> lse_host(
        store_lse ? std::array<ck_tile::index_t, 3>{batches_for_alloc, phy_seqlen_q, num_head}
                  : std::array<ck_tile::index_t, 3>{1, 1, 1});

    ck_tile::HostTensor<InOutDataType> do_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_q, num_head, hdim_v});

    ck_tile::HostTensor<InOutDataType> dq_host_ref(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_q, num_head, hdim_qk});
    ck_tile::HostTensor<InOutDataType> dk_host_ref(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_kv, num_head, hdim_qk});
    ck_tile::HostTensor<InOutDataType> dv_host_ref(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_kv, num_head, hdim_v});

    ck_tile::HostTensor<int8_t> null_mask_host(std::array<ck_tile::index_t, 4>{1, 1, 1, 1});

    if(!initialize_qkv)
    {
        if(use_normal_dist)
        {
            ck_tile::FillNormalDistribution<InOutDataType>{0.f, 1.f, seed}(q_host);
            ck_tile::FillNormalDistribution<InOutDataType>{0.f, 1.f, seed}(k_host);
            ck_tile::FillNormalDistribution<InOutDataType>{0.f, 1.f, seed}(v_host);
            ck_tile::FillNormalDistribution<InOutDataType>{0.f, 1.f, seed}(do_host);
        }
        else
        {
            ck_tile::FillUniformDistribution<InOutDataType>{-1.f, 1.f, seed}(q_host);
            ck_tile::FillUniformDistribution<InOutDataType>{-1.f, 1.f, seed}(k_host);
            ck_tile::FillUniformDistribution<InOutDataType>{-1.f, 1.f, seed}(v_host);
            ck_tile::FillUniformDistribution<InOutDataType>{-1.f, 1.f, seed}(do_host);
        };
    }
    else
    {
        readDataToBufferFromFile(q_host.data(), q_host.get_element_space_size(), "q.dat");
        readDataToBufferFromFile(k_host.data(), k_host.get_element_space_size(), "k.dat");
        readDataToBufferFromFile(v_host.data(), v_host.get_element_space_size(), "v.dat");
        readDataToBufferFromFile(do_host.data(), do_host.get_element_space_size(), "do.dat");
    };

    ck_tile::DeviceMem q_dev(q_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem k_dev(k_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem v_dev(v_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem o_dev(o_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem lse_dev(lse_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem do_dev(do_host.get_element_space_size_in_bytes());

    ck_tile::DeviceMem dq_dev(dq_host_ref.get_element_space_size_in_bytes());
    ck_tile::DeviceMem dk_dev(dk_host_ref.get_element_space_size_in_bytes());
    ck_tile::DeviceMem dv_dev(dv_host_ref.get_element_space_size_in_bytes());

    ck_tile::DeviceMem seq_offsets_q_dev(seq_offsets_q.size() * sizeof(int));
    ck_tile::DeviceMem seq_offsets_kv_dev(seq_offsets_kv.size() * sizeof(int));
    ck_tile::DeviceMem num_targets_dev(num_targets.size() * sizeof(int));

    q_dev.ToDevice(q_host.data());
    k_dev.ToDevice(k_host.data());
    v_dev.ToDevice(v_host.data());
    do_dev.ToDevice(do_host.data());

    if(is_jagged)
    {
        seq_offsets_q_dev.ToDevice(seq_offsets_q.data());
        seq_offsets_kv_dev.ToDevice(seq_offsets_kv.data());
    };
    if(!num_targets.empty())
        num_targets_dev.ToDevice(num_targets.data());

    HstuAttentionNoGroupFwdParams params_fwd;
    HstuAttentionNoGroupBwdParams params_bwd;

    float scale_s = (alpha != 0.f) ? alpha : 1.0f / std::sqrt(hdim_qk);

    if(is_jagged)
    {
        params_fwd.is_cross_attention = is_cross_attention;
        params_fwd.is_jagged          = true;
        params_fwd.num_batch          = num_batch;
        params_fwd.seq_q_offsets_ptr  = seq_offsets_q_dev.GetDeviceBuffer();
        params_fwd.seq_kv_offsets_ptr = seq_offsets_kv_dev.GetDeviceBuffer();
        params_fwd.max_seqlen_q       = max_seqlen_q;
        params_fwd.max_seqlen_kv      = max_seqlen_kv;
        params_fwd.min_seqlen_q       = min_seqlen_q;
        params_fwd.min_seqlen_kv      = min_seqlen_kv;
        params_fwd.q_ptr              = q_dev.GetDeviceBuffer();
        params_fwd.k_ptr              = k_dev.GetDeviceBuffer();
        params_fwd.v_ptr              = v_dev.GetDeviceBuffer();
        params_fwd.bias_ptr           = nullptr; // bias is not supported at present
        params_fwd.o_ptr              = o_dev.GetDeviceBuffer();
        params_fwd.lse_ptr            = use_softmax ? lse_dev.GetDeviceBuffer() : nullptr;
        params_fwd.hdim_qk            = hdim_qk;
        params_fwd.hdim_v             = hdim_v;
        params_fwd.num_head           = num_head;
        params_fwd.scale_s            = scale_s;
        params_fwd.attn_scale         = attn_scale;
        params_fwd.seq_stride_q       = q_host.get_strides()[1];
        params_fwd.seq_stride_k       = k_host.get_strides()[1];
        params_fwd.seq_stride_v       = v_host.get_strides()[1];
        params_fwd.seq_stride_bias    = 0;
        params_fwd.seq_stride_o       = o_host.get_strides()[1];
        params_fwd.seq_stride_lse     = lse_host.get_strides()[1];
        params_fwd.nhead_stride_q     = q_host.get_strides()[2];
        params_fwd.nhead_stride_k     = k_host.get_strides()[2];
        params_fwd.nhead_stride_v     = v_host.get_strides()[2];
        params_fwd.nhead_stride_bias  = 0;
        params_fwd.nhead_stride_o     = o_host.get_strides()[2];
        params_fwd.nhead_stride_lse   = lse_host.get_strides()[2];
        params_fwd.num_targets_ptr =
            num_targets.empty() ? nullptr : num_targets_dev.GetDeviceBuffer();
        params_fwd.use_softmax          = use_softmax;
        params_fwd.is_training          = true;
        params_fwd.use_causal           = use_causal;
        params_fwd.window_size          = window_size;
        params_fwd.contextual_seqlen    = contextual_seqlen;
        params_fwd.min_full_attn_seqlen = min_full_attn_seqlen;
        params_fwd.p_drop               = p_drop;
        params_fwd.philox_seed          = PHILOX_SEED;
        params_fwd.philox_offset        = PHILOX_OFFSET;
    }
    else
    {
        params_fwd.is_cross_attention = is_cross_attention;
        params_fwd.is_jagged          = false;
        params_fwd.num_batch          = num_batch;
        params_fwd.seqlen_q           = phy_seqlen_q;
        params_fwd.seqlen_kv          = phy_seqlen_kv;
        params_fwd.q_ptr              = q_dev.GetDeviceBuffer();
        params_fwd.k_ptr              = k_dev.GetDeviceBuffer();
        params_fwd.v_ptr              = v_dev.GetDeviceBuffer();
        params_fwd.bias_ptr           = nullptr; // bias is not supported at present
        params_fwd.o_ptr              = o_dev.GetDeviceBuffer();
        params_fwd.lse_ptr            = use_softmax ? lse_dev.GetDeviceBuffer() : nullptr;
        params_fwd.hdim_qk            = hdim_qk;
        params_fwd.hdim_v             = hdim_v;
        params_fwd.num_head           = num_head;
        params_fwd.scale_s            = scale_s;
        params_fwd.attn_scale         = attn_scale;
        params_fwd.seq_stride_q       = q_host.get_strides()[1];
        params_fwd.seq_stride_k       = k_host.get_strides()[1];
        params_fwd.seq_stride_v       = v_host.get_strides()[1];
        params_fwd.seq_stride_bias    = 0;
        params_fwd.seq_stride_o       = o_host.get_strides()[1];
        params_fwd.seq_stride_lse     = lse_host.get_strides()[1];
        params_fwd.nhead_stride_q     = q_host.get_strides()[2];
        params_fwd.nhead_stride_k     = k_host.get_strides()[2];
        params_fwd.nhead_stride_v     = v_host.get_strides()[2];
        params_fwd.nhead_stride_bias  = 0;
        params_fwd.nhead_stride_o     = o_host.get_strides()[2];
        params_fwd.nhead_stride_lse   = lse_host.get_strides()[2];
        params_fwd.batch_stride_q     = q_host.get_strides()[0];
        params_fwd.batch_stride_k     = k_host.get_strides()[0];
        params_fwd.batch_stride_v     = v_host.get_strides()[0];
        params_fwd.batch_stride_bias  = 0;
        params_fwd.batch_stride_o     = o_host.get_strides()[0];
        params_fwd.batch_stride_lse   = lse_host.get_strides()[0];
        params_fwd.num_targets_ptr =
            num_targets.empty() ? nullptr : num_targets_dev.GetDeviceBuffer();
        params_fwd.use_softmax          = use_softmax;
        params_fwd.is_training          = true;
        params_fwd.use_causal           = use_causal;
        params_fwd.window_size          = window_size;
        params_fwd.contextual_seqlen    = contextual_seqlen;
        params_fwd.min_full_attn_seqlen = min_full_attn_seqlen;
        params_fwd.p_drop               = p_drop;
        params_fwd.philox_seed          = PHILOX_SEED;
        params_fwd.philox_offset        = PHILOX_OFFSET;
    };

    if(is_jagged)
    {
        params_bwd.is_cross_attention = is_cross_attention;
        params_bwd.is_jagged          = true;
        params_bwd.num_batch          = num_batch;
        params_bwd.seq_q_offsets_ptr  = seq_offsets_q_dev.GetDeviceBuffer();
        params_bwd.seq_kv_offsets_ptr = seq_offsets_kv_dev.GetDeviceBuffer();
        params_bwd.max_seqlen_q       = max_seqlen_q;
        params_bwd.max_seqlen_kv      = max_seqlen_kv;
        params_bwd.min_seqlen_q       = min_seqlen_q;
        params_bwd.min_seqlen_kv      = min_seqlen_kv;
        params_bwd.q_ptr              = q_dev.GetDeviceBuffer();
        params_bwd.k_ptr              = k_dev.GetDeviceBuffer();
        params_bwd.v_ptr              = v_dev.GetDeviceBuffer();
        params_bwd.bias_ptr           = nullptr; // bias is not supported at present
        params_bwd.o_ptr              = o_dev.GetDeviceBuffer();
        params_bwd.do_ptr             = do_dev.GetDeviceBuffer();
        params_bwd.lse_ptr            = use_softmax ? lse_dev.GetDeviceBuffer() : nullptr;
        params_bwd.dq_ptr             = dq_dev.GetDeviceBuffer();
        params_bwd.dk_ptr             = dk_dev.GetDeviceBuffer();
        params_bwd.dv_ptr             = dv_dev.GetDeviceBuffer();
        params_bwd.hdim_qk            = hdim_qk;
        params_bwd.hdim_v             = hdim_v;
        params_bwd.num_head           = num_head;
        params_bwd.scale_s            = scale_s;
        params_bwd.attn_scale         = attn_scale;
        params_bwd.seq_stride_q       = q_host.get_strides()[1];
        params_bwd.seq_stride_k       = k_host.get_strides()[1];
        params_bwd.seq_stride_v       = v_host.get_strides()[1];
        params_bwd.seq_stride_bias    = 0;
        params_bwd.seq_stride_o       = o_host.get_strides()[1];
        params_bwd.seq_stride_do      = do_host.get_strides()[1];
        params_bwd.seq_stride_lse     = lse_host.get_strides()[1];
        params_bwd.seq_stride_dq      = dq_host_ref.get_strides()[1];
        params_bwd.seq_stride_dk      = dk_host_ref.get_strides()[1];
        params_bwd.seq_stride_dv      = dv_host_ref.get_strides()[1];
        params_bwd.nhead_stride_q     = q_host.get_strides()[2];
        params_bwd.nhead_stride_k     = k_host.get_strides()[2];
        params_bwd.nhead_stride_v     = v_host.get_strides()[2];
        params_bwd.nhead_stride_bias  = 0;
        params_bwd.nhead_stride_o     = o_host.get_strides()[2];
        params_bwd.nhead_stride_do    = do_host.get_strides()[2];
        params_bwd.nhead_stride_lse   = lse_host.get_strides()[2];
        params_bwd.nhead_stride_dq    = dq_host_ref.get_strides()[2];
        params_bwd.nhead_stride_dk    = dk_host_ref.get_strides()[2];
        params_bwd.nhead_stride_dv    = dv_host_ref.get_strides()[2];
        params_bwd.num_targets_ptr =
            num_targets.empty() ? nullptr : num_targets_dev.GetDeviceBuffer();
        params_bwd.use_softmax          = use_softmax;
        params_bwd.use_causal           = use_causal;
        params_bwd.window_size          = window_size;
        params_bwd.contextual_seqlen    = contextual_seqlen;
        params_bwd.min_full_attn_seqlen = min_full_attn_seqlen;
        params_bwd.p_drop               = p_drop;
        params_bwd.philox_seed          = PHILOX_SEED;
        params_bwd.philox_offset        = PHILOX_OFFSET;
    }
    else
    {
        params_bwd.is_cross_attention = is_cross_attention;
        params_bwd.is_jagged          = false;
        params_bwd.num_batch          = num_batch;
        params_bwd.seqlen_q           = phy_seqlen_q;
        params_bwd.seqlen_kv          = phy_seqlen_kv;
        params_bwd.q_ptr              = q_dev.GetDeviceBuffer();
        params_bwd.k_ptr              = k_dev.GetDeviceBuffer();
        params_bwd.v_ptr              = v_dev.GetDeviceBuffer();
        params_bwd.bias_ptr           = nullptr; // bias is not supported at present
        params_bwd.o_ptr              = o_dev.GetDeviceBuffer();
        params_bwd.do_ptr             = do_dev.GetDeviceBuffer();
        params_bwd.lse_ptr            = use_softmax ? lse_dev.GetDeviceBuffer() : nullptr;
        params_bwd.dq_ptr             = dq_dev.GetDeviceBuffer();
        params_bwd.dk_ptr             = dk_dev.GetDeviceBuffer();
        params_bwd.dv_ptr             = dv_dev.GetDeviceBuffer();
        params_bwd.hdim_qk            = hdim_qk;
        params_bwd.hdim_v             = hdim_v;
        params_bwd.num_head           = num_head;
        params_bwd.scale_s            = scale_s;
        params_bwd.attn_scale         = attn_scale;
        params_bwd.seq_stride_q       = q_host.get_strides()[1];
        params_bwd.seq_stride_k       = k_host.get_strides()[1];
        params_bwd.seq_stride_v       = v_host.get_strides()[1];
        params_bwd.seq_stride_bias    = 0;
        params_bwd.seq_stride_o       = o_host.get_strides()[1];
        params_bwd.seq_stride_do      = do_host.get_strides()[1];
        params_bwd.seq_stride_lse     = lse_host.get_strides()[1];
        params_bwd.seq_stride_dq      = dq_host_ref.get_strides()[1];
        params_bwd.seq_stride_dk      = dk_host_ref.get_strides()[1];
        params_bwd.seq_stride_dv      = dv_host_ref.get_strides()[1];
        params_bwd.nhead_stride_q     = q_host.get_strides()[2];
        params_bwd.nhead_stride_k     = k_host.get_strides()[2];
        params_bwd.nhead_stride_v     = v_host.get_strides()[2];
        params_bwd.nhead_stride_bias  = 0;
        params_bwd.nhead_stride_o     = o_host.get_strides()[2];
        params_bwd.nhead_stride_do    = do_host.get_strides()[2];
        params_bwd.nhead_stride_lse   = lse_host.get_strides()[2];
        params_bwd.nhead_stride_dq    = dq_host_ref.get_strides()[2];
        params_bwd.nhead_stride_dk    = dk_host_ref.get_strides()[2];
        params_bwd.nhead_stride_dv    = dv_host_ref.get_strides()[2];
        params_bwd.batch_stride_q     = q_host.get_strides()[0];
        params_bwd.batch_stride_k     = k_host.get_strides()[0];
        params_bwd.batch_stride_v     = v_host.get_strides()[0];
        params_bwd.batch_stride_bias  = 0;
        params_bwd.batch_stride_o     = o_host.get_strides()[0];
        params_bwd.batch_stride_do    = do_host.get_strides()[0];
        params_bwd.batch_stride_lse   = lse_host.get_strides()[0];
        params_bwd.batch_stride_dq    = dq_host_ref.get_strides()[0];
        params_bwd.batch_stride_dk    = dk_host_ref.get_strides()[0];
        params_bwd.batch_stride_dv    = dv_host_ref.get_strides()[0];
        params_bwd.num_targets_ptr =
            num_targets.empty() ? nullptr : num_targets_dev.GetDeviceBuffer();
        params_bwd.use_softmax          = use_softmax;
        params_bwd.use_causal           = use_causal;
        params_bwd.window_size          = window_size;
        params_bwd.contextual_seqlen    = contextual_seqlen;
        params_bwd.min_full_attn_seqlen = min_full_attn_seqlen;
        params_bwd.p_drop               = p_drop;
        params_bwd.philox_seed          = PHILOX_SEED;
        params_bwd.philox_offset        = PHILOX_OFFSET;
    };

    bool has_dropout = (p_drop > 0.0f);
    ck_tile::HostTensor<uint8_t> rand_vals_host(
        has_dropout ? std::array<ck_tile::index_t, 4>{batches_for_alloc,
                                                      phy_seqlen_q,
                                                      num_head,
                                                      max_seqlen_kv}
                    : std::array<ck_tile::index_t, 4>{1, 1, 1, 1});

    ck_tile::DeviceMem rand_vals_dev(rand_vals_host.get_element_space_size_in_bytes());

    HstuGenerateRandUniformNumbersParams rv_params;

    if(has_dropout)
    {
        if(is_jagged)
        {
            rv_params.is_jagged         = true;
            rv_params.rand_val_ptr      = rand_vals_dev.GetDeviceBuffer();
            rv_params.num_batch         = num_batch;
            rv_params.seq_q_offsets_ptr = seq_offsets_q_dev.GetDeviceBuffer();
            rv_params.seq_k_offsets_ptr = seq_offsets_kv_dev.GetDeviceBuffer();
            rv_params.max_seqlen_q      = max_seqlen_q;
            rv_params.num_head          = num_head;
            rv_params.stride_seqlen     = rand_vals_host.get_strides()[1];
            rv_params.stride_nhead      = rand_vals_host.get_strides()[2];
            rv_params.philox_seed       = PHILOX_SEED;
            rv_params.philox_offset     = PHILOX_OFFSET;
        }
        else
        {
            rv_params.is_jagged     = false;
            rv_params.rand_val_ptr  = rand_vals_dev.GetDeviceBuffer();
            rv_params.num_batch     = num_batch;
            rv_params.num_head      = num_head;
            rv_params.seqlen_q      = phy_seqlen_q;
            rv_params.seqlen_k      = phy_seqlen_kv;
            rv_params.stride_seqlen = rand_vals_host.get_strides()[1];
            rv_params.stride_nhead  = rand_vals_host.get_strides()[2];
            rv_params.stride_batch  = rand_vals_host.get_strides()[0];
            rv_params.philox_seed   = PHILOX_SEED;
            rv_params.philox_offset = PHILOX_OFFSET;
        }
    }

    hipStream_t stream;

    HIP_CHECK_ERROR(hipStreamCreate(&stream));

    if constexpr(std::is_same<InOutDataType, ck_tile::fp16_t>::value)
    {
        hstu_attention_no_group_forward_fp16(params_fwd, stream);
        hstu_attention_no_group_backward_fp16(params_bwd, stream);
    }
    else if constexpr(std::is_same<InOutDataType, ck_tile::bf16_t>::value)
    {
        hstu_attention_no_group_forward_bf16(params_fwd, stream);
        hstu_attention_no_group_backward_bf16(params_bwd, stream);
    }
    else
        throw std::runtime_error("Other data type is not supported at present!");

    bool res = true;

    if(do_validation)
    {
        if(has_dropout)
        {
            // call a separate kernel to generate the random numbers, the generated random numbers
            // should be same as the random numbers implictly used by the fwd/bwd path for dropping
            if(is_jagged)
                hstu_generate_jagged_random_number_uint8(rv_params, stream);
            else
                hstu_generate_batched_random_number_uint8(rv_params, stream);

            rand_vals_dev.FromDevice(rand_vals_host.data());
        }

        using GemmAccDataType = typename HstuAttentionFwdTypeConfig<InOutDataType>::GemmAccDataType;

        BOOL_SWITCH_2(is_jagged, kIsJagged, use_causal, kUseCausal, [&] {
            ck_tile::reference_no_group_hstu_attention_fwd<InOutDataType,
                                                           GemmAccDataType,
                                                           CompDataType,
                                                           kIsJagged,
                                                           kUseCausal>::Run(is_cross_attention,
                                                                            use_softmax,
                                                                            store_lse,
                                                                            has_dropout,
                                                                            q_host,
                                                                            k_host,
                                                                            v_host,
                                                                            o_host,
                                                                            lse_host,
                                                                            null_mask_host,
                                                                            num_batch,
                                                                            scale_s,
                                                                            attn_scale,
                                                                            max_seqlen_q,
                                                                            max_seqlen_kv,
                                                                            seq_offsets_q,
                                                                            seq_offsets_kv,
                                                                            num_targets,
                                                                            contextual_seqlen,
                                                                            window_size,
                                                                            min_full_attn_seqlen,
                                                                            p_drop,
                                                                            rand_vals_host);

            ck_tile::reference_no_group_hstu_attention_bwd<InOutDataType,
                                                           GemmAccDataType,
                                                           CompDataType,
                                                           kIsJagged,
                                                           kUseCausal>::Run(is_cross_attention,
                                                                            use_softmax,
                                                                            has_dropout,
                                                                            q_host,
                                                                            k_host,
                                                                            v_host,
                                                                            lse_host,
                                                                            o_host,
                                                                            do_host,
                                                                            dq_host_ref,
                                                                            dk_host_ref,
                                                                            dv_host_ref,
                                                                            num_batch,
                                                                            scale_s,
                                                                            attn_scale,
                                                                            max_seqlen_q,
                                                                            seq_offsets_q,
                                                                            seq_offsets_kv,
                                                                            num_targets,
                                                                            contextual_seqlen,
                                                                            window_size,
                                                                            min_full_attn_seqlen,
                                                                            p_drop,
                                                                            rand_vals_host);
        });

        ck_tile::HostTensor<InOutDataType> dq_host(
            std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_q, num_head, hdim_qk});
        ck_tile::HostTensor<InOutDataType> dk_host(
            std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_kv, num_head, hdim_qk});
        ck_tile::HostTensor<InOutDataType> dv_host(
            std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_kv, num_head, hdim_v});

        dq_dev.FromDevice(dq_host.data());
        dk_dev.FromDevice(dk_host.data());
        dv_dev.FromDevice(dv_host.data());

        if(dump_output)
        {
            dumpBufferToFile("dq_dev.dat", dq_host.data(), dq_host.get_element_space_size());
            dumpBufferToFile("dk_dev.dat", dk_host.data(), dk_host.get_element_space_size());
            dumpBufferToFile("dv_dev.dat", dv_host.data(), dv_host.get_element_space_size());
        }

        auto [rtol, atol] = get_elimit<InOutDataType>();

        auto res_q = ck_tile::check_err(
            dq_host, dq_host_ref, std::string("hstu_attention bwd dq error"), rtol, atol);
        auto res_k = ck_tile::check_err(
            dk_host, dk_host_ref, std::string("hstu_attention bwd dk error"), rtol, atol);
        auto res_v = ck_tile::check_err(
            dv_host, dv_host_ref, std::string("hstu_attention bwd dv error"), rtol, atol);

        res = (res_q && res_k && res_v);
    };

    if(measure_perf)
    {
        ck_tile::gpu_timer timer{};

        timer.start(stream);
        for(int i = 0; i < 10; i++)
        {
            if constexpr(std::is_same<InOutDataType, ck_tile::fp16_t>::value)
            {
                hstu_attention_no_group_forward_fp16(params_fwd, stream);
                hstu_attention_no_group_backward_fp16(params_bwd, stream);
            }
            else if constexpr(std::is_same<InOutDataType, ck_tile::bf16_t>::value)
            {
                hstu_attention_no_group_forward_bf16(params_fwd, stream);
                hstu_attention_no_group_backward_bf16(params_bwd, stream);
            }
        }
        timer.stop(stream);

        auto ms = timer.duration() / 10.f;

        std::cout << "Average execution time of the hstu_attention operation is " << ms
                  << " milli-seconds, estimated TFLOPS is "
                  << (static_cast<float>(total_flops) / ms) / 1.0e9 << std::endl;
    }

    return res;
}

template <typename InOutDataType>
bool run_group_hstu_forward_backward(const ck_tile::ArgParser& arg_parser, int num_group)
{
    using CompDataType = typename HstuAttentionFwdTypeConfig<InOutDataType>::CompDataType;

    bool do_validation = static_cast<bool>(arg_parser.get_int("v"));

    int num_batch = arg_parser.get_int("b");

    HSTU_CHECK(num_group > 1, "ru_group_hstu should only be called when num_group > 1 !");
    HSTU_CHECK(num_batch > 0 && num_batch % num_group == 0,
               "number of batches should be a multi-fold value of num_group!");

    int num_batch_per_group = num_batch / num_group;

    int num_head         = arg_parser.get_int("nhead");
    int hdim_qk          = arg_parser.get_int("hdim_qk");
    int hdim_v           = arg_parser.get_int("hdim_v");
    bool use_softmax     = static_cast<bool>(arg_parser.get_int("softmax"));
    bool use_causal      = static_cast<bool>(arg_parser.get_int("causal"));
    float alpha          = arg_parser.get_float("alpha");
    float p_drop         = arg_parser.get_float("p_drop");
    int seed             = arg_parser.get_int("seed");
    bool use_normal_dist = arg_parser.get_int("norm_dist");
    bool measure_perf    = static_cast<bool>(arg_parser.get_int("perf"));
    bool dump_output     = static_cast<bool>(arg_parser.get_int("dump_output"));

    bool initialize_qkv = static_cast<bool>(arg_parser.get_int("init_qkv"));

    std::string str_of_integers;

    str_of_integers              = arg_parser.get_str("targets");
    std::vector<int> num_targets = get_integers_from_string(str_of_integers);

    std::string str_of_lengths_q   = arg_parser.get_str("seqlens");
    std::vector<int> seq_lengths_q = get_integers_from_string(str_of_lengths_q);

    std::string str_of_lengths_kv   = arg_parser.get_str("seqlens_kv");
    std::vector<int> seq_lengths_kv = get_integers_from_string(str_of_lengths_kv);

    bool is_cross_attention = false;

    HSTU_CHECK(!seq_lengths_q.empty(), "sequence lengths shoud be defined!");

    // assume seq_lengths_kv is same as seq_lengths_q if not defined, or else when
    // seq_lengths_kv is explicitly defined, we think the input case is a cross_attention case
    if(seq_lengths_kv.empty())
        seq_lengths_kv = seq_lengths_q;
    else
        is_cross_attention = true;

    str_of_integers                                = arg_parser.get_str("g_max_seqlens");
    std::vector<int> group_input_max_uih_seqlens_q = get_integers_from_string(str_of_integers);

    str_of_integers                                 = arg_parser.get_str("g_max_seqlens_kv");
    std::vector<int> group_input_max_uih_seqlens_kv = get_integers_from_string(str_of_integers);

    // for self-attention, group_input_max_uih_seqlens_kv reuses group_input_max_uih_seqlens_q
    if(!is_cross_attention)
    {
        group_input_max_uih_seqlens_kv = group_input_max_uih_seqlens_q;
    };

    str_of_integers                           = arg_parser.get_str("g_context_lens");
    std::vector<int> group_contextual_seqlens = get_integers_from_string(str_of_integers);

    HSTU_CHECK(!group_contextual_seqlens.empty(), "group contextual seqlens shoud be defined!");

    str_of_integers                     = arg_parser.get_str("g_local_lens");
    std::vector<int> group_window_sizes = get_integers_from_string(str_of_integers);

    HSTU_CHECK(!group_window_sizes.empty(), "group window sizes shoud be defined!");

    str_of_integers                              = arg_parser.get_str("g_minfull_lens");
    std::vector<int> group_min_full_attn_seqlens = get_integers_from_string(str_of_integers);
    HSTU_CHECK(!group_min_full_attn_seqlens.empty(),
               "group min_full_attn seqlens shoud be defined!");

    std::string str_of_floats            = arg_parser.get_str("g_attn_scales");
    std::vector<float> group_attn_scales = get_floats_from_string(str_of_floats);
    HSTU_CHECK(!group_attn_scales.empty(), "group attn_scales shoud be defined!");

    // supplement seq_lengths using the last input value if user-provided lengths not enough
    supplement_array_by_last_element(seq_lengths_q, num_batch);
    supplement_array_by_last_element(seq_lengths_kv, num_batch);

    if(!num_targets.empty())
    {
        // supplement num_targets using the last input value if user-provided lengths not enough
        supplement_array_by_last_element(num_targets, num_batch);
    };

    // supplement group_input_max_uih_seqlens using the last input value if user-provided lengths
    // not enough
    supplement_array_by_last_element(group_input_max_uih_seqlens_q, num_group);
    supplement_array_by_last_element(group_input_max_uih_seqlens_kv, num_group);

    // supplement group_contextual_seqlens using the last input value if user-provided lengths not
    // enough
    supplement_array_by_last_element(group_contextual_seqlens, num_group);

    // supplement group_window_sizes using the last input value if user-provided lengths not enough
    supplement_array_by_last_element(group_window_sizes, num_group);

    // supplement group_min_full_attn_seqlens using the last input value if user-provided lengths
    // not enough
    supplement_array_by_last_element(group_min_full_attn_seqlens, num_group);

    // supplement group_attn_scales using the last input value if user-provided values not enough
    supplement_array_by_last_element(group_attn_scales, num_group);

    int phy_seqlen_q      = 0;
    int phy_seqlen_kv     = 0;
    int max_max_seqlen_q  = 0;
    int max_max_seqlen_kv = 0;

    std::vector<int> group_max_uih_seqlens_q;
    std::vector<int> group_max_uih_seqlens_kv;

    group_max_uih_seqlens_q.resize(num_group);
    group_max_uih_seqlens_kv.resize(num_group);

    for(int i_grp = 0; i_grp < num_group; i_grp++)
    {
        group_max_uih_seqlens_q[i_grp]  = 0;
        group_max_uih_seqlens_kv[i_grp] = 0;

        for(int i_batch = 0; i_batch < num_batch_per_group; i_batch++)
        {
            auto i_global_batch = i_grp * num_batch_per_group + i_batch;

            group_max_uih_seqlens_q[i_grp] =
                max(group_max_uih_seqlens_q[i_grp], seq_lengths_q[i_global_batch]);
            group_max_uih_seqlens_kv[i_grp] =
                max(group_max_uih_seqlens_kv[i_grp], seq_lengths_kv[i_global_batch]);
        };

        HSTU_CHECK(group_input_max_uih_seqlens_q[i_grp] <= 0 ||
                       group_input_max_uih_seqlens_q[i_grp] >= group_max_uih_seqlens_q[i_grp],
                   "the user input of each group max_uih_seqlen can either be ignored or be bigger "
                   "than all uih_seqlens[] of the group");

        HSTU_CHECK(group_input_max_uih_seqlens_kv[i_grp] <= 0 ||
                       group_input_max_uih_seqlens_kv[i_grp] >= group_max_uih_seqlens_kv[i_grp],
                   "the user input of each group max_uih_seqlen can either be ignored or be bigger "
                   "than all uih_seqlens[] of the group");

        group_max_uih_seqlens_q[i_grp]  = group_input_max_uih_seqlens_q[i_grp] > 0
                                              ? group_input_max_uih_seqlens_q[i_grp]
                                              : group_max_uih_seqlens_q[i_grp];
        group_max_uih_seqlens_kv[i_grp] = group_input_max_uih_seqlens_kv[i_grp] > 0
                                              ? group_input_max_uih_seqlens_kv[i_grp]
                                              : group_max_uih_seqlens_kv[i_grp];
    };

    std::vector<int> group_max_seqlens_q;
    std::vector<int> group_max_seqlens_kv;

    group_max_seqlens_q.resize(num_group);
    group_max_seqlens_kv.resize(num_group);

    for(int i_grp = 0; i_grp < num_group; i_grp++)
    {
        int max_num_target = 0;

        if(!num_targets.empty())
        {
            for(int i_batch = 0; i_batch < num_batch_per_group; i_batch++)
            {
                int i_global_batch = i_grp * num_batch_per_group + i_batch;

                max_num_target = max(max_num_target, num_targets[i_global_batch]);
            };
        };

        group_max_seqlens_q[i_grp] =
            group_max_uih_seqlens_q[i_grp] + group_contextual_seqlens[i_grp] + max_num_target;
        max_max_seqlen_q = max(max_max_seqlen_q, group_max_seqlens_q[i_grp]);
        group_max_seqlens_kv[i_grp] =
            group_max_uih_seqlens_kv[i_grp] + group_contextual_seqlens[i_grp] + max_num_target;
        max_max_seqlen_kv = max(max_max_seqlen_kv, group_max_seqlens_kv[i_grp]);
    };

    std::vector<int> seq_offsets_q;
    std::vector<int> seq_offsets_kv;

    seq_offsets_q.push_back(0);

    for(int i = 0; i < num_batch; i++)
    {
        int i_group = i / num_batch_per_group;
        int batch_seqlen =
            num_targets.empty()
                ? seq_lengths_q[i] + group_contextual_seqlens[i_group]
                : seq_lengths_q[i] + num_targets[i] + group_contextual_seqlens[i_group];

        phy_seqlen_q += batch_seqlen;
        seq_offsets_q.push_back(phy_seqlen_q);
    };

    seq_offsets_kv.push_back(0);

    for(int i = 0; i < num_batch; i++)
    {
        if(!is_cross_attention)
        {
            int i_group = i / num_batch_per_group;
            int batch_seqlen =
                num_targets.empty()
                    ? seq_lengths_kv[i] + group_contextual_seqlens[i_group]
                    : seq_lengths_kv[i] + num_targets[i] + group_contextual_seqlens[i_group];

            phy_seqlen_kv += batch_seqlen;
            seq_offsets_kv.push_back(phy_seqlen_kv);
        }
        else // for cross_attention, assume target_in_kv == false
        {
            int i_group      = i / num_batch_per_group;
            int batch_seqlen = seq_lengths_kv[i] + group_contextual_seqlens[i_group];

            phy_seqlen_kv += batch_seqlen;
            seq_offsets_kv.push_back(phy_seqlen_kv);
        }
    };

    int min_seqlen_q  = std::numeric_limits<int>::max();
    int min_seqlen_kv = std::numeric_limits<int>::max();

    for(int i = 0; i < num_batch; i++)
    {
        min_seqlen_q  = min(min_seqlen_q, seq_offsets_q[i + 1] - seq_offsets_q[i]);
        min_seqlen_kv = min(min_seqlen_kv, seq_offsets_kv[i + 1] - seq_offsets_kv[i]);
    };

    long total_flops = 0;

    // estimate the total flops occurred, ignoring the scaling and SILu
    for(int i = 0; i < num_batch; i++)
    {
        int len_q  = seq_offsets_q[i + 1] - seq_offsets_q[i];
        int len_kv = seq_offsets_kv[i + 1] - seq_offsets_kv[i];
        total_flops += (static_cast<long>(len_q) * len_kv * hdim_qk +
                        static_cast<long>(len_q) * hdim_v * len_kv) *
                       2;
    };

    total_flops *= num_head;

    int batches_for_alloc = 1;

    bool store_lse = use_softmax;

    ck_tile::HostTensor<InOutDataType> q_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_q, num_head, hdim_qk});
    ck_tile::HostTensor<InOutDataType> k_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_kv, num_head, hdim_qk});
    ck_tile::HostTensor<InOutDataType> v_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_kv, num_head, hdim_v});
    ck_tile::HostTensor<InOutDataType> o_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_q, num_head, hdim_v});
    ck_tile::HostTensor<CompDataType> lse_host(
        store_lse ? std::array<ck_tile::index_t, 3>{batches_for_alloc, phy_seqlen_q, num_head}
                  : std::array<ck_tile::index_t, 3>{1, 1, 1});

    ck_tile::HostTensor<InOutDataType> do_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_q, num_head, hdim_v});

    ck_tile::HostTensor<InOutDataType> dq_host_ref(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_q, num_head, hdim_qk});
    ck_tile::HostTensor<InOutDataType> dk_host_ref(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_kv, num_head, hdim_qk});
    ck_tile::HostTensor<InOutDataType> dv_host_ref(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_kv, num_head, hdim_v});

    ck_tile::HostTensor<int8_t> null_mask_host(std::array<ck_tile::index_t, 4>{1, 1, 1, 1});

    if(!initialize_qkv)
    {
        if(use_normal_dist)
        {
            ck_tile::FillNormalDistribution<InOutDataType>{0.f, 1.f, seed}(q_host);
            ck_tile::FillNormalDistribution<InOutDataType>{0.f, 1.f, seed}(k_host);
            ck_tile::FillNormalDistribution<InOutDataType>{0.f, 1.f, seed}(v_host);
            ck_tile::FillNormalDistribution<InOutDataType>{0.f, 1.f, seed}(do_host);
        }
        else
        {
            ck_tile::FillUniformDistribution<InOutDataType>{-1.f, 1.f, seed}(q_host);
            ck_tile::FillUniformDistribution<InOutDataType>{-1.f, 1.f, seed}(k_host);
            ck_tile::FillUniformDistribution<InOutDataType>{-1.f, 1.f, seed}(v_host);
            ck_tile::FillUniformDistribution<InOutDataType>{-1.f, 1.f, seed}(do_host);
        };
    }
    else
    {
        readDataToBufferFromFile(q_host.data(), q_host.get_element_space_size(), "q.dat");
        readDataToBufferFromFile(k_host.data(), k_host.get_element_space_size(), "k.dat");
        readDataToBufferFromFile(v_host.data(), v_host.get_element_space_size(), "v.dat");
        readDataToBufferFromFile(do_host.data(), do_host.get_element_space_size(), "do.dat");
    };

    ck_tile::DeviceMem q_dev(q_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem k_dev(k_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem v_dev(v_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem o_dev(o_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem lse_dev(lse_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem do_dev(do_host.get_element_space_size_in_bytes());

    ck_tile::DeviceMem dq_dev(q_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem dk_dev(k_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem dv_dev(v_host.get_element_space_size_in_bytes());

    ck_tile::DeviceMem seq_offsets_q_dev(seq_offsets_q.size() * sizeof(int));
    ck_tile::DeviceMem seq_offsets_kv_dev(seq_offsets_kv.size() * sizeof(int));
    ck_tile::DeviceMem num_targets_dev(num_targets.size() * sizeof(int));

    q_dev.ToDevice(q_host.data());
    k_dev.ToDevice(k_host.data());
    v_dev.ToDevice(v_host.data());
    do_dev.ToDevice(do_host.data());

    seq_offsets_q_dev.ToDevice(seq_offsets_q.data());
    seq_offsets_kv_dev.ToDevice(seq_offsets_kv.data());
    if(!num_targets.empty())
        num_targets_dev.ToDevice(num_targets.data());

    ck_tile::DeviceMem group_max_seqlens_q_dev(group_max_seqlens_q.size() * sizeof(int));
    ck_tile::DeviceMem group_contextual_seqlens_dev(group_contextual_seqlens.size() * sizeof(int));
    ck_tile::DeviceMem group_window_sizes_dev(group_window_sizes.size() * sizeof(int));
    ck_tile::DeviceMem group_min_full_attn_seqlens_dev(group_min_full_attn_seqlens.size() *
                                                       sizeof(int));
    ck_tile::DeviceMem group_attn_scales_dev(group_attn_scales.size() * sizeof(float));

    group_max_seqlens_q_dev.ToDevice(group_max_seqlens_q.data());
    group_contextual_seqlens_dev.ToDevice(group_contextual_seqlens.data());
    group_window_sizes_dev.ToDevice(group_window_sizes.data());
    group_min_full_attn_seqlens_dev.ToDevice(group_min_full_attn_seqlens.data());
    group_attn_scales_dev.ToDevice(group_attn_scales.data());

    HstuAttentionGroupFwdParams params_fwd;
    HstuAttentionGroupBwdParams params_bwd;

    float scale_s = (alpha != 0.f) ? alpha : 1.0f / std::sqrt(hdim_qk);

    params_fwd.is_cross_attention = is_cross_attention;
    params_fwd.num_batch          = num_batch;
    params_fwd.num_group          = num_group;
    params_fwd.seq_q_offsets_ptr  = seq_offsets_q_dev.GetDeviceBuffer();
    params_fwd.seq_kv_offsets_ptr = seq_offsets_kv_dev.GetDeviceBuffer();
    params_fwd.max_seqlen_q       = max_max_seqlen_q;
    params_fwd.max_seqlen_kv      = max_max_seqlen_kv;
    params_fwd.min_seqlen_q       = min_seqlen_q;
    params_fwd.min_seqlen_kv      = min_seqlen_kv;
    params_fwd.q_ptr              = q_dev.GetDeviceBuffer();
    params_fwd.k_ptr              = k_dev.GetDeviceBuffer();
    params_fwd.v_ptr              = v_dev.GetDeviceBuffer();
    params_fwd.bias_ptr           = nullptr; // bias is not supported at present
    params_fwd.o_ptr              = o_dev.GetDeviceBuffer();
    params_fwd.lse_ptr            = use_softmax ? lse_dev.GetDeviceBuffer() : nullptr;
    params_fwd.hdim_qk            = hdim_qk;
    params_fwd.hdim_v             = hdim_v;
    params_fwd.num_head           = num_head;
    params_fwd.scale_s            = scale_s;
    params_fwd.seq_stride_q       = q_host.get_strides()[1];
    params_fwd.seq_stride_k       = k_host.get_strides()[1];
    params_fwd.seq_stride_v       = v_host.get_strides()[1];
    params_fwd.seq_stride_bias    = 0;
    params_fwd.seq_stride_o       = o_host.get_strides()[1];
    params_fwd.seq_stride_lse     = lse_host.get_strides()[1];
    params_fwd.nhead_stride_q     = q_host.get_strides()[2];
    params_fwd.nhead_stride_k     = k_host.get_strides()[2];
    params_fwd.nhead_stride_v     = v_host.get_strides()[2];
    params_fwd.nhead_stride_bias  = 0;
    params_fwd.nhead_stride_o     = o_host.get_strides()[2];
    params_fwd.nhead_stride_lse   = lse_host.get_strides()[2];
    params_fwd.num_targets_ptr = num_targets.empty() ? nullptr : num_targets_dev.GetDeviceBuffer();
    params_fwd.use_softmax     = use_softmax;
    params_fwd.is_training     = true;
    params_fwd.use_causal      = use_causal;
    params_fwd.p_drop          = p_drop;
    params_fwd.philox_seed     = PHILOX_SEED;
    params_fwd.philox_offset   = PHILOX_OFFSET;
    params_fwd.group_max_seqlen_q_ptr         = group_max_seqlens_q_dev.GetDeviceBuffer();
    params_fwd.group_contextual_seqlen_ptr    = group_contextual_seqlens_dev.GetDeviceBuffer();
    params_fwd.group_window_size_ptr          = group_window_sizes_dev.GetDeviceBuffer();
    params_fwd.group_min_full_attn_seqlen_ptr = group_min_full_attn_seqlens_dev.GetDeviceBuffer();
    params_fwd.group_attn_scale_ptr           = group_attn_scales_dev.GetDeviceBuffer();

    params_bwd.is_cross_attention = is_cross_attention;
    params_bwd.num_batch          = num_batch;
    params_bwd.num_group          = num_group;
    params_bwd.seq_q_offsets_ptr  = seq_offsets_q_dev.GetDeviceBuffer();
    params_bwd.seq_kv_offsets_ptr = seq_offsets_kv_dev.GetDeviceBuffer();
    params_bwd.max_seqlen_q       = max_max_seqlen_q;
    params_bwd.max_seqlen_kv      = max_max_seqlen_kv;
    params_bwd.min_seqlen_q       = min_seqlen_q;
    params_bwd.min_seqlen_kv      = min_seqlen_kv;
    params_bwd.q_ptr              = q_dev.GetDeviceBuffer();
    params_bwd.k_ptr              = k_dev.GetDeviceBuffer();
    params_bwd.v_ptr              = v_dev.GetDeviceBuffer();
    params_bwd.bias_ptr           = nullptr; // bias is not supported at present
    params_bwd.o_ptr              = o_dev.GetDeviceBuffer();
    params_bwd.do_ptr             = do_dev.GetDeviceBuffer();
    params_bwd.lse_ptr            = use_softmax ? lse_dev.GetDeviceBuffer() : nullptr;
    params_bwd.dq_ptr             = dq_dev.GetDeviceBuffer();
    params_bwd.dk_ptr             = dk_dev.GetDeviceBuffer();
    params_bwd.dv_ptr             = dv_dev.GetDeviceBuffer();
    params_bwd.hdim_qk            = hdim_qk;
    params_bwd.hdim_v             = hdim_v;
    params_bwd.num_head           = num_head;
    params_bwd.scale_s            = scale_s;
    params_bwd.seq_stride_q       = q_host.get_strides()[1];
    params_bwd.seq_stride_k       = k_host.get_strides()[1];
    params_bwd.seq_stride_v       = v_host.get_strides()[1];
    params_bwd.seq_stride_bias    = 0;
    params_bwd.seq_stride_o       = o_host.get_strides()[1];
    params_bwd.seq_stride_do      = do_host.get_strides()[1];
    params_bwd.seq_stride_lse     = lse_host.get_strides()[1];
    params_bwd.seq_stride_dq      = dq_host_ref.get_strides()[1];
    params_bwd.seq_stride_dk      = dk_host_ref.get_strides()[1];
    params_bwd.seq_stride_dv      = dv_host_ref.get_strides()[1];
    params_bwd.nhead_stride_q     = q_host.get_strides()[2];
    params_bwd.nhead_stride_k     = k_host.get_strides()[2];
    params_bwd.nhead_stride_v     = v_host.get_strides()[2];
    params_bwd.nhead_stride_bias  = 0;
    params_bwd.nhead_stride_o     = o_host.get_strides()[2];
    params_bwd.nhead_stride_do    = do_host.get_strides()[2];
    params_bwd.nhead_stride_lse   = lse_host.get_strides()[2];
    params_bwd.nhead_stride_dq    = dq_host_ref.get_strides()[2];
    params_bwd.nhead_stride_dk    = dk_host_ref.get_strides()[2];
    params_bwd.nhead_stride_dv    = dv_host_ref.get_strides()[2];
    params_bwd.num_targets_ptr = num_targets.empty() ? nullptr : num_targets_dev.GetDeviceBuffer();
    params_bwd.use_softmax     = use_softmax;
    params_bwd.use_causal      = use_causal;
    params_bwd.p_drop          = p_drop;
    params_bwd.philox_seed     = PHILOX_SEED;
    params_bwd.philox_offset   = PHILOX_OFFSET;
    params_bwd.group_max_seqlen_q_ptr         = group_max_seqlens_q_dev.GetDeviceBuffer();
    params_bwd.group_contextual_seqlen_ptr    = group_contextual_seqlens_dev.GetDeviceBuffer();
    params_bwd.group_window_size_ptr          = group_window_sizes_dev.GetDeviceBuffer();
    params_bwd.group_min_full_attn_seqlen_ptr = group_min_full_attn_seqlens_dev.GetDeviceBuffer();
    params_bwd.group_attn_scale_ptr           = group_attn_scales_dev.GetDeviceBuffer();

    bool has_dropout = (p_drop > 0.0f);
    ck_tile::HostTensor<uint8_t> rand_vals_host(
        has_dropout ? std::array<ck_tile::index_t, 4>{batches_for_alloc,
                                                      phy_seqlen_q,
                                                      num_head,
                                                      max_max_seqlen_kv}
                    : std::array<ck_tile::index_t, 4>{1, 1, 1, 1});

    ck_tile::DeviceMem rand_vals_dev(rand_vals_host.get_element_space_size_in_bytes());

    HstuGenerateRandUniformNumbersParams rv_params;

    if(has_dropout)
    {
        rv_params.is_jagged         = true;
        rv_params.rand_val_ptr      = rand_vals_dev.GetDeviceBuffer();
        rv_params.num_batch         = num_batch;
        rv_params.seq_q_offsets_ptr = seq_offsets_q_dev.GetDeviceBuffer();
        rv_params.seq_k_offsets_ptr = seq_offsets_kv_dev.GetDeviceBuffer();
        rv_params.max_seqlen_q      = max_max_seqlen_q;
        rv_params.num_head          = num_head;
        rv_params.stride_seqlen     = rand_vals_host.get_strides()[1];
        rv_params.stride_nhead      = rand_vals_host.get_strides()[2];
        rv_params.philox_seed       = PHILOX_SEED;
        rv_params.philox_offset     = PHILOX_OFFSET;
    }

    hipStream_t stream;

    HIP_CHECK_ERROR(hipStreamCreate(&stream));

    if constexpr(std::is_same<InOutDataType, ck_tile::fp16_t>::value)
    {
        hstu_attention_group_forward_fp16(params_fwd, stream);
        hstu_attention_group_backward_fp16(params_bwd, stream);
    }
    else if constexpr(std::is_same<InOutDataType, ck_tile::bf16_t>::value)
    {
        hstu_attention_group_forward_bf16(params_fwd, stream);
        hstu_attention_group_backward_bf16(params_bwd, stream);
    }
    else
        throw std::runtime_error("Other data type is not supported at present!");

    bool res = true;

    if(do_validation)
    {
        if(has_dropout)
        {
            // call a separate kernel to generate the random numbers, the generated random numbers
            // should be same as the random numbers implictly used by the fwd/bwd path for dropping
            hstu_generate_jagged_random_number_uint8(rv_params, stream);
            rand_vals_dev.FromDevice(rand_vals_host.data());
        }

        using GemmAccDataType = typename HstuAttentionFwdTypeConfig<InOutDataType>::GemmAccDataType;

        BOOL_SWITCH(use_causal, kUseCausal, [&] {
            ck_tile::reference_group_hstu_attention_fwd<
                InOutDataType,
                GemmAccDataType,
                CompDataType,
                kUseCausal>::Run(is_cross_attention,
                                 use_softmax,
                                 store_lse,
                                 has_dropout,
                                 q_host,
                                 k_host,
                                 v_host,
                                 o_host,
                                 lse_host,
                                 null_mask_host,
                                 num_batch,
                                 num_batch / num_group,
                                 scale_s,
                                 max_max_seqlen_q,
                                 max_max_seqlen_kv,
                                 seq_offsets_q,
                                 seq_offsets_kv,
                                 num_targets,
                                 group_max_seqlens_q,
                                 group_contextual_seqlens,
                                 group_window_sizes,
                                 group_min_full_attn_seqlens,
                                 group_attn_scales,
                                 p_drop,
                                 rand_vals_host);

            ck_tile::reference_group_hstu_attention_bwd<
                InOutDataType,
                GemmAccDataType,
                CompDataType,
                kUseCausal>::Run(is_cross_attention,
                                 use_softmax,
                                 has_dropout,
                                 q_host,
                                 k_host,
                                 v_host,
                                 lse_host,
                                 o_host,
                                 do_host,
                                 dq_host_ref,
                                 dk_host_ref,
                                 dv_host_ref,
                                 num_batch,
                                 num_batch / num_group,
                                 scale_s,
                                 seq_offsets_q,
                                 seq_offsets_kv,
                                 num_targets,
                                 group_max_seqlens_q,
                                 group_contextual_seqlens,
                                 group_window_sizes,
                                 group_min_full_attn_seqlens,
                                 group_attn_scales,
                                 p_drop,
                                 rand_vals_host);
        });

        ck_tile::HostTensor<InOutDataType> dq_host(
            std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_q, num_head, hdim_qk});
        ck_tile::HostTensor<InOutDataType> dk_host(
            std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_kv, num_head, hdim_qk});
        ck_tile::HostTensor<InOutDataType> dv_host(
            std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_kv, num_head, hdim_v});

        dq_dev.FromDevice(dq_host.data());
        dk_dev.FromDevice(dk_host.data());
        dv_dev.FromDevice(dv_host.data());

        if(dump_output)
        {
            dumpBufferToFile("dq_dev.dat", dq_host.data(), dq_host.get_element_space_size());
            dumpBufferToFile("dk_dev.dat", dk_host.data(), dk_host.get_element_space_size());
            dumpBufferToFile("dv_dev.dat", dv_host.data(), dv_host.get_element_space_size());
        }

        auto [rtol, atol] = get_elimit<InOutDataType>();

        auto res_q = ck_tile::check_err(
            dq_host, dq_host_ref, std::string("hstu_attention bwd dq error"), rtol, atol);
        auto res_k = ck_tile::check_err(
            dk_host, dk_host_ref, std::string("hstu_attention bwd dk error"), rtol, atol);
        auto res_v = ck_tile::check_err(
            dv_host, dv_host_ref, std::string("hstu_attention bwd dv error"), rtol, atol);
        res = (res_q && res_k && res_v);
    };

    if(measure_perf)
    {
        ck_tile::gpu_timer timer{};

        timer.start(stream);
        for(int i = 0; i < 10; i++)
        {
            if constexpr(std::is_same<InOutDataType, ck_tile::fp16_t>::value)
            {
                hstu_attention_group_forward_fp16(params_fwd, stream);
                hstu_attention_group_backward_fp16(params_bwd, stream);
            }
            else if constexpr(std::is_same<InOutDataType, ck_tile::bf16_t>::value)
            {
                hstu_attention_group_forward_bf16(params_fwd, stream);
                hstu_attention_group_backward_bf16(params_bwd, stream);
            }
        }
        timer.stop(stream);

        auto ms = timer.duration() / 10.f;

        std::cout << "Average execution time of the hstu_attention operation is " << ms
                  << " milli-seconds, estimated TFLOPS is "
                  << (static_cast<float>(total_flops) / ms) / 1.0e9 << std::endl;
    }

    return res;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
    {
        std::cerr << "Invalid arguments, Failed to parse!" << std::endl;
        return -1;
    }

    int num_group               = static_cast<int>(arg_parser.get_int("g"));
    const std::string data_type = arg_parser.get_str("prec");

    if(num_group > 1)
    {
        if(data_type == "fp16")
        {
            return run_group_hstu_forward_backward<ck_tile::half_t>(arg_parser, num_group) ? 0 : -2;
        }
        else if(data_type == "bf16")
        {
            return run_group_hstu_forward_backward<ck_tile::bf16_t>(arg_parser, num_group) ? 0 : -2;
        }
    }
    else
    {
        bool is_jagged = static_cast<bool>(arg_parser.get_int("jagged"));

        if(data_type == "fp16")
        {
            return run_no_group_hstu_forward_backward<ck_tile::half_t>(arg_parser, is_jagged) ? 0
                                                                                              : -2;
        }
        else if(data_type == "bf16")
        {
            return run_no_group_hstu_forward_backward<ck_tile::bf16_t>(arg_parser, is_jagged) ? 0
                                                                                              : -2;
        }
    };

    return -3;
}
