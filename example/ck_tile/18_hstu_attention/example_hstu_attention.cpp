// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <array>
#include <cstring>
#include <functional>
#include <numeric>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <random>

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
#include "reference_hstu_attention.hpp"

#include "hstu_attention_util.hpp"
#include "hstu_attention_api.hpp"

template <typename T>
void dumpBufferToFile(const char* fileName, T* data, size_t dataNumItems)
{
    std::ofstream outFile(fileName, std::ios::binary);
    if(outFile)
    {
        outFile.write(reinterpret_cast<char*>(data), dataNumItems * sizeof(T));
        outFile.close();
        printf("Write output to file %s\n", fileName);
    }
    else
    {
        printf("Could not open file %s for writing\n", fileName);
    }
}

template <typename T>
void readDataToBufferFromFile(T* data, size_t dataNumItems, const std::string& fileName)
{
    std::ifstream infile(fileName, std::ios::binary);
    if(infile)
    {
        try
        {
            infile.read(reinterpret_cast<char*>(data), dataNumItems * sizeof(T));
            infile.close();
        }
        catch(const std::runtime_error& e)
        {
            throw e;
        };
    }
    else
    {
        throw std::runtime_error("could not open the file for reading");
    }
}

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
        .insert("prec", "fp16", "data type. fp16/bf16")
        .insert("jagged", "0", "q/k/v batched sequence is jagged or not")
        .insert("b", "12", "batch size")
        .insert("nhead", "4", "number of heads")
        .insert("hdim_qk", "64", "headdim size of Q/K")
        .insert("hdim_v", "64", "headdim size of V/O")
        .insert("seqlens", "400", "uih seqlen of single or all batches for query tensor, actually allocated seqlen will include the target of each batch and context_len")
        .insert("seqlens_kv", "", "uih seqlen of single or all batches for key/value tensor, actually allocated seqlen will include the target of each batch and context_len")
        .insert("max_seqlen", "0", "max uih_seqlen, can be ignored, or else must be equal or bigger than the maximum of all uih seqlens")
        .insert("max_seqlen_kv", "0", "max uih_seqlen_kv, can be ignored, or else must be equal or bigger than the maximum of all uih seqlens")
        .insert("targets", "", "sequence length at the end of query/key token sequence that should be excluded from attention") 
        .insert("max_target", "0", "max target, can be ignored, or else must be equal of bigger than the maximum of all targets")
        .insert("softmax", "0", "use softmax or not")
        .insert("causal", "1", "enable causal mask or not")
        .insert("local_len", "5", "length of the diagonal window for enabling masking, value 0 to disable") 
        .insert("context_len", "6", "sequence length at the begin of the query sequence the should be included for attention")
        .insert("minfull_len", "6", "sequence length at the end of the query sequence that should be included for attention")
	.insert("seed", "13579", "seed by the uniform or normal distribution generator")
        .insert("norm_dist", "0", "if true, initialize the data in normal distribution, or else in uniform distribution")
        .insert("alpha", "0", "scale factor of S=Q@K. 0 means equal to 1/sqrt(hdim)")
        .insert("attn_scale", "0", "scale factor of SiLU(Q@K). 0 means using 1/max_seqlen for scaling")
        .insert("init_qkv", "0", "initialize q, k, v tensor from local files q.dat, k.dat and v.data")
        .insert("save_mask", "0", "save the mask tensor to disk by the CPU validation codes")
        .insert("perf", "0", "weather measure execution time or not")
        .insert("dump_output", "0", "dump both device and reference hstu attention outputs to files, only used when validation is true");
    // clang-format on

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

static std::vector<int> get_integers_from_string(std::string lengthsStr)
{
    std::vector<int> lengths;
    std::size_t pos = 0;
    std::size_t new_pos;

    new_pos = lengthsStr.find(',', pos);
    while(new_pos != std::string::npos)
    {
        std::string sliceStr = lengthsStr.substr(pos, new_pos - pos);

        int len = std::stoi(sliceStr);

        lengths.push_back(len);

        pos     = new_pos + 1;
        new_pos = lengthsStr.find(',', pos);
    };

    std::string sliceStr = lengthsStr.substr(pos);

    if(!sliceStr.empty())
    {
        int len = std::stoi(sliceStr);

        lengths.push_back(len);
    };

    return (lengths);
};

template <typename T>
void supplement_array_by_last_element(std::vector<T>& arr, int target_num_elements)
{
    if(static_cast<int>(arr.size()) < target_num_elements)
    {
        T last_val = arr.back();

        for(int i = arr.size(); i < target_num_elements; i++)
            arr.push_back(last_val);
    };
};

static void show_hstu_attention_fwd_param(std::ostream& os, HstuAttentionFwdParams& param)
{
    if(param.is_jagged)
    {
        os << "Jagged inputs used! " << std::endl;
        os << "use causal: " << param.use_causal << std::endl;
        os << "Num of batches: " << param.num_batch << std::endl;
        os << "Num of heads: " << param.num_head << std::endl;
        os << "QK hdim: " << param.hdim_qk << " V hdim: " << param.hdim_v << std::endl;
        os << "Q/K/V/O seq stride: " << param.seq_stride_q << " " << param.seq_stride_k << " "
           << param.seq_stride_v << " " << param.seq_stride_o << std::endl;
        os << "Q/K/V/O nhead stride: " << param.nhead_stride_q << " " << param.nhead_stride_k << " "
           << param.nhead_stride_v << " " << param.nhead_stride_o << std::endl;
        os << "contextual_seqlen: " << param.contextual_seqlen << std::endl;
        os << "window_size: " << param.window_size << std::endl;
        os << "min_full_attn_seqlen: " << param.min_full_attn_seqlen << std::endl;
    }
    else
    {
        os << "Batched inputs used! " << std::endl;
        os << "use causal: " << param.use_causal << std::endl;
        os << "Num of batches: " << param.num_batch << std::endl;
        os << "Num of heads: " << param.num_head << std::endl;
        os << "QK hdim: " << param.hdim_qk << " V hdim: " << param.hdim_v << std::endl;
        os << "Q/K/V/O seq stride: " << param.seq_stride_q << " " << param.seq_stride_k << " "
           << param.seq_stride_v << " " << param.seq_stride_o << std::endl;
        os << "Q/K/V/O nhead stride: " << param.nhead_stride_q << " " << param.nhead_stride_k << " "
           << param.nhead_stride_v << " " << param.nhead_stride_o << std::endl;
        os << "Q/K/V/O batch stride: " << param.batch_stride_q << " " << param.batch_stride_k << " "
           << param.batch_stride_v << " " << param.batch_stride_o << std::endl;
        os << "contextual_seqlen: " << param.contextual_seqlen << std::endl;
        os << "window_size: " << param.window_size << std::endl;
        os << "min_full_attn_seqlen: " << param.min_full_attn_seqlen << std::endl;
    };
};

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

template <typename InOutDataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    bool do_validation = static_cast<bool>(arg_parser.get_int("v"));
    bool is_jagged     = static_cast<bool>(arg_parser.get_int("jagged"));
    int num_batch      = arg_parser.get_int("b");
    int num_head       = arg_parser.get_int("nhead");
    int hdim_qk        = arg_parser.get_int("hdim_qk");
    int hdim_v         = arg_parser.get_int("hdim_v");
    bool use_softmax   = static_cast<bool>(arg_parser.get_int("softmax"));
    bool use_causal    = static_cast<bool>(arg_parser.get_int("causal"));

    int window_size = arg_parser.get_int("local_len");

    int contextual_seqlen    = arg_parser.get_int("context_len");
    int min_full_attn_seqlen = arg_parser.get_int("minfull_len");

    float alpha          = arg_parser.get_float("alpha");
    float attn_scale     = arg_parser.get_float("attn_scale");
    int seed             = arg_parser.get_int("seed");
    bool use_normal_dist = arg_parser.get_int("norm_dist");
    bool measure_perf    = static_cast<bool>(arg_parser.get_int("perf"));
    bool dump_output     = static_cast<bool>(arg_parser.get_int("dump_output"));

    bool save_mask      = static_cast<bool>(arg_parser.get_int("save_mask"));
    bool initialize_qkv = static_cast<bool>(arg_parser.get_int("init_qkv"));

    std::string str_of_targets   = arg_parser.get_str("targets");
    std::vector<int> num_targets = get_integers_from_string(str_of_targets);

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

    if(!num_targets.empty())
    {
        // supplement num_targets using the last input value if user-provided lengths not enough
        supplement_array_by_last_element(num_targets, num_batch);

        // only consider num_batch values even if more values are provided by the user
        for(int i = 0; i < num_batch; i++)
            max_target = max(max_target, num_targets[i]);
    };

    HSTU_CHECK(!seq_lengths_q.empty(), "sequence lengths of q shoud be defined!");

    // assume seq_lengths_kv is same as seq_lengths_q if not defined, or else when
    // seq_lengths_kv is explicitly defined, we think the input case is a cross_attention case
    if(seq_lengths_kv.empty())
        seq_lengths_kv = seq_lengths_q;
    else
        is_cross_attention = true;

    // assume input_max_uih_seqlen_kv is same as input_max_uih_seqlen_q if not strictly defined
    if(input_max_uih_seqlen_kv <= 0)
        input_max_uih_seqlen_kv = input_max_uih_seqlen_q;

    if(is_jagged)
    {
        // supplement seq_lengths_q using the last input value if user-provided lengths not
        // enough
        supplement_array_by_last_element(seq_lengths_q, num_batch);

        // supplement seq_lengths_kv using the last input value if user-provided lengths not
        // enough
        supplement_array_by_last_element(seq_lengths_kv, num_batch);

        // only consider num_batch values even if more values are provided by the user
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
    int max_seqlen_kv = max_uih_seqlen_kv + max_target + contextual_seqlen;

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
            int batch_seqlen = num_targets.empty()
                                   ? seq_lengths_kv[i] + contextual_seqlen
                                   : seq_lengths_kv[i] + num_targets[i] + contextual_seqlen;

            phy_seqlen_kv += batch_seqlen;
            seq_offsets_kv.push_back(phy_seqlen_kv);
        };
    }
    else
    {
        phy_seqlen_q  = max_seqlen_q;
        phy_seqlen_kv = max_seqlen_kv;
    };

    long total_flops = 0;

    // estimate the total flops occurred, ignoring the scaling and SILu
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

    ck_tile::HostTensor<InOutDataType> q_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_q, num_head, hdim_qk});
    ck_tile::HostTensor<InOutDataType> k_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_kv, num_head, hdim_qk});
    ck_tile::HostTensor<InOutDataType> v_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_kv, num_head, hdim_v});
    ck_tile::HostTensor<InOutDataType> o_host_ref(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_q, num_head, hdim_v});

    ck_tile::HostTensor<int8_t> mask_host(
        save_mask
            ? std::array<ck_tile::index_t, 4>{num_batch, num_head, max_seqlen_q, max_seqlen_kv}
            : std::array<ck_tile::index_t, 4>{1, 1, 1, 1});

    if(!initialize_qkv)
    {
        if(use_normal_dist)
        {
            ck_tile::FillNormalDistribution<InOutDataType>{0.f, 1.f, seed}(q_host);
            ck_tile::FillNormalDistribution<InOutDataType>{0.f, 1.f, seed}(k_host);
            ck_tile::FillNormalDistribution<InOutDataType>{0.f, 1.f, seed}(v_host);
        }
        else
        {
            ck_tile::FillUniformDistribution<InOutDataType>{-1.f, 1.f, seed}(q_host);
            ck_tile::FillUniformDistribution<InOutDataType>{-1.f, 1.f, seed}(k_host);
            ck_tile::FillUniformDistribution<InOutDataType>{-1.f, 1.f, seed}(v_host);
        };
    }
    else
    {
        readDataToBufferFromFile(q_host.data(), q_host.get_element_space_size(), "q.dat");
        readDataToBufferFromFile(k_host.data(), k_host.get_element_space_size(), "k.dat");
        readDataToBufferFromFile(v_host.data(), v_host.get_element_space_size(), "v.dat");
    };

    ck_tile::DeviceMem q_dev(q_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem k_dev(k_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem v_dev(v_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem o_dev(o_host_ref.get_element_space_size_in_bytes());

    ck_tile::DeviceMem seq_offsets_q_dev(seq_offsets_q.size() * sizeof(int));
    ck_tile::DeviceMem seq_offsets_kv_dev(seq_offsets_kv.size() * sizeof(int));
    ck_tile::DeviceMem num_targets_dev(num_targets.size() * sizeof(int));

    q_dev.ToDevice(q_host.data());
    k_dev.ToDevice(k_host.data());
    v_dev.ToDevice(v_host.data());

    if(is_jagged)
    {
        seq_offsets_q_dev.ToDevice(seq_offsets_q.data());
        seq_offsets_kv_dev.ToDevice(seq_offsets_kv.data());
    };
    if(!num_targets.empty())
        num_targets_dev.ToDevice(num_targets.data());

    HstuAttentionFwdParams params;

    float scale_s = (alpha != 0.f) ? alpha : 1.0f / std::sqrt(hdim_qk);

    if(is_jagged)
    {
        params.is_cross_attention = is_cross_attention;
        params.is_jagged          = true;
        params.num_batch          = num_batch;
        params.seq_q_offsets_ptr  = seq_offsets_q_dev.GetDeviceBuffer();
        params.seq_kv_offsets_ptr = seq_offsets_kv_dev.GetDeviceBuffer();
        params.max_seqlen         = max(max_seqlen_q, max_seqlen_kv);
        params.q_ptr              = q_dev.GetDeviceBuffer();
        params.k_ptr              = k_dev.GetDeviceBuffer();
        params.v_ptr              = v_dev.GetDeviceBuffer();
        params.bias_ptr           = nullptr; // bias is not supported at present
        params.o_ptr              = o_dev.GetDeviceBuffer();
        params.hdim_qk            = hdim_qk;
        params.hdim_v             = hdim_v;
        params.num_head           = num_head;
        params.scale_s            = scale_s;
        params.attn_scale         = attn_scale;
        params.seq_stride_q       = q_host.get_strides()[1];
        params.seq_stride_k       = k_host.get_strides()[1];
        params.seq_stride_v       = v_host.get_strides()[1];
        params.seq_stride_bias    = 0;
        params.seq_stride_o       = o_host_ref.get_strides()[1];
        params.nhead_stride_q     = q_host.get_strides()[2];
        params.nhead_stride_k     = k_host.get_strides()[2];
        params.nhead_stride_v     = v_host.get_strides()[2];
        params.nhead_stride_bias  = 0;
        params.nhead_stride_o     = o_host_ref.get_strides()[2];
        params.num_targets_ptr = num_targets.empty() ? nullptr : num_targets_dev.GetDeviceBuffer();
        params.use_softmax     = use_softmax;
        params.use_causal      = use_causal;
        params.window_size     = window_size;
        params.contextual_seqlen    = contextual_seqlen;
        params.min_full_attn_seqlen = min_full_attn_seqlen;
        params.p_drop               = 0.0f; // dropout is not supported at present
        params.philox_seed          = 0UL;
        params.philox_offset        = 0UL;
    }
    else
    {
        params.is_cross_attention = is_cross_attention;
        params.is_jagged          = false;
        params.num_batch          = num_batch;
        params.seqlen_q           = phy_seqlen_q;
        params.seqlen_kv          = phy_seqlen_kv;
        params.q_ptr              = q_dev.GetDeviceBuffer();
        params.k_ptr              = k_dev.GetDeviceBuffer();
        params.v_ptr              = v_dev.GetDeviceBuffer();
        params.bias_ptr           = nullptr; // bias is not supported at present
        params.o_ptr              = o_dev.GetDeviceBuffer();
        params.hdim_qk            = hdim_qk;
        params.hdim_v             = hdim_v;
        params.num_head           = num_head;
        params.scale_s            = scale_s;
        params.attn_scale         = attn_scale;
        params.seq_stride_q       = q_host.get_strides()[1];
        params.seq_stride_k       = k_host.get_strides()[1];
        params.seq_stride_v       = v_host.get_strides()[1];
        params.seq_stride_bias    = 0;
        params.seq_stride_o       = o_host_ref.get_strides()[1];
        params.nhead_stride_q     = q_host.get_strides()[2];
        params.nhead_stride_k     = k_host.get_strides()[2];
        params.nhead_stride_v     = v_host.get_strides()[2];
        params.nhead_stride_bias  = 0;
        params.nhead_stride_o     = o_host_ref.get_strides()[2];
        params.batch_stride_q     = q_host.get_strides()[0];
        params.batch_stride_k     = k_host.get_strides()[0];
        params.batch_stride_v     = v_host.get_strides()[0];
        params.batch_stride_bias  = 0;
        params.batch_stride_o     = o_host_ref.get_strides()[0];
        params.num_targets_ptr = num_targets.empty() ? nullptr : num_targets_dev.GetDeviceBuffer();
        params.use_softmax     = use_softmax;
        params.use_causal      = use_causal;
        params.window_size     = window_size;
        params.contextual_seqlen    = contextual_seqlen;
        params.min_full_attn_seqlen = min_full_attn_seqlen;
        params.p_drop               = 0.0f; // dropout is not supported at present
        params.philox_seed          = 0UL;
        params.philox_offset        = 0UL;
    };

    // show_hstu_attention_fwd_param(std::cout, params);
    std::ignore = show_hstu_attention_fwd_param;

    hipStream_t stream;

    HIP_CHECK_ERROR(hipStreamCreate(&stream));

    if constexpr(std::is_same<InOutDataType, ck_tile::fp16_t>::value)
    {
        if(is_jagged)
            hstu_attention_jagged_forward_fp16(params, stream);
        else
            hstu_attention_batched_forward_fp16(params, stream);
    }
    else if constexpr(std::is_same<InOutDataType, ck_tile::bf16_t>::value)
    {
        if(is_jagged)
            hstu_attention_jagged_forward_bf16(params, stream);
        else
            hstu_attention_batched_forward_bf16(params, stream);
    }
    else
        throw std::runtime_error("Other data type is not supported at present!");

    bool res = true;

    if(do_validation)
    {
        using GemmAccDataType = typename HstuAttentionFwdTypeConfig<InOutDataType>::GemmAccDataType;
        using CompDataType    = typename HstuAttentionFwdTypeConfig<InOutDataType>::CompDataType;

        BOOL_SWITCH_3(is_jagged, kIsJagged, use_softmax, kUseSoftmax, use_causal, kUseCausal, [&] {
            ck_tile::reference_hstu_attention<InOutDataType,
                                              GemmAccDataType,
                                              CompDataType,
                                              kIsJagged,
                                              kUseSoftmax,
                                              kUseCausal>::Run(is_cross_attention,
                                                               q_host,
                                                               k_host,
                                                               v_host,
                                                               o_host_ref,
                                                               mask_host,
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
                                                               min_full_attn_seqlen);
        });

        ck_tile::HostTensor<InOutDataType> o_host(
            std::array<ck_tile::index_t, 4>{batches_for_alloc, phy_seqlen_q, num_head, hdim_v});

        o_dev.FromDevice(o_host.data());

        if(dump_output)
        {
            dumpBufferToFile("output_dev.dat", o_host.data(), o_host.get_element_space_size());
            dumpBufferToFile("output_host.dat", o_host_ref.data(), o_host.get_element_space_size());
        }

        if(save_mask)
            dumpBufferToFile(
                "ck_hstu_mask.dat", mask_host.data(), mask_host.get_element_space_size());

        auto [rtol, atol] = get_elimit<InOutDataType>();

        res = ck_tile::check_err(
            o_host, o_host_ref, std::string("hstu_attention output error"), rtol, atol);
    };

    if(measure_perf)
    {
        ck_tile::gpu_timer timer{};

        timer.start(stream);
        for(int i = 0; i < 10; i++)
        {
            if constexpr(std::is_same<InOutDataType, ck_tile::fp16_t>::value)
            {
                if(is_jagged)
                    hstu_attention_jagged_forward_fp16(params, stream);
                else
                    hstu_attention_batched_forward_fp16(params, stream);
            }
            else if constexpr(std::is_same<InOutDataType, ck_tile::bf16_t>::value)
            {
                if(is_jagged)
                    hstu_attention_jagged_forward_bf16(params, stream);
                else
                    hstu_attention_batched_forward_bf16(params, stream);
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

    const std::string data_type = arg_parser.get_str("prec");
    if(data_type == "fp16")
    {
        return run<ck_tile::half_t>(arg_parser) ? 0 : -2;
    }
    else if(data_type == "bf16")
    {
        return run<ck_tile::bf16_t>(arg_parser) ? 0 : -2;
    }

    return -3;
}
