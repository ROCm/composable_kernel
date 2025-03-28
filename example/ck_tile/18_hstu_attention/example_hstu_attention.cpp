// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

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
#include <cassert>

#include <ck_tile/host/host_tensor.hpp>
#include <ck_tile/host/fill.hpp>
#include <ck_tile/host/device_memory.hpp>
#include <ck_tile/host/stream_config.hpp>
#include <ck_tile/host/arg_parser.hpp>
#include <ck_tile/host/hip_check_error.hpp>
#include <ck_tile/host/check_err.hpp>
#include <ck_tile/host/timer.hpp>

#include "hstu_attention_setting.hpp"
#include "bool_switch.hpp"
#include "reference_hstu_attention.hpp"

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
        .insert("seqlen", "400", "seqlen of single or all batches for query and key/value tensor")
        .insert("targets", "16", "sequence length at the end of query/key token sequence that should be excluded from attention") 
        .insert("causal", "1", "enable causal mask or not")
        .insert("local_len", "5", "length of the diagonal window for enabling masking, value 0 to disable") 
        .insert("context_len", "6", "sequence length at the begin of the query sequence the should be included for attention")
        .insert("minfull_len", "6", "sequence length at the end of the query sequence that should be included for attention")
	.insert("seed", "13579", "seed by the uniform or normal distribution generator")
        .insert("perf", "0", "weather measure execution time or not");
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
    int len              = std::stoi(sliceStr);

    lengths.push_back(len);

    return (lengths);
};

// threshold for different dtypes
template <typename DataType>
auto get_elimit()
{
    double rtol = 2e-3;
    double atol = 2e-3;

    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<ck_tile::bf16_t>()
{
    double rtol = 1e-2;
    double atol = 1e-2;
    return ck_tile::make_tuple(rtol, atol);
}

template <typename InOutDataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    bool do_validation = static_cast<bool>(arg_parser.get_int("v"));
    bool is_jagged     = static_cast<bool>(arg_parser.get_int("jagged"));
    int num_batch      = arg_parser.get_int("b");
    int nhead          = arg_parser.get_int("nhead");
    int hdim_qk        = arg_parser.get_int("hdim_qk");
    int hdim_v         = arg_parser.get_int("hdim_v");
    bool use_causal    = static_cast<bool>(arg_parser.get_int("causal"));

    int max_attn_len = arg_parser.get_int("local_len");

    bool use_local = (max_attn_len > 0);

    int contextual_seq_len = arg_parser.get_int("context_len");
    int min_full_seq_len   = arg_parser.get_int("minfull_len");

    int seed = arg_parser.get_int("seed");

    bool measure_perf = static_cast<bool>(arg_parser.get_int("perf"));

    (void)do_validation;
    (void)measure_perf;

    std::string str_of_targets   = arg_parser.get_str("targets");
    std::vector<int> num_targets = get_integers_from_string(str_of_targets);

    std::string str_of_lengths   = arg_parser.get_str("seqlen");
    std::vector<int> seq_lengths = get_integers_from_string(str_of_lengths);

    std::vector<int> seq_offsets;

    int seqlen = 0; // means total seq lengths for jagged

    if(is_jagged)
    {
        assert(num_batch == seq_lengths.size());

        seq_offsets.push_back(0);
        for(size_t i = 0; i < seq_lengths.size(); i++)
        {
            seqlen += seq_lengths[i];
            seq_offsets.push_back(seqlen);
        };

        if(!num_targets.empty())
        {
            assert(num_batch == num_targets.size());

            for(size_t i = 0; i < seq_lengths.size(); i++)
            {
                assert(seq_lengths[i] - num_targets[i] >= min_full_seq_len);
                assert(seq_lengths[i] - num_targets[i] >= contextual_seq_len);
            };
        }
        else
        {
            for(size_t i = 0; i < seq_lengths.size(); i++)
            {
                assert(seq_lengths[i] >= min_full_seq_len);
                assert(seq_lengths[i] >= contextual_seq_len);
            };
        };
    }
    else
    {
        assert(1 == seq_lengths.size());
        seqlen = seq_lengths[0];

        if(!num_targets.empty())
        {
            assert(1 == num_targets.size());

            assert(seqlen - num_targets[0] >= min_full_seq_len);
            assert(seqlen - num_targets[0] >= contextual_seq_len);
        }
        else
        {
            assert(seqlen >= min_full_seq_len);
            assert(seqlen >= contextual_seq_len);
        };
    };

    int batches_for_alloc = is_jagged ? 1 : num_batch;

    ck_tile::HostTensor<InOutDataType> q_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, seqlen, nhead, hdim_qk});
    ck_tile::HostTensor<InOutDataType> k_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, seqlen, nhead, hdim_qk});
    ck_tile::HostTensor<InOutDataType> v_host(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, seqlen, nhead, hdim_v});
    ck_tile::HostTensor<InOutDataType> o_host_ref(
        std::array<ck_tile::index_t, 4>{batches_for_alloc, seqlen, nhead, hdim_v});

    ck_tile::FillNormalDistributionIntegerValue<InOutDataType>{-2.f, 2.f, seed}(q_host);
    ck_tile::FillNormalDistributionIntegerValue<InOutDataType>{-2.f, 2.f, seed}(k_host);
    ck_tile::FillNormalDistributionIntegerValue<InOutDataType>{-2.f, 2.f, seed}(v_host);

    using GemmAccDataType   = typename HSTUAttentionTypeConfig<InOutDataType>::GemmAccDataType;
    using SMComputeDataType = typename HSTUAttentionTypeConfig<InOutDataType>::SMComputeDataType;

    BOOL_SWITCH_2(use_causal, USE_CAUSAL_, use_local, USE_LOCAL_, [&] {
        ck_tile::reference_hstu_attention<InOutDataType,
                                          GemmAccDataType,
                                          SMComputeDataType,
                                          USE_CAUSAL_,
                                          USE_LOCAL_>::Run(q_host,
                                                           k_host,
                                                           v_host,
                                                           o_host_ref,
                                                           num_batch,
                                                           1.0f,
                                                           seq_offsets,
                                                           num_targets,
                                                           max_attn_len,
                                                           contextual_seq_len,
                                                           min_full_seq_len);
    });
    return 0;
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
