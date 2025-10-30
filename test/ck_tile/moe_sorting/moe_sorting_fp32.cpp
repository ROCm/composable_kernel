// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <set>
#include <vector>
#include <iostream>
#include <numeric>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <time.h>
#include <unordered_set>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/reduce.hpp"
#include "moe_sorting_api.hpp"

auto create_args(int argc, char* argv[], int index = 0)
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("v", "1", "turn CPU validation on (1) or off (0).")
        .insert("pr_i", "int32", "index data type.  Only int32 is currently supported.")
        .insert("pr_w", "fp32", "output weight data type. Only fp32 is currently supported.")
        .insert("t",
                "128",
                "number of input tokens.\n"
                "If \"local_t\" presents, this value indicates global concurrency of all ranks.")
        .insert(
            "local_t",
            "-1",
            "Number of local input tokens for curent rank.\n"
            "This value must be within range \"[0, t)\", or \"-1\"(no such feature)\n"
            "This feature is to simulate EP case where where each rank has different tokens.\n"
            "Besides, this value will be stored in a GPU buffer, which is friendly for CUDA graph.")
        .insert("e", "8", "number of num_experts")
        .insert("k", "4", "topk")
        .insert("unit", "32", "unit_size")
#if MOE_SORTING_FMOE_2D_BUF
        .insert("moe_buf_interm_dim", "0", "interm_dim(col) of the following fmoe buf")
        .insert(
            "moe_buf_elem_bytes", "2", "fmoe buf element byte size, 1:8bit, 2:16bit, 4:32bit...")
#else
        .insert("moe_buf_size", "0", "moe_buf_size")
#endif
        .insert("ci",
                "1",
                "clear workspace inside API or not(if \"0\", require manually clear outside)")
        .insert(
            "dispatch",
            "0",
            "dispatch policy. 0:automatically pick up kernel, 1:use single kernel, 2:use mp kernel")
        .insert("local_eid",
                "-1",
                "a list of experts enabled as local expert. e.g. \"0,1,4,5\"\n"
                "please make sure eid is in ascending order!")
        .insert("seed",
                "-1",
                "seed to be used. When set to -1, a random seed will be generated each time "
                "invoking this example")
        .insert("kname", "0", "prints the kernel name when set to 1")
        .insert("warmup", "5", "number of iterations before benchmark the kernel")
        .insert("repeat", "20", "number of iterations to benchmark the kernel");

    bool result = arg_parser.parse(argc, argv, index);
    return std::make_tuple(result, arg_parser);
}

template <typename IndexType>
void topid_unique_gen(
    std::vector<IndexType>& host_tensor, int tokens, int topk, int num_expert, int seed)
{
    size_t total_size = topk * tokens;
    std::srand(seed);
    std::set<IndexType> unique_set;
    IndexType current_v;
    for(size_t i = 0; i < total_size; i++)
    {
        if(i % topk == 0)
        {
            unique_set.clear();
        }
        current_v = std::rand() % num_expert;
        while(unique_set.find(current_v) != unique_set.end())
        {
            current_v = std::rand() % num_expert;
        }
        unique_set.insert(current_v);
        host_tensor[i] = current_v;
    }
}

template <typename WeightType, typename IndexType = ck_tile::index_t>
bool test_moe_sorting(ck_tile::ArgParser args)
{
    int validate            = args.get_int("v");
    std::string index_prec  = args.get_str("pr_i");
    std::string weight_prec = args.get_str("pr_w");
    int tokens              = args.get_int("t");
    int local_tokens        = args.get_int("local_t");
    int num_experts         = args.get_int("e");
    int topk                = args.get_int("k");
    int seed                = args.get_int("seed");
    int unit_size           = args.get_int("unit");
#if MOE_SORTING_FMOE_2D_BUF
    int moe_buf_interm_dim = args.get_int("moe_buf_interm_dim");
    int moe_buf_elem_bytes = args.get_int("moe_buf_elem_bytes");
#else
    int64_t moe_buf_size = static_cast<int64_t>(args.get_uint64("moe_buf_size"));
#endif
    int kname           = args.get_int("kname");
    int warmup          = args.get_int("warmup");
    int repeat          = args.get_int("repeat");
    bool clear_inside   = args.get_int("ci") != 0;
    int dispatch_policy = args.get_int("dispatch");

    int max_output_ids =
        ck_tile::integer_least_multiple(topk * tokens + num_experts * unit_size - topk, unit_size);

    if(seed < 0)
    {
        seed = std::time(nullptr);
    }

    if(topk > num_experts)
    {
        printf("topk:%d value should be smaller than, or equal to number of num_experts:%d\n",
               topk,
               num_experts);
        return false;
    }

    // if local_tokens == tokens, not local_token, but better avoid this since no meaning for such
    // case
    bool is_local_token = local_tokens >= 0 && local_tokens < tokens;

    if(local_tokens > tokens)
    {
        printf("local_tokens:%d larger than tokens:%d, invalid\n", local_tokens, tokens);
        return false;
    }

    bool local_expert_masking      = args.get_str("local_eid") != "-1";
    auto local_expert_masking_host = [&]() {
        if(local_expert_masking)
        {
            auto local_eid = args.get_int_vec("local_eid");
            ck_tile::HostTensor<IndexType> v_{{num_experts}};
            v_.SetZero();
            for(auto eid : local_eid)
            {
                if(eid >= num_experts)
                {
                    throw std::runtime_error(
                        "local_eid larger than number of expert, please check");
                }
                v_.mData[eid] = 1;
            }
            return v_;
        }
        else
            return ck_tile::HostTensor<IndexType>{{1}};
    }();

    // tokens already considered batch size
    ck_tile::HostTensor<IndexType> topk_ids_host({tokens, topk}, {topk, 1});
    ck_tile::HostTensor<WeightType> weights_host({tokens, topk}, {topk, 1});
    ck_tile::HostTensor<IndexType> sorted_ids_host({max_output_ids}, {1});
    ck_tile::HostTensor<WeightType> sorted_weights_host({max_output_ids}, {1});
    ck_tile::HostTensor<IndexType> sorted_expert_ids_host({max_output_ids / unit_size}, {1});
    // for simplicity, below buffer allocate 2 dword
    ck_tile::HostTensor<IndexType> sorted_id_cnt_host({2}, {1});
#if MOE_SORTING_FMOE_2D_BUF
    ck_tile::HostTensor<int8_t> moe_buf_host(
        {static_cast<std::size_t>(is_local_token ? local_tokens : tokens) * moe_buf_interm_dim *
         moe_buf_elem_bytes});
    auto moe_buf_bytes = moe_buf_interm_dim == 0 ? static_cast<std::size_t>(0)
                                                 : moe_buf_host.get_element_space_size_in_bytes();
#else
    ck_tile::HostTensor<float> moe_buf_host({moe_buf_size});
    auto moe_buf_bytes = moe_buf_size == 0 ? static_cast<std::size_t>(0)
                                           : moe_buf_host.get_element_space_size_in_bytes();
#endif

    ck_tile::FillUniformDistribution<WeightType>{-.5f, .5f}(weights_host);
#if MOE_SORTING_FMOE_2D_BUF
    ck_tile::FillUniformDistribution<int8_t>{-.5f, .5f}(moe_buf_host);
#else
    ck_tile::FillUniformDistribution<WeightType>{-.5f, .5f}(moe_buf_host);
#endif
    topid_unique_gen<IndexType>(topk_ids_host.mData, tokens, topk, num_experts, seed);

    ck_tile::DeviceMem topk_ids_dev(topk_ids_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem weights_dev(weights_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem sorted_ids_dev(sorted_ids_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem sorted_weights_dev(sorted_weights_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem sorted_expert_ids_dev(
        sorted_expert_ids_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem sorted_id_cnt_dev(sorted_id_cnt_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem moe_buf_dev(moe_buf_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem local_expert_masking_dev(
        local_expert_masking_host.get_element_space_size_in_bytes());

    // used for simulating dynamic_tokens for EP case
    ck_tile::DeviceMem local_tokens_dev(sizeof(ck_tile::index_t));
    if(is_local_token)
    {
        local_tokens_dev.ToDevice(&local_tokens);
    }

    topk_ids_dev.ToDevice(topk_ids_host.data());
    weights_dev.ToDevice(weights_host.data());
    if(moe_buf_bytes > 0)
    {
        moe_buf_dev.ToDevice(moe_buf_host.data());
    }
    if(local_expert_masking)
        local_expert_masking_dev.ToDevice(local_expert_masking_host.data());

    // if return zero, means no need workspace, can set moe_sorting_args.p_ws to nullptr
    ck_tile::index_t workspace_size =
        moe_sorting_get_workspace_size(tokens, num_experts, topk, dispatch_policy);
    ck_tile::DeviceMem moe_sorting_ws(workspace_size != 0 ? workspace_size : 0);
    if(workspace_size != 0 && clear_inside == false)
        moe_sorting_ws.SetZero(); // note, clear here!!!!

    moe_sorting_trait trait{
        index_prec, weight_prec, local_expert_masking, clear_inside, dispatch_policy};

    moe_sorting_args karg{topk_ids_dev.GetDeviceBuffer(),
                          weights_dev.GetDeviceBuffer(),
                          local_expert_masking ? local_expert_masking_dev.GetDeviceBuffer()
                                               : nullptr,
                          is_local_token ? local_tokens_dev.GetDeviceBuffer() : nullptr,
                          sorted_ids_dev.GetDeviceBuffer(),
                          sorted_weights_dev.GetDeviceBuffer(),
                          sorted_expert_ids_dev.GetDeviceBuffer(),
                          sorted_id_cnt_dev.GetDeviceBuffer(),
                          moe_buf_bytes > 0 ? moe_buf_dev.GetDeviceBuffer() : nullptr,
                          workspace_size != 0 ? moe_sorting_ws.GetDeviceBuffer() : nullptr,
                          tokens,
                          unit_size,
                          num_experts,
                          topk,
#if MOE_SORTING_FMOE_2D_BUF
                          moe_buf_interm_dim,
                          moe_buf_elem_bytes
#else
                          static_cast<ck_tile::long_index_t>(moe_buf_size * sizeof(float))
#endif
    };

    ck_tile::stream_config sc{nullptr,
                              true,
                              /* log_level = */ (kname ? 1 : 0),
                              warmup,
                              repeat};

    auto ms = moe_sorting(trait, karg, sc);

    printf("[%s|%s|%s|%d]tokens:%d",
           index_prec.c_str(),
           weight_prec.c_str(),
           workspace_size == 0 ? "cx" : (clear_inside ? "ci" : "co"),
           dispatch_policy,
           tokens);
    if(is_local_token)
    {
        printf("(%d)", local_tokens);
    }
    printf(", num_experts:%d, topk:%d, mp:%d, ", num_experts, topk, workspace_size != 0 ? 1 : 0);

    if(local_expert_masking)
    {
        printf("local_eid:%s, ", args.get_str("local_eid").c_str());
    }

    if(moe_buf_bytes > 0)
    {
#if MOE_SORTING_FMOE_2D_BUF
        printf("moe_buf:%lu(%d,%d), ",
               static_cast<uint64_t>(moe_buf_bytes),
               moe_buf_interm_dim,
               moe_buf_elem_bytes);
#else

        printf("moe_buf:%lu, ", static_cast<uint64_t>(moe_buf_bytes));
#endif
    }

    if(ms < 0)
        printf("not supported\n");
    else
        printf("ms:%f, ", ms);
    fflush(stdout);
    if(ms < 0)
    {
        return false;
    }

    sorted_ids_dev.FromDevice(sorted_ids_host.data());
    sorted_weights_dev.FromDevice(sorted_weights_host.data());
    sorted_expert_ids_dev.FromDevice(sorted_expert_ids_host.data());
    sorted_id_cnt_dev.FromDevice(sorted_id_cnt_host.data());
    if(moe_buf_bytes > 0)
    {
        moe_buf_dev.FromDevice(moe_buf_host.data());
    }

    bool rtn = true;
    if(validate)
    {
        ck_tile::HostTensor<IndexType> sorted_ids_ref({max_output_ids}, {1});
        ck_tile::HostTensor<WeightType> sorted_weights_ref({max_output_ids}, {1});
        ck_tile::HostTensor<IndexType> sorted_expert_ids_ref({max_output_ids / unit_size}, {1});

        int32_t ref_total_tokens_post_pad = 0;
        ck_tile::reference_moe_sorting<WeightType, IndexType>(topk_ids_host,
                                                              weights_host,
                                                              local_expert_masking_host,
                                                              sorted_ids_ref,
                                                              sorted_weights_ref,
                                                              sorted_expert_ids_ref,
                                                              ref_total_tokens_post_pad,
                                                              num_experts,
                                                              unit_size,
                                                              is_local_token ? local_tokens
                                                                             : tokens,
                                                              local_expert_masking);
        printf("total_tokens_post_pad:%d(%d), ",
               ref_total_tokens_post_pad,
               sorted_id_cnt_host.mData[0]);
        if(ref_total_tokens_post_pad == sorted_id_cnt_host.mData[0])
        {
            size_t slen = ref_total_tokens_post_pad;
            rtn &= ck_tile::check_err(sorted_ids_host.slice({0}, {slen}),
                                      sorted_ids_ref.slice({0}, {slen}),
                                      std::string("OUT Error: Incorrect ids!"),
                                      1e-6,
                                      1e-6);
            rtn &= ck_tile::check_err(sorted_weights_host.slice({0}, {slen}),
                                      sorted_weights_ref.slice({0}, {slen}),
                                      std::string("OUT Error: Incorrect w!"),
                                      1e-6,
                                      1e-6);
            rtn &= ck_tile::check_err(sorted_expert_ids_host.slice({0}, {slen / unit_size}),
                                      sorted_expert_ids_ref.slice({0}, {slen / unit_size}),
                                      std::string("OUT Error: Incorrect eid!"),
                                      1e-6,
                                      1e-6);
            // if(is_local_token)
            {
                auto t_ = is_local_token ? local_tokens : tokens;
                bool _f = t_ == sorted_id_cnt_host.mData[1];
                rtn &= _f;
                if(!_f)
                {
                    printf("not equal token buffer pad %d(%d)\n", t_, sorted_id_cnt_host.mData[1]);
                }
            }
        }
        else
        {
            printf("(token size not equal!!)");
            rtn = false;
        }

        if(moe_buf_bytes)
        {
#if MOE_SORTING_FMOE_2D_BUF
            ck_tile::HostTensor<int8_t> moe_buf_ref({moe_buf_bytes});
#else
            ck_tile::HostTensor<WeightType> moe_buf_ref({moe_buf_size});
#endif
            rtn &= ck_tile::check_err(
                moe_buf_host, moe_buf_ref, std::string("OUT Error: Incorrect zero buf!"), 0, 0);
        }
        // rtn &= ref_total_tokens_post_pad == sorted_id_cnt_host.mData[0];
    }

    printf("valid:%s", rtn ? "y" : "n");
    fflush(stdout);
    if(!rtn)
        printf(", (%d)", seed);
    printf("\n");
    fflush(stdout);
    return rtn;
}
template <typename WeightType, typename IndexType = ck_tile::index_t>
bool run_test_case(int argc, char* argv[])
{
    auto [result, args] = create_args(argc, argv);
    if(!result)
        return false;

    return test_moe_sorting<WeightType, IndexType>(args);
}

template <typename WeightType, typename IndexType = ck_tile::index_t>
bool run_test_cases(std::vector<std::vector<std::string>>& test_cases)
{
    bool valid = true;

    for(std::size_t test_idx = 0; test_idx < test_cases.size(); ++test_idx)
    {

        constexpr int max_num_args = 7;
        const int num_args         = test_cases[test_idx].size();

        assert(max_num_args >= num_args && "Invalid number of arguments in test case");

        char* argv[max_num_args];

        for(int arg_idx = 0; arg_idx < num_args; ++arg_idx)
        {
            argv[arg_idx] = test_cases[test_idx][arg_idx].data();
        }

        try
        {
            valid = valid && run_test_case<WeightType, IndexType>(num_args, argv);

            if(!valid)
                break;
        }
        catch(const std::runtime_error& e)
        {
            std::cerr << "Runtime error: " << e.what() << '\n';
            return false;
        }
    }

    return valid;
}

std::vector<std::vector<std::string>> create_test_cases()
{
#if MOE_SORTING_FMOE_2D_BUF
    return {{"-t=80", "-e=17", "-moe_buf_interm_dim=16", "-moe_buf_elem_bytes=4"},
            {"-t=111", "-e=117", "-moe_buf_interm_dim=4", "-moe_buf_elem_bytes=4"},
            {"-t=1000", "-e=55", "-moe_buf_interm_dim=1024", "-moe_buf_elem_bytes=1"},
            {"-t=99", "-e=120", "-moe_buf_interm_dim=10244", "-moe_buf_elem_bytes=2"},
            {"-t=175", "-e=64", "-k=8"},
            {"-t=65", "-e=8", "-k=2"},
            {"-t=1", "-e=25"},
            {"-t=31", "-e=19", "-k=15"},
            {"-t=81", "-e=37", "-k=7"},
            {"-t=23", "-e=1", "-k=1"},
            {"-t=127", "-e=99", "-k=19"},
            {"-t=71", "-e=11", "-k=11"},
            {"-t=1", "-e=1", "-k=1"},
            {"-t=99", "-e=2", "-k=1"},
            {"-t=333", "-e=99", "-k=13"},
            {"-t=11", "-e=256", "-k=5"},
            {"-t=64", "-e=455", "-k=8"},
            {"-t=777", "-e=802", "-k=99"},
            {"-t=4097", "-e=906", "-k=51"},
            {"-t=128", "-e=32", "-k=5", "-local_t=6", "-moe_buf_interm_dim=262144"},
            {"-t=13", "-e=64", "-k=3", "-local_eid=4,5,6,7,8,9,10,11"},
            {"-t=99", "-e=33", "-k=9", "-local_eid=6,10,11,15,19"},
            {"-t=80", "-e=99", "-k=10", "-local_eid=0,8,12,33"},
            {"-t=11", "-e=256", "-k=5", "-local_eid=99,110,129"},
            {"-t=128", "-e=128", "-k=6", "-moe_buf_interm_dim=163840", "-moe_buf_elem_bytes=1"},
            {"-t=8192", "-e=32", "-k=5", "-local_t=11", "-moe_buf_interm_dim=163840"},
            {"-t=8192",
             "-e=32",
             "-k=8",
             "-local_t=12",
             "-moe_buf_interm_dim=163840",
             "-moe_buf_elem_bytes=1"},
            {"-t=8192", "-e=256", "-k=5", "-local_t=13", "-moe_buf_interm_dim=163840"},
            {"-t=8192", "-e=256", "-k=8", "-local_t=8", "-moe_buf_interm_dim=163840"},
            {"-t=163840",
             "-e=256",
             "-k=8",
             "-local_t=4",
             "-moe_buf_interm_dim=163840",
             "-moe_buf_elem_bytes=4"},
            {"-t=12", "-local_t=3", "-e=256", "-k=5", "-local_eid=9,10,199,145"},
            {"-t=67", "-local_t=9", "-e=555", "-k=5", "-local_eid=19,23,24,25,26,99"},
            {"-t=99", "-local_t=93", "-e=121", "-local_t=4", "-moe_buf_interm_dim=10244"},
            {"-t=536", "-local_t=345", "-e=802", "-k=99"},
            {"-t=331", "-local_t=39", "-e=83", "-k=33"},
            {"-t=765", "-local_t=654", "-e=783", "-k=8"},
            {"-t=23", "-local_t=9", "-e=1", "-k=1"},
            {"-t=7", "-local_t=0", "-e=89", "-k=1", "-local_eid=0,8,12,33"},
            {"-t=61", "-local_t=0", "-e=333", "-k=99", "-local_eid=0,8,12,33"},
            {"-t=133940",
             "-local_t=111921",
             "-e=256",
             "-k=17",
             "-local_t=2",
             "-moe_buf_interm_dim=133940",
             "-moe_buf_elem_bytes=1"}};

#else
    return {{"-t=80", "-e=17", "-moe_buf_size=16"},
            {"-t=111", "-e=117", "-moe_buf_size=4"},
            {"-t=1000", "-e=55", "-moe_buf_size=1024"},
            {"-t=99", "-e=120", "-moe_buf_size=10244"},
            {"-t=175", "-e=64", "-k=8"},
            {"-t=65", "-e=8", "-k=2"},
            {"-t=1", "-e=25"},
            {"-t=31", "-e=19", "-k=15"},
            {"-t=81", "-e=37", "-k=7"},
            {"-t=23", "-e=1", "-k=1"},
            {"-t=127", "-e=99", "-k=19"},
            {"-t=71", "-e=11", "-k=11"},
            {"-t=1", "-e=1", "-k=1"},
            {"-t=99", "-e=2", "-k=1"},
            {"-t=333", "-e=99", "-k=13"},
            {"-t=11", "-e=256", "-k=5"},
            {"-t=64", "-e=455", "-k=8"},
            {"-t=777", "-e=802", "-k=99"},
            {"-t=4097", "-e=906", "-k=51"},
            {"-t=128", "-e=32", "-k=5", "-moe_buf_size=262144"},
            {"-t=13", "-e=64", "-k=3", "-local_eid=4,5,6,7,8,9,10,11"},
            {"-t=99", "-e=33", "-k=9", "-local_eid=6,10,11,15,19"},
            {"-t=80", "-e=99", "-k=10", "-local_eid=0,8,12,33"},
            {"-t=11", "-e=256", "-k=5", "-local_eid=99,110,129"},
            {"-t=128", "-e=128", "-k=6", "-moe_buf_size=163840"},
            {"-t=8192", "-e=32", "-k=5", "-moe_buf_size=163840"},
            {"-t=8192", "-e=32", "-k=8", "-moe_buf_size=163840"},
            {"-t=8192", "-e=256", "-k=5", "-moe_buf_size=163840"},
            {"-t=8192", "-e=256", "-k=8", "-moe_buf_size=163840"},
            {"-t=163840", "-e=256", "-k=8", "-moe_buf_size=163840"},
            {"-t=12", "-local_t=3", "-e=256", "-k=5", "-local_eid=9,10,199,145"},
            {"-t=67", "-local_t=9", "-e=555", "-k=5", "-local_eid=19,23,24,25,26,99"},
            {"-t=99", "-local_t=93", "-e=121", "-moe_buf_size=10244"},
            {"-t=536", "-local_t=345", "-e=802", "-k=99"},
            {"-t=331", "-local_t=39", "-e=83", "-k=33"},
            {"-t=765", "-local_t=654", "-e=783", "-k=8"},
            {"-t=23", "-local_t=9", "-e=1", "-k=1"},
            {"-t=7", "-local_t=0", "-e=89", "-k=1", "-local_eid=0,8,12,33"},
            {"-t=61", "-local_t=0", "-e=333", "-k=99", "-local_eid=0,8,12,33"},
            {"-t=133940", "-local_t=111921", "-e=256", "-k=17", "-moe_buf_size=133940"}};
#endif
}

int main()
{
    std::vector<std::vector<std::string>> test_cases = create_test_cases();

    return !run_test_cases<float, ck_tile::index_t>(test_cases);
}
