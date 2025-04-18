// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

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

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("v", "1", "weather do CPU validation or not")
        .insert("pr_i", "int32", "index data type. (currently only int32 supported now)")
        .insert("pr_w", "fp32", "output weight data type(currently only fp32 supported now)")
        .insert("t", "128", "number of input tokens")
        .insert("e", "8", "number of num_experts")
        .insert("k", "4", "topk")
        .insert("unit", "32", "unit_size")
        .insert("moe_buf_size", "0", "moe_buf_size")
        .insert("local_eid",
                "-1",
                "a list of experts enabled as local expert. e.g. \"0,1,4,5\"\n"
                "please make sure eid is in ascending order!")
        .insert("seed", "-1", "seed to be used, -1 means random every time")
        .insert("kname", "0", "when set to 1 it will print kernel name")
        .insert("warmup", "5", "number of iterations before benchmark the kernel")
        .insert("repeat", "20", "number of iterations to benchmark the kernel");

    bool result = arg_parser.parse(argc, argv);
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
    int num_experts         = args.get_int("e");
    int topk                = args.get_int("k");
    int seed                = args.get_int("seed");
    int unit_size           = args.get_int("unit");
    int64_t moe_buf_size    = static_cast<int64_t>(args.get_uint64("moe_buf_size"));
    int kname               = args.get_int("kname");
    int warmup              = args.get_int("warmup");
    int repeat              = args.get_int("repeat");

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

    bool local_expert_masking      = args.get_str("local_eid") != "-1";
    auto local_expert_masking_host = [&]() {
        if(local_expert_masking)
        {
            auto local_eid = args.get_int_vec("local_eid");
            // std::vector<int> v_ {num_experts, 0};
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
            // return std::vector<int>{};
            return ck_tile::HostTensor<IndexType>{{1}};
    }();

    // tokens already considered batch size
    ck_tile::HostTensor<IndexType> topk_ids_host({tokens, topk}, {topk, 1});
    ck_tile::HostTensor<WeightType> weights_host({tokens, topk}, {topk, 1});
    ck_tile::HostTensor<IndexType> sorted_ids_host({max_output_ids}, {1});
    ck_tile::HostTensor<WeightType> sorted_weights_host({max_output_ids}, {1});
    ck_tile::HostTensor<IndexType> sorted_expert_ids_host({max_output_ids / unit_size}, {1});
    ck_tile::HostTensor<IndexType> sorted_id_cnt_host({1}, {1});
    ck_tile::HostTensor<float> moe_buf_host({moe_buf_size});

    ck_tile::FillUniformDistribution<WeightType>{-.5f, .5f}(weights_host);
    ck_tile::FillUniformDistribution<WeightType>{-.5f, .5f}(moe_buf_host);
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

    topk_ids_dev.ToDevice(topk_ids_host.data());
    weights_dev.ToDevice(weights_host.data());
    if(moe_buf_size > 0)
    {
        moe_buf_dev.ToDevice(moe_buf_host.data());
    }
    if(local_expert_masking)
        local_expert_masking_dev.ToDevice(local_expert_masking_host.data());

    // if return zero, means no need workspace, can set moe_sorting_args.p_ws to nullptr
    ck_tile::index_t workspace_size = moe_sorting_get_workspace_size(tokens, num_experts);
    ck_tile::DeviceMem moe_sorting_ws(workspace_size != 0 ? workspace_size : 0);

    if(workspace_size != 0)
        moe_sorting_ws.SetZero(); // note, clear here!!!!

    moe_sorting_trait trait{index_prec, weight_prec, local_expert_masking};

    moe_sorting_args karg{topk_ids_dev.GetDeviceBuffer(),
                          weights_dev.GetDeviceBuffer(),
                          local_expert_masking ? local_expert_masking_dev.GetDeviceBuffer()
                                               : nullptr,
                          sorted_ids_dev.GetDeviceBuffer(),
                          sorted_weights_dev.GetDeviceBuffer(),
                          sorted_expert_ids_dev.GetDeviceBuffer(),
                          sorted_id_cnt_dev.GetDeviceBuffer(),
                          moe_buf_size > 0 ? moe_buf_dev.GetDeviceBuffer() : nullptr,
                          workspace_size != 0 ? moe_sorting_ws.GetDeviceBuffer() : nullptr,
                          tokens,
                          unit_size,
                          num_experts,
                          topk,
                          static_cast<ck_tile::long_index_t>(moe_buf_size * sizeof(float))};

    ck_tile::stream_config sc{nullptr,
                              true,
                              /* log_level = */ (kname ? 1 : 0),
                              warmup,
                              repeat};

    auto ms = moe_sorting(trait, karg, sc);
    // auto ms = moe_sorting_mp(trait, karg, sc);

#if 0
    {
        ck_tile::HostTensor<char> ws_host({workspace_size}, {1});
        moe_sorting_ws.FromDevice(ws_host.data());

        int * p_mesh = reinterpret_cast<int*>(ws_host.data());
        ck_tile::index_t row_size = ck_tile::impl::moe_sorting_mp_mesh_stride(tokens);

        std::cout << "topk_ids:" << std::endl;

        int * p_topk_ids = reinterpret_cast<int*>(topk_ids_host.data());
        for(int i_token = 0; i_token < tokens; i_token++) {
            printf("[t:%2d]", i_token);
            for(int i_topk = 0; i_topk < topk; i_topk++) {
                printf("%d, ",p_topk_ids[i_token * topk + i_topk] );
            }
            printf("\n");
        }
        printf("----------------\n");

        std::vector<int> l_cumsum (num_experts + 1, 0);
        for(int i_expert = 0; i_expert < num_experts; i_expert++  ) {
            printf("[e:%2d]", i_expert);
            int e_cnt = 0;
            for(int i_token = 0; i_token < tokens; i_token++) {
                auto v_mesh = p_mesh[i_expert * row_size + i_token];
                e_cnt += v_mesh != 0 ? 1 : 0;
                printf("%d, ", v_mesh); 
            }
            int e_cnt_unit = (e_cnt + unit_size - 1) / unit_size;
            printf("[%d/%d]", e_cnt, e_cnt_unit);
            printf("\n");
            l_cumsum[i_expert + 1] = l_cumsum[i_expert] + e_cnt_unit;
        }

        printf("----------------\n");
        printf("cumsum:\n");
        for(int i_cc= 0; i_cc < num_experts + 1; i_cc++) {
            printf("%2d, ", l_cumsum[i_cc]);
        }
        printf("\n");
        printf("----------------\n");

        int * p_cumsum = p_mesh + ck_tile::impl::moe_sorting_mp_mesh_elem(tokens, num_experts);
        for(int i_expert = 0; i_expert < num_experts + 1; i_expert++  ) {
            printf("%2d(%d), ",p_cumsum[i_expert], p_cumsum[i_expert] / unit_size);
        }
        printf("\n");
    }
#endif

    printf("[%s|%s]tokens:%d, num_experts:%d, topk:%d, mp:%d, ",
           index_prec.c_str(),
           weight_prec.c_str(),
           tokens,
           num_experts,
           topk,
           workspace_size != 0 ? 1 : 0);

    if(local_expert_masking)
    {
        printf("local_eid:%s, ", args.get_str("local_eid").c_str());
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
    if(moe_buf_size > 0)
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
        }
        else
        {
            printf("(token size not equal!!)");
            rtn = false;
        }

        if(moe_buf_size)
        {
            ck_tile::HostTensor<WeightType> moe_buf_ref({moe_buf_size});
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

int main(int argc, char** argv)
{
    auto [result, args] = create_args(argc, argv);
    if(!result)
        return -1;
    std::string index_prec  = args.get_str("pr_i");
    std::string weight_prec = args.get_str("pr_w");

    bool r = true;
    if(weight_prec.compare("fp32") == 0 && index_prec.compare("int32") == 0)
    {
        r &= test_moe_sorting<float, ck_tile::index_t>(args);
    }
    return r ? 0 : -1;
}
