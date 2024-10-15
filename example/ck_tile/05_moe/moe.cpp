// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
#include "moe.hpp"
#include "ck_tile/host.hpp"
//#include "rotary.hpp"
//#include "utils.hpp"
#include "ck_tile/host/reference/reference_permute.hpp"
#include "include/ck_tile/ops/fused_moe/pipeline/fused_moe_pipeline_nsplit2.hpp"
#include "include/ck_tile/ops/fused_moe/pipeline/fused_moe_pipeline_problem.hpp"
#include "include/ck_tile/ops/fused_moe/pipeline/fused_moe_tile_shape.hpp"
#include "include/ck_tile/ops/fused_moe/pipeline/fused_moe_traits.hpp"
#include "include/ck_tile/ops/fused_moe/pipeline/fused_moe_weight_permute_enum.hpp"
#include "include/ck_tile/ops/fused_moe/kernel/fused_moe_kernel.hpp"

#include <array>
#include <cstring>
#include <functional>
#include <numeric>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
//#include <torch/torch.h>
//test args
auto create_args(int argc, char* argv[])
{
    // get command line data to internal params
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("num_tokens", "10", "")
        .insert("num_experts", "8", "")
        .insert("v", "0", "validation")
        .insert("hidden_size", "4096", "")
        .insert("shard_intermediate_size", "4096", "")
        .insert("topk",
                "2",
                "\n"
                "")
        .insert(
            "dtype",
            "fp16",
            "\n"
            "")
        .insert("use_fp8_w8a8", "0", "")
        .insert("use_int8_w8a16",
                "0",
                "");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

//run args assertion and tensor allocation
    //and slope/scale gen, tensor copy to device
    //init traits/feature instant
    //init args, from tensor to tensor pointer and stride, real args to kernel
    //call moe
    //move result to host
    //referrence gen
    //validation
template <typename DataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    std::string data_type    = arg_parser.get_str("dtype");
    int do_validation        = arg_parser.get_int("v");
    bool use_fp8_w8a8 = arg_parser.get_bool("use_fp8_w8a8");
    bool use_int8_w8a16 = arg_parser.get_bool("use_int8_w8a16");
   // auto mode                = static_cast<mode_enum>(arg_parser.get_uint32("mode"));
    ck_tile::index_t num_tokens   = arg_parser.get_int("num_tokens");
    ck_tile::index_t num_experts   = arg_parser.get_int("num_experts");
    ck_tile::index_t hidden_size = arg_parser.get_int("hidden_size");
    ck_tile::index_t shard_intermediate_size = arg_parser.get_int("shard_intermediate_size");
    ck_tile::index_t topk = arg_parser.get_int("topk");

    int stream_warmup = arg_parser.get_int("warmup");
    int stream_repeat = arg_parser.get_int("repeat");
    bool kname        = arg_parser.get_bool("kname");

    ck_tile::stream_config stream_config{nullptr,
                                         true,
                                         /* log_level = */ (kname ? 1 : 0),
                                         stream_warmup,
                                         stream_repeat,
                                         arg_parser.get_str("timer") == std::string("gpu")};

    //type config, need type config before tensor gen, and define the acc types
    using TypeConfig = MoeTypeConfig<DataType>;
    using ADataType             = typename TypeConfig::ADataType;
    using GDataType             = typename TypeConfig::GDataType;
    using UDataType             = typename TypeConfig::UDataType;
    using DDataType          = typename TypeConfig::DDataType;
    using ODataType          = typename TypeConfig::ODataType;
    using AccDataType = typename TypeConfig::AccDataType;
    using ScaleDataType = typename TypeConfig::ScaleDataType;

    //tensor
    ck_tile::HostTensor<GDataType> g_host_ref({num_experts,
                               shard_intermediate_size/2,
                               hidden_size/16,2,8});
    ck_tile::HostTensor<UDataType> u_host_ref({num_experts,
                               shard_intermediate_size/2,
                               hidden_size/16,2,8});
    ck_tile::HostTensor<DDataType> d_host_ref({num_experts,
                               hidden_size,
                               shard_intermediate_size/2/16,2,8});
    //reference_permute(const HostTensor<DataType>& x, HostTensor<DataType>& y, std::vector<index_t> dims)
    ck_tile::HostTensor<ADataType> a_host({num_tokens, hidden_size});
    ck_tile::HostTensor<GDataType> g_host({num_experts,
                               shard_intermediate_size/2,
                               hidden_size});
    ck_tile::HostTensor<UDataType> u_host({num_experts,
                               shard_intermediate_size/2,
                               hidden_size});
    ck_tile::HostTensor<DDataType> d_host({num_experts,
                               hidden_size,
                               shard_intermediate_size/2});
    ck_tile::reference_permute<GDataType>(g_host_ref, g_host, {0, 1, 3, 4, 2, 5});
    ck_tile::reference_permute<GDataType>(u_host_ref, u_host, {0, 1, 3, 4, 2, 5});
    ck_tile::reference_permute<GDataType>(d_host_ref, d_host, {0, 1, 3, 4, 2, 5});
    ck_tile::HostTensor<ODataType> o_host({num_tokens, hidden_size});

    ck_tile::HostTensor<ck_tile::fp32_t> sorted_weights({num_tokens,topk});
    ck_tile::HostTensor<ck_tile::index_t> sorted_topk_ids({num_tokens,topk});
    ck_tile::HostTensor<ck_tile::index_t> sorted_expert_ids({num_tokens,topk});
    ck_tile::HostTensor<ck_tile::index_t> sorted_num_tokens_post_padded({1});
    //device buffer
    ck_tile::DeviceMem a_buf(a_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem g_buf(g_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem u_buf(u_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d_buf(d_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem o_buf(o_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem sorted_weight_buf(sorted_weights.get_element_space_size_in_bytes());
    ck_tile::DeviceMem sorted_topk_ids_buf(sorted_topk_ids.get_element_space_size_in_bytes());
    ck_tile::DeviceMem sorted_expert_ids_buf(sorted_expert_ids.get_element_space_size_in_bytes());
    ck_tile::DeviceMem sorted_tiles_buf(sorted_num_tokens_post_padded.get_element_space_size_in_bytes());
    a_buf.ToDevice(a_host.data());
    g_buf.ToDevice(g_host.data());
    u_buf.ToDevice(u_host.data());
    d_buf.ToDevice(d_host.data());
    
    //init traits
    const auto init_traits = [&](auto& traits) {
        traits.DownPreShuffled        = 0;   
    };
    //init host args pack internal params to a struct to pass to kernel
    const auto init_args = [&](auto& args) {
        const ck_tile::index_t stride_a = hidden_size;
        const ck_tile::index_t stride_gu = hidden_size;
        const ck_tile::index_t stride_d = shard_intermediate_size/2;
        const ck_tile::index_t stride_o = hidden_size;
        const ck_tile::index_t stride_expert_gu = hidden_size * shard_intermediate_size/2;
        const ck_tile::index_t stride_expert_d = hidden_size * shard_intermediate_size/2;

        args.a_ptr = a_buf.GetDeviceBuffer();
        args.g_ptr = g_buf.GetDeviceBuffer();
        args.u_ptr = u_buf.GetDeviceBuffer();
        args.d_ptr = d_buf.GetDeviceBuffer();
        args.o_ptr = o_buf.GetDeviceBuffer();
        args.sorted_token_ids_ptr = sorted_topk_ids_buf.GetDeviceBuffer();
        args.sorted_weight_ptr = sorted_weight_buf.GetDeviceBuffer();
        args.sorted_expert_ids_ptr = sorted_expert_ids_buf.GetDeviceBuffer();
        args.num_sorted_tiles_ptr = sorted_tiles_buf.GetDeviceBuffer();
        args.stride_a = stride_a;
        args.stride_gu = stride_gu;
        args.stride_d = stride_d;
        args.stride_o = stride_o;
        args.stride_expert_gu = stride_expert_gu;
        args.stride_expert_d = stride_expert_d;
        
//        args.dim_size = dim_size;
        args.hidden_size = hidden_size;
        args.num_tokens = num_tokens;  // input number of tokens for current iteration
        args.num_experts = num_experts; 
    };
    //
   // constexpr ck_tile::index_t ts_experts = experts_;
    //tiling
    using moe_block_tile_0   = ck_tile::sequence<32,  // kM_a
                                        128, // kN_g/u
                                        128, // kN_sub0
                                        32,  // kK_a
                                        128 // kN_d
                                        >;
    using moe_block_warps0_0 = ck_tile::sequence<1, 4, 1>;//mnk
    using moe_block_warps1_0 = ck_tile::sequence<4, 1, 1>;
    using moe_warp_tile_0    = ck_tile::sequence<32, 32, 16>;
    // using fmha_warp_tile_4    = ck::Sequence<32, 32, 8>;

    using moe_shape = ck_tile::FusedMoeTileShape<moe_block_tile_0,
                                                                moe_block_warps0_0,
                                                                moe_warp_tile_0,
                                                                moe_block_warps1_0,
                                                                moe_warp_tile_0>; 
    using moe_traits = ck_tile::FusedMoeTraits<false,//down preshuffle
          -1, // index_t kBlockPerCu_  = ,overwrite occupancy if not -1
          0,//index_t OAtomic_
          ck_tile::FusedMoeWeightPermuteEnum::permute_b_nr_kr_kw_nw_kv//FusedMoeWeightPermuteEnum WeightPermute_ =
          >;
    using moe_problem  = ck_tile::FusedMoePipelineProblem<ADataType, GDataType, UDataType, DDataType,
          ODataType, AccDataType, ScaleDataType, ck::tensor_operation::element_wise::Silu, moe_shape, moe_traits>; 
    using moe_pipeline = ck_tile::FusedMoePipelineNSplit2<moe_problem>;                     
    using Hargs = ck_tile::FusedMoeKernel::FusedMoeCommonHargs;
    using moe_partitioner = ck_tile::FusedMoeTilePartitioner_PersistentSplitD<moe_shape>;                                                                                            \
    using kernel = ck_tile::FusedMoeKernel<moe_partitioner,moe_pipeline>;
    using Kargs = ck_tile::FusedMoeKernel::FusedMoeCommonKargs;
    Hargs hargs;
    Kargs kargs;
    //args to hargs                                     
    init_args[](hargs);                                                                                                         \
    auto kargs = kernel::MakeKargs(hargs);
    int cu_count = getAvailableComputeUnitCount(stream_config);                                                                                                                                       \
    const dim3 grids      = kernel::GridSize(cu_count,moe_pipeline::kBlockPerCu);                                                
    constexpr dim3 blocks = kernel::BlockSize();                                                
                                                                                                
    float ave_time = ck_tile::launch_kernel(                                                    
        s, ck_tile::make_kernel<blocks.x, 1>(kernel{}, grids, blocks, 0, kargs));               
                                                                                                
    return ave_time;


}

//main
int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if (!result)
        return -1;
    const std::string data_type = arg_parser.get_str("prec");
    if(data_type == "fp16")
    {
        return run<ck_tile::half_t>(arg_parser) ? 0 : -2;
    }
    else if(data_type == "bf16")
    {
        return run<ck_tile::bf16_t>(arg_parser) ? 0 : -2;
    }
    else if(data_type == "fp8")
    {
        return run<ck_tile::fp8_t>(arg_parser) ? 0 : -2;
    }

    return -3;
}
    //call creat args
    //call run
     

    //return
