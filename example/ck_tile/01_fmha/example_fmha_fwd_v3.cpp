// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <iomanip>
#include <iostream>
#include <optional>
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

#include "fmha_fwd.hpp"
#include "fmha_fwd_v3.hpp"
#include "mask.hpp"

auto parse_cmd_args(int argc, char* argv[]) -> std::pair<bool, ck_tile::ArgParser>
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("prec", "fp16", "data type. fp16/bf16")
        .insert("b", "2", "batch size")
        .insert("h", "8", "num of head, for q")
        .insert("h_k",
                "-1",
                "num of head, for k/v, -1 means equal to h\n"
                "if not equal to h, then this is GQA/MQA case")
        .insert("s", "3328", "seqlen_q")
        .insert("s_k", "-1", "seqlen_k, -1 means equal to s")
        .insert("d", "128", "head dim for q & k")
        .insert("scale_s", "0", "scale factor of S. 0 means equal to 1/sqrt(hdim)")
        .insert("iperm",
                "0",
                "permute input\n"
                "if true, will be b*h*s*d, else b*s*h*d")
        .insert("operm", "0", "permute output")
        .insert("causal", "0", "0: no mask, 1: causal mask")
        .insert("v", "1", "0:no verify, 1:verify")
        .insert("seed",
                "11939",
                "random seed used for initializing input tensors. 0 for "
                "non-deterministic seed")
        .insert("warmup", "5", "number of iterations before benchmark the kernel")
        .insert("repeat", "30", "number of iterations to benchmark the kernel");

    bool result = arg_parser.parse(argc, argv);
    return std::make_pair(result, arg_parser);
}

enum class TensorLayout
{
    bhsd,
    bshd,
};

std::ostream& operator<<(std::ostream& stream, TensorLayout layout)
{
    switch(layout)
    {
    case TensorLayout::bhsd: return stream << "bhsd";
    case TensorLayout::bshd: return stream << "bshd";
    default: return stream << "unknown";
    }
}

struct Problem
{
    explicit Problem(const ck_tile::ArgParser& args)
    {
        data_type = args.get_str("prec") == "fp16"
                        ? ck_tile::fmha_fwd_v3_args::data_type_enum::fp16
                        : ck_tile::fmha_fwd_v3_args::data_type_enum::bf16;
        batch     = args.get_int("b");
        seqlen_q  = args.get_int("s");
        seqlen_k  = args.get_int("s_k");
        if(seqlen_k < 0)
        {
            seqlen_k = seqlen_q;
        }
        nhead_q  = args.get_int("h");
        nhead_kv = args.get_int("h_k");
        if(nhead_kv < 0)
        {
            nhead_kv = nhead_q;
        }
        hdim          = args.get_int("d");
        softmax_scale = args.get_float("scale_s");
        if(softmax_scale == .0f)
            softmax_scale = 1.0 / ck_tile::sqrt(static_cast<float>(hdim));

        const auto is_causal = args.get_bool("causal");
        if(is_causal)
        {
            mask = mask_info::decode("b:-1,0", seqlen_q, seqlen_k);
        }
        else
        {
            mask = mask_info::decode("0", seqlen_q, seqlen_k);
        }

        input_layout  = args.get_int("iperm") == 1 ? TensorLayout::bhsd : TensorLayout::bshd;
        output_layout = args.get_int("operm") == 1 ? TensorLayout::bhsd : TensorLayout::bshd;
    }

    std::vector<ck_tile::index_t> get_query_shape() const
    {
        if(input_layout == TensorLayout::bhsd)
        {
            return {batch, nhead_q, seqlen_q, hdim};
        }
        else
        {
            return {batch, seqlen_q, nhead_q, hdim};
        }
    }

    std::vector<ck_tile::index_t> get_key_shape() const
    {
        if(input_layout == TensorLayout::bhsd)
        {
            return {batch, nhead_kv, seqlen_k, hdim};
        }
        else
        {
            return {batch, seqlen_k, nhead_kv, hdim};
        }
    }

    std::vector<ck_tile::index_t> get_value_shape() const
    {
        if(input_layout == TensorLayout::bhsd)
        {
            return {batch, nhead_kv, seqlen_k, hdim};
        }
        else
        {
            return {batch, seqlen_k, nhead_kv, hdim};
        }
    }

    std::vector<ck_tile::index_t> get_output_shape() const
    {
        if(output_layout == TensorLayout::bhsd)
        {
            return {batch, nhead_q, seqlen_q, hdim};
        }
        else
        {
            return {batch, seqlen_q, nhead_q, hdim};
        }
    }

    ck_tile::fmha_fwd_v3_args::data_type_enum data_type;
    ck_tile::index_t batch;
    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_kv;
    ck_tile::index_t hdim;
    float softmax_scale;
    mask_info mask;
    TensorLayout input_layout;
    TensorLayout output_layout;
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
        verify        = args.get_bool("v");
    }

    std::optional<uint32_t> seed;
    int kernel_warmup;
    int kernel_repeat;
    bool verify;
};

template <typename DataType>
auto generate_qkv(const Problem& problem,
                  [[maybe_unused]] std::optional<uint32_t> seed = std::nullopt)
    -> std::tuple<ck_tile::HostTensor<DataType>,
                  ck_tile::HostTensor<DataType>,
                  ck_tile::HostTensor<DataType>>
{
    ck_tile::HostTensor<DataType> q(problem.get_query_shape());
    ck_tile::HostTensor<DataType> k(problem.get_key_shape());
    ck_tile::HostTensor<DataType> v(problem.get_value_shape());

    ck_tile::FillNormalDistribution<DataType>{0.f, 3.f, seed}(q);
    ck_tile::FillNormalDistribution<DataType>{0.f, 3.f, seed}(k);
    ck_tile::FillNormalDistribution<DataType>{0.f, 3.f, seed}(v);

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
                           const SAccElementOp& s_acc_element_op = {})
{
    const int batch_size = q_bshd.mDesc.get_lengths()[0];
    const int seqlen_q   = q_bshd.mDesc.get_lengths()[1];
    const int seqlen_kv  = k_bshd.mDesc.get_lengths()[1];
    const int nhead_q    = q_bshd.mDesc.get_lengths()[2];
    const int nhead_kv   = k_bshd.mDesc.get_lengths()[2];
    const int hdim_qk    = q_bshd.mDesc.get_lengths()[3];
    const int hdim_v     = v_bshd.mDesc.get_lengths()[3];

    const int nr = nhead_q / nhead_kv;

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
        q_host_ref.ForEach([&](auto& self, auto idx) { self(idx) = q_bshd(b, idx[1], idx[0]     , idx[2]); });
        k_host_ref.ForEach([&](auto& self, auto idx) { self(idx) = k_bshd(b, idx[1], idx[0] / nr, idx[2]); });
        v_host_ref.ForEach([&](auto& self, auto idx) { self(idx) = v_bshd(b, idx[2], idx[0] / nr, idx[1]); });
        // clang-format on
        ck_tile::reference_batched_gemm<QDataType, KDataType, AccDataType>(
            q_host_ref, k_host_ref, s_host_ref, q_element_op, k_element_op, s_acc_element_op);

        if(mask.type == mask_enum::no_mask)
        {
            ck_tile::reference_batched_masking(s_host_ref, FmhaMasks::NoMask{seqlen_q, seqlen_kv});
        }
        else if(mask.type == mask_enum::window_generic)
        {
            ck_tile::reference_batched_masking(
                s_host_ref,
                ck_tile::make_generic_attention_mask_from_lr_window<FmhaMasks::GenericMask>(
                    mask.left, mask.right, seqlen_q, seqlen_kv));
        }
        else
        {
            // if left window size is negative, means causal
            // else means generic (for current batch)
            if(mask.left < 0)
                ck_tile::reference_batched_masking(
                    s_host_ref,
                    ck_tile::make_generic_attention_mask_from_lr_window<FmhaMasks::CausalMask>(
                        mask.left,
                        mask.right,
                        seqlen_q,
                        seqlen_kv,
                        mask.type == mask_enum::mask_top_left));
            else
                ck_tile::reference_batched_masking(
                    s_host_ref,
                    ck_tile::make_generic_attention_mask_from_lr_window<FmhaMasks::GenericMask>(
                        mask.left,
                        mask.right,
                        seqlen_q,
                        seqlen_kv,
                        mask.type == mask_enum::mask_top_left));
        }

        ck_tile::reference_batched_softmax<AccDataType, AccDataType>(
            s_host_ref, p_host_ref, ck_tile::identity{});

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
    auto [q, k, v] = generate_qkv<DataType>(problem, run_config.seed);

    ck_tile::DeviceMem q_buf(q.get_element_space_size_in_bytes());
    ck_tile::DeviceMem k_buf(k.get_element_space_size_in_bytes());
    ck_tile::DeviceMem v_buf(v.get_element_space_size_in_bytes());
    /// FIXME: use correct size for output tensor. just use q size for now since hidm_qk = hdim_v
    ck_tile::DeviceMem o_buf(q.get_element_space_size_in_bytes());

    q_buf.ToDevice(q.data());
    k_buf.ToDevice(k.data());
    v_buf.ToDevice(v.data());

    ck_tile::fmha_fwd_v3_args args;

    args.data_type     = problem.data_type;
    args.batch         = problem.batch;
    args.seqlen_q      = problem.seqlen_q;
    args.seqlen_k      = problem.seqlen_k;
    args.nhead_q       = problem.nhead_q;
    args.nhead_kv      = problem.nhead_kv;
    args.hdim_qk       = problem.hdim;
    args.hdim_v        = problem.hdim;
    args.softmax_scale = problem.softmax_scale;

    args.window_size_left  = problem.mask.left;
    args.window_size_right = problem.mask.right;
    args.mask_type         = static_cast<ck_tile::index_t>(problem.mask.type);

    // bshd: (batch, seqlen_q, nhead_q, hdim)
    // bhsd: (batch, nhead_q, seqlen_q, hdim)
    args.q_ptr = q_buf.GetDeviceBuffer();
    args.stride_q =
        problem.input_layout == TensorLayout::bshd ? problem.nhead_q * problem.hdim : problem.hdim;
    args.nhead_stride_q =
        problem.input_layout == TensorLayout::bshd ? problem.hdim : problem.seqlen_q * problem.hdim;
    args.batch_stride_q = problem.seqlen_q * problem.nhead_q * problem.hdim;

    // bshd: (batch, seqlen_k, nhead_kv, hdim)
    // bhsd: (batch, nhead_kv, seqlen_k, hdim)
    args.k_ptr = k_buf.GetDeviceBuffer();
    args.stride_k =
        problem.input_layout == TensorLayout::bshd ? problem.nhead_kv * problem.hdim : problem.hdim;
    args.nhead_stride_k =
        problem.input_layout == TensorLayout::bshd ? problem.hdim : problem.seqlen_k * problem.hdim;
    args.batch_stride_k = problem.seqlen_k * problem.nhead_kv * problem.hdim;

    // bshd: (batch, seqlen_k, nhead_kv, hdim)
    // bhsd: (batch, nhead_kv, seqlen_k, hdim)
    args.v_ptr = v_buf.GetDeviceBuffer();
    args.stride_v =
        problem.input_layout == TensorLayout::bshd ? problem.nhead_kv * problem.hdim : problem.hdim;
    args.nhead_stride_v =
        problem.input_layout == TensorLayout::bshd ? problem.hdim : problem.seqlen_k * problem.hdim;
    args.batch_stride_v = problem.seqlen_k * problem.nhead_kv * problem.hdim;

    // bshd: (batch, seqlen_q, nhead_q, hdim)
    // bhsd: (batch, nhead_q, seqlen_q, hdim)
    args.o_ptr = o_buf.GetDeviceBuffer();
    args.stride_o =
        problem.output_layout == TensorLayout::bshd ? problem.nhead_q * problem.hdim : problem.hdim;
    args.nhead_stride_o = problem.output_layout == TensorLayout::bshd
                              ? problem.hdim
                              : problem.seqlen_q * problem.hdim;
    args.batch_stride_o = problem.seqlen_q * problem.nhead_q * problem.hdim;

    ck_tile::stream_config stream_config{nullptr,
                                         true,
                                         /*log_level=*/0,
                                         run_config.kernel_warmup,
                                         run_config.kernel_repeat};

    auto [result, time] = ck_tile::fmha_fwd_v3(args, stream_config);
    if(!result)
    {
        std::cerr << "faild to run fmha_fwd_v3()" << std::endl;
        return false;
    }

    std::size_t flop = [&] {
        if(problem.mask.type == mask_enum::no_mask)
        {
            return 4 * problem.batch * problem.nhead_q * problem.seqlen_q * problem.seqlen_k *
                   problem.hdim;
        }
        else
        {
            /// FIXME: Use a more accurate method; for now, we’re just dividing the flop by 2.
            return 2 * problem.batch * problem.nhead_q * problem.seqlen_q * problem.seqlen_k *
                   problem.hdim;
        }
    }();
    float tflops = static_cast<float>(flop) / 1.e9 / time;

    std::cout << "[" << problem.data_type << "|";
    if(problem.input_layout == problem.output_layout)
    {
        std::cout << problem.input_layout;
    }
    else
    {
        std::cout << problem.input_layout << "-" << problem.output_layout;
    }
    std::cout << "] b:" << problem.batch << ", h:" << problem.nhead_q << "/" << problem.nhead_kv
              << ", s:" << problem.seqlen_q << "/" << problem.seqlen_k << ", d:" << problem.hdim
              << ", scale_s:" << problem.softmax_scale << ", mask:" << problem.mask << std::fixed
              << ", " << std::setprecision(3) << time << " ms, " << std::setprecision(2) << tflops
              << " TFlops" << std::endl;

    if(!run_config.verify)
    {
        return true;
    }

    // transpose tensor descriptors from bhsd to bshd if necessary
    if(problem.input_layout != TensorLayout::bshd)
    {
        q = q.transpose({0, 2, 1, 3});
        k = k.transpose({0, 2, 1, 3});
        v = v.transpose({0, 2, 1, 3});
    }

    ck_tile::HostTensor<DataType> o_ref(problem.get_output_shape());
    if(problem.output_layout != TensorLayout::bshd)
    {
        o_ref = o_ref.transpose({0, 2, 1, 3});
    }

    host::fmha_fwd<float, DataType>(q,
                                    k,
                                    v,
                                    problem.mask,
                                    o_ref,
                                    ck_tile::identity{},
                                    ck_tile::identity{},
                                    ck_tile::identity{},
                                    ck_tile::scales{problem.softmax_scale});

    ck_tile::HostTensor<DataType> o(problem.get_output_shape());
    o_buf.FromDevice(o.data());

    const auto [rtol, atol] = [&] {
        if constexpr(std::is_same_v<DataType, ck_tile::fp16_t>)
            return std::make_tuple(1e-3, 1e-3);
        else
            return std::make_tuple(1e-2, 1e-2);
    }();
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
        if(problem.data_type == ck_tile::fmha_fwd_v3_args::data_type_enum::fp16)
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
