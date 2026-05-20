// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "mask.hpp"
#include "bias.hpp"

#include <functional>
#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>
#include <variant>

struct FmhaBwdFp32
{
};

struct FmhaBwdFp16
{
};

struct FmhaBwdBf16
{
};

template <typename DataType>
struct FmhaBwdTypeConfig;

template <>
struct FmhaBwdTypeConfig<FmhaBwdFp32>
{
    using QDataType             = float;
    using KDataType             = float;
    using VDataType             = float;
    using GemmDataType          = float;
    using BiasDataType          = float;
    using LSEDataType           = float;
    using AccDataType           = float; // data type for gemm accumulation
    using DDataType             = float;
    using RandValOutputDataType = uint8_t;
    using ODataType             = float;
    using OGradDataType         = float;
    using QGradDataType         = float;
    using KGradDataType         = float;
    using VGradDataType         = float;
    using BiasGradDataType      = float;
};

template <>
struct FmhaBwdTypeConfig<FmhaBwdFp16>
{
    using QDataType             = ck_tile::half_t;
    using KDataType             = ck_tile::half_t;
    using VDataType             = ck_tile::half_t;
    using GemmDataType          = ck_tile::half_t;
    using BiasDataType          = ck_tile::half_t;
    using LSEDataType           = float;
    using AccDataType           = float; // data type for gemm accumulation
    using DDataType             = float;
    using RandValOutputDataType = uint8_t;
    using ODataType             = ck_tile::half_t;
    using OGradDataType         = ck_tile::half_t;
    using QGradDataType         = ck_tile::half_t;
    using KGradDataType         = ck_tile::half_t;
    using VGradDataType         = ck_tile::half_t;
    using BiasGradDataType      = ck_tile::half_t;
};

template <>
struct FmhaBwdTypeConfig<FmhaBwdBf16>
{
    using QDataType             = ck_tile::bf16_t;
    using KDataType             = ck_tile::bf16_t;
    using VDataType             = ck_tile::bf16_t;
    using GemmDataType          = ck_tile::bf16_t;
    using BiasDataType          = ck_tile::bf16_t;
    using LSEDataType           = float;
    using AccDataType           = float; // data type for gemm accumulation
    using DDataType             = float;
    using RandValOutputDataType = uint8_t;
    using ODataType             = ck_tile::bf16_t;
    using OGradDataType         = ck_tile::bf16_t;
    using QGradDataType         = ck_tile::bf16_t;
    using KGradDataType         = ck_tile::bf16_t;
    using VGradDataType         = ck_tile::bf16_t;
    using BiasGradDataType      = ck_tile::bf16_t;
};

struct FmhaMasks
{
    using NoMask      = ck_tile::GenericAttentionMask<false>;
    using GenericMask = ck_tile::GenericAttentionMask<true, true>;
    using CausalMask  = ck_tile::GenericAttentionMask<true, false>;
};

// runtime args, some will passed to karg, some will used to compute grids/blocks
struct fmha_bwd_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr; // bias or alibi_slope pointer
    const void* o_ptr;
    const void* lse_ptr;
    const void* do_ptr;
    void* d_ptr;
    void* rand_val_ptr;
    void* dq_ptr;
    void* dk_ptr;
    void* dv_ptr;
    void* dbias_ptr;
    void* workspace_ptr;
    const void*
        sink_ptr; // sink scores [batch, nhead] in log-space (LSEDataType); nullptr disables sink
    void* d_sink_ptr; // sink gradient output [nhead] (LSEDataType); nullptr disables sink gradient

    // Usage notes for sequence length pointer parameters:
    //
    // [Note: Define "Group mode" vs "Batch mode" here if possible, e.g., "Group mode handles
    // MQA/GQA..."]
    //
    // With padding:
    //   Group mode:
    //     - seqstart_q_ptr, seqstart_k_ptr: Record cumulative physical (including padding) sequence
    //       lengths. [array size: batch + 1]
    //     - seqlen_q_ptr/seqlen_k_ptr: Records logical (excluding padding) length for each
    //       sequence. [array size: batch]
    //     - cu_seqlen_q_ptr/cu_seqlen_k_ptr: Records cumulative logical (excluding padding)
    //       sequence lengths. [array size: batch + 1]
    //     - seqlen_q_ptr (per-sequence) and cu_seqlen_q_ptr (cumulative logical) are mutually
    //       exclusive. Use one set, not both.
    //
    //   Batch mode:
    //     - cu_seqlen_q_ptr/cu_seqlen_k_ptr: Records cumulative logical (excluding padding)
    //     sequence lengths. [array size: batch + 1]
    //     - seqstart_* and seqlen_* pointers must be nullptr.
    //
    // Without padding:
    //   (Note: Physical length equals logical length)
    //
    //   Group mode:
    //     - seqstart_q_ptr, seqstart_k_ptr: Record cumulative physical sequence lengths. [array
    //     size: batch + 1]
    //     - seqlen_q_ptr/seqlen_k_ptr and cu_seqlen_q_ptr/cu_seqlen_k_ptr must be nullptr.
    //
    //   Batch mode:
    //     - All sequence length pointers (seqstart_*, seqlen_*, cu_seqlen_*) must be nullptr.
    //
    const void* seqstart_q_ptr =
        nullptr; // Cumulative physical sequence length array [batch + 1]. (Used in Group mode)
    const void* seqstart_k_ptr =
        nullptr; // Cumulative physical sequence length array [batch + 1]. (Used in Group mode)
    const void* seqlen_q_ptr = nullptr;    // Per-sequence logical (excluding padding) length array
                                           // [batch]. (Used in Group mode with padding)
    const void* seqlen_k_ptr = nullptr;    // Per-sequence logical (excluding padding) length array
                                           // [batch]. (Used in Group mode with padding)
    const void* cu_seqlen_q_ptr = nullptr; // Cumulative logical (excluding padding) sequence length
                                           // array [batch + 1]. (Used with padding)
    const void* cu_seqlen_k_ptr = nullptr; // Cumulative logical (excluding padding) sequence length
                                           // array [batch + 1]. (Used with padding)
    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t batch;
    ck_tile::index_t max_seqlen_q;
    ck_tile::index_t max_seqlen_k;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;
    float scale;
    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_bias; // if alibi, b*h need set this to h, 1*h need set this to 0
    ck_tile::index_t stride_o;
    ck_tile::index_t stride_randval;
    ck_tile::index_t stride_do;
    ck_tile::index_t stride_dq;
    ck_tile::index_t stride_dk;
    ck_tile::index_t stride_dv;
    ck_tile::index_t stride_dbias;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_bias;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t nhead_stride_randval;
    ck_tile::index_t nhead_stride_do;
    ck_tile::index_t nhead_stride_lsed;
    ck_tile::index_t nhead_stride_dq;
    ck_tile::index_t nhead_stride_dk;
    ck_tile::index_t nhead_stride_dv;
    ck_tile::index_t nhead_stride_dbias;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_bias;
    ck_tile::index_t batch_stride_o;
    ck_tile::index_t batch_stride_randval;
    ck_tile::index_t batch_stride_do;
    ck_tile::index_t batch_stride_lsed;
    ck_tile::index_t batch_stride_dq;
    ck_tile::index_t batch_stride_dk;
    ck_tile::index_t batch_stride_dv;
    ck_tile::index_t batch_stride_dbias;
    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t mask_type;
    float p_drop;
    float p_undrop;
    std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>>
        drop_seed_offset;
};

template <typename FmhaBwdDQDKDVKernel>
auto fmha_bwd_dq_dk_dv_create_kargs_and_grids(fmha_bwd_args args)
{
    assert(args.nhead_q % args.nhead_k == 0);
    auto kargs = [&] {
        // create group mode kernel arguments
        if constexpr(FmhaBwdDQDKDVKernel::kIsGroupMode)
        {
            return FmhaBwdDQDKDVKernel::MakeKargsImpl(args.q_ptr,
                                                      args.k_ptr,
                                                      args.v_ptr,
                                                      args.bias_ptr,
                                                      args.lse_ptr,
                                                      args.do_ptr,
                                                      args.d_ptr,
                                                      args.rand_val_ptr,
                                                      args.dq_ptr,
                                                      args.dk_ptr,
                                                      args.dv_ptr,
                                                      args.dbias_ptr,
                                                      args.workspace_ptr,
                                                      args.seqstart_q_ptr,
                                                      args.seqstart_k_ptr,
                                                      args.seqlen_q_ptr,
                                                      args.seqlen_k_ptr,
                                                      args.cu_seqlen_q_ptr,
                                                      args.cu_seqlen_k_ptr,
                                                      args.batch,
                                                      args.hdim_q,
                                                      args.hdim_v,
                                                      args.nhead_q,
                                                      args.nhead_q / args.nhead_k,
                                                      args.scale,
                                                      args.stride_q,
                                                      args.stride_k,
                                                      args.stride_v,
                                                      args.stride_bias,
                                                      args.stride_randval,
                                                      args.stride_do,
                                                      args.stride_dq,
                                                      args.stride_dk,
                                                      args.stride_dv,
                                                      args.stride_dbias,
                                                      args.nhead_stride_q,
                                                      args.nhead_stride_k,
                                                      args.nhead_stride_v,
                                                      args.nhead_stride_bias,
                                                      args.nhead_stride_randval,
                                                      args.nhead_stride_do,
                                                      args.nhead_stride_lsed,
                                                      args.nhead_stride_dq,
                                                      args.nhead_stride_dk,
                                                      args.nhead_stride_dv,
                                                      args.nhead_stride_dbias,
                                                      args.window_size_left,
                                                      args.window_size_right,
                                                      args.mask_type,
                                                      args.p_drop,
                                                      args.drop_seed_offset);
        }
        else
        { // create batch mode kernel arguments
            return FmhaBwdDQDKDVKernel::MakeKargsImpl(args.q_ptr,
                                                      args.k_ptr,
                                                      args.v_ptr,
                                                      args.bias_ptr,
                                                      args.lse_ptr,
                                                      args.do_ptr,
                                                      args.d_ptr,
                                                      args.rand_val_ptr,
                                                      args.dq_ptr,
                                                      args.dk_ptr,
                                                      args.dv_ptr,
                                                      args.dbias_ptr,
                                                      args.workspace_ptr,
                                                      args.seqlen_q,
                                                      args.seqlen_k,
                                                      args.batch,
                                                      args.hdim_q,
                                                      args.hdim_v,
                                                      args.nhead_q,
                                                      args.nhead_q / args.nhead_k,
                                                      args.scale,
                                                      args.stride_q,
                                                      args.stride_k,
                                                      args.stride_v,
                                                      args.stride_bias,
                                                      args.stride_randval,
                                                      args.stride_do,
                                                      args.stride_dq,
                                                      args.stride_dk,
                                                      args.stride_dv,
                                                      args.stride_dbias,
                                                      args.nhead_stride_q,
                                                      args.nhead_stride_k,
                                                      args.nhead_stride_v,
                                                      args.nhead_stride_bias,
                                                      args.nhead_stride_randval,
                                                      args.nhead_stride_do,
                                                      args.nhead_stride_lsed,
                                                      args.nhead_stride_dq,
                                                      args.nhead_stride_dk,
                                                      args.nhead_stride_dv,
                                                      args.nhead_stride_dbias,
                                                      args.batch_stride_q,
                                                      args.batch_stride_k,
                                                      args.batch_stride_v,
                                                      args.batch_stride_bias,
                                                      args.batch_stride_randval,
                                                      args.batch_stride_do,
                                                      args.batch_stride_lsed,
                                                      args.batch_stride_dq,
                                                      args.batch_stride_dk,
                                                      args.batch_stride_dv,
                                                      args.batch_stride_dbias,
                                                      args.window_size_left,
                                                      args.window_size_right,
                                                      args.mask_type,
                                                      args.p_drop,
                                                      args.drop_seed_offset);
        }
    }();

    dim3 grids = FmhaBwdDQDKDVKernel::GridSize(args.batch, args.nhead_q, args.max_seqlen_k);
    return ck_tile::make_tuple(kargs, grids);
}

template <typename FmhaBwdOGradDotOKernel>
auto fmha_bwd_dot_do_o_create_kargs_and_grids(fmha_bwd_args args)
{
    auto kargs = [&] {
        // create group mode kernel arguments
        if constexpr(FmhaBwdOGradDotOKernel::kIsGroupMode)
        {
            return FmhaBwdOGradDotOKernel::MakeKargs(args.o_ptr,
                                                     args.do_ptr,
                                                     args.d_ptr,
                                                     args.lse_ptr,
                                                     args.sink_ptr,
                                                     args.d_sink_ptr,
                                                     args.p_undrop,
                                                     args.seqstart_q_ptr,
                                                     args.seqlen_q_ptr,
                                                     args.cu_seqlen_q_ptr,
                                                     args.hdim_v,
                                                     args.nhead_q,
                                                     args.stride_do,
                                                     args.stride_o,
                                                     args.nhead_stride_do,
                                                     args.nhead_stride_o,
                                                     args.nhead_stride_lsed);
        }
        else
        { // create batch mode kernel arguments
            return FmhaBwdOGradDotOKernel::MakeKargs(args.o_ptr,
                                                     args.do_ptr,
                                                     args.d_ptr,
                                                     args.lse_ptr,
                                                     args.sink_ptr,
                                                     args.d_sink_ptr,
                                                     args.p_undrop,
                                                     args.seqlen_q,
                                                     args.hdim_v,
                                                     args.nhead_q,
                                                     args.stride_do,
                                                     args.stride_o,
                                                     args.nhead_stride_do,
                                                     args.nhead_stride_o,
                                                     args.nhead_stride_lsed,
                                                     args.batch_stride_do,
                                                     args.batch_stride_o,
                                                     args.batch_stride_lsed);
        }
    }();

    dim3 grids = FmhaBwdOGradDotOKernel::GridSize(args.batch, args.nhead_q, args.max_seqlen_q);
    return ck_tile::make_tuple(kargs, grids);
}

template <typename FmhaBwdConvertQGradKernel>
auto fmha_bwd_convert_dq_create_kargs_and_grids(fmha_bwd_args args)
{
    auto kargs = [&] {
        // create group mode kernel arguments
        if constexpr(FmhaBwdConvertQGradKernel::kIsGroupMode)
        {
            return FmhaBwdConvertQGradKernel::MakeKargs(args.workspace_ptr,
                                                        args.dq_ptr,
                                                        args.batch,
                                                        args.nhead_q,
                                                        args.seqstart_q_ptr,
                                                        args.seqstart_k_ptr,
                                                        args.seqlen_q_ptr,
                                                        args.seqlen_k_ptr,
                                                        args.cu_seqlen_q_ptr,
                                                        args.cu_seqlen_k_ptr,
                                                        args.hdim_q,
                                                        args.stride_dq,
                                                        args.nhead_stride_dq);
        }
        else
        { // create batch mode kernel arguments
            return FmhaBwdConvertQGradKernel::MakeKargs(args.workspace_ptr,
                                                        args.dq_ptr,
                                                        args.batch,
                                                        args.nhead_q,
                                                        args.seqlen_q,
                                                        args.seqlen_k,
                                                        args.hdim_q,
                                                        args.stride_dq,
                                                        args.nhead_stride_dq,
                                                        args.batch_stride_dq);
        }
    }();

    dim3 grids = FmhaBwdConvertQGradKernel::GridSize(args.batch, args.nhead_q, args.max_seqlen_q);
    return ck_tile::make_tuple(kargs, grids);
}

// this is used to pattern-match internl kernel implementation, not to instantiate kernel
template <ck_tile::index_t HDim_,
          typename DataType_,
          bool kIsGroupMode_,
          typename FmhaMask_,
          typename FmhaDropout_,
          ck_tile::BlockAttentionBiasEnum BiasEnum_,
          bool kHasBiasGrad_,
          ck_tile::index_t kPadD_,
          ck_tile::index_t kPadDv_,
          bool kIsDeterministic_,
          bool kUseTrLoad_,
          ck_tile::index_t MaxSeqLenQ_,
          ck_tile::index_t kN0>
struct fmha_bwd_dq_dk_dv_traits_
{
};

template <typename Traits_, typename Arch = void>
float fmha_bwd_dq_dk_dv_(const ck_tile::stream_config&, fmha_bwd_args);

template <typename Traits_, typename Arch = void>
void fmha_bwd_dq_dk_dv_oneshot_(const ck_tile::stream_config&, fmha_bwd_args);

template <typename Traits_, typename Arch = void>
std::string fmha_bwd_dq_dk_dv_get_name_();
template <typename Traits_, typename Arch = void>
int fmha_bwd_dq_dk_dv_maxq_();
struct fmha_bwd_traits;
template <typename Traits_, typename Arch = void>
size_t fmha_bwd_dq_dk_dv_dq_ws_host_size_(int batch_size);
// `total_seqlen_q_padded` is total q tokens across all batches (incl. per-batch padding):
//   - batch mode: max_batch * seqlen_q
//   - group mode: seqstart_q[batch] (== varlen q tensor's first dim)
template <typename Traits_, typename Arch = void>
size_t fmha_bwd_dq_dk_dv_dq_ws_device_upper_bound_(ck_tile::index_t max_batch,
                                                   ck_tile::index_t hdim_q,
                                                   ck_tile::index_t nhead_q,
                                                   ck_tile::index_t total_seqlen_q_padded,
                                                   ck_tile::index_t max_seqlen_k);
template <typename Traits_, typename Arch = void>
size_t fmha_bwd_dq_dk_dv_dq_prepare_ws_host_(void* cpu_ws,
                                             ck_tile::index_t batch_size,
                                             ck_tile::index_t hdim_q,
                                             ck_tile::index_t nhead_q,
                                             ck_tile::index_t seqlen_q,
                                             ck_tile::index_t seqlen_k,
                                             const ck_tile::index_t* seqstart_qs,
                                             const ck_tile::index_t* seqstart_ks);
template <typename Traits_, typename Arch = void>
bool fmha_bwd_dq_dk_dv_needs_zero_dq_acc_();

template <ck_tile::index_t HDim_, typename DataType_, bool kIsGroupMode_, bool kPadS_, bool kPadDv_>
struct fmha_bwd_dot_do_o_traits_
{
    static constexpr ck_tile::index_t HDim = HDim_;
    using DataType                         = ck_tile::remove_cvref_t<DataType_>;
    static constexpr bool kIsGroupMode     = kIsGroupMode_;
    static constexpr bool kPadS            = kPadS_;
    static constexpr bool kPadDv           = kPadDv_;
};

template <typename Traits_, typename Arch = void>
float fmha_bwd_dot_do_o_(const ck_tile::stream_config&, fmha_bwd_args);

template <typename Traits_, typename Arch = void>
void fmha_bwd_dot_do_o_oneshot_(const ck_tile::stream_config&, fmha_bwd_args);

template <typename Traits_, typename Arch = void>
std::string fmha_bwd_dot_do_o_get_name_();

template <ck_tile::index_t HDim_,
          typename DataType_,
          bool kIsGroupMode_,
          bool kPadS_,
          bool kPadD_,
          bool kIsDeterministic_>
struct fmha_bwd_convert_dq_traits_
{
};

template <typename Traits_, typename Arch = void>
float fmha_bwd_convert_dq_(const ck_tile::stream_config&, fmha_bwd_args);

template <typename Traits_, typename Arch = void>
void fmha_bwd_convert_dq_oneshot_(const ck_tile::stream_config&, fmha_bwd_args);

template <typename Traits_, typename Arch = void>
std::string fmha_bwd_convert_dq_get_name_();

// Traits that are used to dispatch different kernel implementations for fmha backward
struct fmha_bwd_traits
{
    int seqlen_q;
    int seqlen_k;
    int batch;
    int max_seqlen_q;
    int max_seqlen_k;
    int hdim_q;
    int hdim_v;
    int nhead_q;
    int nhead_k;
    std::string data_type;
    bool is_group_mode;
    mask_enum mask_type;
    bias_enum bias_type; // 0:no bias, 1:elementwise bias, 2:alibi. sync with BlockAttentionBiasEnum
    bool has_dbias;
    bool has_dropout;
    bool is_store_randval;
    bool is_deterministic;
    // TODO: padding check is inside this api
};

template <typename T0 /*dot_do_o_trait*/,
          typename T1 /*dq_dk_dv_trait*/,
          typename T2 /*convert_dq_trait*/,
          typename Arch>
float fmha_bwd_(const ck_tile::stream_config& s, fmha_bwd_args a)
{
    if constexpr(!std::is_same_v<T2, void>)
    {
        if(s.log_level_ > 0)
            std::cout << ", " << fmha_bwd_dot_do_o_get_name_<T0, Arch>() << "@"
                      << fmha_bwd_convert_dq_get_name_<T2, Arch>() << "@"
                      << fmha_bwd_dq_dk_dv_get_name_<T1, Arch>() << std::flush;
        return ck_tile::launch_kernel(
            s,
            [=](const ck_tile::stream_config& s_) { fmha_bwd_dot_do_o_oneshot_<T0, Arch>(s_, a); },
            [=](const ck_tile::stream_config& s_) { fmha_bwd_dq_dk_dv_oneshot_<T1, Arch>(s_, a); },
            [=](const ck_tile::stream_config& s_) {
                fmha_bwd_convert_dq_oneshot_<T2, Arch>(s_, a);
            });
    }
    else
    {
        if(s.log_level_ > 0)
            std::cout << ", " << fmha_bwd_dot_do_o_get_name_<T0, Arch>() << "@"
                      << fmha_bwd_dq_dk_dv_get_name_<T1, Arch>() << std::flush;
        return ck_tile::launch_kernel(
            s,
            [=](const ck_tile::stream_config& s_) { fmha_bwd_dot_do_o_oneshot_<T0, Arch>(s_, a); },
            [=](const ck_tile::stream_config& s_) { fmha_bwd_dq_dk_dv_oneshot_<T1, Arch>(s_, a); });
    }
}

template <int Version = 2>
float fmha_bwd(const fmha_bwd_traits&, fmha_bwd_args, const ck_tile::stream_config&);

struct fmha_bwd_launcher
{
    std::function<float(fmha_bwd_args, const ck_tile::stream_config&)> run{
        [](fmha_bwd_args, const ck_tile::stream_config&) {
            std::cerr << "fmha_bwd: no kernel found for given traits, skipping run\n";
            return -1.0f;
        }};
    // Layout: [host_ws_size_ bytes (host-prepared metadata)][dq_acc region]
    size_t workspace_size = 0;

    fmha_bwd_launcher(const fmha_bwd_traits&);
    fmha_bwd_launcher(fmha_bwd_launcher&&)            = delete;
    fmha_bwd_launcher& operator=(fmha_bwd_launcher&&) = delete;

    ~fmha_bwd_launcher() noexcept { schedule_pin_staging_release(); }

    // Stream-async: zero dq_acc, D2H seqstart, host-pack metadata, H2D into device_ws.
    // `pinned_host_alloc` returns a shared_ptr to a pinned host buffer; its deleter
    // is invoked on the stream tail after the H2D completes.
    void prepare_workspace_async( //
        void* device_ws_ptr,
        const int* seqstart_q_dev,
        const int* seqstart_k_dev,
        const ck_tile::stream_config& s,
        const std::function<std::shared_ptr<void>(size_t)>& pinned_host_alloc)
    {
        hipStream_t stream = s.stream_id_;

        // Fast path: no host-side metadata to stage; just zero dq_acc if needed.
        if(host_ws_size_ == 0)
        {
            if(needs_zero_dq_acc_ && workspace_size > 0)
                HIP_CHECK_ERROR(hipMemsetAsync(device_ws_ptr, 0, workspace_size, stream));
            return;
        }

        if(!pinned_host_alloc)
            throw std::runtime_error(
                "fmha_bwd_launcher::prepare_workspace_async: pinned_host_alloc is required");

        // Allocate pinned host staging first: if it throws we haven't issued any
        // stream work yet, leaving the workspace cleanly un-prepared.
        const size_t seqstart_bytes = traits_.is_group_mode ? sizeof(int) * (traits_.batch + 1) : 0;
        const size_t total_bytes    = 2 * seqstart_bytes + host_ws_size_;
        auto pin_base               = pinned_host_alloc(total_bytes);

        if(needs_zero_dq_acc_ && workspace_size > host_ws_size_)
            HIP_CHECK_ERROR(hipMemsetAsync(static_cast<char*>(device_ws_ptr) + host_ws_size_,
                                           0,
                                           workspace_size - host_ws_size_,
                                           stream));

        char* base                   = static_cast<char*>(pin_base.get());
        int* pin_q                   = reinterpret_cast<int*>(base);
        int* pin_k                   = reinterpret_cast<int*>(base + seqstart_bytes);
        void* pin_w                  = base + 2 * seqstart_bytes;
        const int* seqstart_q_pinned = traits_.is_group_mode ? pin_q : nullptr;
        const int* seqstart_k_pinned = traits_.is_group_mode ? pin_k : nullptr;

        if(traits_.is_group_mode)
        {
            if(!seqstart_q_dev || !seqstart_k_dev)
                throw std::runtime_error("fmha_bwd_launcher::prepare_workspace_async: "
                                         "seqstart_q_dev and seqstart_k_dev are required in "
                                         "group mode");
            HIP_CHECK_ERROR(hipMemcpyAsync(
                pin_q, seqstart_q_dev, seqstart_bytes, hipMemcpyDeviceToHost, stream));
            HIP_CHECK_ERROR(hipMemcpyAsync(
                pin_k, seqstart_k_dev, seqstart_bytes, hipMemcpyDeviceToHost, stream));
        }

        auto pack_closure = std::make_unique<std::function<void()>>(
            [=, fn = pack_workspace_host_]() { fn(pin_w, seqstart_q_pinned, seqstart_k_pinned); });
        // Callback runs on the HIP driver helper thread across a C ABI boundary;
        // any exception escaping it would call std::terminate.
        HIP_CHECK_ERROR(hipLaunchHostFunc(
            stream,
            [](void* ud) {
                std::unique_ptr<std::function<void()>> c{static_cast<std::function<void()>*>(ud)};
                try
                {
                    (*c)();
                }
                catch(const std::exception& e)
                {
                    // The H2D queued after this callback will copy indeterminate
                    // metadata to device and the kernel will produce wrong results;
                    // unlikely in practice since pack_workspace_host_ only throws on
                    // precondition violations.
                    std::cerr << "fmha_bwd_launcher: pack_workspace_host threw: " << e.what()
                              << '\n';
                }
                catch(...)
                {
                    std::cerr << "fmha_bwd_launcher: pack_workspace_host threw unknown\n";
                }
            },
            pack_closure.get()));
        // Ownership transferred to the callback only after a successful launch.
        pack_closure.release();

        HIP_CHECK_ERROR(
            hipMemcpyAsync(device_ws_ptr, pin_w, host_ws_size_, hipMemcpyHostToDevice, stream));

        // Release any previous in-flight buffer before taking a new one.
        schedule_pin_staging_release();
        pin_staging_    = std::move(pin_base);
        release_stream_ = stream;
    }

    private:
    fmha_bwd_traits traits_{};
    size_t host_ws_size_    = 0;
    bool needs_zero_dq_acc_ = false;
    // Pure CPU; safe to invoke from a hipLaunchHostFunc callback.
    std::function<void(void* host_ws, const int* seqstart_q, const int* seqstart_k)>
        pack_workspace_host_{[](void*, const int*, const int*) {
            std::cerr
                << "fmha_bwd: no kernel found for given traits, skipping pack_workspace_host\n";
        }};
    std::shared_ptr<void> pin_staging_;
    hipStream_t release_stream_ = nullptr;

    // The pin_staging_ deleter MUST NOT call any HIP API: it fires from the
    // hipLaunchHostFunc callback on the driver helper thread, which holds
    // runtime locks (would deadlock against main-thread hipFree). PyTorch's
    // CachingHostAllocator is safe; bare hipHostMalloc users should defer
    // hipHostFree via ck_tile::pinned_host_releaser.
    void schedule_pin_staging_release() noexcept
    {
        if(!pin_staging_)
            return;
        auto* heap_ref       = new std::shared_ptr<void>(std::move(pin_staging_));
        const hipError_t err = hipLaunchHostFunc(
            release_stream_,
            [](void* ud) { delete static_cast<std::shared_ptr<void>*>(ud); },
            heap_ref);
        if(err != hipSuccess)
        {
            std::cerr << "fmha_bwd_launcher: hipLaunchHostFunc failed: " << hipGetErrorString(err)
                      << "; releasing eagerly\n";
            delete heap_ref;
        }
    }

    template <typename T0 /*dot_do_o_trait*/,
              typename T1 /*dq_dk_dv_trait*/,
              typename T2 /*convert_dq_trait*/,
              typename Arch>
    void init(const fmha_bwd_traits& t)
    {
        traits_ = t;
        run     = [](fmha_bwd_args a, const ck_tile::stream_config& s) {
            return fmha_bwd_<T0, T1, T2, Arch>(s, a);
        };
        host_ws_size_         = fmha_bwd_dq_dk_dv_dq_ws_host_size_<T1, Arch>(t.batch);
        size_t device_ws_size = 0;
        if(host_ws_size_ > 0)
        {
            // In group mode t.seqlen_q is already the padded total (== seqstart_q[batch]);
            // in batch mode it's per-batch and the total is batch * seqlen_q.
            const ck_tile::index_t total_seqlen_q_padded =
                t.is_group_mode ? t.seqlen_q : t.batch * t.seqlen_q;
            device_ws_size = fmha_bwd_dq_dk_dv_dq_ws_device_upper_bound_<T1, Arch>(
                t.batch, t.hdim_q, t.nhead_q, total_seqlen_q_padded, t.max_seqlen_k);
            pack_workspace_host_ = [batch    = t.batch,
                                    hdim_q   = t.hdim_q,
                                    nhead_q  = t.nhead_q,
                                    seqlen_q = t.seqlen_q,
                                    seqlen_k = t.seqlen_k //
            ](void* host_ws, const int* seqstart_q, const int* seqstart_k) {
                fmha_bwd_dq_dk_dv_dq_prepare_ws_host_<T1, Arch>(
                    host_ws, batch, hdim_q, nhead_q, seqlen_q, seqlen_k, seqstart_q, seqstart_k);
            };
        }
        workspace_size     = host_ws_size_ + device_ws_size;
        needs_zero_dq_acc_ = fmha_bwd_dq_dk_dv_needs_zero_dq_acc_<T1, Arch>();
    }

    public:
    template <typename... Args>
    float operator()(Args&&... args) const
    {
        return run(std::forward<Args>(args)...);
    }
};
