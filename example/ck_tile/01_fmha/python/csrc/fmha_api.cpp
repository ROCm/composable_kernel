// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "fmha_common.hpp"

namespace ck_tile::fmha::torch_api {

std::vector<at::Tensor>
mha_fwd(at::Tensor& q,       // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        const at::Tensor& k, // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        const at::Tensor& v, // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        std::optional<at::Tensor>&
            out_, // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        std::optional<at::Tensor>& alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        std::optional<at::Generator> gen_,
        std::optional<at::Tensor> bias_,
        std::optional<at::Tensor> q_descale_,
        std::optional<at::Tensor> k_descale_,
        std::optional<at::Tensor> v_descale_,
        std::string qscale,
        bool return_lse,
        int block_size_q,
        int block_size_kv);

std::vector<at::Tensor> mha_varlen_fwd(
    at::Tensor& q, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor&
        k, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x
           // page_block_size x num_heads_k x head_size if there's a block_table.
    const at::Tensor&
        v, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x
           // page_block_size x num_heads_k x head_size if there's a block_table.
    std::optional<at::Tensor>&
        out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor& cu_seqlens_q, // b+1
    const at::Tensor& cu_seqlens_k, // b+1
    std::optional<at::Tensor>&
        seqused_k, // b. If given, only this many elements of each batch element's keys are used.
    std::optional<const at::Tensor>& leftpad_k_, // batch_size
    std::optional<at::Tensor>& block_table_,     // batch_size x max_num_blocks_per_seq
    std::optional<at::Tensor>& alibi_slopes_,    // num_heads or b x num_heads
    int max_seqlen_q,
    const int max_seqlen_k,
    const float p_dropout,
    const float softmax_scale,
    const bool zero_tensors,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool return_softmax,
    std::optional<at::Generator> gen_,
    int num_splits,
    std::optional<at::Tensor> q_descale_,
    std::optional<at::Tensor> k_descale_,
    std::optional<at::Tensor> v_descale_,
    std::string qscale,
    bool return_lse);

std::vector<at::Tensor>
mha_bwd(const at::Tensor& dout, // batch_size x seqlen_q x num_heads, x multiple_of(head_size_og, 8)
        const at::Tensor& q,    // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor& k,    // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor& v,    // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor& out,  // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor& softmax_lse,            // b x h x seqlen_q
        std::optional<at::Tensor>& dq_,           // batch_size x seqlen_q x num_heads x head_size
        std::optional<at::Tensor>& dk_,           // batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor>& dv_,           // batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor>& alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,                    // probability to drop
        const float softmax_scale,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool deterministic,
        std::optional<at::Generator> gen_,
        std::optional<at::Tensor>& rng_state,
        std::optional<at::Tensor> bias_,
        std::optional<at::Tensor> dbias_);

std::vector<at::Tensor> mha_varlen_bwd(
    const at::Tensor& dout, // total_q x num_heads x head_size
    const at::Tensor& q,    // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor& k,    // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor& v,    // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor& out,  // total_q x num_heads x head_size
    const at::Tensor& softmax_lse, // b x h x s   softmax logsumexp
    std::optional<at::Tensor>&
        dq_, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    std::optional<at::Tensor>&
        dk_, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    std::optional<at::Tensor>&
        dv_, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor& cu_seqlens_q,           // b+1
    const at::Tensor& cu_seqlens_k,           // b+1
    std::optional<at::Tensor>& alibi_slopes_, // num_heads or b x num_heads
    const int max_seqlen_q,
    const int max_seqlen_k, // max sequence length to choose the kernel
    const float p_dropout,  // probability to drop
    const float softmax_scale,
    const bool zero_tensors,
    const bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool deterministic,
    std::optional<at::Generator> gen_,
    std::optional<at::Tensor>& rng_state);

std::vector<at::Tensor> mha_fwd_kvcache(
    at::Tensor& q,            // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor& kcache, // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x
                              // page_block_size x num_heads_k x head_size if there's a block_table.
    const at::Tensor& vcache, // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x
                              // page_block_size x num_heads_k x head_size if there's a block_table.
    std::optional<const at::Tensor>& k_, // batch_size x seqlen_knew x num_heads_k x head_size
    std::optional<const at::Tensor>& v_, // batch_size x seqlen_knew x num_heads_k x head_size
    std::optional<const at::Tensor>& seqlens_k_,       // batch_size
    std::optional<const at::Tensor>& rotary_cos_,      // seqlen_ro x (rotary_dim / 2)
    std::optional<const at::Tensor>& rotary_sin_,      // seqlen_ro x (rotary_dim / 2)
    std::optional<const at::Tensor>& cache_batch_idx_, // indices to index into the KV cache
    std::optional<const at::Tensor>& leftpad_k_,       // batch_size
    std::optional<at::Tensor>& block_table_,           // batch_size x max_num_blocks_per_seq
    std::optional<at::Tensor>& alibi_slopes_,          // num_heads or batch_size x num_heads
    std::optional<at::Tensor>& out_, // batch_size x seqlen_q x num_heads x head_size
    const float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    bool is_rotary_interleaved, // if true, rotary combines indices 0 & 1, else indices 0 &
                                // rotary_dim / 2
    int num_splits);

} // namespace ck_tile::fmha::torch_api

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "CK Tile FMHA PyTorch bindings";
    m.def("fwd",
          &ck_tile::fmha::torch_api::mha_fwd,
          "Forward pass",
          pybind11::arg("q"),
          pybind11::arg("k"),
          pybind11::arg("v"),
          pybind11::arg("out"),
          pybind11::arg("alibi_slopes"),
          pybind11::arg("dropout_p"),
          pybind11::arg("softmax_scale"),
          pybind11::arg("causal"),
          pybind11::arg("window_size_left"),
          pybind11::arg("window_size_right"),
          pybind11::arg("softcap"),
          pybind11::arg("return_softmax"),
          pybind11::arg("gen"),
          pybind11::arg("bias")          = pybind11::none(),
          pybind11::arg("q_descale")     = pybind11::none(),
          pybind11::arg("k_descale")     = pybind11::none(),
          pybind11::arg("v_descale")     = pybind11::none(),
          pybind11::arg("qscale")        = "no_scale",
          pybind11::arg("return_lse")    = true,
          pybind11::arg("block_size_q")  = 128,
          pybind11::arg("block_size_kv") = 128);
    m.def("varlen_fwd",
          &ck_tile::fmha::torch_api::mha_varlen_fwd,
          "Forward pass (variable length)",
          pybind11::arg("q"),
          pybind11::arg("k"),
          pybind11::arg("v"),
          pybind11::arg("out"),
          pybind11::arg("cu_seqlens_q"),
          pybind11::arg("cu_seqlens_k"),
          pybind11::arg("seqused_k"),
          pybind11::arg("leftpad_k"),
          pybind11::arg("block_table"),
          pybind11::arg("alibi_slopes"),
          pybind11::arg("max_seqlen_q"),
          pybind11::arg("max_seqlen_k"),
          pybind11::arg("dropout_p"),
          pybind11::arg("softmax_scale"),
          pybind11::arg("zero_tensors"),
          pybind11::arg("causal"),
          pybind11::arg("window_size_left"),
          pybind11::arg("window_size_right"),
          pybind11::arg("softcap"),
          pybind11::arg("return_softmax"),
          pybind11::arg("gen"),
          pybind11::arg("num_splits"),
          pybind11::arg("q_descale")  = pybind11::none(),
          pybind11::arg("k_descale")  = pybind11::none(),
          pybind11::arg("v_descale")  = pybind11::none(),
          pybind11::arg("qscale")     = "no_scale",
          pybind11::arg("return_lse") = true);
    m.def("bwd",
          &ck_tile::fmha::torch_api::mha_bwd,
          "Backward pass",
          pybind11::arg("dout"),
          pybind11::arg("q"),
          pybind11::arg("k"),
          pybind11::arg("v"),
          pybind11::arg("out"),
          pybind11::arg("softmax_lse"),
          pybind11::arg("dq"),
          pybind11::arg("dk"),
          pybind11::arg("dv"),
          pybind11::arg("alibi_slopes"),
          pybind11::arg("dropout_p"),
          pybind11::arg("softmax_scale"),
          pybind11::arg("causal"),
          pybind11::arg("window_size_left"),
          pybind11::arg("window_size_right"),
          pybind11::arg("softcap"),
          pybind11::arg("deterministic"),
          pybind11::arg("gen"),
          pybind11::arg("rng_state"),
          pybind11::arg("bias")  = pybind11::none(),
          pybind11::arg("dbias") = pybind11::none());
    m.def(
        "varlen_bwd", &ck_tile::fmha::torch_api::mha_varlen_bwd, "Backward pass (variable length)");
    m.def("fwd_kvcache", &ck_tile::fmha::torch_api::mha_fwd_kvcache, "Forward pass, with KV-cache");
}
