// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp"
#include "ck_tile/ops/fmha/block/variants.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_output.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_traits.hpp"
#include "ck_tile/ops/fmha/kernel/fmha_fwd_workgroup_mapping.hpp"

#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#define CK_TILE_FMHA_HANDLE_XOR_LENGTH_FOLD 0

#if !defined(CK_TILE_FMHA_FORCE_HEAD_MAJOR)
#if defined(__HIP_DEVICE_COMPILE__) && \
    (defined(__gfx11__) || defined(__gfx12__) || defined(__gfx117__))
#define CK_TILE_FMHA_FORCE_HEAD_MAJOR 1
#else
#define CK_TILE_FMHA_FORCE_HEAD_MAJOR 0
#endif
#endif

// S[seqlen_q, seqlen_k] = Q[seqlen_q, hdim_q] @ K[seqlen_k, hdim_q]
// S'[seqlen_q, seqlen_k] = S[seqlen_q, seqlen_k] * Scale[1]
// S''[seqlen_q, seqlen_k] = S'[seqlen_q, seqlen_k] + Bias[seqlen_q, seqlen_k]
// P[seqlen_q, seqlen_k] = Softmax(S''[seqlen_q, seqlen_k])
// O[seqlen_q, hdim_v] = P[seqlen_q, seqlen_k] @ V^T[hdim_v, seqlen_k]

namespace ck_tile {

namespace detail {

template <typename Pipeline, typename = void>
struct uses_qr_tdm_lds_arena : std::false_type
{
};

template <typename Pipeline>
struct uses_qr_tdm_lds_arena<Pipeline, std::void_t<decltype(Pipeline::kUsesLdsArena)>>
    : std::bool_constant<Pipeline::kUsesLdsArena>
{
};

template <typename Pipeline>
inline constexpr bool uses_qr_tdm_lds_arena_v = uses_qr_tdm_lds_arena<Pipeline>::value;

// Whether the K/V views take the NT_HT cache hint on gfx1250 (see
// FmhaFwdKernel::kKVCoherence): TDM pipelines that stage K/V in their LDS arena,
// fp16/bf16, head dims below 256, tiles of at most 64 rows.
template <typename Pipeline>
inline constexpr bool uses_kv_nt_ht_hint_v =
    uses_qr_tdm_lds_arena_v<Pipeline> && Pipeline::kM0 <= 64 &&
    Pipeline::BlockFmhaShape::kQKHeaddim < 256 &&
    (std::is_same_v<typename Pipeline::KDataType, half_t> ||
     std::is_same_v<typename Pipeline::KDataType, bf16_t>);

// A helper struct for detecting n0loop
template <typename T, typename = void>
struct has_n0loop_flag : std::false_type
{
};

template <typename T>
struct has_n0loop_flag<
    T,
    std::enable_if_t<std::is_convertible_v<decltype(T::kUseN0Loop), bool> && T::kUseN0Loop>>
    : std::true_type
{
};

template <typename T>
static inline constexpr bool is_n0loop_pipeline_v = has_n0loop_flag<T>::value;

// A helper struct for detecting ignore_fast_exp2 flag
template <typename T, typename = void>
struct has_ignore_fast_exp2_flag : std::false_type
{
};

// IgnoreFastExp2 is used by some pipeline which explicitly chooses not to use FAST_EXP2;
// By detecting the kIgnoreFastExp2 from the pipeline, the kernel's MakeKargsImpl() interface
// is able to avoid passing an in-correct scale_s parameter to the kernel layer
template <typename T>
struct has_ignore_fast_exp2_flag<
    T,
    std::enable_if_t<std::is_convertible_v<decltype(T::kIgnoreFastExp2), bool> &&
                     T::kIgnoreFastExp2>> : std::true_type
{
};

template <typename T>
static inline constexpr bool ignore_fast_exp2_v = has_ignore_fast_exp2_flag<T>::value;

// A helper struct for detecting naive_hdim_load, naive_hdim_load means load tiles of
// hdim96/hdim160/hdim192 without padding the tensor_view/tile_window to hdim128/hdim256
// naive_hdim_load is current supported by the qr_ks_vs_whole_k_prefetch_pipeline
template <typename T, typename = void>
struct has_naive_hdim_load_flag : std::false_type
{
};

template <typename T>
struct has_naive_hdim_load_flag<
    T,
    std::enable_if_t<std::is_convertible_v<decltype(T::kIsNaiveHDimLoad), bool> &&
                     T::kIsNaiveHDimLoad>> : std::true_type
{
};

template <typename T>
static inline constexpr bool is_naive_hdim_load_v = has_naive_hdim_load_flag<T>::value;

// A helper struct for detecting kUseTrLoad
template <typename T, typename = void>
struct has_use_trload_flag : std::false_type
{
};

template <typename T>
struct has_use_trload_flag<
    T,
    std::enable_if_t<std::is_convertible_v<decltype(T::kUseTrLoad), bool> && T::kUseTrLoad>>
    : std::true_type
{
};

template <typename T>
static inline constexpr bool is_using_trload_v = has_use_trload_flag<T>::value;

template <typename T>
using has_untransposed_v_kernel_path = decltype(T::kUsesUntransposedVKernelPath);

template <typename T>
using has_tdm_affine_dram_path = decltype(T::kUsesTdmAffineDramPath);

template <typename T>
using has_fixed_segmented_lds_arena = decltype(T::kUsesFixedSegmentedLdsArena);

// qr_tdm used these kernel paths before opt-in markers were introduced.
template <typename T>
static inline constexpr bool uses_untransposed_v_kernel_path_v = []() {
    if constexpr(is_detected<has_untransposed_v_kernel_path, T>::value)
        return static_cast<bool>(T::kUsesUntransposedVKernelPath);
    else
        return std::string_view{T::name} == "qr_tdm";
}();

template <typename T>
static inline constexpr bool uses_tdm_affine_dram_path_v = []() {
    if constexpr(is_detected<has_tdm_affine_dram_path, T>::value)
        return static_cast<bool>(T::kUsesTdmAffineDramPath);
    else
        return std::string_view{T::name} == "qr_tdm";
}();

template <typename T>
using has_tdm_output_store = decltype(T::kStoresOutputViaTdm);

template <typename T>
static inline constexpr bool stores_output_via_tdm_v = []() {
    if constexpr(is_detected<has_tdm_output_store, T>::value)
        return static_cast<bool>(T::kStoresOutputViaTdm);
    else
        return false;
}();

template <typename T>
static inline constexpr bool uses_fixed_segmented_lds_arena_v = []() {
    if constexpr(is_detected<has_fixed_segmented_lds_arena, T>::value)
        return static_cast<bool>(T::kUsesFixedSegmentedLdsArena);
    else
        return false;
}();

template <typename T>
using has_global_causal_order = decltype(T::kUsesGlobalCausalOrder);

template <typename T>
static inline constexpr bool uses_global_causal_order_v = []() {
    if constexpr(is_detected<has_global_causal_order, T>::value)
        return static_cast<bool>(T::kUsesGlobalCausalOrder);
    else
        return false;
}();

// Whether an epilogue takes the kernel's LDS as a 4th operator() argument
// (kUsesSmem). An arity probe would also match Default2DEpilogue's defaulted
// `void*` parameter, and GetSmemSize() != 0 would match DynamicQuantEpilogue,
// which reserves its own LDS.
template <typename Epilogue, typename = void>
struct epilogue_uses_smem : std::false_type
{
};

template <typename Epilogue>
struct epilogue_uses_smem<Epilogue, std::void_t<decltype(Epilogue::kUsesSmem)>>
    : std::bool_constant<Epilogue::kUsesSmem>
{
};

template <typename Epilogue>
inline constexpr bool epilogue_uses_smem_v = epilogue_uses_smem<Epilogue>::value;

} // namespace detail

template <typename FmhaPipeline_, typename EpiloguePipeline_>
struct FmhaFwdKernel
{
    using FmhaPipeline                           = ck_tile::remove_cvref_t<FmhaPipeline_>;
    using EpiloguePipeline                       = ck_tile::remove_cvref_t<EpiloguePipeline_>;
    static constexpr ck_tile::index_t kBlockSize = FmhaPipeline::kBlockSize;

    template <typename T>
    using has_hdim_tail_args = decltype(T::kUseHdimTailArgs);

    static constexpr ck_tile::index_t kBlockPerCu = FmhaPipeline::kBlockPerCu;
    static_assert(kBlockPerCu > 0);
    static constexpr ck_tile::index_t kBlockPerCuInput = FmhaPipeline::Problem::kBlockPerCu;
    // Query heads of one KV head folded into the query rows (row = g * seqlen_q + s).
    static constexpr bool kMergeNumHeadGroupsSeqLenQ =
        FmhaPipeline::Problem::kMergeNumHeadGroupsSeqLenQ;

    using QDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::QDataType>;
    using KDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::KDataType>;
    using VDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::VDataType>;
    using PDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::PDataType>;
    using BiasDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::BiasDataType>;
    using RandValOutputDataType =
        ck_tile::remove_cvref_t<typename FmhaPipeline::RandValOutputDataType>;
    using LSEDataType  = ck_tile::remove_cvref_t<typename FmhaPipeline::LSEDataType>;
    using ODataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::ODataType>;
    using SaccDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::SaccDataType>;

    using VLayout = ck_tile::remove_cvref_t<typename FmhaPipeline::VLayout>;

    static constexpr bool kIsGroupMode      = FmhaPipeline::kIsGroupMode;
    static constexpr bool kPadSeqLenQ       = FmhaPipeline::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK       = FmhaPipeline::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ      = FmhaPipeline::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV      = FmhaPipeline::kPadHeadDimV;
    static constexpr bool kHasLogitsSoftCap = FmhaPipeline::kHasLogitsSoftCap;
    static constexpr auto BiasEnum          = FmhaPipeline::BiasEnum;
    static constexpr bool kStoreLSE         = FmhaPipeline::kStoreLSE;
    static constexpr bool kHasDropout       = FmhaPipeline::kHasDropout;
    static constexpr auto QScaleEnum        = FmhaPipeline::Problem::QScaleEnum;
    static constexpr bool kSkipMinSeqlenQ   = FmhaPipeline::Problem::kSkipMinSeqlenQ;
    static constexpr bool kHasSink          = FmhaPipeline::kHasSink;
    static constexpr bool kUsesUntransposedVKernelPath =
        detail::uses_untransposed_v_kernel_path_v<FmhaPipeline>;
    static constexpr bool kUsesTdmAffineDramPath =
        detail::uses_tdm_affine_dram_path_v<FmhaPipeline>;
    static constexpr bool kUsesFixedSegmentedLdsArena =
        detail::uses_fixed_segmented_lds_arena_v<FmhaPipeline>;

    static constexpr std::string_view kPipelineName = FmhaPipeline::name;

    // Pipelines that take the descale arguments; GetSoftmaxScale folds them as for qr_tdm.
    static constexpr bool kTakesTdmDescales =
        kPipelineName == "qr_tdm" || kPipelineName == "qr_tdm_fp8_sched";

    static_assert(kTakesTdmDescales || QScaleEnum != BlockAttentionQuantScaleEnum::PERHEAD,
                  "perhead scale is only supported on qr_tdm and qr_tdm_fp8_sched");
    static_assert(!kMergeNumHeadGroupsSeqLenQ || kPipelineName == "qr_tdm_fp8_sched",
                  "merged query heads are implemented for qr_tdm_fp8_sched only");

    using AttentionVariant = ck_tile::remove_cvref_t<typename FmhaPipeline::AttentionVariant>;
    using FmhaMask         = ck_tile::remove_cvref_t<typename FmhaPipeline::FmhaMask>;
    static constexpr bool kHasMask = FmhaMask::IsMasking;

    // Cache hint for the K/V tensor views. qr_tdm moves K/V straight from HBM into
    // LDS through the TDM and never reads them back through the vector path, so
    // allocating them in the near cache only evicts other data. NT_HT drops the
    // near-cache allocation and keeps K/V high priority in the far cache, where
    // the m-tiles of one (batch, head) share them. Only the bm0 <= 64 tiles take
    // it: the bm0=128 tile of long sequences relies on K/V reuse in the near cache
    // and is slower with the hint. Measured on fp16/bf16 below d256 only (d256 is
    // neutral, fp8 unmeasured); see detail::uses_kv_nt_ht_hint_v.
#if defined(__gfx125__)
    static constexpr auto kKVCoherence = detail::uses_kv_nt_ht_hint_v<FmhaPipeline>
                                             ? amd_buffer_coherence_enum::NT_HT
                                             : amd_buffer_coherence_enum::coherence_default;
#else
    static constexpr auto kKVCoherence = amd_buffer_coherence_enum::coherence_default;
#endif

    static constexpr bool kUseAsyncCopy = FmhaPipeline::Policy::AsyncCopy;
    static constexpr bool kUseTrLoad    = detail::is_using_trload_v<FmhaPipeline>;
    static constexpr bool kIsTdmSchedD128 =
        kPipelineName == "qr_tdm_sched" && FmhaPipeline::BlockFmhaShape::kQKHeaddim == 128;
    static constexpr bool kUseAsyncLdsOStore =
        kIsTdmSchedD128 && kUsesFixedSegmentedLdsArena && sizeof(ODataType) == 2;
    static constexpr index_t kAsyncOStoreBytes          = 16;
    static constexpr index_t kAsyncOStoreVec            = kAsyncOStoreBytes / sizeof(ODataType);
    static constexpr uint64_t kAsyncOTileSpanLimitBytes = uint64_t{1} << 32;
    // Match qr_tdm plus the validated sched D64/D128 variants. Sched D192 keeps
    // the existing head-major remap. GetTileIndex additionally requires batch
    // BHSD, square causal attention, one V tile, and no dropout for this order.
    static constexpr bool kUseBatchLocalCausalOrder =
        detail::uses_qr_tdm_lds_arena_v<FmhaPipeline> || kIsTdmSchedD128 ||
        (kPipelineName == "qr_tdm_sched" && FmhaPipeline::BlockFmhaShape::kQKHeaddim == 64);

    // This family was measured without LSE output. Keep LSE-writing variants
    // on their existing mapping until their performance is validated separately.
    // GetTileIndex applies architecture and runtime layout/length/topology gates per workgroup.
    static constexpr bool kSupportsXcdCausalOrder =
        kHasMask && !kIsGroupMode && kIsTdmSchedD128 && FmhaPipeline::kM0 == 128 &&
        FmhaPipeline::kN0 == 128 && std::is_same_v<QDataType, ck_tile::bf16_t> &&
        std::is_same_v<KDataType, ck_tile::bf16_t> && std::is_same_v<VDataType, ck_tile::bf16_t> &&
        !kStoreLSE && !kHasDropout && !kHasSink && !kHasLogitsSoftCap &&
        BiasEnum == BlockAttentionBiasEnum::NO_BIAS &&
        QScaleEnum == BlockAttentionQuantScaleEnum::NO_SCALE;

#if defined(__gfx950__)
    static constexpr bool kIsAvailable = true;
#else
    static constexpr bool kIsAvailable = !kUseTrLoad;
#endif

    template <ck_tile::index_t I> // to avoid duplicated base class prblem, introduce an template
                                  // arg
    struct FmhaFwdEmptyKargs
    {
    };

    // kargs use aggregate initializer, so no constructor will provided
    // use inheritance to minimize karg size
    // user need to use MakeKargs() function to create kargs.
    struct FmhaFwdCommonKargs
    {
        const void* q_ptr;
        const void* k_ptr;
        const void* v_ptr;
        void* o_ptr;
        const void* sink_ptr;

        ck_tile::index_t seqlen_q;
        ck_tile::index_t seqlen_k;
        ck_tile::index_t hdim_q;
        ck_tile::index_t hdim_v;

        ck_tile::index_t num_head_q;
        // for MQA/GQA, nhead could be different. This parameter is nhead_q / nhead_k
        // if this param is larger than 1, indicate MQA/GQA case
        ck_tile::index_t nhead_ratio_qk;
        float scale_s;

        ck_tile::index_t stride_q;
        ck_tile::index_t stride_k;
        ck_tile::index_t stride_v;
        ck_tile::index_t stride_o;

        ck_tile::index_t nhead_stride_q;
        ck_tile::index_t nhead_stride_k;
        ck_tile::index_t nhead_stride_v;
        ck_tile::index_t nhead_stride_o;

        // Optional global head count and head offset (for grouped launches & RNG correctness)
        ck_tile::index_t num_head_q_total = 0;
        ck_tile::index_t head_start       = 0;
    };

    struct FmhaFwdLogitsSoftCapKargs
    {
        FmhaFwdLogitsSoftCapKargs() = default;

        void init_logits_soft_cap(float logits_soft_cap_)
        {
            if(0 < logits_soft_cap_)
            {
                logits_soft_cap     = logits_soft_cap_;
                logits_soft_cap_rcp = 1.f / logits_soft_cap;
            }
            else
            {
                logits_soft_cap     = 0.f;
                logits_soft_cap_rcp = 0.f;
            }
        }

        float logits_soft_cap;
        float logits_soft_cap_rcp;
    };

    struct FmhaFwdCommonBiasKargs
    {
        const void* bias_ptr               = nullptr;
        ck_tile::index_t stride_bias       = 0;
        ck_tile::index_t nhead_stride_bias = 0;
    };

    struct FmhaFwdBatchModeBiasKargs : FmhaFwdCommonBiasKargs
    {
        ck_tile::index_t batch_stride_bias = 0;
    };

    struct FmhaFwdAlibiKargs
    {
        // alibi is batch*nhead*1, no matter in batch/group mode, they are the same
        const void* alibi_slope_ptr;
        ck_tile::index_t alibi_slope_stride; // stride in batch, or 0 for all batch share same slope
    };

    struct FmhaFwdMaskKargs
    {
        // ck_tile::index_t window_size_left, window_size_right;
        ck_tile::index_t window_size_left, window_size_right, sink_size;
        ck_tile::GenericAttentionMaskEnum mask_type;
    };

    struct FmhaFwdCommonQScaleKargs
    {
        const void* q_descale_ptr = nullptr;
        const void* k_descale_ptr = nullptr;
        const void* v_descale_ptr = nullptr;
    };

    struct FmhaFwdCommonPerHeadKargs : public FmhaFwdCommonQScaleKargs
    {
        ck_tile::index_t nhead_stride_q_descale;
        ck_tile::index_t nhead_stride_k_descale;
        ck_tile::index_t nhead_stride_v_descale;

        ck_tile::index_t batch_stride_q_descale;
        ck_tile::index_t batch_stride_k_descale;
        ck_tile::index_t batch_stride_v_descale;
    };

    struct FmhaFwdCommonBlockScaleKargs : public FmhaFwdCommonQScaleKargs
    {
        ck_tile::index_t nhead_stride_q_descale;
        ck_tile::index_t nhead_stride_k_descale;
        ck_tile::index_t nhead_stride_v_descale;

        ck_tile::index_t block_scale_size_q;
        ck_tile::index_t block_scale_size_kv;
    };

    struct FmhaFwdBatchBlockScaleKargs : public FmhaFwdCommonBlockScaleKargs
    {
        ck_tile::index_t batch_stride_q_descale;
        ck_tile::index_t batch_stride_k_descale;
        ck_tile::index_t batch_stride_v_descale;
    };

    struct FmhaFwdGroupBlockScaleKargs : public FmhaFwdCommonBlockScaleKargs
    {
        const int32_t* block_scale_seqstart_q_ptr;
        const int32_t* block_scale_seqstart_k_ptr;
    };

    struct FmhaFwdCommonMXKargs : FmhaFwdCommonQScaleKargs
    {
        ck_tile::index_t stride_q_descale;
        ck_tile::index_t stride_k_descale;
        ck_tile::index_t stride_v_descale;

        ck_tile::index_t nhead_stride_q_descale;
        ck_tile::index_t nhead_stride_k_descale;
        ck_tile::index_t nhead_stride_v_descale;
    };

    struct FmhaFwdBatchMXKargs : FmhaFwdCommonMXKargs
    {
        ck_tile::index_t batch_stride_q_descale;
        ck_tile::index_t batch_stride_k_descale;
        ck_tile::index_t batch_stride_v_descale;
    };

    struct FmhaFwdGroupMXKargs : FmhaFwdCommonMXKargs
    {
        const int32_t* seqstart_v_scale_ptr;
    };

    struct FmhaFwdCommonLSEKargs
    {
        void* lse_ptr                     = nullptr;
        ck_tile::index_t nhead_stride_lse = 0;
        ck_tile::index_t batch_stride_lse = 0;
    };

    struct FmhaFwdDropoutSeedOffset
    {
        template <typename T>
        union ValueOrPointer
        {
            T val;
            const T* ptr;
        };

        ValueOrPointer<uint64_t> drop_seed;
        ValueOrPointer<uint64_t> drop_offset;
        bool is_drop_seed_offset_from_host;
    };

    struct FmhaFwdCommonDropoutKargs : FmhaFwdDropoutSeedOffset
    {
        void init_dropout(float p_drop, uint64_t seed, uint64_t offset)
        {
            float p_undrop = 1.0 - p_drop;
            p_undrop_in_uint8_t =
                uint8_t(std::floor(p_undrop * std::numeric_limits<uint8_t>::max()));
            rp_undrop = 1.0 / p_undrop;

            this->drop_seed.val                 = seed;
            this->drop_offset.val               = offset;
            this->is_drop_seed_offset_from_host = true;
        }

        void init_dropout(float p_drop, const uint64_t* seed_ptr, const uint64_t* offset_ptr)
        {
            float p_undrop = 1.0 - p_drop;
            p_undrop_in_uint8_t =
                uint8_t(std::floor(p_undrop * std::numeric_limits<uint8_t>::max()));
            rp_undrop = 1.0 / p_undrop;

            this->drop_seed.ptr                 = seed_ptr;
            this->drop_offset.ptr               = offset_ptr;
            this->is_drop_seed_offset_from_host = false;
        }

        float rp_undrop             = 1;
        uint8_t p_undrop_in_uint8_t = std::numeric_limits<uint8_t>::max();
        bool is_store_randval       = false;
        void* rand_val_ptr          = nullptr;

        ck_tile::index_t stride_randval       = 0;
        ck_tile::index_t nhead_stride_randval = 0;
    };

    struct FmhaFwdBatchModeDropoutKargs : FmhaFwdCommonDropoutKargs
    {
        ck_tile::index_t batch_stride_randval = 0;
    };

    struct FmhaFwdSkipMinSeqlenQKargs
    {
        ck_tile::index_t min_seqlen_q = 0;
    };

    struct FmhaFwdBatchModeKargs
        : FmhaFwdCommonKargs,
          std::conditional_t<BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS,
                             FmhaFwdBatchModeBiasKargs,
                             std::conditional_t<BiasEnum == BlockAttentionBiasEnum::ALIBI,
                                                FmhaFwdAlibiKargs,
                                                FmhaFwdEmptyKargs<0>>>,
          std::conditional_t<kHasMask, FmhaFwdMaskKargs, FmhaFwdEmptyKargs<1>>,
          std::conditional_t<kStoreLSE, FmhaFwdCommonLSEKargs, FmhaFwdEmptyKargs<2>>,
          std::conditional_t<
              QScaleEnum == BlockAttentionQuantScaleEnum::PERTENSOR,
              FmhaFwdCommonQScaleKargs,
              std::conditional_t<
                  QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE,
                  FmhaFwdBatchBlockScaleKargs,
                  std::conditional_t<
                      QScaleEnum == BlockAttentionQuantScaleEnum::MX,
                      FmhaFwdBatchMXKargs,
                      std::conditional_t<QScaleEnum == BlockAttentionQuantScaleEnum::PERHEAD,
                                         FmhaFwdCommonPerHeadKargs,
                                         FmhaFwdEmptyKargs<3>>>>>,
          std::conditional_t<kHasDropout, FmhaFwdBatchModeDropoutKargs, FmhaFwdEmptyKargs<4>>,
          std::conditional_t<kHasLogitsSoftCap, FmhaFwdLogitsSoftCapKargs, FmhaFwdEmptyKargs<5>>
    {
        ck_tile::index_t batch_stride_q;
        ck_tile::index_t batch_stride_k;
        ck_tile::index_t batch_stride_v;
        ck_tile::index_t batch_stride_o;

        // Optional cumulative sequence length pointers for batch mode
        // If provided, they override seqlen_q / seqlen_k per-batch to skip tail padding.
        const int32_t* cu_seqlen_q_ptr = nullptr; // cumulative, length without PAD
        const int32_t* cu_seqlen_k_ptr = nullptr; // cumulative, length without PAD
    };

    struct FmhaFwdGroupModeKargs
        : FmhaFwdCommonKargs,
          std::conditional_t<BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS,
                             FmhaFwdCommonBiasKargs,
                             std::conditional_t<BiasEnum == BlockAttentionBiasEnum::ALIBI,
                                                FmhaFwdAlibiKargs,
                                                FmhaFwdEmptyKargs<0>>>,
          std::conditional_t<kHasMask, FmhaFwdMaskKargs, FmhaFwdEmptyKargs<1>>,
          std::conditional_t<kStoreLSE, FmhaFwdCommonLSEKargs, FmhaFwdEmptyKargs<2>>,
          std::conditional_t<
              QScaleEnum == BlockAttentionQuantScaleEnum::PERTENSOR,
              FmhaFwdCommonQScaleKargs,
              std::conditional_t<
                  QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE,
                  FmhaFwdGroupBlockScaleKargs,
                  std::conditional_t<
                      QScaleEnum == BlockAttentionQuantScaleEnum::MX,
                      FmhaFwdGroupMXKargs,
                      std::conditional_t<QScaleEnum == BlockAttentionQuantScaleEnum::PERHEAD,
                                         FmhaFwdCommonPerHeadKargs,
                                         FmhaFwdEmptyKargs<3>>>>>,
          std::conditional_t<kHasDropout, FmhaFwdCommonDropoutKargs, FmhaFwdEmptyKargs<4>>,
          std::conditional_t<kHasLogitsSoftCap, FmhaFwdLogitsSoftCapKargs, FmhaFwdEmptyKargs<5>>,
          std::conditional_t<kSkipMinSeqlenQ, FmhaFwdSkipMinSeqlenQKargs, FmhaFwdEmptyKargs<6>>
    {
        const int32_t* seqstart_q_ptr;
        const int32_t* seqstart_k_ptr;
        const int32_t* seqlen_q_ptr;
        const int32_t* seqlen_k_ptr;

        // Optional per-sequence and cumulative logical (excluding padding) sequence length arrays
        const int32_t* cu_seqlen_q_ptr = nullptr;
        const int32_t* cu_seqlen_k_ptr = nullptr;
    };

    using Kargs = std::conditional_t<kIsGroupMode, FmhaFwdGroupModeKargs, FmhaFwdBatchModeKargs>;

    struct BlockIndices
    {
        ck_tile::index_t batch_idx;
        ck_tile::index_t qo_head_idx;
        ck_tile::index_t kv_head_idx;
    };

    template <bool Cond = !kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargsImpl(const void* q_ptr,
                  const void* k_ptr,
                  const void* v_ptr,
                  const void* bias_ptr,
                  const void* q_descale_ptr,
                  const void* k_descale_ptr,
                  const void* v_descale_ptr,
                  void* rand_val_ptr,
                  void* lse_ptr,
                  void* o_ptr,
                  ck_tile::index_t seqlen_q,
                  ck_tile::index_t seqlen_k,
                  ck_tile::index_t hdim_q,
                  ck_tile::index_t hdim_v,
                  ck_tile::index_t num_head_q,
                  ck_tile::index_t nhead_ratio_qk,
                  float scale_s,
                  float logits_soft_cap,
                  ck_tile::index_t stride_q,
                  ck_tile::index_t stride_k,
                  ck_tile::index_t stride_v,
                  ck_tile::index_t stride_bias,
                  ck_tile::index_t stride_randval,
                  ck_tile::index_t stride_o,
                  ck_tile::index_t stride_q_descale,
                  ck_tile::index_t stride_k_descale,
                  ck_tile::index_t stride_v_descale,
                  ck_tile::index_t nhead_stride_q,
                  ck_tile::index_t nhead_stride_k,
                  ck_tile::index_t nhead_stride_v,
                  ck_tile::index_t nhead_stride_bias,
                  ck_tile::index_t nhead_stride_randval,
                  ck_tile::index_t nhead_stride_lse,
                  ck_tile::index_t nhead_stride_o,
                  ck_tile::index_t nhead_stride_q_descale,
                  ck_tile::index_t nhead_stride_k_descale,
                  ck_tile::index_t nhead_stride_v_descale,
                  ck_tile::index_t batch_stride_q,
                  ck_tile::index_t batch_stride_k,
                  ck_tile::index_t batch_stride_v,
                  ck_tile::index_t batch_stride_bias,
                  ck_tile::index_t batch_stride_randval,
                  ck_tile::index_t batch_stride_lse,
                  ck_tile::index_t batch_stride_o,
                  ck_tile::index_t batch_stride_q_descale,
                  ck_tile::index_t batch_stride_k_descale,
                  ck_tile::index_t batch_stride_v_descale,
                  ck_tile::index_t window_size_left,
                  ck_tile::index_t window_size_right,
                  ck_tile::index_t sink_size,
                  ck_tile::index_t mask_type,
                  float p_drop,
                  bool s_randval,
                  std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>>
                      drop_seed_offset,
                  ck_tile::index_t block_scale_size_q,
                  ck_tile::index_t block_scale_size_kv,
                  const void* cu_seqlen_q_ptr       = nullptr,
                  const void* cu_seqlen_k_ptr       = nullptr,
                  const void* sink_ptr              = nullptr,
                  ck_tile::index_t num_head_q_total = 0,
                  ck_tile::index_t head_start       = 0)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     o_ptr,
                     sink_ptr,
                     seqlen_q,
                     seqlen_k,
                     hdim_q,
                     hdim_v,
                     num_head_q,
                     nhead_ratio_qk,
#if CK_TILE_FMHA_FWD_FAST_EXP2
                     detail::ignore_fast_exp2_v<FmhaPipeline>
                         ? scale_s
                         : static_cast<float>(scale_s * ck_tile::log2e_v<>),
#else
                     scale_s,
#endif
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_o,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_o}, // args for common karg
                    {},               // placeholder for bias
                    {},               // placeholder for mask
                    {},               // placeholder for lse
                    {},               // placeholder for qscale
                    {},               // placeholder for dropout
                    {},               // placeholder for logits_soft_cap
                    batch_stride_q,
                    batch_stride_k,
                    batch_stride_v,
                    batch_stride_o};
        kargs.num_head_q_total = num_head_q_total;
        kargs.head_start       = head_start;

        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
        {
            kargs.bias_ptr          = bias_ptr;
            kargs.stride_bias       = stride_bias;
            kargs.nhead_stride_bias = nhead_stride_bias;
            kargs.batch_stride_bias = batch_stride_bias;
        }
        else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
        {
            kargs.alibi_slope_ptr    = bias_ptr;
            kargs.alibi_slope_stride = stride_bias;
        }
        if constexpr(kHasMask)
        {
            kargs.window_size_left  = window_size_left;
            kargs.window_size_right = window_size_right;
            kargs.sink_size         = sink_size;
            kargs.mask_type         = static_cast<ck_tile::GenericAttentionMaskEnum>(mask_type);
        }
        if constexpr(kStoreLSE)
        {
            kargs.lse_ptr          = lse_ptr;
            kargs.nhead_stride_lse = nhead_stride_lse;
            kargs.batch_stride_lse = batch_stride_lse;
        }
        if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::PERTENSOR)
        {
            kargs.q_descale_ptr = q_descale_ptr;
            kargs.k_descale_ptr = k_descale_ptr;
            kargs.v_descale_ptr = v_descale_ptr;
        }
        else if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE)
        {
            kargs.q_descale_ptr = q_descale_ptr;
            kargs.k_descale_ptr = k_descale_ptr;
            kargs.v_descale_ptr = v_descale_ptr;

            kargs.nhead_stride_q_descale = nhead_stride_q_descale;
            kargs.nhead_stride_k_descale = nhead_stride_k_descale;
            kargs.nhead_stride_v_descale = nhead_stride_v_descale;

            kargs.batch_stride_q_descale = batch_stride_q_descale;
            kargs.batch_stride_k_descale = batch_stride_k_descale;
            kargs.batch_stride_v_descale = batch_stride_v_descale;

            kargs.block_scale_size_q  = block_scale_size_q;
            kargs.block_scale_size_kv = block_scale_size_kv;
        }
        else if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::MX)
        {
            kargs.q_descale_ptr = q_descale_ptr;
            kargs.k_descale_ptr = k_descale_ptr;
            kargs.v_descale_ptr = v_descale_ptr;

            kargs.stride_q_descale = stride_q_descale;
            kargs.stride_k_descale = stride_k_descale;
            kargs.stride_v_descale = stride_v_descale;

            kargs.nhead_stride_q_descale = nhead_stride_q_descale;
            kargs.nhead_stride_k_descale = nhead_stride_k_descale;
            kargs.nhead_stride_v_descale = nhead_stride_v_descale;

            kargs.batch_stride_q_descale = batch_stride_q_descale;
            kargs.batch_stride_k_descale = batch_stride_k_descale;
            kargs.batch_stride_v_descale = batch_stride_v_descale;
        }
        else if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::PERHEAD)
        {
            kargs.q_descale_ptr = q_descale_ptr;
            kargs.k_descale_ptr = k_descale_ptr;
            kargs.v_descale_ptr = v_descale_ptr;

            kargs.nhead_stride_q_descale = nhead_stride_q_descale;
            kargs.nhead_stride_k_descale = nhead_stride_k_descale;
            kargs.nhead_stride_v_descale = nhead_stride_v_descale;

            kargs.batch_stride_q_descale = batch_stride_q_descale;
            kargs.batch_stride_k_descale = batch_stride_k_descale;
            kargs.batch_stride_v_descale = batch_stride_v_descale;
        }
        if constexpr(kHasDropout)
        {
            if(drop_seed_offset.index() == 0) // seed & offset come from host
            {
                const auto& [seed, offset] = std::get<0>(drop_seed_offset);
                kargs.init_dropout(p_drop, seed, offset);
            }
            else // seed & offset come from device
            {
                const auto& [seed_ptr, offset_ptr] = std::get<1>(drop_seed_offset);
                kargs.init_dropout(p_drop,
                                   reinterpret_cast<const uint64_t*>(seed_ptr),
                                   reinterpret_cast<const uint64_t*>(offset_ptr));
            }

            kargs.rand_val_ptr         = rand_val_ptr;
            kargs.stride_randval       = stride_randval;
            kargs.nhead_stride_randval = nhead_stride_randval;
            kargs.batch_stride_randval = batch_stride_randval;
            kargs.is_store_randval     = s_randval;
        }
        if constexpr(kHasLogitsSoftCap)
        {
            kargs.init_logits_soft_cap(logits_soft_cap);
        }

        kargs.cu_seqlen_q_ptr = reinterpret_cast<const int32_t*>(cu_seqlen_q_ptr);
        kargs.cu_seqlen_k_ptr = reinterpret_cast<const int32_t*>(cu_seqlen_k_ptr);
        return kargs;
    }

    // std::variant<> can't take in a list initializer, overload for backward compatibility
    template <bool Cond = !kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              const void* bias_ptr,
              const void* q_descale_ptr,
              const void* k_descale_ptr,
              const void* v_descale_ptr,
              void* rand_val_ptr,
              void* lse_ptr,
              void* o_ptr,
              ck_tile::index_t seqlen_q,
              ck_tile::index_t seqlen_k,
              ck_tile::index_t hdim_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head_q,
              ck_tile::index_t nhead_ratio_qk,
              float scale_s,
              float logits_soft_cap,
              ck_tile::index_t stride_q,
              ck_tile::index_t stride_k,
              ck_tile::index_t stride_v,
              ck_tile::index_t stride_bias,
              ck_tile::index_t stride_randval,
              ck_tile::index_t stride_o,
              ck_tile::index_t stride_q_descale,
              ck_tile::index_t stride_k_descale,
              ck_tile::index_t stride_v_descale,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_bias,
              ck_tile::index_t nhead_stride_randval,
              ck_tile::index_t nhead_stride_lse,
              ck_tile::index_t nhead_stride_o,
              ck_tile::index_t nhead_stride_q_descale,
              ck_tile::index_t nhead_stride_k_descale,
              ck_tile::index_t nhead_stride_v_descale,
              ck_tile::index_t batch_stride_q,
              ck_tile::index_t batch_stride_k,
              ck_tile::index_t batch_stride_v,
              ck_tile::index_t batch_stride_bias,
              ck_tile::index_t batch_stride_randval,
              ck_tile::index_t batch_stride_lse,
              ck_tile::index_t batch_stride_o,
              ck_tile::index_t batch_stride_q_descale,
              ck_tile::index_t batch_stride_k_descale,
              ck_tile::index_t batch_stride_v_descale,
              ck_tile::index_t window_size_left,
              ck_tile::index_t window_size_right,
              ck_tile::index_t sink_size,
              ck_tile::index_t mask_type,
              float p_drop,
              bool s_randval,
              const std::tuple<uint64_t, uint64_t>& drop_seed_offset,
              ck_tile::index_t block_scale_size_q,
              ck_tile::index_t block_scale_size_kv,
              const void* cu_seqlen_q_ptr       = nullptr,
              const void* cu_seqlen_k_ptr       = nullptr,
              const void* sink_ptr              = nullptr,
              ck_tile::index_t num_head_q_total = 0,
              ck_tile::index_t head_start       = 0)
    {
        return MakeKargsImpl(
            q_ptr,
            k_ptr,
            v_ptr,
            bias_ptr,
            q_descale_ptr,
            k_descale_ptr,
            v_descale_ptr,
            rand_val_ptr,
            lse_ptr,
            o_ptr,
            seqlen_q,
            seqlen_k,
            hdim_q,
            hdim_v,
            num_head_q,
            nhead_ratio_qk,
            scale_s,
            logits_soft_cap,
            stride_q,
            stride_k,
            stride_v,
            stride_bias,
            stride_randval,
            stride_o,
            stride_q_descale,
            stride_k_descale,
            stride_v_descale,
            nhead_stride_q,
            nhead_stride_k,
            nhead_stride_v,
            nhead_stride_bias,
            nhead_stride_randval,
            nhead_stride_lse,
            nhead_stride_o,
            nhead_stride_q_descale,
            nhead_stride_k_descale,
            nhead_stride_v_descale,
            batch_stride_q,
            batch_stride_k,
            batch_stride_v,
            batch_stride_bias,
            batch_stride_randval,
            batch_stride_lse,
            batch_stride_o,
            batch_stride_q_descale,
            batch_stride_k_descale,
            batch_stride_v_descale,
            window_size_left,
            window_size_right,
            sink_size,
            mask_type,
            p_drop,
            s_randval,
            std::make_pair(std::get<0>(drop_seed_offset), std::get<1>(drop_seed_offset)),
            block_scale_size_q,
            block_scale_size_kv,
            cu_seqlen_q_ptr,
            cu_seqlen_k_ptr,
            sink_ptr,
            num_head_q_total,
            head_start);
    }

    // std::variant<> can't take in a list initializer, overload for backward compatibility
    template <bool Cond = !kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              const void* bias_ptr,
              const void* q_descale_ptr,
              const void* k_descale_ptr,
              const void* v_descale_ptr,
              void* rand_val_ptr,
              void* lse_ptr,
              void* o_ptr,
              ck_tile::index_t seqlen_q,
              ck_tile::index_t seqlen_k,
              ck_tile::index_t hdim_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head_q,
              ck_tile::index_t nhead_ratio_qk,
              float scale_s,
              float logits_soft_cap,
              ck_tile::index_t stride_q,
              ck_tile::index_t stride_k,
              ck_tile::index_t stride_v,
              ck_tile::index_t stride_bias,
              ck_tile::index_t stride_randval,
              ck_tile::index_t stride_o,
              ck_tile::index_t stride_q_descale,
              ck_tile::index_t stride_k_descale,
              ck_tile::index_t stride_v_descale,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_bias,
              ck_tile::index_t nhead_stride_randval,
              ck_tile::index_t nhead_stride_lse,
              ck_tile::index_t nhead_stride_o,
              ck_tile::index_t nhead_stride_q_descale,
              ck_tile::index_t nhead_stride_k_descale,
              ck_tile::index_t nhead_stride_v_descale,
              ck_tile::index_t batch_stride_q,
              ck_tile::index_t batch_stride_k,
              ck_tile::index_t batch_stride_v,
              ck_tile::index_t batch_stride_bias,
              ck_tile::index_t batch_stride_randval,
              ck_tile::index_t batch_stride_lse,
              ck_tile::index_t batch_stride_o,
              ck_tile::index_t batch_stride_q_descale,
              ck_tile::index_t batch_stride_k_descale,
              ck_tile::index_t batch_stride_v_descale,
              ck_tile::index_t window_size_left,
              ck_tile::index_t window_size_right,
              ck_tile::index_t sink_size,
              ck_tile::index_t mask_type,
              float p_drop,
              bool s_randval,
              const std::tuple<const void*, const void*>& drop_seed_offset,
              ck_tile::index_t block_scale_size_q,
              ck_tile::index_t block_scale_size_kv,
              const void* cu_seqlen_q_ptr       = nullptr,
              const void* cu_seqlen_k_ptr       = nullptr,
              const void* sink_ptr              = nullptr,
              ck_tile::index_t num_head_q_total = 0,
              ck_tile::index_t head_start       = 0)
    {
        return MakeKargsImpl(
            q_ptr,
            k_ptr,
            v_ptr,
            bias_ptr,
            q_descale_ptr,
            k_descale_ptr,
            v_descale_ptr,
            rand_val_ptr,
            lse_ptr,
            o_ptr,
            seqlen_q,
            seqlen_k,
            hdim_q,
            hdim_v,
            num_head_q,
            nhead_ratio_qk,
            scale_s,
            logits_soft_cap,
            stride_q,
            stride_k,
            stride_v,
            stride_bias,
            stride_randval,
            stride_o,
            stride_q_descale,
            stride_k_descale,
            stride_v_descale,
            nhead_stride_q,
            nhead_stride_k,
            nhead_stride_v,
            nhead_stride_bias,
            nhead_stride_randval,
            nhead_stride_lse,
            nhead_stride_o,
            nhead_stride_q_descale,
            nhead_stride_k_descale,
            nhead_stride_v_descale,
            batch_stride_q,
            batch_stride_k,
            batch_stride_v,
            batch_stride_bias,
            batch_stride_randval,
            batch_stride_lse,
            batch_stride_o,
            batch_stride_q_descale,
            batch_stride_k_descale,
            batch_stride_v_descale,
            window_size_left,
            window_size_right,
            sink_size,
            mask_type,
            p_drop,
            s_randval,
            std::make_pair(std::get<0>(drop_seed_offset), std::get<1>(drop_seed_offset)),
            block_scale_size_q,
            block_scale_size_kv,
            cu_seqlen_q_ptr,
            cu_seqlen_k_ptr,
            sink_ptr,
            num_head_q_total,
            head_start);
    }

    template <bool Cond = kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargsImpl(const void* q_ptr,
                  const void* k_ptr,
                  const void* v_ptr,
                  const void* bias_ptr,
                  const void* q_descale_ptr,
                  const void* k_descale_ptr,
                  const void* v_descale_ptr,
                  void* rand_val_ptr,
                  void* lse_ptr,
                  void* o_ptr,
                  const void* seqstart_q_ptr,
                  const void* seqstart_k_ptr,
                  const void* seqlen_q_ptr,
                  const void* seqlen_k_ptr,
                  const void* block_scale_seqstart_q_ptr,
                  const void* block_scale_seqstart_k_ptr,
                  const void* seqstart_v_scale_ptr,
                  ck_tile::index_t hdim_q,
                  ck_tile::index_t hdim_v,
                  ck_tile::index_t num_head_q,
                  ck_tile::index_t nhead_ratio_qk,
                  float scale_s,
                  float logits_soft_cap,
                  ck_tile::index_t stride_q,
                  ck_tile::index_t stride_k,
                  ck_tile::index_t stride_v,
                  ck_tile::index_t stride_bias,
                  ck_tile::index_t stride_randval,
                  ck_tile::index_t stride_o,
                  ck_tile::index_t stride_q_descale,
                  ck_tile::index_t stride_k_descale,
                  ck_tile::index_t stride_v_descale,
                  ck_tile::index_t nhead_stride_q,
                  ck_tile::index_t nhead_stride_k,
                  ck_tile::index_t nhead_stride_v,
                  ck_tile::index_t nhead_stride_bias,
                  ck_tile::index_t nhead_stride_randval,
                  ck_tile::index_t nhead_stride_lse,
                  ck_tile::index_t nhead_stride_o,
                  ck_tile::index_t nhead_stride_q_descale,
                  ck_tile::index_t nhead_stride_k_descale,
                  ck_tile::index_t nhead_stride_v_descale,
                  ck_tile::index_t batch_stride_q_descale,
                  ck_tile::index_t batch_stride_k_descale,
                  ck_tile::index_t batch_stride_v_descale,
                  ck_tile::index_t window_size_left,
                  ck_tile::index_t window_size_right,
                  ck_tile::index_t sink_size,
                  ck_tile::index_t mask_type,
                  ck_tile::index_t min_seqlen_q,
                  float p_drop,
                  bool s_randval,
                  std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>>
                      drop_seed_offset,
                  ck_tile::index_t block_scale_size_q,
                  ck_tile::index_t block_scale_size_kv,
                  const void* cu_seqlen_q_ptr       = nullptr,
                  const void* cu_seqlen_k_ptr       = nullptr,
                  const void* sink_ptr              = nullptr,
                  ck_tile::index_t num_head_q_total = 0,
                  ck_tile::index_t head_start       = 0)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     o_ptr,
                     sink_ptr,
                     -1, // seqlen will be updated by another pointer
                     -1, //
                     hdim_q,
                     hdim_v,
                     num_head_q,
                     nhead_ratio_qk,
#if CK_TILE_FMHA_FWD_FAST_EXP2
                     detail::ignore_fast_exp2_v<FmhaPipeline>
                         ? scale_s
                         : static_cast<float>(scale_s * ck_tile::log2e_v<>),
#else
                     scale_s,
#endif
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_o,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_o}, // args for common karg
                    {},               // placeholder for bias
                    {},               // placeholder for mask
                    {},               // placeholder for lse
                    {},               // placeholder for qscale
                    {},               // placeholder for dropout
                    {},               // placeholder for logits_soft_cap
                    {},               // placeholder for min_seqlen_q
                    reinterpret_cast<const int32_t*>(seqstart_q_ptr),
                    reinterpret_cast<const int32_t*>(seqstart_k_ptr),
                    reinterpret_cast<const int32_t*>(seqlen_q_ptr),
                    reinterpret_cast<const int32_t*>(seqlen_k_ptr)};
        kargs.num_head_q_total = num_head_q_total;
        kargs.head_start       = head_start;

        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
        {
            kargs.bias_ptr          = bias_ptr;
            kargs.stride_bias       = stride_bias;
            kargs.nhead_stride_bias = nhead_stride_bias;
        }
        else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
        {
            kargs.alibi_slope_ptr    = bias_ptr;
            kargs.alibi_slope_stride = stride_bias;
        }
        if constexpr(kHasMask)
        {
            kargs.window_size_left  = window_size_left;
            kargs.window_size_right = window_size_right;
            kargs.sink_size         = sink_size;
            kargs.mask_type         = static_cast<ck_tile::GenericAttentionMaskEnum>(mask_type);
        }
        if constexpr(kStoreLSE)
        {
            kargs.lse_ptr          = lse_ptr;
            kargs.nhead_stride_lse = nhead_stride_lse;
        }
        if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::PERTENSOR)
        {
            kargs.q_descale_ptr = q_descale_ptr;
            kargs.k_descale_ptr = k_descale_ptr;
            kargs.v_descale_ptr = v_descale_ptr;
        }
        else if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE)
        {
            kargs.q_descale_ptr = q_descale_ptr;
            kargs.k_descale_ptr = k_descale_ptr;
            kargs.v_descale_ptr = v_descale_ptr;

            kargs.nhead_stride_q_descale = nhead_stride_q_descale;
            kargs.nhead_stride_k_descale = nhead_stride_k_descale;
            kargs.nhead_stride_v_descale = nhead_stride_v_descale;

            kargs.block_scale_size_q  = block_scale_size_q;
            kargs.block_scale_size_kv = block_scale_size_kv;

            kargs.block_scale_seqstart_q_ptr =
                reinterpret_cast<const int32_t*>(block_scale_seqstart_q_ptr);
            kargs.block_scale_seqstart_k_ptr =
                reinterpret_cast<const int32_t*>(block_scale_seqstart_k_ptr);
        }
        else if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::MX)
        {
            kargs.q_descale_ptr = q_descale_ptr;
            kargs.k_descale_ptr = k_descale_ptr;
            kargs.v_descale_ptr = v_descale_ptr;

            kargs.stride_q_descale = stride_q_descale;
            kargs.stride_k_descale = stride_k_descale;
            kargs.stride_v_descale = stride_v_descale;

            kargs.nhead_stride_q_descale = nhead_stride_q_descale;
            kargs.nhead_stride_k_descale = nhead_stride_k_descale;
            kargs.nhead_stride_v_descale = nhead_stride_v_descale;

            kargs.seqstart_v_scale_ptr = reinterpret_cast<const int32_t*>(seqstart_v_scale_ptr);
        }
        else if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::PERHEAD)
        {
            kargs.q_descale_ptr = q_descale_ptr;
            kargs.k_descale_ptr = k_descale_ptr;
            kargs.v_descale_ptr = v_descale_ptr;

            kargs.nhead_stride_q_descale = nhead_stride_q_descale;
            kargs.nhead_stride_k_descale = nhead_stride_k_descale;
            kargs.nhead_stride_v_descale = nhead_stride_v_descale;

            kargs.batch_stride_q_descale = batch_stride_q_descale;
            kargs.batch_stride_k_descale = batch_stride_k_descale;
            kargs.batch_stride_v_descale = batch_stride_v_descale;
        }
        if constexpr(kHasDropout)
        {
            if(drop_seed_offset.index() == 0) // seed & offset come from host
            {
                const auto& [seed, offset] = std::get<0>(drop_seed_offset);
                kargs.init_dropout(p_drop, seed, offset);
            }
            else // seed & offset come from device
            {
                const auto& [seed_ptr, offset_ptr] = std::get<1>(drop_seed_offset);
                kargs.init_dropout(p_drop,
                                   reinterpret_cast<const uint64_t*>(seed_ptr),
                                   reinterpret_cast<const uint64_t*>(offset_ptr));
            }

            kargs.rand_val_ptr         = rand_val_ptr;
            kargs.stride_randval       = stride_randval;
            kargs.nhead_stride_randval = nhead_stride_randval;
            kargs.is_store_randval     = s_randval;
        }
        if constexpr(kHasLogitsSoftCap)
        {
            kargs.init_logits_soft_cap(logits_soft_cap);
        }
        if constexpr(kSkipMinSeqlenQ)
        {
            kargs.min_seqlen_q = min_seqlen_q;
        }

        kargs.cu_seqlen_q_ptr = reinterpret_cast<const int32_t*>(cu_seqlen_q_ptr);
        kargs.cu_seqlen_k_ptr = reinterpret_cast<const int32_t*>(cu_seqlen_k_ptr);
        return kargs;
    }

    // std::variant<> can't take in a list initializer, overload for backward compatibility
    template <bool Cond = kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              const void* bias_ptr,
              const void* q_descale_ptr,
              const void* k_descale_ptr,
              const void* v_descale_ptr,
              void* rand_val_ptr,
              void* lse_ptr,
              void* o_ptr,
              const void* seqstart_q_ptr,
              const void* seqstart_k_ptr,
              const void* seqlen_q_ptr,
              const void* seqlen_k_ptr,
              const void* block_scale_seqstart_q_ptr,
              const void* block_scale_seqstart_k_ptr,
              const void* seqstart_v_scale_ptr,
              ck_tile::index_t hdim_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head_q,
              ck_tile::index_t nhead_ratio_qk,
              float scale_s,
              float logits_soft_cap,
              ck_tile::index_t stride_q,
              ck_tile::index_t stride_k,
              ck_tile::index_t stride_v,
              ck_tile::index_t stride_bias,
              ck_tile::index_t stride_randval,
              ck_tile::index_t stride_o,
              ck_tile::index_t stride_q_descale,
              ck_tile::index_t stride_k_descale,
              ck_tile::index_t stride_v_descale,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_bias,
              ck_tile::index_t nhead_stride_randval,
              ck_tile::index_t nhead_stride_lse,
              ck_tile::index_t nhead_stride_o,
              ck_tile::index_t nhead_stride_q_descale,
              ck_tile::index_t nhead_stride_k_descale,
              ck_tile::index_t nhead_stride_v_descale,
              ck_tile::index_t batch_stride_q_descale,
              ck_tile::index_t batch_stride_k_descale,
              ck_tile::index_t batch_stride_v_descale,
              ck_tile::index_t window_size_left,
              ck_tile::index_t window_size_right,
              ck_tile::index_t sink_size,
              ck_tile::index_t mask_type,
              ck_tile::index_t min_seqlen_q,
              float p_drop,
              bool s_randval,
              const std::tuple<uint64_t, uint64_t>& drop_seed_offset,
              ck_tile::index_t block_scale_size_q,
              ck_tile::index_t block_scale_size_kv,
              const void* cu_seqlen_q_ptr       = nullptr,
              const void* cu_seqlen_k_ptr       = nullptr,
              const void* sink_ptr              = nullptr,
              ck_tile::index_t num_head_q_total = 0,
              ck_tile::index_t head_start       = 0)
    {
        return MakeKargsImpl(
            q_ptr,
            k_ptr,
            v_ptr,
            bias_ptr,
            q_descale_ptr,
            k_descale_ptr,
            v_descale_ptr,
            rand_val_ptr,
            lse_ptr,
            o_ptr,
            seqstart_q_ptr,
            seqstart_k_ptr,
            seqlen_q_ptr,
            seqlen_k_ptr,
            block_scale_seqstart_q_ptr,
            block_scale_seqstart_k_ptr,
            seqstart_v_scale_ptr,
            hdim_q,
            hdim_v,
            num_head_q,
            nhead_ratio_qk,
            scale_s,
            logits_soft_cap,
            stride_q,
            stride_k,
            stride_v,
            stride_bias,
            stride_randval,
            stride_o,
            stride_q_descale,
            stride_k_descale,
            stride_v_descale,
            nhead_stride_q,
            nhead_stride_k,
            nhead_stride_v,
            nhead_stride_bias,
            nhead_stride_randval,
            nhead_stride_lse,
            nhead_stride_o,
            nhead_stride_q_descale,
            nhead_stride_k_descale,
            nhead_stride_v_descale,
            batch_stride_q_descale,
            batch_stride_k_descale,
            batch_stride_v_descale,
            window_size_left,
            window_size_right,
            sink_size,
            mask_type,
            min_seqlen_q,
            p_drop,
            s_randval,
            std::make_pair(std::get<0>(drop_seed_offset), std::get<1>(drop_seed_offset)),
            block_scale_size_q,
            block_scale_size_kv,
            cu_seqlen_q_ptr,
            cu_seqlen_k_ptr,
            sink_ptr,
            num_head_q_total,
            head_start);
    }

    // std::variant<> can't take in a list initializer, overload for backward compatibility
    template <bool Cond = kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              const void* bias_ptr,
              const void* q_descale_ptr,
              const void* k_descale_ptr,
              const void* v_descale_ptr,
              void* rand_val_ptr,
              void* lse_ptr,
              void* o_ptr,
              const void* seqstart_q_ptr,
              const void* seqstart_k_ptr,
              const void* seqlen_q_ptr,
              const void* seqlen_k_ptr,
              const void* block_scale_seqstart_q_ptr,
              const void* block_scale_seqstart_k_ptr,
              const void* seqstart_v_scale_ptr,
              ck_tile::index_t hdim_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head_q,
              ck_tile::index_t nhead_ratio_qk,
              float scale_s,
              float logits_soft_cap,
              ck_tile::index_t stride_q,
              ck_tile::index_t stride_k,
              ck_tile::index_t stride_v,
              ck_tile::index_t stride_bias,
              ck_tile::index_t stride_randval,
              ck_tile::index_t stride_o,
              ck_tile::index_t stride_q_descale,
              ck_tile::index_t stride_k_descale,
              ck_tile::index_t stride_v_descale,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_bias,
              ck_tile::index_t nhead_stride_randval,
              ck_tile::index_t nhead_stride_lse,
              ck_tile::index_t nhead_stride_o,
              ck_tile::index_t nhead_stride_q_descale,
              ck_tile::index_t nhead_stride_k_descale,
              ck_tile::index_t nhead_stride_v_descale,
              ck_tile::index_t batch_stride_q_descale,
              ck_tile::index_t batch_stride_k_descale,
              ck_tile::index_t batch_stride_v_descale,
              ck_tile::index_t window_size_left,
              ck_tile::index_t window_size_right,
              ck_tile::index_t sink_size,
              ck_tile::index_t mask_type,
              ck_tile::index_t min_seqlen_q,
              float p_drop,
              bool s_randval,
              const std::tuple<const void*, const void*>& drop_seed_offset,
              ck_tile::index_t block_scale_size_q,
              ck_tile::index_t block_scale_size_kv,
              const void* cu_seqlen_q_ptr       = nullptr,
              const void* cu_seqlen_k_ptr       = nullptr,
              const void* sink_ptr              = nullptr,
              ck_tile::index_t num_head_q_total = 0,
              ck_tile::index_t head_start       = 0)
    {
        return MakeKargsImpl(
            q_ptr,
            k_ptr,
            v_ptr,
            bias_ptr,
            q_descale_ptr,
            k_descale_ptr,
            v_descale_ptr,
            rand_val_ptr,
            lse_ptr,
            o_ptr,
            seqstart_q_ptr,
            seqstart_k_ptr,
            seqlen_q_ptr,
            seqlen_k_ptr,
            block_scale_seqstart_q_ptr,
            block_scale_seqstart_k_ptr,
            seqstart_v_scale_ptr,
            hdim_q,
            hdim_v,
            num_head_q,
            nhead_ratio_qk,
            scale_s,
            logits_soft_cap,
            stride_q,
            stride_k,
            stride_v,
            stride_bias,
            stride_randval,
            stride_o,
            stride_q_descale,
            stride_k_descale,
            stride_v_descale,
            nhead_stride_q,
            nhead_stride_k,
            nhead_stride_v,
            nhead_stride_bias,
            nhead_stride_randval,
            nhead_stride_lse,
            nhead_stride_o,
            nhead_stride_q_descale,
            nhead_stride_k_descale,
            nhead_stride_v_descale,
            batch_stride_q_descale,
            batch_stride_k_descale,
            batch_stride_v_descale,
            window_size_left,
            window_size_right,
            sink_size,
            mask_type,
            min_seqlen_q,
            p_drop,
            s_randval,
            std::make_pair(std::get<0>(drop_seed_offset), std::get<1>(drop_seed_offset)),
            block_scale_size_q,
            block_scale_size_kv,
            cu_seqlen_q_ptr,
            cu_seqlen_k_ptr,
            sink_ptr,
            num_head_q_total,
            head_start);
    }

    CK_TILE_HOST static bool IsSupportedArgument([[maybe_unused]] const Kargs& kargs)
    {
        if constexpr(kPipelineName == "qr_tdm_sched")
        {
            // The generated no-virtual-sink specialization omits the finite
            // sink normalization. kHasSink controls retained mask tokens and
            // cannot be used to infer whether sink_ptr is present.
            if constexpr(!FmhaPipeline::kEnableVirtualSink)
            {
                if(kargs.sink_ptr != nullptr)
                    return false;
            }
            // The codegen constraint repeats these checks so the dispatcher falls back to qr_tdm.
            if constexpr(kUseAsyncLdsOStore)
            {
                if(kargs.hdim_v != FmhaPipeline::kN1)
                    return false;
                if(reinterpret_cast<uintptr_t>(kargs.o_ptr) % kAsyncOStoreBytes != 0 ||
                   kargs.stride_o < 0 || kargs.stride_o % kAsyncOStoreVec != 0 ||
                   kargs.nhead_stride_o % kAsyncOStoreVec != 0)
                {
                    return false;
                }
                if constexpr(!kIsGroupMode)
                {
                    if(kargs.batch_stride_o % kAsyncOStoreVec != 0)
                        return false;
                }
                const uint64_t tile_span_bytes = (FmhaPipeline::kM0 - 1) *
                                                     static_cast<uint64_t>(kargs.stride_o) *
                                                     sizeof(ODataType) +
                                                 FmhaPipeline::kN1 * sizeof(ODataType);
                if(tile_span_bytes > kAsyncOTileSpanLimitBytes)
                    return false;
            }
        }

        if constexpr(kPipelineName == "qr_tdm_fp8_sched")
        {
            const bool log    = ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING));
            const auto reject = [&](const char* reason) {
                if(log)
                    CK_TILE_ERROR("FMHA fwd qr_tdm_fp8_sched: ", reason);
                return false;
            };
            // No head-dim padding: Q/K/V TDM tiles and WMMAs cover exactly the tile head dims.
            if(kargs.hdim_q != FmhaPipeline::kQKHeaddim || kargs.hdim_v != FmhaPipeline::kN1)
                return reject("hdim_q and hdim_v must equal the tile head dims.");
            if constexpr(!FmhaPipeline::kEnableVirtualSink)
            {
                if(kargs.sink_ptr != nullptr)
                    return reject("sink_ptr is not supported.");
            }
            if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE)
            {
                if(kargs.block_scale_size_q <= 0 || kargs.block_scale_size_kv <= 0)
                    return reject("block_scale_size_q and block_scale_size_kv must be positive.");
            }
            if constexpr(kMergeNumHeadGroupsSeqLenQ)
            {
                // Merged heads must form one row stride: a single query, or seqlen_q-row heads.
                const auto s  = kargs.seqlen_q;
                bool foldable = s == 1 || (kargs.nhead_stride_q == s * kargs.stride_q &&
                                           kargs.nhead_stride_o == s * kargs.stride_o);
                if constexpr(kStoreLSE)
                    foldable = foldable && (s == 1 || kargs.nhead_stride_lse == s);
                if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::MX)
                    foldable = foldable && (s == 1 || kargs.nhead_stride_q_descale ==
                                                          s * kargs.stride_q_descale);
                if(!foldable)
                    return reject("merged query heads need foldable Q/O/LSE/scale strides.");
                if(s * kargs.nhead_ratio_qk > FmhaPipeline::kM0)
                    return reject("merged query heads must fit one tile of query rows.");
                if(kargs.cu_seqlen_q_ptr != nullptr || kargs.cu_seqlen_k_ptr != nullptr)
                    return reject("merged query heads do not support cu_seqlen_q/k_ptr.");
            }
            if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::MX)
            {
                // Q/K E8M0 rows are read per lane as int32s (D128/D256) or an int16 (D64).
                const auto bits  = [](auto v) { return static_cast<uint64_t>(v); };
                uint64_t offsets = bits(reinterpret_cast<uintptr_t>(kargs.q_descale_ptr)) |
                                   bits(reinterpret_cast<uintptr_t>(kargs.k_descale_ptr)) |
                                   bits(kargs.stride_q_descale) | bits(kargs.stride_k_descale) |
                                   bits(kargs.nhead_stride_q_descale) |
                                   bits(kargs.nhead_stride_k_descale);
                if constexpr(!kIsGroupMode)
                    offsets |=
                        bits(kargs.batch_stride_q_descale) | bits(kargs.batch_stride_k_descale);
                if(offsets % FmhaPipeline::kMxQkScaleAlign != 0)
                    return reject("MX Q/K descale pointers and strides must align to a scale row.");
                if constexpr(kIsGroupMode)
                {
                    if(kargs.seqstart_v_scale_ptr == nullptr)
                        return reject("group mode MX needs seqstart_v_scale_ptr.");
                }
            }
        }

        if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE)
        {
            const bool log = ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING));

            if constexpr(kTakesTdmDescales)
            {
                if(kargs.block_scale_size_q <= 0 || kargs.block_scale_size_kv <= 0)
                {
                    if(log)
                        CK_TILE_ERROR("FMHA fwd BLOCKSCALE: block sizes must be positive.");
                    return false;
                }
            }

            if(kargs.block_scale_size_q % FmhaPipeline::kM0 != 0)
            {
                if(log)
                    CK_TILE_ERROR("FMHA fwd BLOCKSCALE: block_scale_size_q (",
                                  kargs.block_scale_size_q,
                                  ") must be a multiple of the M tile size (",
                                  FmhaPipeline::kM0,
                                  ").");
                return false;
            }

            if constexpr(kTakesTdmDescales)
            {
                if(kargs.block_scale_size_kv % FmhaPipeline::kKVScaleAlign != 0)
                {
                    if(log)
                        CK_TILE_ERROR("FMHA fwd BLOCKSCALE: block_scale_size_kv (",
                                      kargs.block_scale_size_kv,
                                      ") must be a multiple of the MMA scale operand's KV "
                                      "resolution (",
                                      FmhaPipeline::kKVScaleAlign,
                                      ").");
                    return false;
                }
            }
            else
            {
                if(kargs.block_scale_size_kv % FmhaPipeline::kN0 != 0)
                {
                    if(log)
                        CK_TILE_ERROR("FMHA fwd BLOCKSCALE: block_scale_size_kv (",
                                      kargs.block_scale_size_kv,
                                      ") must be a multiple of the N tile size (",
                                      FmhaPipeline::kN0,
                                      ").");
                    return false;
                }
            }
        }

        return true;
    }

    CK_TILE_HOST static constexpr auto GridSize(ck_tile::index_t batch_size_,
                                                ck_tile::index_t nhead_,
                                                ck_tile::index_t seqlen_q_,
                                                ck_tile::index_t hdim_v_,
                                                bool has_padded_seqlen_k = false)
    {
        // has_padded_seqlen_k is determined by checking (seqlen_k_ptr != nullptr)
        if(has_padded_seqlen_k)
        {
            // TODO: this may need tuning
            return dim3(nhead_,
                        batch_size_,
                        ck_tile::integer_divide_ceil(seqlen_q_, FmhaPipeline::kM0) *
                            ck_tile::integer_divide_ceil(hdim_v_, FmhaPipeline::kN1));
        }
        else
        {
            // TODO: this may need tuning
            return dim3(nhead_,
                        ck_tile::integer_divide_ceil(seqlen_q_, FmhaPipeline::kM0) *
                            ck_tile::integer_divide_ceil(hdim_v_, FmhaPipeline::kN1),
                        batch_size_);
        }
    }

    // Uniform sequence lengths allow locality-oriented XCD partitioning. Causal
    // tiles within one sequence still have different amounts of work.
    CK_TILE_DEVICE static bool HasUniformSequenceLengths([[maybe_unused]] const Kargs& kargs)
    {
        if constexpr(kIsGroupMode)
            return false;
        else
            return kargs.cu_seqlen_q_ptr == nullptr && kargs.cu_seqlen_k_ptr == nullptr;
    }

    CK_TILE_DEVICE static constexpr auto GetTileIndex(const Kargs& kargs)
    {
        bool has_padded_seqlen_k = false;
        if constexpr(kIsGroupMode)
            has_padded_seqlen_k = kargs.seqlen_k_ptr != nullptr;

        const FmhaFwdWorkgroupGrid grid{
            static_cast<index_t>(gridDim.x),
            static_cast<index_t>(has_padded_seqlen_k ? gridDim.z : gridDim.y),
            static_cast<index_t>(has_padded_seqlen_k ? gridDim.y : gridDim.z),
            ck_tile::integer_divide_ceil(kargs.hdim_v, FmhaPipeline::kN1),
            has_padded_seqlen_k};
        const FmhaFwdWorkgroup block{
            static_cast<index_t>(blockIdx.x),
            static_cast<index_t>(has_padded_seqlen_k ? blockIdx.z : blockIdx.y),
            static_cast<index_t>(has_padded_seqlen_k ? blockIdx.y : blockIdx.z)};

        FmhaFwdWorkgroupFeatures features{};
#if CK_TILE_FMHA_FORCE_HEAD_MAJOR
        // Preserve the ROCm 7.1 + gfx12 compiler-workaround gate.
#if defined(__gfx12__) && (HIP_VERSION_MAJOR == 7) && (HIP_VERSION_MINOR == 1)
        constexpr bool kSkipHeadMajor = kIsGroupMode && kHasMask && !kHasDropout &&
                                        (BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS) &&
                                        kPadHeadDimQ && kPadHeadDimV &&
                                        (FmhaPipeline::kN1 == 256) &&
                                        std::is_same_v<QDataType, ck_tile::fp16_t> &&
                                        std::is_same_v<KDataType, ck_tile::fp16_t> &&
                                        std::is_same_v<VDataType, ck_tile::fp16_t>;
#else
        constexpr bool kSkipHeadMajor = false;
#endif
        constexpr bool kEnableHeadMajor = !kSkipHeadMajor;
#else
        constexpr bool kEnableHeadMajor = false;
#endif
        constexpr bool kEnableBatchCausal =
            kHasMask && !kIsGroupMode && !kHasDropout && kUseBatchLocalCausalOrder;
        // The head-stride guard avoids the single-head BSHD false positive.
        features.is_bhsd = kargs.stride_q == kargs.hdim_q && kargs.nhead_stride_q > kargs.hdim_q;
        features.uniform_lengths = HasUniformSequenceLengths(kargs);
        if constexpr(kHasMask)
            features.square_causal = kargs.seqlen_q == kargs.seqlen_k &&
                                     kargs.window_size_left < 0 && kargs.window_size_right == 0;

        if constexpr(kEnableHeadMajor && kHasMask && !kIsGroupMode && !kHasDropout &&
                     detail::uses_global_causal_order_v<FmhaPipeline>)
        {
            // Square causal: visit the longest tiles across all batches and heads first.
            if(features.is_bhsd && features.square_causal && grid.num_v_tiles == 1)
            {
                const index_t num_bh    = grid.num_heads * grid.num_batches;
                const index_t linear_id = FmhaFwdLinearWorkgroupId(grid, block);
                const index_t rank      = linear_id / num_bh;
                const index_t bh        = linear_id - rank * num_bh;
                const index_t i_batch   = bh / grid.num_heads;
                // The divisions are lowered to VALU; keep the indices in SGPRs.
                return ck_tile::make_tuple(amd_wave_read_first_lane(grid.num_qv_tiles - 1 - rank),
                                           index_t{0},
                                           amd_wave_read_first_lane(bh - i_batch * grid.num_heads),
                                           amd_wave_read_first_lane(i_batch));
            }
        }

#if defined(CK_TILE_FMHA_PAIRED_XCC_ORDER) && CK_TILE_FMHA_PAIRED_XCC_ORDER && defined(__gfx1250__)
        constexpr bool kEnablePaired = kHasMask && !kIsGroupMode && kIsTdmSchedD128 &&
                                       std::is_same_v<QDataType, ck_tile::bf16_t> &&
                                       FmhaPipeline::BlockFmhaShape::kQKHeaddim == 128 &&
                                       !kHasDropout && !kHasSink && !kHasLogitsSoftCap &&
                                       BiasEnum == BlockAttentionBiasEnum::NO_BIAS &&
                                       QScaleEnum == BlockAttentionQuantScaleEnum::NO_SCALE;
#else
        constexpr bool kEnablePaired = false;
#endif
// The measured device pass is gfx1250; gfx1250-strict retains its fallback.
#if defined(__gfx1250__)
        constexpr bool kEnableXcdCausal = kSupportsXcdCausalOrder;
#else
        constexpr bool kEnableXcdCausal = false;
#endif
        const auto tile = MapFmhaFwdWorkgroup<kHasMask,
                                              kEnableHeadMajor,
                                              kEnableBatchCausal,
                                              kEnablePaired,
                                              kEnableXcdCausal>(
            features, grid, block, CK_TILE_XCD_REMAP_NUM_XCDS, kargs.nhead_ratio_qk);
        return ck_tile::make_tuple(tile.q_tile, tile.v_tile, tile.head, tile.batch);
    }

    CK_TILE_HOST static dim3 BlockSize()
    {
        if(is_wave32())
        {
            return dim3(kBlockSize / 2);
        }
        else
        {
            return dim3(kBlockSize);
        }
    }

    // Arena bytes only; qr_tdm_fp8_sched adds static LDS for O staging, MX and merged row scales.
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        if constexpr(detail::uses_qr_tdm_lds_arena_v<FmhaPipeline>)
        {
            using Layout = typename FmhaPipeline::Policy::template LdsArenaLayout<
                typename FmhaPipeline::Problem>;
            static_assert(FmhaPipeline::GetSmemSize() == Layout::kArenaBytes);
            return ck_tile::max(Layout::kArenaBytes, EpiloguePipeline::GetSmemSize());
        }
        else
        {
            return ck_tile::max(FmhaPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
        }
    }

    // PERTENSOR folds q_descale*k_descale into scale_s for every pipeline (its offset is always
    // 0 - one scale for the whole tensor). kTakesTdmDescales pipelines additionally fold PERHEAD's
    // q_descale*k_descale (per-head) and BLOCKSCALE's q_descale (per-M-block, via i_m0) into
    // scale_s - other pipelines never instantiate PERHEAD, and leave BLOCKSCALE's
    // q_descale/k_descale to be applied elsewhere, so scale_s stays unfolded there.
    // descale_offset_q/k are the caller's already-computed nhead/batch (and, in group mode,
    // block-scale-seqstart) offsets - 0 wherever the caller's pipeline never indexes them.
    CK_TILE_DEVICE static constexpr float GetSoftmaxScale(const Kargs& kargs,
                                                          long_index_t descale_offset_q,
                                                          long_index_t descale_offset_k,
                                                          index_t i_m0)
    {
        if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::PERTENSOR)
        {
            const float q_descale = *(reinterpret_cast<const float*>(kargs.q_descale_ptr));
            const float k_descale = *(reinterpret_cast<const float*>(kargs.k_descale_ptr));

            return kargs.scale_s * q_descale * k_descale;
        }
        else if constexpr(kTakesTdmDescales && QScaleEnum == BlockAttentionQuantScaleEnum::PERHEAD)
        {
            const float* q_descale_ptr = reinterpret_cast<const float*>(kargs.q_descale_ptr);
            const float* k_descale_ptr = reinterpret_cast<const float*>(kargs.k_descale_ptr);

            return kargs.scale_s * q_descale_ptr[descale_offset_q] *
                   k_descale_ptr[descale_offset_k];
        }
        else if constexpr(kTakesTdmDescales &&
                          QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE)
        {
            const float* q_descale_ptr = reinterpret_cast<const float*>(kargs.q_descale_ptr);
            const float q_descale =
                q_descale_ptr[descale_offset_q + i_m0 / kargs.block_scale_size_q];

            return kargs.scale_s * q_descale;
        }
        else
        {
            return kargs.scale_s;
        }
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        if constexpr(kIsAvailable)
        {
            run_(std::move(kargs));
        }
    }

    CK_TILE_DEVICE void run_(Kargs kargs) const
    {
        // Pipelines with native row-major V consume (seqlen, hdim_v) windows
        // without the explicit DRAM transpose used by the generic path.
        if constexpr(!kUsesUntransposedVKernelPath)
        {
            // allocate LDS
            __shared__ char smem_ptr[GetSmemSize()];
            // divide problem
            const auto [i_tile_m, i_tile_n, i_nhead, i_batch] = GetTileIndex(kargs);
            const index_t i_m0 = amd_wave_read_first_lane(i_tile_m * FmhaPipeline::kM0);
            const index_t i_n1 = amd_wave_read_first_lane(i_tile_n * FmhaPipeline::kN1);

            long_index_t batch_offset_q         = 0;
            long_index_t batch_offset_k         = 0;
            long_index_t batch_offset_v         = 0;
            long_index_t batch_offset_bias      = 0;
            long_index_t batch_offset_randval   = 0;
            long_index_t batch_offset_lse       = 0;
            long_index_t batch_offset_o         = 0;
            long_index_t batch_offset_q_descale = 0;
            long_index_t batch_offset_k_descale = 0;
            long_index_t batch_offset_v_descale = 0;
            float sink_value                    = -numeric<float>::infinity();
            if constexpr(kHasMask && !kHasSink)
            {
                sink_value = -numeric<float>::infinity();
            }
            else
            {
                // This branch is never qr_tdm, so we can use 0,0 as perhead is not supported
                // and blockscale is handled elementwise in the pipeline.
                sink_value = kargs.sink_ptr != nullptr
                                 ? (*(static_cast<const float*>(kargs.sink_ptr) + i_nhead)) /
                                       GetSoftmaxScale(kargs, 0, 0, i_m0)
                                 : -numeric<float>::infinity();
            }

            if constexpr(kIsGroupMode)
            {
                // Use seqstart_q_ptr and seqstart_k_ptr for physical starts
                const long_index_t query_start = kargs.seqstart_q_ptr[i_batch];
                const long_index_t key_start   = kargs.seqstart_k_ptr[i_batch];

                // DRAM base offsets use physical starts
                batch_offset_q = query_start * kargs.stride_q;
                batch_offset_k = key_start * kargs.stride_k;
                if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
                {
                    batch_offset_v = key_start * kargs.stride_v;
                }
                else
                {
                    batch_offset_v = key_start;
                }
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                {
                    batch_offset_bias = query_start * kargs.stride_bias;
                }
                if constexpr(kStoreLSE)
                {
                    // LSE follows the physical layout to stay consistent with other tensors
                    batch_offset_lse = query_start;
                }
                if constexpr(kHasDropout)
                {
                    batch_offset_randval = query_start * kargs.stride_randval;
                }
                if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE)
                {
                    const long_index_t bquery_start = kargs.block_scale_seqstart_q_ptr[i_batch];
                    const long_index_t bkey_start   = kargs.block_scale_seqstart_k_ptr[i_batch];
                    batch_offset_q_descale          = bquery_start;
                    batch_offset_k_descale          = bkey_start;
                    batch_offset_v_descale          = bkey_start;
                }
                else if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::MX)
                {
                    batch_offset_q_descale = query_start * kargs.stride_q_descale;
                    batch_offset_k_descale = key_start * kargs.stride_k_descale;
                    batch_offset_v_descale = kargs.seqstart_v_scale_ptr[i_batch];
                }
                batch_offset_o = query_start * kargs.stride_o;

                // real logical lengths (exclude PAD)
                // Priority: seqlen_q_ptr > cu_seqlen_q_ptr > calculated from seqstart_q_ptr
                if(kargs.seqlen_q_ptr != nullptr)
                {
                    kargs.seqlen_q = kargs.seqlen_q_ptr[i_batch];
                }
                else if(kargs.cu_seqlen_q_ptr != nullptr)
                {
                    kargs.seqlen_q =
                        kargs.cu_seqlen_q_ptr[i_batch + 1] - kargs.cu_seqlen_q_ptr[i_batch];
                }
                else
                {
                    const auto adjusted_seqstart_q_ptr = kargs.seqstart_q_ptr + i_batch;
                    kargs.seqlen_q = adjusted_seqstart_q_ptr[1] - adjusted_seqstart_q_ptr[0];
                }

                if constexpr(kSkipMinSeqlenQ)
                {
                    if(kargs.seqlen_q <= kargs.min_seqlen_q)
                    {
                        return;
                    }
                }

                // terminate unnecessary blocks earlier
                if(kargs.seqlen_q <= i_m0)
                {
                    return;
                }

                if(kargs.seqlen_k_ptr != nullptr)
                {
                    kargs.seqlen_k = kargs.seqlen_k_ptr[i_batch];
                }
                else if(kargs.cu_seqlen_k_ptr != nullptr)
                {
                    kargs.seqlen_k =
                        kargs.cu_seqlen_k_ptr[i_batch + 1] - kargs.cu_seqlen_k_ptr[i_batch];
                }
                else
                {
                    const auto adjusted_seqstart_k_ptr = kargs.seqstart_k_ptr + i_batch;
                    kargs.seqlen_k = adjusted_seqstart_k_ptr[1] - adjusted_seqstart_k_ptr[0];
                }
            }
            else
            {
                batch_offset_q = static_cast<long_index_t>(i_batch) * kargs.batch_stride_q;
                batch_offset_k = static_cast<long_index_t>(i_batch) * kargs.batch_stride_k;
                batch_offset_v = static_cast<long_index_t>(i_batch) * kargs.batch_stride_v;
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                {
                    batch_offset_bias =
                        static_cast<long_index_t>(i_batch) * kargs.batch_stride_bias;
                }
                if constexpr(kStoreLSE)
                {
                    batch_offset_lse = static_cast<long_index_t>(i_batch) * kargs.batch_stride_lse;
                }
                if constexpr(kHasDropout)
                {
                    batch_offset_randval =
                        static_cast<long_index_t>(i_batch) * kargs.batch_stride_randval;
                }
                if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE ||
                             QScaleEnum == BlockAttentionQuantScaleEnum::MX)
                {
                    batch_offset_q_descale =
                        static_cast<long_index_t>(i_batch) * kargs.batch_stride_q_descale;
                    batch_offset_k_descale =
                        static_cast<long_index_t>(i_batch) * kargs.batch_stride_k_descale;
                    batch_offset_v_descale =
                        static_cast<long_index_t>(i_batch) * kargs.batch_stride_v_descale;
                }
                batch_offset_o = static_cast<long_index_t>(i_batch) * kargs.batch_stride_o;

                // If cumulative seqlen pointers are provided, override per-batch effective lengths
                if(kargs.cu_seqlen_q_ptr != nullptr)
                {
                    kargs.seqlen_q =
                        kargs.cu_seqlen_q_ptr[i_batch + 1] - kargs.cu_seqlen_q_ptr[i_batch];

                    // the grid covers the padded seqlen_q; return early for the blocks
                    // past this batch's effective length, as group mode does
                    if(kargs.seqlen_q <= i_m0)
                    {
                        return;
                    }
                }
                if(kargs.cu_seqlen_k_ptr != nullptr)
                {
                    kargs.seqlen_k =
                        kargs.cu_seqlen_k_ptr[i_batch + 1] - kargs.cu_seqlen_k_ptr[i_batch];
                }
            }

            // for simplicity, batch stride we just modify the pointer
            const index_t i_nhead_k = i_nhead / kargs.nhead_ratio_qk;

            const QDataType* q_ptr =
                reinterpret_cast<const QDataType*>(kargs.q_ptr) +
                (static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q + batch_offset_q) /
                    numeric_traits<QDataType>::PackedSize;
            const KDataType* k_ptr =
                reinterpret_cast<const KDataType*>(kargs.k_ptr) +
                (static_cast<long_index_t>(i_nhead_k) * kargs.nhead_stride_k + batch_offset_k) /
                    numeric_traits<KDataType>::PackedSize;
            const VDataType* v_ptr =
                reinterpret_cast<const VDataType*>(kargs.v_ptr) +
                (static_cast<long_index_t>(i_nhead_k) * kargs.nhead_stride_v + batch_offset_v) /
                    numeric_traits<VDataType>::PackedSize;
            ODataType* o_ptr = reinterpret_cast<ODataType*>(kargs.o_ptr) +
                               static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_o +
                               batch_offset_o;

            constexpr index_t kQKHeaddimToUse = detail::is_naive_hdim_load_v<FmhaPipeline>
                                                    ? FmhaPipeline::kQKHeaddim
                                                    : FmhaPipeline::kSubQKHeaddim;

            // Q/K/V DRAM and DRAM window
            const auto q_dram = [&]() {
                const auto q_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    q_ptr,
                    make_tuple(kargs.seqlen_q, kargs.hdim_q),
                    make_tuple(kargs.stride_q, 1),
                    number<FmhaPipeline::kAlignmentQ>{},
                    number<1>{});
                if constexpr(FmhaPipeline::kQLoadOnce)
                {
                    return pad_tensor_view(
                        q_dram_naive,
                        make_tuple(number<FmhaPipeline::kM0>{}, number<kQKHeaddimToUse>{}),
                        sequence<kPadSeqLenQ, kPadHeadDimQ>{});
                }
                else
                {
                    return pad_tensor_view(
                        q_dram_naive,
                        make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kK0>{}),
                        sequence<kPadSeqLenQ, kPadHeadDimQ>{});
                }
            }();
            const auto k_dram = [&]() {
                const auto k_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    k_ptr,
                    make_tuple(kargs.seqlen_k, kargs.hdim_q),
                    make_tuple(kargs.stride_k, 1),
                    number<FmhaPipeline::kAlignmentK>{},
                    number<1>{});

                constexpr bool kPadSeqLenK_ = kUseAsyncCopy ? kPadSeqLenK : false;

                if constexpr(detail::is_n0loop_pipeline_v<FmhaPipeline>)
                {
                    return pad_tensor_view(
                        k_dram_naive,
                        make_tuple(number<FmhaPipeline::kN0Sub>{}, number<kQKHeaddimToUse>{}),
                        sequence<kPadSeqLenK_, kPadHeadDimQ>{});
                }
                else
                {
                    return pad_tensor_view(
                        k_dram_naive,
                        make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0>{}),
                        sequence<kPadSeqLenK_, kPadHeadDimQ>{});
                }
            }();
            const auto v_dram = [&]() {
                if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
                {
                    const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                        v_ptr,
                        make_tuple(kargs.seqlen_k, kargs.hdim_v),
                        make_tuple(kargs.stride_v, 1),
                        number<FmhaPipeline::kAlignmentV>{},
                        number<1>{});

                    if constexpr(!kUseTrLoad)
                    {
                        const auto v_dram_transposed = transform_tensor_view(
                            v_dram_naive,
                            make_tuple(make_pass_through_transform(kargs.hdim_v),
                                       make_pass_through_transform(kargs.seqlen_k)),
                            make_tuple(sequence<1>{}, sequence<0>{}),
                            make_tuple(sequence<0>{}, sequence<1>{}));

                        constexpr bool kPadSeqLenK_ = kUseAsyncCopy ? kPadSeqLenK : false;

                        return pad_tensor_view(
                            v_dram_transposed,
                            make_tuple(number<FmhaPipeline::kN1>{}, number<FmhaPipeline::kK1>{}),
                            sequence<kPadHeadDimV, kPadSeqLenK_>{});
                    }
                    else
                    {
                        return pad_tensor_view(
                            v_dram_naive,
                            make_tuple(number<FmhaPipeline::kK1>{}, number<FmhaPipeline::kN1>{}),
                            sequence<false, kPadHeadDimV>{});
                    };
                }
                else
                {
                    const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                        v_ptr,
                        make_tuple(kargs.hdim_v, kargs.seqlen_k),
                        make_tuple(kargs.stride_v, 1),
                        number<FmhaPipeline::kAlignmentV>{},
                        number<1>{});

                    constexpr bool kPadHeadDimV_ = kUseAsyncCopy ? kPadHeadDimV : false;
                    return pad_tensor_view(
                        v_dram_naive,
                        make_tuple(number<FmhaPipeline::kN1>{}, number<FmhaPipeline::kK1>{}),
                        sequence<kPadHeadDimV_, kPadSeqLenK>{});
                }
            }();

            auto q_dram_window = make_tile_window(
                q_dram,
                [&]() {
                    if constexpr(FmhaPipeline::kQLoadOnce)
                        return make_tuple(number<FmhaPipeline::kM0>{}, number<kQKHeaddimToUse>{});
                    else
                        return make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kK0>{});
                }(),
                {i_m0, 0});

            auto k_dram_window = [&]() {
                if constexpr(detail::is_n0loop_pipeline_v<FmhaPipeline>)
                {
                    return make_tile_window(
                        k_dram,
                        make_tuple(number<FmhaPipeline::kN0Sub>{}, number<kQKHeaddimToUse>{}),
                        {0, 0});
                }
                else
                {
                    return make_tile_window(
                        k_dram,
                        make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0>{}),
                        {0, 0});
                }
            }();

            auto v_dram_window = make_tile_window(
                v_dram,
                make_tuple(number<FmhaPipeline::kN1>{}, number<FmhaPipeline::kK1>{}),
                {i_n1, 0});
            /// FIXME: Before C++20, capturing structured binding variables are not supported.
            /// Remove following copy capture of the 'i_nhead' if in C++20
            const auto bias_dram_window = [&, i_nhead_ = i_nhead]() {
                constexpr auto bias_dram_window_lengths =
                    make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN0>{});
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                {
                    const BiasDataType* bias_ptr =
                        reinterpret_cast<const BiasDataType*>(kargs.bias_ptr) +
                        static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_bias +
                        batch_offset_bias;

                    const auto bias_dram = [&]() {
                        const auto bias_dram_naive =
                            make_naive_tensor_view<address_space_enum::global>(
                                bias_ptr,
                                make_tuple(kargs.seqlen_q, kargs.seqlen_k),
                                make_tuple(kargs.stride_bias, 1),
                                number<FmhaPipeline::kAlignmentBias>{},
                                number<1>{});

                        return pad_tensor_view(bias_dram_naive,
                                               bias_dram_window_lengths,
                                               sequence<kPadSeqLenQ, kPadSeqLenK>{});
                    }();

                    return make_tile_window(bias_dram, bias_dram_window_lengths, {i_m0, 0});
                }
                else
                {
                    return make_null_tile_window(bias_dram_window_lengths);
                }
            }();

            // lse
            auto lse_dram_window = [&, i_nhead_ = i_nhead]() {
                constexpr auto lse_dram_window_lengths = make_tuple(number<FmhaPipeline::kM0>{});
                if constexpr(kStoreLSE)
                {
                    LSEDataType* lse_ptr =
                        reinterpret_cast<LSEDataType*>(kargs.lse_ptr) +
                        static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_lse +
                        batch_offset_lse;

                    const auto lse_dram = [&]() {
                        const auto lse_dram_naive =
                            make_naive_tensor_view<address_space_enum::global>(
                                lse_ptr,
                                make_tuple(kargs.seqlen_q),
                                make_tuple(1),
                                number<1>{},
                                number<1>{});

                        return pad_tensor_view(
                            lse_dram_naive, lse_dram_window_lengths, sequence<kPadSeqLenQ>{});
                    }();

                    return make_tile_window(lse_dram, lse_dram_window_lengths, {i_m0});
                }
                else
                {
                    return make_null_tile_window(lse_dram_window_lengths);
                }
            }();

            auto dropout = [&, i_nhead_ = i_nhead, i_batch_ = i_batch]() {
                if constexpr(kHasDropout)
                {
                    const auto num_head_q_total =
                        (kargs.num_head_q_total > 0 ? kargs.num_head_q_total : kargs.num_head_q);
                    const auto i_head_global = kargs.head_start + i_nhead_;
                    return BlockDropout{i_batch_,
                                        i_head_global,
                                        num_head_q_total,
                                        kargs.is_drop_seed_offset_from_host ? kargs.drop_seed.val
                                                                            : *kargs.drop_seed.ptr,
                                        kargs.is_drop_seed_offset_from_host
                                            ? kargs.drop_offset.val
                                            : *kargs.drop_offset.ptr,
                                        kargs.rp_undrop,
                                        kargs.p_undrop_in_uint8_t,
                                        kargs.is_store_randval};
                }
                else
                {
                    return NullBlockDropout{};
                };
            }();

            auto randval_dram_window = [&, i_nhead_ = i_nhead]() {
                constexpr auto randval_dram_window_lengths =
                    make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN0>{});
                if constexpr(kHasDropout)
                {
                    RandValOutputDataType* rand_val_ptr =
                        reinterpret_cast<RandValOutputDataType*>(kargs.rand_val_ptr) +
                        static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_randval +
                        batch_offset_randval;

                    const auto randval_dram = [&]() {
                        const auto randval_dram_naive =
                            make_naive_tensor_view<address_space_enum::global>(
                                rand_val_ptr,
                                make_tuple(kargs.seqlen_q, kargs.seqlen_k),
                                make_tuple(kargs.stride_randval, 1),
                                number<FmhaPipeline::kAlignmentRandVal>{},
                                number<1>{});

                        return pad_tensor_view(randval_dram_naive,
                                               randval_dram_window_lengths,
                                               sequence<kPadSeqLenQ, kPadSeqLenK>{});
                    }();

                    return make_tile_window(randval_dram, randval_dram_window_lengths, {i_m0, 0});
                }
                else
                {
                    return make_null_tile_window(randval_dram_window_lengths);
                }
            }();

            FmhaMask mask = [&]() {
                if constexpr(kHasMask)
                    return ck_tile::make_generic_attention_mask_from_lr_window<FmhaMask>(
                        kargs.window_size_left,
                        kargs.window_size_right,
                        kargs.sink_size,
                        kargs.seqlen_q,
                        kargs.seqlen_k,
                        kargs.mask_type == GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
                else
                    return FmhaMask{kargs.seqlen_q, kargs.seqlen_k};
            }();

            // WA i_batch capture structure binding before c++20
            auto position_encoding = [&, i_batch_ = i_batch, i_nhead_ = i_nhead]() {
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
                {
                    // data loading, shared by entire wg
                    // TODO: how to use s_read?
                    SaccDataType slope =
                        *(reinterpret_cast<const SaccDataType*>(kargs.alibi_slope_ptr) +
                          i_batch_ * kargs.alibi_slope_stride + i_nhead_);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                    if constexpr(!detail::ignore_fast_exp2_v<FmhaPipeline>)
                    {
                        slope *= ck_tile::log2e_v<>;
                    }
#endif
                    if constexpr(kHasMask)
                    {
                        return make_alibi_from_lr_mask<SaccDataType, true>(slope,
                                                                           kargs.window_size_left,
                                                                           kargs.window_size_right,
                                                                           kargs.seqlen_q,
                                                                           kargs.seqlen_k,
                                                                           kargs.mask_type);
                    }
                    else
                    {
                        return Alibi<SaccDataType, true>{
                            slope, kargs.seqlen_q, kargs.seqlen_k, AlibiMode::FROM_BOTTOM_RIGHT};
                    }
                }
                else
                {
                    return EmptyPositionEncoding<SaccDataType>{};
                }
            }();

            AttentionVariant variant;
            const auto variant_params = [&] {
                // This branch is never qr_tdm, so we can use 0,0 as perhead is not supported and
                // blockscale is handled elementwise in the pipeline.
                const float scale_s = GetSoftmaxScale(kargs, 0, 0, i_m0);

                if constexpr(kHasLogitsSoftCap)
                {
                    return ck_tile::LogitsSoftCapParams<FmhaMask, CK_TILE_FMHA_FWD_FAST_EXP2>{
                        mask, scale_s, kargs.logits_soft_cap, kargs.logits_soft_cap_rcp};
                }
                else
                {
                    return ck_tile::StandardAttentionParams<FmhaMask>{mask, scale_s};
                }
            }();

            BlockIndices block_indices{i_batch, i_nhead, i_nhead_k};
            constexpr bool kPassHdimTailArgs = [] {
                if constexpr(ck_tile::is_detected<has_hdim_tail_args, FmhaPipeline>::value)
                    return static_cast<bool>(FmhaPipeline::kUseHdimTailArgs);
                else
                    return false;
            }();
            auto invoke_fmha_pipeline = [&](auto&&... args) -> decltype(auto) {
                if constexpr(kPassHdimTailArgs)
                {
                    const ck_tile::index_t valid_k0_loops =
                        ck_tile::integer_divide_ceil(kargs.hdim_q, FmhaPipeline::kK0);
                    const ck_tile::index_t valid_last_k0_length =
                        kargs.hdim_q - (valid_k0_loops - 1) * FmhaPipeline::kK0;
                    const ck_tile::index_t valid_n1_length = [&]() {
                        const ck_tile::index_t remaining_n1 = kargs.hdim_v - i_n1;
                        return ck_tile::min(remaining_n1,
                                            static_cast<ck_tile::index_t>(FmhaPipeline::kN1));
                    }();
                    return FmhaPipeline{}(static_cast<decltype(args)&&>(args)...,
                                          sink_value,
                                          valid_k0_loops,
                                          valid_last_k0_length,
                                          valid_n1_length);
                }
                else
                {
                    return FmhaPipeline{}(static_cast<decltype(args)&&>(args)..., sink_value);
                }
            };

            auto o_acc_tile = [&, i_nhead_ = i_nhead, i_nhead_k_ = i_nhead_k]() {
                if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::PERTENSOR)
                {
                    // TODO - move global load of descale to pipeline
                    float v_descale = *(reinterpret_cast<const float*>(kargs.v_descale_ptr));

                    float scale_p =
                        ck_tile::type_convert<float>(ck_tile::numeric<PDataType>::max());
                    float scale_o = v_descale / scale_p;

                    auto o_acc_element_func = [&]() {
                        if constexpr(std::is_same_v<ODataType, ck_tile::fp8_t>)
                            return make_composes(
                                ck_tile::saturates<ck_tile::fp8_t>{},
                                ck_tile::scales<remove_cvref_t<decltype(scale_o)>>{scale_o});
                        else
                            return ck_tile::scales<remove_cvref_t<decltype(scale_o)>>{scale_o};
                    }();
                    return invoke_fmha_pipeline(q_dram_window,
                                                identity{}, // q_element_func
                                                k_dram_window,
                                                identity{}, // k_element_func
                                                v_dram_window,
                                                identity{}, // v_element_func
                                                bias_dram_window,
                                                identity{}, // bias_element_func
                                                randval_dram_window,
                                                lse_dram_window,
                                                identity{}, // lse_element_func
                                                identity{}, // s_acc_element_func
                                                scales<remove_cvref_t<decltype(scale_p)>>{
                                                    scale_p},       // p_compute_element_func
                                                o_acc_element_func, // o_acc_element_func
                                                mask,
                                                position_encoding,
                                                variant_params.sm_scale,
                                                variant,
                                                variant_params,
                                                block_indices,
                                                smem_ptr,
                                                dropout,
                                                nullptr,
                                                nullptr,
                                                1,
                                                make_null_tile_window(make_tuple()),
                                                make_null_tile_window(make_tuple()),
                                                make_null_tile_window(make_tuple()));
                }
                else if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE)
                {
                    const float* q_descale_ptr =
                        reinterpret_cast<const float*>(kargs.q_descale_ptr) +
                        static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_q_descale +
                        batch_offset_q_descale;
                    const float* k_descale_ptr =
                        reinterpret_cast<const float*>(kargs.k_descale_ptr) +
                        static_cast<long_index_t>(i_nhead_ / kargs.nhead_ratio_qk) *
                            kargs.nhead_stride_k_descale +
                        batch_offset_k_descale;
                    const float* v_descale_ptr =
                        reinterpret_cast<const float*>(kargs.v_descale_ptr) +
                        static_cast<long_index_t>(i_nhead_ / kargs.nhead_ratio_qk) *
                            kargs.nhead_stride_v_descale +
                        batch_offset_v_descale;

                    size_t idx      = i_m0 / kargs.block_scale_size_q;
                    float q_descale = q_descale_ptr[idx];
                    // BLOCKSCALE: P is scaled in exp2(x+shift) where shift=7 or 8
                    // Both P and rowsum are scaled by 2^shift, canceling in normalization
                    // No additional scaling needed in p_compute_element_func or o_acc_element_func

                    return invoke_fmha_pipeline(
                        q_dram_window,
                        identity{}, // q_element_func
                        k_dram_window,
                        identity{}, // k_element_func
                        v_dram_window,
                        identity{}, // v_element_func
                        bias_dram_window,
                        identity{}, // bias_element_func
                        randval_dram_window,
                        lse_dram_window,
                        identity{},               // lse_element_func
                        scales<float>(q_descale), // s_acc_element_func
                        identity{}, // p_compute_element_func - No scaling (done in exp2)
                        identity{}, // o_acc_element_func - No dequant needed (canceled by rowsum)
                        mask,
                        position_encoding,
                        kargs.scale_s,
                        variant,
                        variant_params,
                        block_indices,
                        smem_ptr,
                        dropout,
                        k_descale_ptr,
                        v_descale_ptr,
                        kargs.block_scale_size_kv,
                        make_null_tile_window(make_tuple()),
                        make_null_tile_window(make_tuple()),
                        make_null_tile_window(make_tuple()));
                }
                else if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::MX)
                {
                    using QScaleDataType = typename FmhaPipeline::QScaleDataType;
                    using KScaleDataType = typename FmhaPipeline::KScaleDataType;
                    using VScaleDataType = typename FmhaPipeline::VScaleDataType;

                    constexpr ck_tile::index_t kQKScaleGranularity =
                        FmhaPipeline::kQKScaleGranularity;
                    constexpr ck_tile::index_t kVScaleGranularity =
                        FmhaPipeline::kVScaleGranularity;

                    const QScaleDataType* q_descale_ptr =
                        reinterpret_cast<const QScaleDataType*>(kargs.q_descale_ptr) +
                        static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_q_descale +
                        batch_offset_q_descale;
                    const KScaleDataType* k_descale_ptr =
                        reinterpret_cast<const KScaleDataType*>(kargs.k_descale_ptr) +
                        static_cast<long_index_t>(i_nhead_k_) * kargs.nhead_stride_k_descale +
                        batch_offset_k_descale;
                    const VScaleDataType* v_descale_ptr =
                        reinterpret_cast<const VScaleDataType*>(kargs.v_descale_ptr) +
                        static_cast<long_index_t>(i_nhead_k_) * kargs.nhead_stride_v_descale +
                        batch_offset_v_descale;

                    const ck_tile::index_t hdim_q_scale =
                        ck_tile::integer_divide_ceil(kargs.hdim_q, kQKScaleGranularity);
                    const ck_tile::index_t seqlen_v_scale =
                        ck_tile::integer_divide_ceil(kargs.seqlen_k, kVScaleGranularity);

                    // Custom invalid_element_value is required for e8m0_t scales because
                    // the default (numeric<e8m0_t>>::zero()) is NaN
                    const auto q_scale_dram = [&]() {
                        auto desc =
                            make_naive_tensor_descriptor(make_tuple(kargs.seqlen_q, hdim_q_scale),
                                                         make_tuple(kargs.stride_q_descale, 1),
                                                         number<1>{},
                                                         number<1>{});
                        auto buffer_view = make_buffer_view<address_space_enum::global>(
                            q_descale_ptr,
                            desc.get_element_space_size(),
                            type_convert<QScaleDataType>(1.0f));
                        return pad_tensor_view(
                            tensor_view<decltype(buffer_view), decltype(desc)>{buffer_view, desc},
                            make_tuple(
                                number<FmhaPipeline::kM0>{},
                                number<(FmhaPipeline::kQLoadOnce ? FmhaPipeline::kSubQKHeaddim
                                                                 : FmhaPipeline::kK0) /
                                       kQKScaleGranularity>{}),
                            sequence<kPadSeqLenQ, kPadHeadDimQ>{});
                    }();
                    const auto k_scale_dram = [&]() {
                        auto desc =
                            make_naive_tensor_descriptor(make_tuple(kargs.seqlen_k, hdim_q_scale),
                                                         make_tuple(kargs.stride_k_descale, 1),
                                                         number<1>{},
                                                         number<1>{});
                        auto buffer_view = make_buffer_view<address_space_enum::global>(
                            k_descale_ptr,
                            desc.get_element_space_size(),
                            type_convert<KScaleDataType>(1.0f));
                        return pad_tensor_view(
                            tensor_view<decltype(buffer_view), decltype(desc)>{buffer_view, desc},
                            make_tuple(number<FmhaPipeline::kN0>{},
                                       number<FmhaPipeline::kK0 / kQKScaleGranularity>{}),
                            sequence<false, kPadHeadDimQ>{});
                    }();
                    const auto v_scale_dram = [&]() {
                        static_assert(
                            std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::ColumnMajor>);
                        auto desc =
                            make_naive_tensor_descriptor(make_tuple(kargs.hdim_v, seqlen_v_scale),
                                                         make_tuple(kargs.stride_v_descale, 1),
                                                         number<1>{},
                                                         number<1>{});
                        auto buffer_view = make_buffer_view<address_space_enum::global>(
                            v_descale_ptr,
                            desc.get_element_space_size(),
                            type_convert<VScaleDataType>(1.0f));
                        return pad_tensor_view(
                            tensor_view<decltype(buffer_view), decltype(desc)>{buffer_view, desc},
                            make_tuple(number<FmhaPipeline::kN1>{},
                                       number<FmhaPipeline::kK1 / kVScaleGranularity>{}),
                            sequence<false, kPadSeqLenK>{});
                    }();

                    auto q_scale_dram_window = make_tile_window(
                        q_scale_dram,
                        make_tuple(number<FmhaPipeline::kM0>{},
                                   number<(FmhaPipeline::kQLoadOnce ? FmhaPipeline::kSubQKHeaddim
                                                                    : FmhaPipeline::kK0) /
                                          kQKScaleGranularity>{}),
                        {i_m0, 0});
                    auto k_scale_dram_window = make_tile_window(
                        k_scale_dram,
                        make_tuple(number<FmhaPipeline::kN0>{},
                                   number<FmhaPipeline::kK0 / kQKScaleGranularity>{}),
                        {0, 0});
                    auto v_scale_dram_window = make_tile_window(
                        v_scale_dram,
                        make_tuple(number<FmhaPipeline::kN1>{},
                                   number<FmhaPipeline::kK1 / kVScaleGranularity>{}),
                        {i_n1, 0});

                    return invoke_fmha_pipeline(q_dram_window,
                                                identity{}, // q_element_func
                                                k_dram_window,
                                                identity{}, // k_element_func
                                                v_dram_window,
                                                identity{}, // v_element_func
                                                bias_dram_window,
                                                identity{}, // bias_element_func
                                                randval_dram_window,
                                                lse_dram_window,
                                                identity{}, // lse_element_func
                                                identity{}, // s_acc_element_func
                                                identity{}, // p_compute_element_func
                                                identity{}, // o_acc_element_func
                                                mask,
                                                position_encoding,
                                                kargs.scale_s,
                                                variant,
                                                variant_params,
                                                block_indices,
                                                smem_ptr,
                                                dropout,
                                                nullptr,
                                                nullptr,
                                                1,
                                                q_scale_dram_window,
                                                k_scale_dram_window,
                                                v_scale_dram_window);
                }
                else
                {
                    return invoke_fmha_pipeline(q_dram_window,
                                                k_dram_window,
                                                v_dram_window,
                                                bias_dram_window,
                                                randval_dram_window,
                                                lse_dram_window,
                                                mask,
                                                position_encoding,
                                                variant_params.sm_scale,
                                                variant,
                                                variant_params,
                                                block_indices,
                                                smem_ptr,
                                                dropout);
                }
            }();

            // O DRAM and O DRAM window
            auto o_dram = [&]() {
                const auto o_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    o_ptr,
                    make_tuple(kargs.seqlen_q, kargs.hdim_v),
                    make_tuple(kargs.stride_o, 1),
                    number<FmhaPipeline::kAlignmentO>{},
                    number<1>{});

                return pad_tensor_view(
                    o_dram_naive,
                    make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN1>{}),
                    sequence<kPadSeqLenQ, kPadHeadDimV>{});
            }();

            auto o_dram_window = make_tile_window(
                o_dram,
                make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN1>{}),
                {i_m0, i_n1});

            EpiloguePipeline{}(o_dram_window, o_acc_tile, nullptr);
        }
        else
        {
            // TODO: Refine the logical here.
            // In Decode case
            //     1. we don't expect KV data reused by different ThreadGroups, bypass the cache
            //     2. limit the LDS usage, as we want higher occupancy
            // In Prefill case
            //     1. we expect KV data reused by different ThreadGroups, use cache
            //     2. use more LDS, as we want better memory latency hiding
            // If SplitKV off, we don't expect Q data reused by different ThreadGroups, bypass the
            // cache
            constexpr bool PrefillCase =
                FmhaPipeline::kM0 > 64 && FmhaPipeline::BlockFmhaShape::kQKHeaddim < 256;
            // qr_tdm_fp8_sched takes the fixed LDS arena path, one-wavefront decode tiles included.
            constexpr bool kPrefillLdsPath = PrefillCase || kPipelineName == "qr_tdm_fp8_sched";
            // divide problem
            const auto [i_tile_m, i_tile_n, i_nhead, i_batch] = GetTileIndex(kargs);
            const index_t i_m0                                = i_tile_m * FmhaPipeline::kM0;
            const index_t i_n1                                = i_tile_n * FmhaPipeline::kN1;

            long_index_t batch_offset_q    = 0;
            long_index_t batch_offset_k    = 0; // unused for paged-kvcache
            long_index_t batch_offset_v    = 0; // unused for paged-kvcache
            long_index_t batch_offset_bias = 0;
            long_index_t batch_offset_lse  = 0;
            long_index_t batch_offset_o    = 0;
            // index_t kv_l2p_offset =
            //     0; // logical-to-physical offset of seqlen_k coordinate. only used for
            //     paged-kvcache

            if constexpr(kIsGroupMode)
            {
                // get starting offset for each batch - use seqstart_q_ptr/seqstart_k_ptr for
                // physical starts
                const long_index_t query_start = kargs.seqstart_q_ptr[i_batch];
                const long_index_t key_start   = kargs.seqstart_k_ptr[i_batch];

                batch_offset_q = query_start * kargs.stride_q;
                batch_offset_k = key_start * kargs.stride_k;
                if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
                {
                    batch_offset_v = key_start * kargs.stride_v;
                }
                else
                {
                    // col-major V: offset along seqlen dimension is scalar index
                    batch_offset_v = key_start;
                }
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                {
                    batch_offset_bias = query_start * kargs.stride_bias;
                }

                // LSE layout is [nhead, total_seqlen] following the physical layout for Q/O
                batch_offset_lse = query_start;
                batch_offset_o   = query_start * kargs.stride_o;

                // get real # queries & # keys under group mode
                if(kargs.seqlen_q_ptr != nullptr)
                {
                    kargs.seqlen_q = kargs.seqlen_q_ptr[i_batch];
                }
                else if(kargs.cu_seqlen_q_ptr != nullptr)
                {
                    kargs.seqlen_q =
                        kargs.cu_seqlen_q_ptr[i_batch + 1] - kargs.cu_seqlen_q_ptr[i_batch];
                }
                else
                {
                    kargs.seqlen_q =
                        kargs.seqstart_q_ptr[i_batch + 1] - kargs.seqstart_q_ptr[i_batch];
                }

                if constexpr(kSkipMinSeqlenQ)
                {
                    if(kargs.seqlen_q <= kargs.min_seqlen_q)
                        return;
                }

                // # of required blocks is different in each groups, terminate unnecessary blocks
                // earlier
                if(kargs.seqlen_q <= i_m0)
                {
                    return;
                }

                if(kargs.seqlen_k_ptr != nullptr)
                {
                    kargs.seqlen_k = kargs.seqlen_k_ptr[i_batch];
                }
                else if(kargs.cu_seqlen_k_ptr != nullptr)
                {
                    kargs.seqlen_k =
                        kargs.cu_seqlen_k_ptr[i_batch + 1] - kargs.cu_seqlen_k_ptr[i_batch];
                }
                else
                {
                    kargs.seqlen_k =
                        kargs.seqstart_k_ptr[i_batch + 1] - kargs.seqstart_k_ptr[i_batch];
                }
            }
            else
            {
                batch_offset_q = static_cast<long_index_t>(i_batch) * kargs.batch_stride_q;
                batch_offset_k = static_cast<long_index_t>(i_batch) * kargs.batch_stride_k;
                batch_offset_v = static_cast<long_index_t>(i_batch) * kargs.batch_stride_v;
                if constexpr(kStoreLSE)
                {
                    batch_offset_lse = static_cast<long_index_t>(i_batch) * kargs.batch_stride_lse;
                }
                batch_offset_o = static_cast<long_index_t>(i_batch) * kargs.batch_stride_o;

                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                {
                    batch_offset_bias =
                        static_cast<long_index_t>(i_batch) * kargs.batch_stride_bias;
                }

                // If cumulative seqlen pointers are provided, override per-batch effective lengths
                if(kargs.cu_seqlen_q_ptr != nullptr)
                {
                    kargs.seqlen_q =
                        kargs.cu_seqlen_q_ptr[i_batch + 1] - kargs.cu_seqlen_q_ptr[i_batch];

                    // the grid covers the padded seqlen_q; return early for the blocks
                    // past this batch's effective length, as group mode does
                    if(kargs.seqlen_q <= i_m0)
                    {
                        return;
                    }
                }
                if(kargs.cu_seqlen_k_ptr != nullptr)
                {
                    kargs.seqlen_k =
                        kargs.cu_seqlen_k_ptr[i_batch + 1] - kargs.cu_seqlen_k_ptr[i_batch];
                }
            }

            // for simplicity, batch stride we just modify the pointer
            const index_t i_nhead_k =
                kMergeNumHeadGroupsSeqLenQ ? i_nhead : i_nhead / kargs.nhead_ratio_qk;
            // Merged: the workgroup owns query heads [i_nhead_q, +nhead_ratio_qk) as its rows.
            const index_t i_nhead_q =
                kMergeNumHeadGroupsSeqLenQ ? i_nhead * kargs.nhead_ratio_qk : i_nhead;
            const index_t num_rows_q =
                kMergeNumHeadGroupsSeqLenQ ? kargs.seqlen_q * kargs.nhead_ratio_qk : kargs.seqlen_q;
            // With one query per head the rows step by head; otherwise heads are seqlen_q rows.
            const bool rows_by_head    = kMergeNumHeadGroupsSeqLenQ && kargs.seqlen_q == 1;
            const index_t row_stride_q = rows_by_head ? kargs.nhead_stride_q : kargs.stride_q;
            const index_t row_stride_o = rows_by_head ? kargs.nhead_stride_o : kargs.stride_o;

            const QDataType* q_ptr =
                reinterpret_cast<const QDataType*>(kargs.q_ptr) +
                (static_cast<long_index_t>(i_nhead_q) * kargs.nhead_stride_q + batch_offset_q) /
                    numeric_traits<QDataType>::PackedSize;
            const KDataType* k_ptr =
                reinterpret_cast<const KDataType*>(kargs.k_ptr) +
                (static_cast<long_index_t>(i_nhead_k) * kargs.nhead_stride_k + batch_offset_k) /
                    numeric_traits<KDataType>::PackedSize;
            const VDataType* v_ptr =
                reinterpret_cast<const VDataType*>(kargs.v_ptr) +
                (static_cast<long_index_t>(i_nhead_k) * kargs.nhead_stride_v + batch_offset_v) /
                    numeric_traits<VDataType>::PackedSize;

            ODataType* o_ptr = reinterpret_cast<ODataType*>(kargs.o_ptr) +
                               static_cast<long_index_t>(i_nhead_q) * kargs.nhead_stride_o +
                               batch_offset_o;

            // Q/K/V DRAM and DRAM window
            const auto q_dram = [&] {
                const auto q_dram_naive = [&] {
                    {
                        return make_naive_tensor_view<address_space_enum::global,
                                                      memory_operation_enum::set,
                                                      amd_buffer_coherence_enum::SYSTEM_NT1>(
                            q_ptr,
                            make_tuple(num_rows_q, kargs.hdim_q),
                            make_tuple(row_stride_q, 1),
                            number<FmhaPipeline::kAlignmentQ>{},
                            number<1>{});
                    }
                }();

                if constexpr(FmhaPipeline::kQLoadOnce)
                {
                    const auto seqlen_q   = kargs.seqlen_q;
                    const auto q_dram_pad = pad_tensor_view(
                        q_dram_naive,
                        make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kK0>{}),
                        sequence<false, kPadHeadDimQ>{});

                    // TDM box-major DMA cannot honor a software XOR layout
                    // on the dram side, so affine-TDM pipelines must consume
                    // an affine pad-only view. Bypass the unmerge/xor/merge_v3
                    // chain below: that chain (i) is dead code for TDM (TDM
                    // box write can't produce XOR'd LDS -- see
                    // qr_ks_vs_tdm_policy.hpp), and (ii) trips
                    // get_cached_global_strides because
                    // calculate_offset(unit_vec) walks the full chain and
                    // returns an XOR-polluted offset (e.g. stride_q + alignQ
                    // = 136 instead of 128) which then drives the box-copy
                    // to read garbage rows. Returning the affine naive view
                    // (no head-dim pad) keeps get_lengths()[hdim] at the true
                    // head-dim so the TDM box clamp zero-fills the OOB tail.
                    if constexpr(kUsesTdmAffineDramPath)
                    {
                        return q_dram_naive;
                    }
                    else
                    {
#if CK_TILE_FMHA_HANDLE_XOR_LENGTH_FOLD
                        constexpr index_t LDSLayerSize =
                            256 * numeric_traits<QDataType>::PackedSize / sizeof(QDataType);
                        constexpr index_t XorLengthFold = LDSLayerSize / (FmhaPipeline::kQKHeaddim);

                        if constexpr(XorLengthFold > 1)
                        {
                            const auto q_dram_unmerged = transform_tensor_view(
                                q_dram_pad,
                                make_tuple(make_unmerge_transform(
                                               make_tuple(seqlen_q / XorLengthFold, XorLengthFold)),
                                           make_pass_through_transform(
                                               number<FmhaPipeline::kQKHeaddim>{})),
                                make_tuple(sequence<0>{}, sequence<1>{}),
                                make_tuple(sequence<0, 1>{}, sequence<2>{}));

                            const auto q_dram_merged = transform_tensor_view(
                                q_dram_unmerged,
                                make_tuple(make_pass_through_transform(seqlen_q / XorLengthFold),
                                           make_merge_transform_v3_division_mod(make_tuple(
                                               XorLengthFold, number<FmhaPipeline::kQKHeaddim>{}))),
                                make_tuple(sequence<0>{}, sequence<1, 2>{}),
                                make_tuple(sequence<0>{}, sequence<1>{}));

                            const auto q_dram_unmerged_xor = transform_tensor_view(
                                q_dram_merged,
                                make_tuple(make_pass_through_transform(seqlen_q / XorLengthFold),
                                           make_unmerge_transform(make_tuple(
                                               number<LDSLayerSize / FmhaPipeline::kAlignmentQ>{},
                                               number<FmhaPipeline::kAlignmentQ>{}))),
                                make_tuple(sequence<0>{}, sequence<1>{}),
                                make_tuple(sequence<0>{}, sequence<1, 2>{}));

                            const auto q_dram_permuted = transform_tensor_view(
                                q_dram_unmerged_xor,
                                make_tuple(make_xor_transform(make_tuple(
                                               seqlen_q / XorLengthFold,
                                               number<LDSLayerSize / FmhaPipeline::kAlignmentQ>{})),
                                           make_pass_through_transform(
                                               number<FmhaPipeline::kAlignmentQ>{})),
                                make_tuple(sequence<0, 1>{}, sequence<2>{}),
                                make_tuple(sequence<0, 1>{}, sequence<2>{}));

                            const auto q_dram_tmp = transform_tensor_view(
                                q_dram_permuted,
                                make_tuple(make_pass_through_transform(seqlen_q / XorLengthFold),
                                           make_unmerge_transform(
                                               make_tuple(number<XorLengthFold>{},
                                                          number<FmhaPipeline::kQKHeaddim /
                                                                 FmhaPipeline::kAlignmentQ>{})),
                                           make_pass_through_transform(
                                               number<FmhaPipeline::kAlignmentQ>{})),
                                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                                make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

                            return transform_tensor_view(
                                q_dram_tmp,
                                make_tuple(make_merge_transform_v3_division_mod(make_tuple(
                                               seqlen_q / XorLengthFold, number<XorLengthFold>{})),
                                           make_merge_transform_v3_division_mod(
                                               make_tuple(number<FmhaPipeline::kQKHeaddim /
                                                                 FmhaPipeline::kAlignmentQ>{},
                                                          number<FmhaPipeline::kAlignmentQ>{}))),
                                make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
                                make_tuple(sequence<0>{}, sequence<1>{}));
                        }
                        else
#endif // CK_TILE_FMHA_HANDLE_XOR_LENGTH_FOLD
                        {
                            const auto q_dram_unmerged = transform_tensor_view(
                                q_dram_pad,
                                make_tuple(make_pass_through_transform(seqlen_q),
                                           make_unmerge_transform(
                                               make_tuple(number<FmhaPipeline::kQKHeaddim /
                                                                 FmhaPipeline::kAlignmentQ>{},
                                                          number<FmhaPipeline::kAlignmentQ>{}))),
                                make_tuple(sequence<0>{}, sequence<1>{}),
                                make_tuple(sequence<0>{}, sequence<1, 2>{}));

                            const auto q_dram_permuted = transform_tensor_view(
                                q_dram_unmerged,
                                make_tuple(make_xor_transform(
                                               make_tuple(seqlen_q,
                                                          number<FmhaPipeline::kQKHeaddim /
                                                                 FmhaPipeline::kAlignmentQ>{})),
                                           make_pass_through_transform(
                                               number<FmhaPipeline::kAlignmentQ>{})),
                                make_tuple(sequence<0, 1>{}, sequence<2>{}),
                                make_tuple(sequence<0, 1>{}, sequence<2>{}));

                            return transform_tensor_view(
                                q_dram_permuted,
                                make_tuple(make_pass_through_transform(seqlen_q),
                                           make_merge_transform_v3_division_mod(
                                               make_tuple(number<FmhaPipeline::kQKHeaddim /
                                                                 FmhaPipeline::kAlignmentQ>{},
                                                          number<FmhaPipeline::kAlignmentQ>{}))),
                                make_tuple(sequence<0>{}, sequence<1, 2>{}),
                                make_tuple(sequence<0>{}, sequence<1>{}));
                        }
                    } // end else (qr_tdm dispatch above)
                }
                else
                {
                    return pad_tensor_view(
                        q_dram_naive,
                        make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kK0>{}),
                        sequence<false, kPadHeadDimQ>{});
                }
            }();

            const auto make_k_dram = [&](const KDataType* data, index_t height) {
                const auto k_dram_naive = make_naive_tensor_view<address_space_enum::global,
                                                                 memory_operation_enum::set,
                                                                 kKVCoherence>(
                    data, // will update this pointer if using paged-kvcache
                    make_tuple(height, kargs.hdim_q),
                    make_tuple(kargs.stride_k, 1),
                    number<FmhaPipeline::kAlignmentK>{},
                    number<1>{});

                const auto k_dram_pad = pad_tensor_view(
                    k_dram_naive,
                    make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0>{}),
                    sequence<false, kPadHeadDimQ>{});

                // Same rationale as the affine-TDM dispatch in make_q_dram above:
                // TDM box-major DMA can't honor software XOR'd dram views,
                // the unmerge/xor/merge_v3 chain below is dead code for TDM,
                // and calculate_offset(unit_vec) would otherwise produce an
                // XOR-polluted stride. Return the affine naive view (no
                // head-dim pad) so get_lengths()[hdim] stays at the true
                // head-dim and the TDM box clamp zero-fills the OOB tail.
                if constexpr(kUsesTdmAffineDramPath)
                {
                    return k_dram_naive;
                }
                else
                {
                    constexpr auto kDramTileK =
                        FmhaPipeline::kKLoadOnce ? FmhaPipeline::kQKHeaddim : FmhaPipeline::kK0;

#if CK_TILE_FMHA_HANDLE_XOR_LENGTH_FOLD
                    constexpr index_t LDSLayerSize =
                        256 * numeric_traits<KDataType>::PackedSize / sizeof(KDataType);
                    constexpr index_t XorLengthFold = LDSLayerSize / (FmhaPipeline::kQKHeaddim);

                    if constexpr(XorLengthFold > 1)
                    {
                        const auto k_dram_unmerged = transform_tensor_view(
                            k_dram_pad,
                            make_tuple(
                                make_unmerge_transform(
                                    make_tuple(height / XorLengthFold, XorLengthFold)),
                                make_pass_through_transform(number<FmhaPipeline::kQKHeaddim>{})),
                            make_tuple(sequence<0>{}, sequence<1>{}),
                            make_tuple(sequence<0, 1>{}, sequence<2>{}));

                        const auto k_dram_merged = transform_tensor_view(
                            k_dram_unmerged,
                            make_tuple(make_pass_through_transform(height / XorLengthFold),
                                       make_merge_transform_v3_division_mod(make_tuple(
                                           XorLengthFold, number<FmhaPipeline::kQKHeaddim>{}))),
                            make_tuple(sequence<0>{}, sequence<1, 2>{}),
                            make_tuple(sequence<0>{}, sequence<1>{}));

                        const auto k_dram_unmerged_xor = transform_tensor_view(
                            k_dram_merged,
                            make_tuple(make_pass_through_transform(height / XorLengthFold),
                                       make_unmerge_transform(make_tuple(
                                           number<LDSLayerSize / FmhaPipeline::kAlignmentK>{},
                                           number<FmhaPipeline::kAlignmentK>{}))),
                            make_tuple(sequence<0>{}, sequence<1>{}),
                            make_tuple(sequence<0>{}, sequence<1, 2>{}));

                        const auto k_dram_permuted = transform_tensor_view(
                            k_dram_unmerged_xor,
                            make_tuple(
                                make_xor_transform(
                                    make_tuple(height / XorLengthFold,
                                               number<LDSLayerSize / FmhaPipeline::kAlignmentK>{})),
                                make_pass_through_transform(number<FmhaPipeline::kAlignmentK>{})),
                            make_tuple(sequence<0, 1>{}, sequence<2>{}),
                            make_tuple(sequence<0, 1>{}, sequence<2>{}));

                        const auto k_dram_tmp = transform_tensor_view(
                            k_dram_permuted,
                            make_tuple(
                                make_pass_through_transform(height / XorLengthFold),
                                make_unmerge_transform(
                                    make_tuple(number<XorLengthFold>{},
                                               number<FmhaPipeline::kQKHeaddim /
                                                      FmhaPipeline::kAlignmentK>{})),
                                make_pass_through_transform(number<FmhaPipeline::kAlignmentK>{})),
                            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                            make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

                        return transform_tensor_view(
                            k_dram_tmp,
                            make_tuple(
                                make_merge_transform_v3_division_mod(
                                    make_tuple(height / XorLengthFold, number<XorLengthFold>{})),
                                make_merge_transform_v3_division_mod(make_tuple(
                                    number<FmhaPipeline::kQKHeaddim / FmhaPipeline::kAlignmentK>{},
                                    number<FmhaPipeline::kAlignmentK>{}))),
                            make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
                            make_tuple(sequence<0>{}, sequence<1>{}));
                    }
                    else
#endif // CK_TILE_FMHA_HANDLE_XOR_LENGTH_FOLD
                    {
                        // The leading dim of the unmerge was previously
                        // `kQKHeaddim / kDramTileK / kAlignmentK`, which under
                        // typical configs (e.g. fp16 hdim=128, kDramTileK=kK0=32,
                        // kAlignmentK=8 -> 128/32/8 = 0) folds to 0 by integer
                        // division. That made the resulting dram tensor lengths
                        // `tuple<int, constant<0>>` whose product (0) no longer
                        // matches the real hdim_q (128). The async-load path never
                        // read those lengths so the bug stayed dormant; the TDM
                        // path reads them via get_cached_global_strides and
                        // hits a hard compile failure. Semantically the leading
                        // dim should be "how many K-iter tiles fit into hdim" =
                        // `kQKHeaddim / kDramTileK`. Removing the spurious
                        // `/ kAlignmentK` makes product = 4 * 4 * 8 = 128 again.
                        // Mirror the same fix in the matching make_pass_through
                        // and make_merge calls below so the 3-axis layout stays
                        // consistent.
                        const auto k_dram_unmerged = transform_tensor_view(
                            k_dram_pad,
                            make_tuple(make_pass_through_transform(height),
                                       make_unmerge_transform(make_tuple(
                                           number<FmhaPipeline::kQKHeaddim / kDramTileK>{},
                                           number<kDramTileK / FmhaPipeline::kAlignmentK>{},
                                           number<FmhaPipeline::kAlignmentK>{}))),
                            make_tuple(sequence<0>{}, sequence<1>{}),
                            make_tuple(sequence<0>{}, sequence<1, 2, 3>{}));

                        const auto k_dram_permuted = transform_tensor_view(
                            k_dram_unmerged,
                            make_tuple(
                                make_xor_transform(make_tuple(
                                    height, number<kDramTileK / FmhaPipeline::kAlignmentK>{})),
                                make_pass_through_transform(
                                    number<FmhaPipeline::kQKHeaddim / kDramTileK>{}),
                                make_pass_through_transform(number<FmhaPipeline::kAlignmentK>{})),
                            make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}),
                            make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

                        return transform_tensor_view(
                            k_dram_permuted,
                            make_tuple(make_pass_through_transform(height),
                                       make_merge_transform_v3_division_mod(make_tuple(
                                           number<FmhaPipeline::kQKHeaddim / kDramTileK>{},
                                           number<kDramTileK / FmhaPipeline::kAlignmentK>{},
                                           number<FmhaPipeline::kAlignmentK>{}))),
                            make_tuple(sequence<0>{}, sequence<1, 2, 3>{}),
                            make_tuple(sequence<0>{}, sequence<1>{}));
                    }
                } // end else (qr_tdm dispatch above)
            };
            const auto k_dram = [&]() {
                {
                    return make_k_dram(k_ptr, kargs.seqlen_k);
                }
            }();

            const auto make_v_dram = [&](const VDataType* data, index_t length) {
                const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global,
                                                                 memory_operation_enum::set,
                                                                 kKVCoherence>(
                    data, // will update this pointer if using paged-kvcache
                    make_tuple(length, kargs.hdim_v),
                    make_tuple(kargs.stride_v, 1),
                    number<FmhaPipeline::kAlignmentV>{},
                    number<1>{});

                // TODO: Add kVHeadDim
                constexpr index_t XorGroupSize =
                    FmhaPipeline::Problem::BlockFmhaShape::Gemm1WarpTile::at(number<0>{});

                const auto v_dram_pad = pad_tensor_view(
                    v_dram_naive,
                    make_tuple(number<FmhaPipeline::kK1>{}, number<FmhaPipeline::kN1>{}),
                    sequence<kPadSeqLenK, false>{});

                // TDM needs affine DRAM strides and the logical sequence bounds;
                // software padding would make invalid V rows readable by DMA.
                // LDS padding is configured separately by the pipeline policy.
                if constexpr(kUsesTdmAffineDramPath)
                {
                    return v_dram_naive;
                }
                else
                {
#if CK_TILE_FMHA_HANDLE_XOR_LENGTH_FOLD
                    constexpr index_t LDSLayerSize =
                        256 * numeric_traits<VDataType>::PackedSize / sizeof(VDataType);
                    constexpr index_t XorLengthFold = LDSLayerSize / (FmhaPipeline::kQKHeaddim);

                    if constexpr(XorLengthFold > 1)
                    {
                        const auto v_dram_unmerged = transform_tensor_view(
                            v_dram_pad,
                            make_tuple(
                                make_unmerge_transform(
                                    make_tuple(length / XorLengthFold, XorLengthFold)),
                                make_pass_through_transform(number<FmhaPipeline::kQKHeaddim>{})),
                            make_tuple(sequence<0>{}, sequence<1>{}),
                            make_tuple(sequence<0, 1>{}, sequence<2>{}));

                        const auto v_dram_merged = transform_tensor_view(
                            v_dram_unmerged,
                            make_tuple(make_pass_through_transform(length / XorLengthFold),
                                       make_merge_transform_v3_division_mod(make_tuple(
                                           XorLengthFold, number<FmhaPipeline::kQKHeaddim>{}))),
                            make_tuple(sequence<0>{}, sequence<1, 2>{}),
                            make_tuple(sequence<0>{}, sequence<1>{}));

                        const auto v_dram_unmerged_xor = transform_tensor_view(
                            v_dram_merged,
                            make_tuple(make_pass_through_transform(length / XorLengthFold),
                                       make_unmerge_transform(
                                           make_tuple(number<LDSLayerSize / XorGroupSize>{},
                                                      number<XorGroupSize>{}))),
                            make_tuple(sequence<0>{}, sequence<1>{}),
                            make_tuple(sequence<0>{}, sequence<1, 2>{}));

                        const auto v_dram_permuted = transform_tensor_view(
                            v_dram_unmerged_xor,
                            make_tuple(
                                make_xor_transform(make_tuple(
                                    length / XorLengthFold, number<LDSLayerSize / XorGroupSize>{})),
                                make_pass_through_transform(number<XorGroupSize>{})),
                            make_tuple(sequence<0, 1>{}, sequence<2>{}),
                            make_tuple(sequence<0, 1>{}, sequence<2>{}));

                        const auto v_dram_tmp = transform_tensor_view(
                            v_dram_permuted,
                            make_tuple(make_pass_through_transform(length / XorLengthFold),
                                       make_unmerge_transform(make_tuple(
                                           number<XorLengthFold>{},
                                           number<FmhaPipeline::kQKHeaddim / XorGroupSize>{})),
                                       make_pass_through_transform(number<XorGroupSize>{})),
                            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                            make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

                        return transform_tensor_view(
                            v_dram_tmp,
                            make_tuple(make_merge_transform_v3_division_mod(make_tuple(
                                           length / XorLengthFold, number<XorLengthFold>{})),
                                       make_merge_transform_v3_division_mod(make_tuple(
                                           number<FmhaPipeline::kQKHeaddim / XorGroupSize>{},
                                           number<XorGroupSize>{}))),
                            make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
                            make_tuple(sequence<0>{}, sequence<1>{}));
                    }
                    else
#endif // CK_TILE_FMHA_HANDLE_XOR_LENGTH_FOLD
                    {
                        const auto v_dram_unmerged = transform_tensor_view(
                            v_dram_pad,
                            make_tuple(make_pass_through_transform(length),
                                       make_unmerge_transform(make_tuple(
                                           number<FmhaPipeline::kQKHeaddim / XorGroupSize>{},
                                           number<XorGroupSize>{}))),
                            make_tuple(sequence<0>{}, sequence<1>{}),
                            make_tuple(sequence<0>{}, sequence<1, 2>{}));

                        const auto v_dram_permuted = transform_tensor_view(
                            v_dram_unmerged,
                            make_tuple(
                                make_xor_transform(make_tuple(
                                    length, number<FmhaPipeline::kQKHeaddim / XorGroupSize>{})),
                                make_pass_through_transform(number<XorGroupSize>{})),
                            make_tuple(sequence<0, 1>{}, sequence<2>{}),
                            make_tuple(sequence<0, 1>{}, sequence<2>{}));

                        return transform_tensor_view(
                            v_dram_permuted,
                            make_tuple(make_pass_through_transform(length),
                                       make_merge_transform_v3_division_mod(make_tuple(
                                           number<FmhaPipeline::kQKHeaddim / XorGroupSize>{},
                                           number<XorGroupSize>{}))),
                            make_tuple(sequence<0>{}, sequence<1, 2>{}),
                            make_tuple(sequence<0>{}, sequence<1>{}));
                    }
                } // end else (affine-TDM dispatch above returns v_dram_naive early)
            };

            const auto v_dram = [&]() {
                {
                    return make_v_dram(v_ptr, kargs.seqlen_k);
                }
            }();

            auto q_dram_window = make_tile_window(
                q_dram,
                [&]() {
                    if constexpr(FmhaPipeline::kQLoadOnce)
                        return make_tuple(number<FmhaPipeline::kM0>{},
                                          number<FmhaPipeline::kSubQKHeaddim>{});
                    else
                        return make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kK0>{});
                }(),
                {i_m0, 0});

            auto k_dram_window = make_tile_window(
                k_dram,
                make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0>{}),
                {0, 0});

            auto v_dram_window = make_tile_window(
                v_dram,
                make_tuple(number<FmhaPipeline::kN1>{}, number<FmhaPipeline::kK1>{}),
                {0, 0});

            /// FIXME: Before C++20, capturing structured binding variables are not supported.
            /// Remove following copy capture of the 'i_nhead' if in C++20
            const auto bias_dram_window = [&, i_nhead_ = i_nhead]() {
                constexpr auto bias_dram_window_lengths =
                    make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN0>{});
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                {
                    const BiasDataType* bias_ptr =
                        reinterpret_cast<const BiasDataType*>(kargs.bias_ptr) +
                        static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_bias +
                        batch_offset_bias;

                    const auto bias_dram = [&]() {
                        const auto bias_dram_naive =
                            make_naive_tensor_view<address_space_enum::global>(
                                bias_ptr,
                                make_tuple(kargs.seqlen_q, kargs.seqlen_k),
                                make_tuple(kargs.stride_bias, 1),
                                number<FmhaPipeline::kAlignmentBias>{},
                                number<1>{});

                        return pad_tensor_view(bias_dram_naive,
                                               bias_dram_window_lengths,
                                               sequence<false, kPadSeqLenK>{});
                    }();

                    return make_tile_window(bias_dram, bias_dram_window_lengths, {i_m0, 0});
                }
                else
                {
                    return make_null_tile_window(bias_dram_window_lengths);
                }
            }();

            // lse acc
            auto lse_dram_window = [&, i_nhead_ = i_nhead_q]() {
                constexpr auto lse_dram_window_lengths = make_tuple(number<FmhaPipeline::kM0>{});
                if constexpr(kStoreLSE)
                {
                    LSEDataType* lse_ptr =
                        reinterpret_cast<LSEDataType*>(kargs.lse_ptr) +
                        static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_lse +
                        batch_offset_lse;

                    const auto lse_dram = [&] {
                        const auto lse_dram_naive = [&] {
                            {
                                return make_naive_tensor_view<address_space_enum::global>(
                                    lse_ptr,
                                    make_tuple(num_rows_q),
                                    make_tuple(rows_by_head ? kargs.nhead_stride_lse : 1),
                                    number<1>{},
                                    number<1>{});
                            }
                        }();
                        return pad_tensor_view(
                            lse_dram_naive, lse_dram_window_lengths, sequence<kPadSeqLenQ>{});
                    }();

                    return make_tile_window(lse_dram, lse_dram_window_lengths, {i_m0});
                }
                else
                {
                    return make_null_tile_window(lse_dram_window_lengths);
                }
            }();

            FmhaMask mask = [&]() {
                if constexpr(kHasMask)
                    return ck_tile::make_generic_attention_mask_from_lr_window<FmhaMask>(
                        kargs.window_size_left,
                        kargs.window_size_right,
                        kargs.sink_size,
                        kargs.seqlen_q,
                        kargs.seqlen_k,
                        kargs.mask_type == GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
                else
                    return FmhaMask{kargs.seqlen_q, kargs.seqlen_k};
            }();

            // WA i_batch capture structure binding before c++20
            auto position_encoding = [&, i_batch_ = i_batch, i_nhead_ = i_nhead]() {
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
                {
                    // data loading, shared by entire wg
                    // TODO: how to use s_read?
                    SaccDataType slope =
                        *(reinterpret_cast<const SaccDataType*>(kargs.alibi_slope_ptr) +
                          i_batch_ * kargs.alibi_slope_stride + i_nhead_);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                    if constexpr(!detail::ignore_fast_exp2_v<FmhaPipeline>)
                    {
                        slope *= ck_tile::log2e_v<>;
                    }
#endif
                    if constexpr(kHasMask)
                    {
                        return make_alibi_from_lr_mask<SaccDataType, true, 32>(
                            slope,
                            kargs.window_size_left,
                            kargs.window_size_right,
                            kargs.seqlen_q,
                            kargs.seqlen_k,
                            kargs.mask_type);
                    }
                    else
                    {
                        return Alibi<SaccDataType, true, 32>{
                            slope, kargs.seqlen_q, kargs.seqlen_k, AlibiMode::FROM_BOTTOM_RIGHT};
                    }
                }
                else
                {
                    return EmptyPositionEncoding<SaccDataType>{};
                }
            }();

            constexpr bool kPerHeadQScale = QScaleEnum == BlockAttentionQuantScaleEnum::PERHEAD;
            constexpr bool kBlockQScale   = QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE;
            // MX descales: E8M0 per 32 elements, consumed as WMMA scale operands.
            constexpr bool kMxTdmSched = kPipelineName == "qr_tdm_fp8_sched" &&
                                         QScaleEnum == BlockAttentionQuantScaleEnum::MX;
            [[maybe_unused]] long_index_t descale_offset_q = 0;
            [[maybe_unused]] long_index_t descale_offset_k = 0;
            [[maybe_unused]] long_index_t descale_offset_v = 0;
            if constexpr(kPerHeadQScale || kBlockQScale || kMxTdmSched)
            {
                descale_offset_q =
                    static_cast<long_index_t>(i_nhead_q) * kargs.nhead_stride_q_descale;
                descale_offset_k =
                    static_cast<long_index_t>(i_nhead_k) * kargs.nhead_stride_k_descale;
                descale_offset_v =
                    static_cast<long_index_t>(i_nhead_k) * kargs.nhead_stride_v_descale;
                if constexpr(kPerHeadQScale || !kIsGroupMode)
                {
                    descale_offset_q +=
                        static_cast<long_index_t>(i_batch) * kargs.batch_stride_q_descale;
                    descale_offset_k +=
                        static_cast<long_index_t>(i_batch) * kargs.batch_stride_k_descale;
                    descale_offset_v +=
                        static_cast<long_index_t>(i_batch) * kargs.batch_stride_v_descale;
                }
                else if constexpr(kMxTdmSched)
                {
                    // Group MX: Q/K scale rows follow the physical token starts, V its own.
                    descale_offset_q += static_cast<long_index_t>(kargs.seqstart_q_ptr[i_batch]) *
                                        kargs.stride_q_descale;
                    descale_offset_k += static_cast<long_index_t>(kargs.seqstart_k_ptr[i_batch]) *
                                        kargs.stride_k_descale;
                    descale_offset_v += kargs.seqstart_v_scale_ptr[i_batch];
                }
                else
                {
                    descale_offset_q += kargs.block_scale_seqstart_q_ptr[i_batch];
                    descale_offset_k += kargs.block_scale_seqstart_k_ptr[i_batch];
                    descale_offset_v += kargs.block_scale_seqstart_k_ptr[i_batch];
                }
            }

            static_assert(kTakesTdmDescales || QScaleEnum == BlockAttentionQuantScaleEnum::NO_SCALE,
                          "only qr_tdm and qr_tdm_fp8_sched implement quantization descales in "
                          "this branch");

            constexpr bool kFoldedQScale =
                QScaleEnum == BlockAttentionQuantScaleEnum::PERTENSOR || kPerHeadQScale;
            const float v_descale = [&] {
                if constexpr(kFoldedQScale)
                    return reinterpret_cast<const float*>(kargs.v_descale_ptr)[descale_offset_v];
                else
                    return 1.0f;
            }();

            // Same scale_s the pipeline receives, so the sink recovers its logit in the same
            // units it was seeded in. Dividing by the folded scale_s the pipeline re-applies is
            // what makes the descales cancel.
            const float scale_s = GetSoftmaxScale(kargs, descale_offset_q, descale_offset_k, i_m0);
            // Merged rows read their sinks per head in the pipeline; a finite value marks one.
            const float sink_value =
                kargs.sink_ptr != nullptr
                    ? (kMergeNumHeadGroupsSeqLenQ
                           ? 0.0f
                           : (*(static_cast<const float*>(kargs.sink_ptr) + i_nhead)) / scale_s)
                    : -numeric<float>::infinity();

            auto invoke_fmha_pipeline = [&,
                                         i_batch_   = i_batch,
                                         i_nhead_   = i_nhead,
                                         i_nhead_k_ = i_nhead_k](auto&&... args) -> decltype(auto) {
                if constexpr(kMergeNumHeadGroupsSeqLenQ)
                {
                    typename FmhaPipeline::MxDescales mx;
                    if constexpr(kMxTdmSched)
                    {
                        mx.q_ptr =
                            reinterpret_cast<const e8m0_t*>(kargs.q_descale_ptr) + descale_offset_q;
                        mx.k_ptr =
                            reinterpret_cast<const e8m0_t*>(kargs.k_descale_ptr) + descale_offset_k;
                        mx.v_ptr =
                            reinterpret_cast<const e8m0_t*>(kargs.v_descale_ptr) + descale_offset_v;
                        mx.stride_q =
                            rows_by_head ? kargs.nhead_stride_q_descale : kargs.stride_q_descale;
                        mx.stride_k = kargs.stride_k_descale;
                        mx.stride_v = kargs.stride_v_descale;
                        mx.seqlen_q = num_rows_q;
                    }
                    typename FmhaPipeline::MergedHeads merged;
                    merged.seqlen_q             = kargs.seqlen_q;
                    merged.num_groups           = kargs.nhead_ratio_qk;
                    merged.sink_ptr             = kargs.sink_ptr != nullptr
                                                      ? static_cast<const float*>(kargs.sink_ptr) + i_nhead_q
                                                      : nullptr;
                    const float* k_descale_ptr  = nullptr;
                    const float* v_descale_ptr  = nullptr;
                    index_t block_scale_size_kv = 1;
                    if constexpr(kPerHeadQScale || kBlockQScale)
                    {
                        merged.q_descale_ptr =
                            reinterpret_cast<const float*>(kargs.q_descale_ptr) + descale_offset_q;
                        merged.q_descale_head_stride = kargs.nhead_stride_q_descale;
                        merged.scale_s               = kargs.scale_s;
                    }
                    if constexpr(kPerHeadQScale)
                        merged.k_descale =
                            reinterpret_cast<const float*>(kargs.k_descale_ptr)[descale_offset_k];
                    if constexpr(kBlockQScale)
                    {
                        k_descale_ptr =
                            reinterpret_cast<const float*>(kargs.k_descale_ptr) + descale_offset_k;
                        v_descale_ptr =
                            reinterpret_cast<const float*>(kargs.v_descale_ptr) + descale_offset_v;
                        block_scale_size_kv = kargs.block_scale_size_kv;
                    }
                    return FmhaPipeline{}(static_cast<decltype(args)&&>(args)...,
                                          k_descale_ptr,
                                          v_descale_ptr,
                                          block_scale_size_kv,
                                          v_descale,
                                          mx,
                                          merged);
                }
                else if constexpr(kTakesTdmDescales && kBlockQScale)
                {
                    const float* k_descale_ptr =
                        reinterpret_cast<const float*>(kargs.k_descale_ptr) + descale_offset_k;
                    const float* v_descale_ptr =
                        reinterpret_cast<const float*>(kargs.v_descale_ptr) + descale_offset_v;
                    return FmhaPipeline{}(static_cast<decltype(args)&&>(args)...,
                                          k_descale_ptr,
                                          v_descale_ptr,
                                          kargs.block_scale_size_kv,
                                          v_descale);
                }
                else if constexpr(kMxTdmSched)
                {
                    typename FmhaPipeline::MxDescales mx;
                    mx.q_ptr =
                        reinterpret_cast<const e8m0_t*>(kargs.q_descale_ptr) + descale_offset_q;
                    mx.k_ptr =
                        reinterpret_cast<const e8m0_t*>(kargs.k_descale_ptr) + descale_offset_k;
                    mx.v_ptr =
                        reinterpret_cast<const e8m0_t*>(kargs.v_descale_ptr) + descale_offset_v;
                    mx.stride_q = kargs.stride_q_descale;
                    mx.stride_k = kargs.stride_k_descale;
                    mx.stride_v = kargs.stride_v_descale;
                    mx.seqlen_q = kargs.seqlen_q;
                    return FmhaPipeline{}(
                        static_cast<decltype(args)&&>(args)..., nullptr, nullptr, 1, 1.0f, mx);
                }
                else if constexpr(kTakesTdmDescales && kFoldedQScale)
                    return FmhaPipeline{}(
                        static_cast<decltype(args)&&>(args)..., nullptr, nullptr, 1, v_descale);
                else if constexpr(kPipelineName == "qr_tdm" && (kHasLogitsSoftCap || kHasDropout))
                {
                    auto transform = [&](auto x) {
                        if constexpr(kHasLogitsSoftCap)
                        {
                            const LogitsSoftCapParams<FmhaMask, CK_TILE_FMHA_FWD_FAST_EXP2> params{
                                mask, scale_s, kargs.logits_soft_cap, kargs.logits_soft_cap_rcp};
                            AttentionVariant variant;
                            return variant.LogitsTransform(params,
                                                           variant.QueryTransform(params, x),
                                                           i_batch_,
                                                           i_nhead_,
                                                           i_nhead_k_);
                        }
                        else
                            return x;
                    };
                    auto apply_dropout = [&] {
                        if constexpr(kHasDropout)
                        {
                            const auto heads = kargs.num_head_q_total > 0 ? kargs.num_head_q_total
                                                                          : kargs.num_head_q;
                            const BlockDropout dropout{
                                i_batch_,
                                kargs.head_start + i_nhead_,
                                heads,
                                kargs.is_drop_seed_offset_from_host ? kargs.drop_seed.val
                                                                    : *kargs.drop_seed.ptr,
                                kargs.is_drop_seed_offset_from_host ? kargs.drop_offset.val
                                                                    : *kargs.drop_offset.ptr,
                                kargs.rp_undrop,
                                kargs.p_undrop_in_uint8_t,
                                kargs.is_store_randval};
                            RandValOutputDataType* randval = nullptr;
                            if(kargs.is_store_randval)
                            {
                                const long_index_t batch_offset = [&] {
                                    if constexpr(kIsGroupMode)
                                        return static_cast<long_index_t>(
                                                   kargs.seqstart_q_ptr[i_batch_]) *
                                               kargs.stride_randval;
                                    else
                                        return static_cast<long_index_t>(i_batch_) *
                                               kargs.batch_stride_randval;
                                }();
                                randval = static_cast<RandValOutputDataType*>(kargs.rand_val_ptr) +
                                          batch_offset +
                                          static_cast<long_index_t>(i_nhead_) *
                                              kargs.nhead_stride_randval;
                            }
                            return [dropout,
                                    randval,
                                    stride   = kargs.stride_randval,
                                    seqlen_q = kargs.seqlen_q,
                                    seqlen_k = kargs.seqlen_k](auto& p, index_t m, index_t n) {
                                dropout.RunOnTile(p, m, n, randval, stride, seqlen_q, seqlen_k);
                            };
                        }
                        else
                            return [](auto&, index_t, index_t) {};
                    }();
                    return FmhaPipeline{}(static_cast<decltype(args)&&>(args)...,
                                          make_tuple(transform, apply_dropout));
                }
                else
                    return FmhaPipeline{}(static_cast<decltype(args)&&>(args)...);
            };

            // Declared here so the epilogue can reuse the pipeline's LDS. The
            // non-arena prefill pipelines and the fixed segmented arena declare
            // their own arrays.
            constexpr bool kPipelineDeclaresLds =
                kPrefillLdsPath &&
                (kUsesFixedSegmentedLdsArena || !detail::uses_qr_tdm_lds_arena_v<FmhaPipeline>);
            char* smem_ptr = nullptr;
            if constexpr(!kPipelineDeclaresLds)
            {
                alignas(256) __shared__ char smem[GetSmemSize()];
                smem_ptr = smem;
            }
            [[maybe_unused]] void* o_lds_staging = nullptr;

            auto o_acc_tile = [&]() {
                if constexpr(kPrefillLdsPath)
                {
                    if constexpr(kUsesFixedSegmentedLdsArena)
                    {
                        using Policy = typename FmhaPipeline::Policy;
                        static_assert(FmhaPipeline::GetSmemSize() == Policy::GetLdsArenaSize());
                        alignas(16) __shared__ char smem_arena[Policy::GetLdsArenaSize()];
                        o_lds_staging = smem_arena + Policy::GetLdsOffsetK0();
                        return invoke_fmha_pipeline(q_dram_window,
                                                    k_dram_window,
                                                    v_dram_window,
                                                    bias_dram_window,
                                                    lse_dram_window,
                                                    mask,
                                                    position_encoding,
                                                    scale_s,
                                                    sink_value,
                                                    smem_arena + Policy::GetLdsOffsetK0(),
                                                    smem_arena + Policy::GetLdsOffsetK1(),
                                                    smem_arena + Policy::GetLdsOffsetV0(),
                                                    smem_arena + Policy::GetLdsOffsetV1());
                    }
                    else if constexpr(detail::uses_qr_tdm_lds_arena_v<FmhaPipeline>)
                    {
                        return invoke_fmha_pipeline(q_dram_window,
                                                    k_dram_window,
                                                    v_dram_window,
                                                    bias_dram_window,
                                                    lse_dram_window,
                                                    mask,
                                                    position_encoding,
                                                    scale_s,
                                                    sink_value,
                                                    smem_ptr);
                    }
                    else
                    {
                        // allocate double lds
                        // add __restrict__ here to avoid aliasing
                        __shared__ char smem_ptrk0[FmhaPipeline::Policy::template GetSmemSizeK<
                            typename FmhaPipeline::Problem,
                            true>()];
                        __shared__ char smem_ptrk1[FmhaPipeline::Policy::template GetSmemSizeK<
                            typename FmhaPipeline::Problem,
                            true>()];
                        __shared__ char smem_ptrv0[FmhaPipeline::Policy::template GetSmemSizeV<
                            typename FmhaPipeline::Problem>()];
                        __shared__ char smem_ptrv1[FmhaPipeline::Policy::template GetSmemSizeV<
                            typename FmhaPipeline::Problem>()];

                        return invoke_fmha_pipeline(q_dram_window,
                                                    k_dram_window,
                                                    v_dram_window,
                                                    bias_dram_window,
                                                    lse_dram_window,
                                                    mask,
                                                    position_encoding,
                                                    scale_s,
                                                    sink_value,
                                                    smem_ptrk0,
                                                    smem_ptrk1,
                                                    smem_ptrv0,
                                                    smem_ptrv1);
                    }
                }
                else
                {
                    // The arena pipeline and the single-buffer ones both take
                    // one LDS block here: the hoisted smem_ptr.
                    return invoke_fmha_pipeline(q_dram_window,
                                                k_dram_window,
                                                v_dram_window,
                                                bias_dram_window,
                                                lse_dram_window,
                                                mask,
                                                position_encoding,
                                                scale_s,
                                                smem_ptr,
                                                sink_value);
                }
            }();

            if constexpr(detail::stores_output_via_tdm_v<FmhaPipeline>)
            {
                // o_ptr already holds the group's physical query start; rows stop at seqlen_q.
                alignas(16) __shared__ char smem_o[FmhaPipeline::kOLdsBytes];
                FmhaPipeline::StoreOutput(o_acc_tile,
                                          o_ptr + static_cast<long_index_t>(i_m0) * row_stride_o +
                                              i_n1,
                                          num_rows_q - i_m0,
                                          kargs.hdim_v - i_n1,
                                          row_stride_o,
                                          smem_o);
            }
            else
            {
                auto store_output = [&](auto alignment) {
                    // Oacc DRAM and Oacc DRAM window
                    auto o_dram = [&] {
                        const auto o_dram_naive = [&] {
                            {
                                return make_naive_tensor_view<address_space_enum::global>(
                                    o_ptr,
                                    make_tuple(kargs.seqlen_q, kargs.hdim_v),
                                    make_tuple(kargs.stride_o, 1),
                                    alignment,
                                    number<1>{});
                            }
                        }();

                        return pad_tensor_view(
                            o_dram_naive,
                            make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN1>{}),
                            sequence<kPadSeqLenQ, kPadHeadDimV>{});
                    }();

                    auto o_dram_window = make_tile_window(
                        o_dram,
                        make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN1>{}),
                        {i_m0, i_n1});

                    // Same logical output as the current Default2DEpilogue (cast to ODataType,
                    // overwrite valid rows, no scale or atomic); bounds come from
                    // clamping/skipping. Mirror epilogue changes here.
                    if constexpr(kUseAsyncLdsOStore && PrefillCase)
                    {
                        fmha_tdm_sched_store_o_tile_async_from_lds<FmhaPipeline, kAsyncOStoreBytes>(
                            o_lds_staging,
                            o_ptr + static_cast<long_index_t>(i_m0) * kargs.stride_o + i_n1,
                            kargs.stride_o,
                            kargs.seqlen_q - i_m0,
                            o_acc_tile);
                    }
                    else if constexpr(detail::epilogue_uses_smem_v<EpiloguePipeline>)
                    {
                        // The pipeline is done with its LDS; pipelines that declare
                        // their own arrays leave none to the epilogue (direct store).
                        if constexpr(kPipelineDeclaresLds)
                            EpiloguePipeline{}(o_dram_window, o_acc_tile, nullptr, nullptr);
                        else
                            EpiloguePipeline{}(o_dram_window, o_acc_tile, nullptr, smem_ptr);
                    }
                    else
                    {
                        EpiloguePipeline{}(o_dram_window, o_acc_tile, nullptr);
                    }
                };
                if constexpr(kPipelineName == "qr_tdm" && kPadHeadDimV)
                {
                    // Keep vector stores for aligned heads, and predicate individual
                    // elements for arbitrary tails and row strides.
                    constexpr index_t alignment = FmhaPipeline::kAlignmentOacc;
                    if(kargs.hdim_v % alignment == 0 && kargs.stride_o % alignment == 0 &&
                       reinterpret_cast<uintptr_t>(o_ptr) % (alignment * sizeof(ODataType)) == 0)
                        store_output(number<alignment>{});
                    else
                        store_output(number<1>{});
                }
                else
                    store_output(number<FmhaPipeline::kAlignmentOacc>{});
            }
        }
    }
};

} // namespace ck_tile
