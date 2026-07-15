// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/dispatcher/kernel_instance.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/streamk_gemm/streamk_gemm_kernel.hpp"
#include "ck_tile/ops/common/streamk_common.hpp"
#include <hip/hip_runtime.h>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ck_tile {
namespace dispatcher {
namespace backends {

// Lock the dispatcher's ReductionStrategy (defined in kernel_key.hpp, which is
// deliberately kept ck_tile-free -- same policy as the void* workspace in
// dispatcher.hpp) to ck_tile::StreamKReductionStrategy so the two enums cannot
// silently drift. The dispatcher enum carries an extra None=0 sentinel, so the
// three real strategies are offset by one. This backend header is the single
// place that includes both definitions, so the check belongs here rather than in
// the public key header.
static_assert(static_cast<std::uint32_t>(ReductionStrategy::Atomic) ==
                  static_cast<std::uint32_t>(ck_tile::StreamKReductionStrategy::Atomic) + 1u,
              "dispatcher ReductionStrategy drifted from ck_tile::StreamKReductionStrategy");
static_assert(static_cast<std::uint32_t>(ReductionStrategy::Linear) ==
                  static_cast<std::uint32_t>(ck_tile::StreamKReductionStrategy::Linear) + 1u,
              "dispatcher ReductionStrategy drifted from ck_tile::StreamKReductionStrategy");
static_assert(static_cast<std::uint32_t>(ReductionStrategy::Tree) ==
                  static_cast<std::uint32_t>(ck_tile::StreamKReductionStrategy::Tree) + 1u,
              "dispatcher ReductionStrategy drifted from ck_tile::StreamKReductionStrategy");

/**
 * Kernel-instance wrapper for unified_gemm_codegen.py Stream-K kernels.
 *
 * Counterpart of GeneratedTileKernelInstance (regular GEMM) for the Stream-K
 * variant. The difference is the host-args type: Stream-K needs
 * ck_tile::StreamKHostArgs (workspace pointer + reduction strategy), which is
 * ABI-incompatible with the GemmHostArgs path -- this is exactly why Stream-K
 * could not previously ride the registry. With this backend it can: the
 * Dispatcher selects the instance by KernelKey (which now carries streamk +
 * reduction_strategy) and calls run().
 *
 * supports() gates on the requested reduction strategy so that the registry can
 * hold atomic/linear/tree side by side and the Dispatcher's first-fit selection
 * picks the one the caller asked for via Problem::reduction_strategy.
 *
 * NOTE (PR-C): the generated SelectedKernel::launch(StreamKHostArgs, stream)
 * still owns the reduction workspace internally (DeviceMem) and does the
 * per-iter reset. PR-D relocates workspace ownership + reset to Dispatcher::run()
 * via get_workspace_size()/the workspace-aware run() overload.
 */
template <typename SelectedKernelType,
          typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename AccDataType_>
class GeneratedStreamKKernelInstance : public KernelInstance
{
    public:
    using ADataType      = ADataType_;
    using BDataType      = BDataType_;
    using CDataType      = CDataType_;
    using AccDataType    = AccDataType_;
    using SelectedKernel = SelectedKernelType;

    GeneratedStreamKKernelInstance(const KernelKey& key, const std::string& name)
        : key_(key), name_(name)
    {
    }

    const KernelKey& get_key() const override { return key_; }

    std::string get_name() const override { return name_; }

    /// Accept ONLY when the caller requested a Stream-K kernel with THIS
    /// instance's reduction strategy. Lets atomic/linear/tree coexist in the
    /// registry and be selected by Problem::reduction_strategy.
    bool supports(const Problem& problem) const override
    {
        if(!problem.streamk)
            return false;
        if(problem.reduction_strategy != key_.algorithm.reduction_strategy)
            return false;

        // Stream-K distributes K-iterations across workgroups; padding flags
        // mirror the regular backend's divisibility guard.
        constexpr bool pad_m = SelectedKernel::kPadM;
        constexpr bool pad_n = SelectedKernel::kPadN;
        constexpr bool pad_k = SelectedKernel::kPadK;
        if(!pad_m && problem.M % SelectedKernel::TileM != 0)
            return false;
        if(!pad_n && problem.N % SelectedKernel::TileN != 0)
            return false;
        if(!pad_k && problem.K % SelectedKernel::TileK != 0)
            return false;

        // Final feasibility: enough tiles to partition across CUs. Rejecting here
        // (instead of throwing at launch) lets the dispatcher's first-fit fall back
        // to a non-Stream-K kernel for too-small problems.
        return SelectedKernel::IsSupported(make_args(problem));
    }

    /// Device workspace (bytes) needed for `problem`. 0 for Atomic; >0 for
    /// Linear/Tree. The Dispatcher uses this to size the buffer it owns and then
    /// passes that buffer to the workspace-aware run() below.
    std::size_t get_workspace_size(const Problem& problem) const override
    {
        return SelectedKernel::GetWorkSpaceSize(make_args(problem));
    }

    /// No-workspace entry point: delegates to the workspace-aware overload with a
    /// null buffer, so the generated launch() falls back to its internal
    /// (self-allocating) path. Used when the caller does not own a workspace.
    float run(const void* a_ptr,
              const void* b_ptr,
              void* c_ptr,
              const void** d_ptrs,
              const Problem& problem,
              void* stream) const override
    {
        return run(a_ptr, b_ptr, c_ptr, d_ptrs, /*workspace=*/nullptr, problem, stream);
    }

    /// Workspace-aware execution (PR-D). `workspace` is the Dispatcher-owned
    /// reduction buffer (may be null for Atomic, which needs none). When non-null
    /// the generated launch() binds it instead of allocating its own DeviceMem.
    float run(const void* a_ptr,
              const void* b_ptr,
              void* c_ptr,
              const void** d_ptrs,
              void* workspace,
              const Problem& problem,
              void* stream) const override
    {
        (void)d_ptrs; // Not used for Stream-K GEMM

        auto args = make_args(problem, a_ptr, b_ptr, c_ptr);

        const bool bench = this->benchmarking_;
        ck_tile::stream_config stream_cfg;
        stream_cfg.stream_id_    = reinterpret_cast<hipStream_t>(stream);
        stream_cfg.time_kernel_  = bench;
        stream_cfg.log_level_    = 0;
        stream_cfg.cold_niters_  = bench ? 5 : 0;
        stream_cfg.nrepeat_      = bench ? 10 : 1;
        stream_cfg.is_gpu_timer_ = bench;
        // Flush the L2 between timed iterations so the measurement is cold, like
        // tile_engine and the standalone 03 driver. Leaving the cache warm here was
        // the methodology artifact that over-reported TFlops and produced the
        // spurious dispatcher-vs-TE "performance gap"; do not present a warm number
        // as parity evidence.
        stream_cfg.flush_cache_ = bench;
        // NOTE: input-buffer rotation is intentionally NOT enabled (rotating_count
        // = 1). Atomic reduction accumulates straight into C, and this same run()
        // serves the functional path that callers verify against the reference, so
        // rotating/accumulating would corrupt the output left on the device. This
        // means the timing here is cold-but-non-rotated and is therefore NOT the
        // fully apple-to-apple surface: for TE-calibrated numbers use the 03 driver
        // (or a --validate 0 pass) which rotates 1000 input copies like tile_engine.
        stream_cfg.rotating_count_ = 1;

        if(workspace != nullptr)
            return SelectedKernel::launch(args, stream_cfg, workspace);
        return SelectedKernel::launch(args, stream_cfg);
    }

    bool validate(const void* a_ptr,
                  const void* b_ptr,
                  const void* c_ptr,
                  const void** d_ptrs,
                  const Problem& problem,
                  float tolerance) const override
    {
        (void)d_ptrs;
        (void)tolerance;
        // This backend owns no host reference, so a numeric correctness check is
        // out of scope here (the TE/driver harness does that). But returning a
        // blind "true" would mis-report an unrunnable config as valid, so validate
        // what we CAN without a reference: non-null operands, a well-formed
        // problem, and that THIS Stream-K instance actually supports it.
        if(a_ptr == nullptr || b_ptr == nullptr || c_ptr == nullptr)
            return false;
        if(!problem.is_valid())
            return false;
        return supports(problem);
    }

    private:
    /// Build StreamKHostArgs for `problem`. Leading dims are derived from the
    /// kernel key's layouts so every layout works (rcr/rrr/ccr/crr, ...), not
    /// just rcr: A is MxK (row->K, col->M), B is KxN (row->N, col->K), C is MxN
    /// (row->N, col->M). k_batch is owned by the Stream-K tile partitioner, not
    /// passed here. Pointers default to null for sizing-only use
    /// (GetWorkSpaceSize). StreamKHostArgs uses ck_tile::index_t (int32); cast
    /// from Problem's int64.
    ck_tile::StreamKHostArgs make_args(const Problem& problem,
                                       const void* a_ptr = nullptr,
                                       const void* b_ptr = nullptr,
                                       void* c_ptr       = nullptr) const
    {
        using idx = ck_tile::index_t;
        // StreamKHostArgs uses int32 index_t while Problem carries int64 dims.
        // Guard the narrowing so an oversized M/N/K (or a derived leading dim)
        // fails loudly instead of silently wrapping to a negative/garbage extent.
        // The dimension parser was widened to std::stoll specifically to avoid
        // overflow, so dropping back to int32 here must be checked, not assumed.
        auto to_idx = [](std::int64_t v, const char* what) -> idx {
            if(v < 0 || v > static_cast<std::int64_t>(std::numeric_limits<idx>::max()))
                throw std::runtime_error(std::string("StreamK make_args: ") + what + " (" +
                                         std::to_string(v) +
                                         ") exceeds int32 ck_tile::index_t range");
            return static_cast<idx>(v);
        };

        const auto& sig    = key_.signature;
        const bool a_row   = sig.layout_a == LayoutTag::RowMajor;
        const bool b_row   = sig.layout_b == LayoutTag::RowMajor;
        const bool c_row   = sig.layout_c == LayoutTag::RowMajor;
        const idx M        = to_idx(problem.M, "M");
        const idx N        = to_idx(problem.N, "N");
        const idx K        = to_idx(problem.K, "K");
        const idx stride_a = to_idx(a_row ? problem.K : problem.M, "stride_a");
        const idx stride_b = to_idx(b_row ? problem.N : problem.K, "stride_b");
        const idx stride_c = to_idx(c_row ? problem.N : problem.M, "stride_c");
        return ck_tile::StreamKHostArgs{a_ptr, b_ptr, c_ptr, M, N, K, stride_a, stride_b, stride_c};
    }

    KernelKey key_;
    std::string name_;
};

/// Helper to create a Stream-K kernel-instance wrapper.
template <typename SelectedKernel,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType>
std::shared_ptr<KernelInstance> create_generated_streamk_kernel(const KernelKey& key,
                                                                const std::string& name)
{
    return std::make_shared<GeneratedStreamKKernelInstance<SelectedKernel,
                                                           ADataType,
                                                           BDataType,
                                                           CDataType,
                                                           AccDataType>>(key, name);
}

} // namespace backends
} // namespace dispatcher
} // namespace ck_tile
