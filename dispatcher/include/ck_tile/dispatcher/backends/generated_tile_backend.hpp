// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/dispatcher/kernel_instance.hpp"
#include "ck_tile/dispatcher/backends/gfx125_gate.hpp"
#include "ck_tile/dispatcher/validation/reference_kernels.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/epilogue/tdm_epilogue.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include <hip/hip_runtime.h>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <string>
#include <type_traits>

namespace ck_tile {
namespace dispatcher {
namespace backends {

// True for kernels whose launcher rejects partial N/K tiles (preshuffle compute TDM).
template <typename Kernel, typename = void>
struct kernel_requires_full_nk_tiles : std::false_type
{
};
template <typename Kernel>
struct kernel_requires_full_nk_tiles<Kernel, std::void_t<decltype(Kernel::RequiresFullNKTiles)>>
    : std::bool_constant<Kernel::RequiresFullNKTiles>
{
};

// True when SelectedKernel declares GemmEpilogue (all codegen kernels do;
// hand-written test stubs without it fall back to key_.algorithm.epilogue).
template <typename Kernel, typename = void>
struct kernel_declares_epilogue : std::false_type
{
};
template <typename Kernel>
struct kernel_declares_epilogue<Kernel, std::void_t<typename Kernel::GemmEpilogue>> : std::true_type
{
};

/// Shared GEMM support check for both GeneratedTileKernelInstance (ctypes single-kernel
/// bridge) and GeneratedKernelInstance (C++ multi-kernel register_all_kernels.hpp).
template <typename SelectedKernel>
bool generated_gemm_supports(const KernelKey& key, const Problem& problem)
{
    if(!problem.is_valid())
        return false;

    // Tile-divisibility gate, mirroring ck_tile::GemmKernel::IsSupportedArgument
    // exactly. A dimension only needs to be a multiple of its tile size when an
    // operand whose contiguous (inner) axis is that dimension participates AND
    // padding for it is disabled. This is layout-dependent:
    //
    //   layout RowMajor A -> inner axis K   | layout ColMajor A -> inner axis M
    //   layout RowMajor B -> inner axis N   | layout ColMajor B -> inner axis K
    //   layout RowMajor C -> inner axis N   | layout ColMajor C -> inner axis M
    //
    // Pipelines that skip launch checks (TDM) clip A/B boxes in hardware instead.
    using GemmPipe    = typename SelectedKernel::GemmPipeline;
    const auto is_row = [](LayoutTag l) { return l == LayoutTag::RowMajor; };
    const bool row_c  = is_row(key.signature.layout_c);
    if constexpr(!pipeline_skips_launch_checks<GemmPipe>::value)
    {
        constexpr bool pad_m = SelectedKernel::kPadM;
        constexpr bool pad_n = SelectedKernel::kPadN;
        constexpr bool pad_k = SelectedKernel::kPadK;

        constexpr int tile_m = SelectedKernel::TileM;
        constexpr int tile_n = SelectedKernel::TileN;
        constexpr int tile_k = SelectedKernel::TileK;

        const bool row_a = is_row(key.signature.layout_a);
        const bool row_b = is_row(key.signature.layout_b);

        const bool require_m = (!row_a) || (!row_c); // ColMajor A or C gate M
        const bool require_n = row_b || row_c;       // RowMajor B or C gate N
        const bool require_k = row_a || (!row_b);    // RowMajor A or ColMajor B gate K

        const std::int64_t k_grain = static_cast<std::int64_t>(tile_k) * problem.k_batch;

        if(require_m && !pad_m && problem.M % tile_m != 0)
            return false;
        if(require_n && !pad_n && problem.N % tile_n != 0)
            return false;
        if(require_k && !pad_k && problem.K % k_grain != 0)
            return false;
    }
    else
    {
        const bool tdm_epilogue = [&]() {
            if constexpr(kernel_declares_epilogue<SelectedKernel>::value)
                return ck_tile::is_tdm_epilogue_v<typename SelectedKernel::GemmEpilogue>;
            else
                return key.algorithm.epilogue == Epilogue::Tdm ||
                       key.algorithm.epilogue == Epilogue::TdmStaged;
        }();
        // The TDM epilogues overwrite C, and non-PC TDM launchers emit a host k_batch == 1
        // guard. Plain CompTDMPC with CShuffle omits that guard for split-K, where
        // UniversalGemmKernel::IsSupportedArgument requires whole TileK splits per batch.
        const bool pc_cshuffle_split_k = !tdm_epilogue &&
                                         !kernel_requires_full_nk_tiles<SelectedKernel>::value &&
                                         key.algorithm.pipeline == Pipeline::CompTDMPC;
        if(problem.k_batch != 1)
        {
            if(!pc_cshuffle_split_k)
                return false;
            if(problem.K % problem.k_batch != 0 ||
               (problem.K / problem.k_batch) % SelectedKernel::TileK != 0)
                return false;
        }
        if constexpr(kernel_requires_full_nk_tiles<SelectedKernel>::value)
        {
            if(problem.N % SelectedKernel::TileN != 0 || problem.K % SelectedKernel::TileK != 0)
                return false;
        }
        // Mirror UniversalGemmKernel::IsCTensorSupported for non-TDM epilogues
        // (e.g. CShuffle on comp_tdm_v3 or comp_tdm_pc): A/B loads stay hardware-clipped,
        // while an unpadded contiguous C dimension must still divide the C tile.
        if(!tdm_epilogue)
        {
            const std::int64_t c_extent = row_c ? problem.N : problem.M;
            const int c_tile            = row_c ? SelectedKernel::TileN : SelectedKernel::TileM;
            const bool pad_c            = row_c ? SelectedKernel::kPadN : SelectedKernel::kPadM;
            if(!pad_c && c_extent % c_tile != 0)
                return false;
        }
    }

    // Select the host-side A/B distribution for this kernel's target.
    // This avoids a HIP device query for every candidate in the registry.
    const bool wave32  = key.gfx_arch.rfind("gfx9", 0) != 0;
    const auto width_a = wave32 ? GemmPipe::template GetVectorSizeA<true>()
                                : GemmPipe::template GetVectorSizeA<false>();
    const auto width_b = wave32 ? GemmPipe::template GetVectorSizeB<true>()
                                : GemmPipe::template GetVectorSizeB<false>();
    return vector_widths_divide(
        key, problem.M, problem.N, problem.K, width_a, width_b, SelectedKernel::VectorSizeC);
}

/// Derive layout-accurate GemmHostArgs strides from KernelKey (RowMajor A/B/C ->
/// K/N/N; ColMajor -> M/K/M) so non-square rrr/crr/ccr launches never assume rcr.
inline ck_tile::GemmHostArgs make_generated_gemm_host_args(
    const KernelKey& key, const Problem& problem, const void* a_ptr, const void* b_ptr, void* c_ptr)
{
    const auto is_row   = [](LayoutTag l) { return l == LayoutTag::RowMajor; };
    const auto stride_a = is_row(key.signature.layout_a) ? problem.K : problem.M;
    const auto stride_b = is_row(key.signature.layout_b) ? problem.N : problem.K;
    const auto stride_c = is_row(key.signature.layout_c) ? problem.N : problem.M;
    return ck_tile::GemmHostArgs(a_ptr,
                                 b_ptr,
                                 c_ptr,
                                 problem.k_batch,
                                 problem.M,
                                 problem.N,
                                 problem.K,
                                 stride_a,
                                 stride_b,
                                 stride_c);
}

/**
 * Kernel instance wrapper for unified_gemm_codegen.py generated kernels
 *
 * These kernels have structure:
 * - Types defined outside: using ADataType = ...; using BDataType = ...;
 * - struct SelectedKernel with static constexpr config and launch() method
 * - constexpr const char* KERNEL_NAME = "...";
 *
 * This is different from tile_engine style where everything is in SelectedKernel.
 */
template <typename SelectedKernelType,
          typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename AccDataType_>
class GeneratedTileKernelInstance : public KernelInstance
{
    public:
    using ADataType      = ADataType_;
    using BDataType      = BDataType_;
    using CDataType      = CDataType_;
    using AccDataType    = AccDataType_;
    using SelectedKernel = SelectedKernelType;

    GeneratedTileKernelInstance(const KernelKey& key, const std::string& name)
        : key_(key), name_(name)
    {
    }

    const KernelKey& get_key() const override { return key_; }

    bool supports(const Problem& problem) const override
    {
        return device_supports_pipeline<typename SelectedKernel::GemmPipeline>() &&
               generated_gemm_supports<SelectedKernel>(key_, problem);
    }

    std::string get_name() const override { return name_; }

    float run(const void* a_ptr,
              const void* b_ptr,
              void* c_ptr,
              const void** d_ptrs,
              const Problem& problem,
              void* stream) const override
    {
        (void)d_ptrs; // Not used in basic GEMM

        ck_tile::GemmHostArgs args =
            make_generated_gemm_host_args(key_, problem, a_ptr, b_ptr, c_ptr);

        const bool bench = this->benchmarking_;
        ck_tile::stream_config stream_cfg;
        stream_cfg.stream_id_      = reinterpret_cast<hipStream_t>(stream);
        stream_cfg.time_kernel_    = bench;
        stream_cfg.log_level_      = 0;
        stream_cfg.cold_niters_    = bench ? env_int("CK_TILE_BENCH_WARMUP", 50) : 0;
        stream_cfg.nrepeat_        = bench ? env_int("CK_TILE_BENCH_REPEAT", 100) : 1;
        stream_cfg.is_gpu_timer_   = bench;
        stream_cfg.flush_cache_    = bench && env_bool("CK_TILE_BENCH_FLUSH", true);
        stream_cfg.rotating_count_ = bench ? env_int("CK_TILE_BENCH_ROTATING", 1000) : 1;

        // Call the generated kernel's launch method
        return SelectedKernel::launch(args, stream_cfg);
    }

    bool validate(const void* a_ptr,
                  const void* b_ptr,
                  const void* c_ptr,
                  const void** d_ptrs,
                  const Problem& problem,
                  float tolerance) const override
    {
        (void)a_ptr;
        (void)b_ptr;
        (void)c_ptr;
        (void)d_ptrs;
        (void)problem;
        (void)tolerance;
        // Validation would require reference implementation
        return true;
    }

    private:
    // Read an integer benchmark knob from the environment, falling back to
    // `fallback` when unset or unparseable.
    static int env_int(const char* name, int fallback)
    {
        const char* v = std::getenv(name);
        if(v == nullptr || *v == '\0')
            return fallback;
        char* end      = nullptr;
        const long out = std::strtol(v, &end, 10);
        if(end == v)
            return fallback;
        return static_cast<int>(out);
    }

    // Read a boolean benchmark knob ("0"/"false"/"off", any case => false, else true).
    static bool env_bool(const char* name, bool fallback)
    {
        const char* v = std::getenv(name);
        if(v == nullptr || *v == '\0')
            return fallback;
        std::string s(v);
        for(char& c : s)
            if(c >= 'A' && c <= 'Z')
                c = static_cast<char>(c - 'A' + 'a');
        return !(s == "0" || s == "false" || s == "off");
    }

    KernelKey key_;
    std::string name_;
};

/// Helper function to create a generated tile kernel instance wrapper
template <typename SelectedKernel,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType>
std::shared_ptr<KernelInstance> create_generated_tile_kernel(const KernelKey& key,
                                                             const std::string& name)
{
    return std::make_shared<
        GeneratedTileKernelInstance<SelectedKernel, ADataType, BDataType, CDataType, AccDataType>>(
        key, name);
}

} // namespace backends
} // namespace dispatcher
} // namespace ck_tile