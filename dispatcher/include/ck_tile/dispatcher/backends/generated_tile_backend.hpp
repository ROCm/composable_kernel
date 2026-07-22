// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/dispatcher/kernel_instance.hpp"
#include "ck_tile/dispatcher/validation/reference_kernels.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include <hip/hip_runtime.h>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <string>

namespace ck_tile {
namespace dispatcher {
namespace backends {

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
        // Tile-divisibility gate, mirroring ck_tile::GemmKernel::IsSupportedArgument
        // exactly. A dimension only needs to be a multiple of its tile size when an
        // operand whose contiguous (inner) axis is that dimension participates AND
        // padding for it is disabled. This is layout-dependent:
        //
        //   layout RowMajor A -> inner axis K   | layout ColMajor A -> inner axis M
        //   layout RowMajor B -> inner axis N   | layout ColMajor B -> inner axis K
        //   layout RowMajor C -> inner axis N   | layout ColMajor C -> inner axis M
        //
        // The old check blindly required M % TileM == 0 for every layout, which
        // wrongly rejected e.g. rcr kernels (RowMajor A & C never gate M) on
        // M-indivisible problems that Old-TE runs fine. Anything this lets through
        // is still validated by the kernel's own IsSupportedArgument inside launch(),
        // so the bridge stays a strict functional equivalent of Old-TE.
        constexpr bool pad_m = SelectedKernel::kPadM;
        constexpr bool pad_n = SelectedKernel::kPadN;
        constexpr bool pad_k = SelectedKernel::kPadK;

        constexpr int tile_m = SelectedKernel::TileM;
        constexpr int tile_n = SelectedKernel::TileN;
        constexpr int tile_k = SelectedKernel::TileK;

        const auto is_row = [](LayoutTag l) { return l == LayoutTag::RowMajor; };
        const bool row_a  = is_row(key_.signature.layout_a);
        const bool row_b  = is_row(key_.signature.layout_b);
        const bool row_c  = is_row(key_.signature.layout_c);

        // Which problem dimensions are actually constrained for this layout combo.
        const bool require_m = (!row_a) || (!row_c); // ColMajor A or C gate M
        const bool require_n = row_b || row_c;       // RowMajor B or C gate N
        const bool require_k = row_a || (!row_b);    // RowMajor A or ColMajor B gate K

        const std::int64_t k_grain =
            static_cast<std::int64_t>(tile_k) * (problem.k_batch > 0 ? problem.k_batch : 1);

        if(require_m && !pad_m && problem.M % tile_m != 0)
            return false;
        if(require_n && !pad_n && problem.N % tile_n != 0)
            return false;
        if(require_k && !pad_k && problem.K % k_grain != 0)
            return false;

        return true;
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

        // Leading dimensions depend on each operand's layout, NOT a fixed
        // rcr assumption: RowMajor A/B/C -> inner axis is K/N/N; ColMajor ->
        // M/K/M. Hard-coding {K, K, N} only happens to be right for rcr and for
        // square problems (M==N==K); it corrupts every non-square rrr/ccr/crr
        // launch. Derive each stride from the kernel's real layout instead.
        const auto is_row   = [](LayoutTag l) { return l == LayoutTag::RowMajor; };
        const auto stride_a = is_row(key_.signature.layout_a) ? problem.K : problem.M;
        const auto stride_b = is_row(key_.signature.layout_b) ? problem.N : problem.K;
        const auto stride_c = is_row(key_.signature.layout_c) ? problem.N : problem.M;

        // Order from GemmHostArgs constructor: a_ptr, b_ptr, e_ptr, k_batch, M, N, K, stride_A,
        // stride_B, stride_E
        ck_tile::GemmHostArgs args(a_ptr,           // a_ptr
                                   b_ptr,           // b_ptr
                                   c_ptr,           // e_ptr/c_ptr
                                   problem.k_batch, // k_batch (4th argument!)
                                   problem.M,       // M
                                   problem.N,       // N
                                   problem.K,       // K
                                   stride_a,        // stride_A
                                   stride_b,        // stride_B
                                   stride_c         // stride_E/C
        );

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