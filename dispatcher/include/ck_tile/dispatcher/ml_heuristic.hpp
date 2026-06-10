// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/kernel_key.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
namespace ck_tile {
namespace dispatcher {
extern "C" {
int LGBM_BoosterCreateFromModelfile(const char*, int*, void**);
int LGBM_BoosterPredictForMat(
    void*, const void*, int, int, int, int, int, int, int, const char*, int64_t*, double*);
int LGBM_BoosterFree(void*);
}
inline int encode_pipeline(Pipeline p)
{
    switch(p)
    {
    case Pipeline::CompV3: return 0;
    case Pipeline::CompV4: return 1;
    case Pipeline::CompV5: return 2;
    case Pipeline::Mem: return 3;
    case Pipeline::PreShuffleV2: return 4;
    case Pipeline::CompV6: return 5;
    default: return 0;
    }
}
inline int encode_scheduler(Scheduler s)
{
    switch(s)
    {
    case Scheduler::Intrawave: return 0;
    case Scheduler::Interwave: return 1;
    default: return 0;
    }
}
inline int encode_epilogue(Epilogue e)
{
    switch(e)
    {
    case Epilogue::Default: return 0;
    case Epilogue::CShuffle: return 1;
    default: return 0;
    }
}
inline int encode_layout(LayoutTag a, LayoutTag b, LayoutTag c)
{
    bool ra = (a == LayoutTag::RowMajor), rb = (b == LayoutTag::RowMajor);
    if(ra && !rb)
        return 0; // RCR
    if(ra && rb)
        return 1; // RRR
    if(!ra && rb)
        return 2; // CCR
    return 3;     // CRR
}
inline double dtype_bytes_ml(DataType dt)
{
    switch(dt)
    {
    case DataType::FP32: return 4;
    case DataType::FP16:
    case DataType::BF16: return 2;
    case DataType::FP8:
    case DataType::BF8:
    case DataType::INT8: return 1;
    case DataType::INT4: return 0.5;
    default: return 2;
    }
}
struct HardwareProfile
{
    int num_cus = 256, simds_per_cu = 4, shader_engines = 32, max_clock_mhz = 2400,
        max_waves_per_cu = 32, wavefront_size = 64, lds_capacity = 65536, l1_cache_kb = 32,
        l2_cache_kb = 4096, l3_cache_kb = 262144, num_xcd = 8;
    int total_simds() const { return num_cus * simds_per_cu; }
};

// CRITICAL: Feature count MUST match feature_spec.json
// Python training uses 72 features - this header MUST extract exactly 72 features in the same order
static constexpr int NUM_FEATURES = 72;

inline std::array<double, NUM_FEATURES>
extract_features(const Problem& prob, const KernelKey& key, const HardwareProfile& hw)
{
    // Problem dimensions
    double M = prob.M, N = prob.N, K = prob.K;
    double sk  = (prob.k_batch > 0 ? prob.k_batch : 1);
    double bpe = dtype_bytes_ml(key.signature.dtype_a);

    // Log-scale features
    double l2M   = std::log2(std::max(M, 1.0));
    double l2N   = std::log2(std::max(N, 1.0));
    double l2K   = std::log2(std::max(K, 1.0));
    double l2MNK = std::log2(std::max(M * N * K, 1.0));

    // Arithmetic intensity
    double mem = (M * K + K * N + M * N) * bpe;
    double ai  = 2.0 * M * N * K / std::max(mem, 1.0);

    // Aspect ratios
    double ar_mn = M / std::max(N, 1.0);
    double ar_mk = M / std::max(K, 1.0);
    double ar_nk = N / std::max(K, 1.0);

    // Layout encoding
    double layout = (double)encode_layout(
        key.signature.layout_a, key.signature.layout_b, key.signature.layout_c);

    // Tile dimensions
    double tm = key.algorithm.tile_shape.m;
    double tn = key.algorithm.tile_shape.n;
    double tk = key.algorithm.tile_shape.k;

    // Wave/warp dimensions
    double wm = key.algorithm.wave_shape.m;
    double wn = key.algorithm.wave_shape.n;
    double wk = key.algorithm.wave_shape.k;

    // Warp tile dimensions
    double wtm = key.algorithm.warp_tile_shape.m;
    double wtn = key.algorithm.warp_tile_shape.n;
    double wtk = key.algorithm.warp_tile_shape.k;

    // Algorithm encoding
    double pipeline  = (double)encode_pipeline(key.algorithm.pipeline);
    double scheduler = (double)encode_scheduler(key.algorithm.scheduler);
    double epilogue  = (double)encode_epilogue(key.algorithm.epilogue);

    // Padding flags - read from KernelKey
    double pad_m = key.algorithm.pad_m ? 1.0 : 0.0;
    double pad_n = key.algorithm.pad_n ? 1.0 : 0.0;
    double pad_k = key.algorithm.pad_k ? 1.0 : 0.0;

    // Persistent kernel flag
    double persistent = key.algorithm.persistent ? 1.0 : 0.0;

    // Derived features
    double num_warps   = wm * wn * wk;
    double tile_volume = tm * tn * tk;
    double tile_mn     = tm * tn;

    // LDS usage estimation
    double lest = (tm * tk + tn * tk) * bpe;
    double lcap = (key.algorithm.pipeline == Pipeline::CompV4) ? 32768.0 : (double)hw.lds_capacity;
    double lds_ratio = lest / std::max(lcap, 1.0);

    // Tile counts
    double ntm                = std::ceil(M / std::max(tm, 1.0));
    double ntn                = std::ceil(N / std::max(tn, 1.0));
    double ntk                = std::ceil(K / std::max(tk, 1.0));
    double total_output_tiles = ntm * ntn;

    // Tile efficiency (fractional remainder utilization)
    auto ef = [](double d, double t) -> double {
        if(t <= 0)
            return 1.0;
        double r = std::fmod(d, t);
        return r > 0 ? r / t : 1.0;
    };
    double tile_eff_m              = ef(M, tm);
    double tile_eff_n              = ef(N, tn);
    double tile_eff_k              = ef(K, tk);
    double overall_tile_efficiency = tile_eff_m * tile_eff_n * tile_eff_k;

    // CU utilization
    double cu_utilization = total_output_tiles / std::max((double)hw.num_cus, 1.0);

    // P0 FIX: Problem-to-tile ratio features (critical for small problems)
    double ratio_M_to_tile_m = M / std::max(tm, 1.0);
    double ratio_N_to_tile_n = N / std::max(tn, 1.0);
    double ratio_K_to_tile_k = K / std::max(tk, 1.0);

    // Binary features: is problem dimension smaller than tile?
    double problem_smaller_than_tile_m = (M < tm) ? 1.0 : 0.0;
    double problem_smaller_than_tile_n = (N < tn) ? 1.0 : 0.0;
    double problem_smaller_than_tile_k = (K < tk) ? 1.0 : 0.0;
    double any_dim_too_small           = ((M < tm) || (N < tn) || (K < tk)) ? 1.0 : 0.0;

    // P1 FIX: Padding requirement features
    double needs_padding_m = (tm > 0 && std::fmod(M, tm) != 0.0) ? 1.0 : 0.0;
    double needs_padding_n = (tn > 0 && std::fmod(N, tn) != 0.0) ? 1.0 : 0.0;
    double needs_padding_k = (tk > 0 && std::fmod(K, tk) != 0.0) ? 1.0 : 0.0;

    // Interaction features: kernel has padding when problem needs it
    double has_padding_when_needed_m = (needs_padding_m && pad_m) ? 1.0 : 0.0;
    double has_padding_when_needed_n = (needs_padding_n && pad_n) ? 1.0 : 0.0;
    double has_padding_when_needed_k = (needs_padding_k && pad_k) ? 1.0 : 0.0;

    // Critical feature: missing required padding (kernel will likely fail)
    double missing_required_padding_m = (needs_padding_m && !pad_m) ? 1.0 : 0.0;
    double missing_required_padding_n = (needs_padding_n && !pad_n) ? 1.0 : 0.0;
    double missing_required_padding_k = (needs_padding_k && !pad_k) ? 1.0 : 0.0;
    double missing_any_required_padding =
        (missing_required_padding_m || missing_required_padding_n || missing_required_padding_k)
            ? 1.0
            : 0.0;

    // Hardware features
    double hw_num_cus          = (double)hw.num_cus;
    double hw_simds_per_cu     = (double)hw.simds_per_cu;
    double hw_total_simds      = (double)hw.total_simds();
    double hw_shader_engines   = (double)hw.shader_engines;
    double hw_max_clock_mhz    = (double)hw.max_clock_mhz;
    double hw_max_waves_per_cu = (double)hw.max_waves_per_cu;
    double hw_wavefront_size   = (double)hw.wavefront_size;
    double hw_lds_capacity     = (double)hw.lds_capacity;
    double hw_l1_cache_kb      = (double)hw.l1_cache_kb;
    double hw_l2_cache_kb      = (double)hw.l2_cache_kb;
    double hw_l3_cache_kb      = (double)hw.l3_cache_kb;
    double hw_num_xcd          = (double)hw.num_xcd;

    // Feature vector in EXACT order from feature_spec.json
    // This order MUST match Python feature_engine.py::get_feature_names()
    return {{
        M,                            // 0
        N,                            // 1
        K,                            // 2
        sk,                           // 3 (split_k)
        l2M,                          // 4 (log2_M)
        l2N,                          // 5 (log2_N)
        l2K,                          // 6 (log2_K)
        l2MNK,                        // 7 (log2_MNK)
        ai,                           // 8 (arithmetic_intensity)
        ar_mn,                        // 9 (aspect_ratio_mn)
        ar_mk,                        // 10 (aspect_ratio_mk)
        ar_nk,                        // 11 (aspect_ratio_nk)
        layout,                       // 12 (layout)
        tm,                           // 13 (tile_m)
        tn,                           // 14 (tile_n)
        tk,                           // 15 (tile_k)
        wm,                           // 16 (warp_m)
        wn,                           // 17 (warp_n)
        wk,                           // 18 (warp_k)
        wtm,                          // 19 (warp_tile_m)
        wtn,                          // 20 (warp_tile_n)
        wtk,                          // 21 (warp_tile_k)
        pipeline,                     // 22 (pipeline)
        scheduler,                    // 23 (scheduler)
        epilogue,                     // 24 (epilogue)
        pad_m,                        // 25 (pad_m)
        pad_n,                        // 26 (pad_n)
        pad_k,                        // 27 (pad_k)
        persistent,                   // 28 (persistent)
        num_warps,                    // 29 (num_warps)
        tile_volume,                  // 30 (tile_volume)
        tile_mn,                      // 31 (tile_mn)
        lest,                         // 32 (lds_usage_estimate)
        lds_ratio,                    // 33 (lds_usage_ratio)
        ntm,                          // 34 (num_tiles_m)
        ntn,                          // 35 (num_tiles_n)
        ntk,                          // 36 (num_tiles_k)
        total_output_tiles,           // 37 (total_output_tiles)
        tile_eff_m,                   // 38 (tile_eff_m)
        tile_eff_n,                   // 39 (tile_eff_n)
        tile_eff_k,                   // 40 (tile_eff_k)
        overall_tile_efficiency,      // 41 (overall_tile_efficiency)
        cu_utilization,               // 42 (cu_utilization)
        ratio_M_to_tile_m,            // 43 (ratio_M_to_tile_m)
        ratio_N_to_tile_n,            // 44 (ratio_N_to_tile_n)
        ratio_K_to_tile_k,            // 45 (ratio_K_to_tile_k)
        problem_smaller_than_tile_m,  // 46 (problem_smaller_than_tile_m)
        problem_smaller_than_tile_n,  // 47 (problem_smaller_than_tile_n)
        problem_smaller_than_tile_k,  // 48 (problem_smaller_than_tile_k)
        any_dim_too_small,            // 49 (any_dim_too_small)
        needs_padding_m,              // 50 (needs_padding_m)
        needs_padding_n,              // 51 (needs_padding_n)
        needs_padding_k,              // 52 (needs_padding_k)
        has_padding_when_needed_m,    // 53 (has_padding_when_needed_m)
        has_padding_when_needed_n,    // 54 (has_padding_when_needed_n)
        has_padding_when_needed_k,    // 55 (has_padding_when_needed_k)
        missing_required_padding_m,   // 56 (missing_required_padding_m)
        missing_required_padding_n,   // 57 (missing_required_padding_n)
        missing_required_padding_k,   // 58 (missing_required_padding_k)
        missing_any_required_padding, // 59 (missing_any_required_padding)
        hw_num_cus,                   // 60 (hw_num_cus)
        hw_simds_per_cu,              // 61 (hw_simds_per_cu)
        hw_total_simds,               // 62 (hw_total_simds)
        hw_shader_engines,            // 63 (hw_shader_engines)
        hw_max_clock_mhz,             // 64 (hw_max_clock_mhz)
        hw_max_waves_per_cu,          // 65 (hw_max_waves_per_cu)
        hw_wavefront_size,            // 66 (hw_wavefront_size)
        hw_lds_capacity,              // 67 (hw_lds_capacity)
        hw_l1_cache_kb,               // 68 (hw_l1_cache_kb)
        hw_l2_cache_kb,               // 69 (hw_l2_cache_kb)
        hw_l3_cache_kb,               // 70 (hw_l3_cache_kb)
        hw_num_xcd,                   // 71 (hw_num_xcd)
    }};
}

class MLHeuristic
{
    public:
    MLHeuristic(const std::string& path,
                const Registry* reg,
                HardwareProfile hw = {},
                bool log_t         = false)
        : registry_(reg), hw_(hw), log_t_(log_t)
    {
        int iters = 0;
        if(LGBM_BoosterCreateFromModelfile(path.c_str(), &iters, &b_) != 0 || !b_)
        {
            std::cerr << "MLHeuristic: Failed to load " << path << std::endl;

            // Check if a compressed .gz version exists
            std::string gz_path = path + ".gz";
            std::ifstream gz_check(gz_path);
            if(gz_check.good())
            {
                std::cerr << "MLHeuristic: Found compressed model at " << gz_path << std::endl;
                std::cerr << "MLHeuristic: Please decompress it first:" << std::endl;
                std::cerr << "  gunzip " << gz_path << std::endl;
            }

            b_ = nullptr;
        }
        else
            std::cout << "MLHeuristic: Loaded (" << iters << " iters)" << std::endl;
    }
    ~MLHeuristic()
    {
        if(b_)
            LGBM_BoosterFree(b_);
    }
    MLHeuristic(const MLHeuristic&)            = delete;
    MLHeuristic& operator=(const MLHeuristic&) = delete;
    bool is_loaded() const { return b_ != nullptr; }
    double predict_tflops(const Problem& prob, const KernelKey& key) const
    {
        if(!b_)
            return 0;
        auto f      = extract_features(prob, key, hw_);
        int64_t ol  = 0;
        double pred = 0;
        // data_type=1 (C_API_DTYPE_FLOAT64): f is std::array<double>.
        if(LGBM_BoosterPredictForMat(
               b_, f.data(), 1, 1, NUM_FEATURES, 1, 0, 0, 0, "", &ol, &pred) != 0)
            return 0;
        return log_t_ ? std::expm1(pred) : pred;
    }
    std::vector<std::string> operator()(const Problem& prob) const
    {
        if(!b_ || !registry_)
            return {};
        auto insts = registry_->get_all();
        struct C
        {
            std::string id;
            double t;
        };
        std::vector<C> cs;
        cs.reserve(insts.size());
        for(auto& i : insts)
        {
            auto& k = i->get_key();
            cs.push_back({k.encode_identifier(), predict_tflops(prob, k)});
        }
        std::sort(cs.begin(), cs.end(), [](auto& a, auto& b) { return a.t > b.t; });
        std::vector<std::string> r;
        r.reserve(cs.size());
        for(auto& c : cs)
            r.push_back(std::move(c.id));
        return r;
    }

    private:
    void* b_                  = nullptr;
    const Registry* registry_ = nullptr;
    HardwareProfile hw_;
    bool log_t_ = false;
};
} // namespace dispatcher
} // namespace ck_tile
