// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/host.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>

#ifdef __linux__
#include <dirent.h>
#endif

#ifndef CK_TILE_FMHA_ENABLE_HEAD_GROUPING
#define CK_TILE_FMHA_ENABLE_HEAD_GROUPING 1
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wno-unknown-warning-option"
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
#if CK_TILE_FMHA_ENABLE_HEAD_GROUPING
CK_TILE_DECLARE_ENV_VAR_BOOL(CK_TILE_FMHA_HEAD_GROUP_LOG)
CK_TILE_DECLARE_ENV_VAR_BOOL(CK_TILE_FMHA_DISABLE_HEAD_GROUPING)
CK_TILE_DECLARE_ENV_VAR_UINT64(CK_TILE_FMHA_LLC_CACHE_MB)

namespace fmha_fwd_head_grouping {

inline bool log_enabled()
{
    return ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_FMHA_HEAD_GROUP_LOG));
}

inline bool disabled_by_env()
{
    return ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_FMHA_DISABLE_HEAD_GROUPING));
}

inline bool is_decimal_string(const std::string& s)
{
    if(s.empty())
        return false;
    return std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
}

inline std::optional<long long> read_property_value(const std::string& filepath,
                                                    const std::string& key)
{
    std::ifstream fs(filepath);
    if(!fs.is_open())
        return std::nullopt;

    std::string k, v;
    while(fs >> k >> v)
    {
        if(k == key)
        {
            try
            {
                return std::stoll(v, nullptr, 0);
            }
            catch(...)
            {
                return std::nullopt;
            }
        }
        std::string rest;
        std::getline(fs, rest);
    }
    return std::nullopt;
}

#if defined(__linux__)

struct kfd_device_location
{
    int domain      = 0;
    int location_id = 0;
};

inline std::optional<kfd_device_location> get_current_kfd_location()
{
    int device = 0;
    if(hipGetDevice(&device) != hipSuccess)
        return std::nullopt;

    char bdf[64] = {};
    if(hipDeviceGetPCIBusId(bdf, sizeof(bdf), device) == hipSuccess)
    {
        unsigned int domain = 0, bus = 0, dev = 0, fn = 0;
        if(std::sscanf(bdf, "%x:%x:%x.%x", &domain, &bus, &dev, &fn) == 4)
        {
            return kfd_device_location{
                static_cast<int>(domain),
                static_cast<int>(((bus & 0xff) << 8) | ((dev & 0x1f) << 3) | (fn & 0x7))};
        }
    }

    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, device) != hipSuccess)
        return std::nullopt;

    return kfd_device_location{props.pciDomainID,
                               ((props.pciBusID & 0xff) << 8) | ((props.pciDeviceID & 0x1f) << 3)};
}

inline std::optional<std::string> find_matching_kfd_node(const kfd_device_location& loc)
{
    constexpr const char* kKfdNodesDir = "/sys/class/kfd/kfd/topology/nodes";
    DIR* dir                           = opendir(kKfdNodesDir);
    if(dir == nullptr)
        return std::nullopt;

    std::optional<std::string> matched;
    while(auto* ent = readdir(dir))
    {
        const std::string node_name(ent->d_name);
        if(!is_decimal_string(node_name))
            continue;

        const std::string prop_path = std::string(kKfdNodesDir) + "/" + node_name + "/properties";
        const auto location_val     = read_property_value(prop_path, "location_id");
        if(!location_val.has_value() || static_cast<int>(*location_val) != loc.location_id)
            continue;

        const auto domain_val = read_property_value(prop_path, "domain");
        if(domain_val.has_value() && static_cast<int>(*domain_val) != loc.domain)
            continue;

        matched = node_name;
        break;
    }

    closedir(dir);
    return matched;
}

inline size_t read_kfd_node_l3_bytes(const std::string& node_name)
{
    const std::string caches_dir = "/sys/class/kfd/kfd/topology/nodes/" + node_name + "/caches";
    DIR* dir                     = opendir(caches_dir.c_str());
    if(dir == nullptr)
        return 0;

    size_t l3_kb = 0;
    while(auto* ent = readdir(dir))
    {
        const std::string cache_name(ent->d_name);
        if(!is_decimal_string(cache_name))
            continue;

        const std::string prop_path = caches_dir + "/" + cache_name + "/properties";
        const auto level_val        = read_property_value(prop_path, "level");
        if(!level_val.has_value() || *level_val != 3)
            continue;

        const auto size_val = read_property_value(prop_path, "size");
        if(!size_val.has_value() || *size_val <= 0)
            continue;

        l3_kb = std::max(l3_kb, static_cast<size_t>(*size_val));
    }

    closedir(dir);
    return l3_kb * 1024ull;
}

inline size_t get_kfd_sysfs_llc_cache_bytes()
{
    const auto loc = get_current_kfd_location();
    if(!loc.has_value())
        return 0;

    const auto node = find_matching_kfd_node(*loc);
    if(!node.has_value())
        return 0;

    return read_kfd_node_l3_bytes(*node);
}

#else

inline size_t get_kfd_sysfs_llc_cache_bytes() { return 0; }

#endif

inline size_t get_default_llc_cache_bytes_for_arch(const std::string& arch);

inline size_t resolve_llc_cache_bytes_uncached(const std::string& arch)
{
    // If parsed LLC looks invalidly tiny, ignore it and fallback.
    constexpr size_t kMinValidKfdLlcBytes = 32ull * 1024ull;

    const size_t kfd_llc_bytes = get_kfd_sysfs_llc_cache_bytes();
    if(kfd_llc_bytes >= kMinValidKfdLlcBytes)
        return kfd_llc_bytes;

    const size_t default_cache_bytes = get_default_llc_cache_bytes_for_arch(arch);
    if(default_cache_bytes > 0)
        return default_cache_bytes;

    // No default configured -> no grouping.
    return 0;
}

inline bool ck_tile_is_rdna_arch(const std::string& arch)
{
    return arch.rfind("gfx11", 0) == 0 || arch.rfind("gfx12", 0) == 0;
}

inline size_t get_default_llc_cache_bytes_for_arch(const std::string& arch)
{
    if(arch == "gfx1100")
        return 96ull * 1024ull * 1024ull;
    if(arch == "gfx1101")
        return 64ull * 1024ull * 1024ull;
    if(arch == "gfx1102")
        return 32ull * 1024ull * 1024ull;
    if(arch == "gfx1151")
        return 32ull * 1024ull * 1024ull;
    if(arch == "gfx1200")
        return 32ull * 1024ull * 1024ull;
    if(arch == "gfx1201")
        return 64ull * 1024ull * 1024ull;
    return 0;
}

inline size_t get_llc_cache_bytes(const std::string& arch)
{
    // resolve once and reuse.
    static const size_t resolved_llc_bytes = [&]() -> size_t {
        const uint64_t llc_mb = ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_FMHA_LLC_CACHE_MB));
        if(llc_mb > 0)
        {
            constexpr uint64_t kBytesPerMb   = 1024ull * 1024ull;
            const uint64_t max_mb_for_size_t = static_cast<uint64_t>(
                std::numeric_limits<size_t>::max() / static_cast<size_t>(kBytesPerMb));

            if(llc_mb <= max_mb_for_size_t)
                return static_cast<size_t>(llc_mb * kBytesPerMb);
        }

        return resolve_llc_cache_bytes_uncached(arch);
    }();

    return resolved_llc_bytes;
}

inline std::optional<ck_tile::index_t> get_head_group_size(ck_tile::index_t nhead_q,
                                                           ck_tile::index_t nhead_k,
                                                           ck_tile::index_t batch,
                                                           ck_tile::index_t seqlen_k,
                                                           ck_tile::index_t hdim_q,
                                                           ck_tile::index_t hdim_v,
                                                           size_t elem_bytes_k,
                                                           size_t elem_bytes_v)
{
    if(disabled_by_env())
        return std::nullopt;

    const std::string arch = ck_tile::get_device_name();
    if(arch.empty() || !ck_tile_is_rdna_arch(arch))
        return std::nullopt;

    const size_t llc_bytes = get_llc_cache_bytes(arch);
    if(llc_bytes == 0)
        return std::nullopt;

    if(nhead_k <= 0 || nhead_q <= 0 || (nhead_q % nhead_k) != 0)
        return std::nullopt;
    if(seqlen_k <= 0 || hdim_q <= 0 || hdim_v <= 0 || batch <= 0)
        return std::nullopt;

    const size_t kv_bytes_per_head =
        static_cast<size_t>(seqlen_k) *
        (static_cast<size_t>(hdim_q) * elem_bytes_k + static_cast<size_t>(hdim_v) * elem_bytes_v);
    if(kv_bytes_per_head == 0)
        return std::nullopt;

    // large LLC GPUs (>= 64MB): slightly more cache-resident grouping
    constexpr size_t kLargeLlcThresholdBytes = 64ull * 1024ull * 1024ull;
    const bool is_large_llc                  = llc_bytes >= kLargeLlcThresholdBytes;
    const long double llc_utilization        = is_large_llc ? 0.85L : 1.0L;
    const long double threshold_ratio        = is_large_llc ? 1.3L : 1.5L;
    const size_t target_llc_bytes =
        static_cast<size_t>(static_cast<long double>(llc_bytes) * llc_utilization);
    if(target_llc_bytes == 0)
        return std::nullopt;

    const size_t total_kv_bytes = static_cast<size_t>(nhead_q) * kv_bytes_per_head;
    if(static_cast<long double>(total_kv_bytes) <
       static_cast<long double>(target_llc_bytes) * threshold_ratio)
        return std::nullopt;

    ck_tile::index_t group = static_cast<ck_tile::index_t>(target_llc_bytes / kv_bytes_per_head);
    if(group < 1)
        group = 1;

    const ck_tile::index_t min_group_size = std::max<ck_tile::index_t>(1, nhead_q / 16);
    if(group < min_group_size)
        group = min_group_size;

    // Cap the number of groups to avoid excessive launch overhead.
    constexpr ck_tile::index_t kMaxGroups = 8;
    const ck_tile::index_t min_group_for_max_groups =
        ck_tile::integer_divide_ceil(nhead_q, kMaxGroups);
    if(group < min_group_for_max_groups)
        group = min_group_for_max_groups;

    const ck_tile::index_t gqa_ratio = nhead_q / nhead_k;
    if(gqa_ratio > 1)
    {
        group = ((group + gqa_ratio - 1) / gqa_ratio) * gqa_ratio;
    }

    group = std::min(group, nhead_q);
    if(group >= nhead_q)
        return std::nullopt;

    return group;
}

template <typename T>
inline const void* ptr_offset(const void* base, ck_tile::index_t offset_elems)
{
    if(base == nullptr)
        return nullptr;
    return static_cast<const void*>(reinterpret_cast<const T*>(base) + offset_elems);
}

template <typename T>
inline void* ptr_offset(void* base, ck_tile::index_t offset_elems)
{
    if(base == nullptr)
        return nullptr;
    return static_cast<void*>(reinterpret_cast<T*>(base) + offset_elems);
}

template <typename QDataType,
          typename KDataType,
          typename VDataType,
          typename ODataType,
          typename BiasDataType,
          typename LSEDataType,
          typename RandValOutputDataType,
          typename FmhaFwdTraits,
          typename FmhaFwdArgs,
          typename RunKernelFn>
float run_fwd_head_grouped(const ck_tile::stream_config& sc,
                           const FmhaFwdTraits& fmha_traits,
                           const FmhaFwdArgs& base_args_in,
                           ck_tile::index_t nhead,
                           ck_tile::index_t nhead_k,
                           ck_tile::index_t group_size_q,
                           bool use_blockscale_qscale,
                           RunKernelFn&& run_kernel_fn)
{
    auto base_args                   = base_args_in;
    base_args.num_head_q_total       = nhead;
    const ck_tile::index_t gqa_ratio = (nhead_k > 0 ? (nhead / nhead_k) : 1);
    const ck_tile::index_t group_sz  = std::min(group_size_q, nhead);
    const ck_tile::index_t n_groups  = ck_tile::integer_divide_ceil(nhead, group_sz);

    float total_time = 0.0f;
    for(ck_tile::index_t head_start = 0; head_start < nhead; head_start += group_sz)
    {
        const ck_tile::index_t q_heads = std::min(group_sz, nhead - head_start);
        const ck_tile::index_t k_head_start =
            (gqa_ratio >= 1 ? head_start / gqa_ratio : head_start);
        const ck_tile::index_t k_heads = (gqa_ratio >= 1 ? q_heads / gqa_ratio : q_heads);

        auto args       = base_args;
        args.nhead_q    = q_heads;
        args.nhead_k    = k_heads;
        args.head_start = head_start;

        args.q_ptr = ptr_offset<QDataType>(base_args.q_ptr, head_start * base_args.nhead_stride_q);
        args.k_ptr =
            ptr_offset<KDataType>(base_args.k_ptr, k_head_start * base_args.nhead_stride_k);
        args.v_ptr =
            ptr_offset<VDataType>(base_args.v_ptr, k_head_start * base_args.nhead_stride_v);
        args.o_ptr = ptr_offset<ODataType>(base_args.o_ptr, head_start * base_args.nhead_stride_o);

        args.bias_ptr =
            ptr_offset<BiasDataType>(base_args.bias_ptr, head_start * base_args.nhead_stride_bias);
        args.lse_ptr =
            ptr_offset<LSEDataType>(base_args.lse_ptr, head_start * base_args.nhead_stride_lse);
        args.rand_val_ptr = ptr_offset<RandValOutputDataType>(
            base_args.rand_val_ptr, head_start * base_args.nhead_stride_randval);

        if(use_blockscale_qscale)
        {
            args.q_descale_ptr = ptr_offset<float>(base_args.q_descale_ptr,
                                                   head_start * base_args.nhead_stride_q_descale);
            args.k_descale_ptr = ptr_offset<float>(base_args.k_descale_ptr,
                                                   k_head_start * base_args.nhead_stride_k_descale);
            args.v_descale_ptr = ptr_offset<float>(base_args.v_descale_ptr,
                                                   k_head_start * base_args.nhead_stride_v_descale);
        }
        else
        {
            args.q_descale_ptr = base_args.q_descale_ptr;
            args.k_descale_ptr = base_args.k_descale_ptr;
            args.v_descale_ptr = base_args.v_descale_ptr;
        }

        args.sink_ptr = ptr_offset<float>(base_args.sink_ptr, head_start);

        if(log_enabled())
        {
            const ck_tile::index_t head_end = head_start + q_heads;
            std::cout << "[LLC Head Grouping] group " << (head_start / group_sz) << "/" << n_groups
                      << " heads_q=[" << head_start << ", " << head_end << ") heads_k=["
                      << k_head_start << ", " << (k_head_start + k_heads) << ")" << std::endl;
        }

        const float t = run_kernel_fn(fmha_traits, args, sc);
        if(t < 0.0f)
            return t;
        total_time += t;
    }
    return total_time;
}

} // namespace fmha_fwd_head_grouping
#endif // CK_TILE_FMHA_ENABLE_HEAD_GROUPING
#pragma clang diagnostic pop
