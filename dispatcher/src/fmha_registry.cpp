// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/dispatcher/fmha_registry.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <sstream>

namespace ck_tile {
namespace dispatcher {

namespace {

std::string json_escape(const std::string& str)
{
    std::ostringstream oss;
    for(unsigned char c : str)
    {
        switch(c)
        {
        case '"': oss << "\\\""; break;
        case '\\': oss << "\\\\"; break;
        case '\b': oss << "\\b"; break;
        case '\f': oss << "\\f"; break;
        case '\n': oss << "\\n"; break;
        case '\r': oss << "\\r"; break;
        case '\t': oss << "\\t"; break;
        default:
            if(c < 0x20)
            {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                oss << buf;
            }
            else
            {
                oss << static_cast<char>(c);
            }
            break;
        }
    }
    return oss.str();
}

} // namespace

bool FmhaRegistry::register_kernel(FmhaKernelInstancePtr instance, Priority priority)
{
    if(!instance)
    {
        return false;
    }
    bool ok = Base::register_kernel(
        instance->get_key().encode_identifier(), std::move(instance), priority);
    if(ok)
    {
        perform_auto_export();
    }
    return ok;
}

FmhaKernelInstancePtr FmhaRegistry::lookup(const std::string& identifier) const
{
    std::lock_guard<std::mutex> lock(mutex());
    auto it = entries().find(identifier);
    return it != entries().end() ? it->second.instance : nullptr;
}

FmhaKernelInstancePtr FmhaRegistry::lookup(const FmhaKernelKey& key) const
{
    return lookup(key.encode_identifier());
}

std::vector<FmhaKernelInstancePtr> FmhaRegistry::get_all() const
{
    std::lock_guard<std::mutex> lock(mutex());

    struct RankedKernel
    {
        FmhaKernelInstancePtr instance;
        Priority priority;
    };

    std::vector<RankedKernel> ranked;
    ranked.reserve(entries().size());
    for(const auto& [name, entry] : entries())
    {
        ranked.push_back({entry.instance, entry.priority});
    }

    std::stable_sort(
        ranked.begin(), ranked.end(), [](const RankedKernel& lhs, const RankedKernel& rhs) {
            if(lhs.priority != rhs.priority)
            {
                return static_cast<int>(lhs.priority) > static_cast<int>(rhs.priority);
            }

            const auto& lkey = lhs.instance->get_key();
            const auto& rkey = rhs.instance->get_key();
            if(lkey.algorithm.selection_rank != rkey.algorithm.selection_rank)
            {
                return lkey.algorithm.selection_rank < rkey.algorithm.selection_rank;
            }

            if(lkey.signature.hdim_q != rkey.signature.hdim_q)
            {
                return lkey.signature.hdim_q < rkey.signature.hdim_q;
            }

            if(lkey.signature.hdim_v != rkey.signature.hdim_v)
            {
                return lkey.signature.hdim_v < rkey.signature.hdim_v;
            }

            if(lkey.algorithm.tile_shape.m0 != rkey.algorithm.tile_shape.m0)
            {
                return lkey.algorithm.tile_shape.m0 < rkey.algorithm.tile_shape.m0;
            }

            return lhs.instance->get_name() < rhs.instance->get_name();
        });

    std::vector<FmhaKernelInstancePtr> result;
    result.reserve(ranked.size());
    for(const auto& entry : ranked)
    {
        result.push_back(entry.instance);
    }
    return result;
}

std::vector<FmhaKernelInstancePtr>
FmhaRegistry::filter(std::function<bool(const FmhaKernelInstance&)> predicate) const
{
    auto all = get_all();
    std::vector<FmhaKernelInstancePtr> result;
    result.reserve(all.size());
    for(const auto& instance : all)
    {
        if(predicate(*instance))
        {
            result.push_back(instance);
        }
    }
    return result;
}

std::string FmhaRegistry::export_json(bool include_statistics) const
{
    auto all = get_all();

    std::ostringstream json;
    json << "{\n";
    json << "  \"metadata\": {\n";
    json << "    \"registry_name\": \"" << json_escape(get_name()) << "\",\n";
    json << "    \"total_kernels\": " << all.size() << "\n";
    json << "  }";

    if(include_statistics)
    {
        std::map<std::string, int> by_family;
        std::map<std::string, int> by_dtype;
        std::map<std::string, int> by_pipeline;

        for(const auto& kernel : all)
        {
            const auto& key = kernel->get_key();
            by_family[to_string(key.signature.family)]++;
            by_dtype[key.signature.data_type]++;
            by_pipeline[key.algorithm.pipeline]++;
        }

        json << ",\n  \"statistics\": {\n";
        auto emit_map = [&](const char* label, const auto& values, bool last) {
            json << "    \"" << label << "\": {";
            bool first = true;
            for(const auto& [name, count] : values)
            {
                if(!first)
                {
                    json << ",";
                }
                json << "\"" << json_escape(name) << "\":" << count;
                first = false;
            }
            json << "}";
            json << (last ? "\n" : ",\n");
        };

        emit_map("by_family", by_family, false);
        emit_map("by_dtype", by_dtype, false);
        emit_map("by_pipeline", by_pipeline, true);
        json << "  }";
    }

    json << ",\n  \"kernels\": [\n";
    for(std::size_t i = 0; i < all.size(); ++i)
    {
        const auto& kernel = all[i];
        const auto& key    = kernel->get_key();
        json << "    {\n";
        json << "      \"name\": \"" << json_escape(kernel->get_name()) << "\",\n";
        json << "      \"identifier\": \"" << json_escape(key.encode_identifier()) << "\",\n";
        json << "      \"family\": \"" << to_string(key.signature.family) << "\",\n";
        json << "      \"dtype\": \"" << json_escape(key.signature.data_type) << "\",\n";
        json << "      \"pipeline\": \"" << json_escape(key.algorithm.pipeline) << "\",\n";
        json << "      \"gfx_arch\": \"" << json_escape(key.gfx_arch) << "\"\n";
        json << "    }";
        if(i + 1 < all.size())
        {
            json << ",";
        }
        json << "\n";
    }
    json << "  ]\n";
    json << "}\n";
    return json.str();
}

bool FmhaRegistry::export_json_to_file(const std::string& filename, bool include_statistics) const
{
    std::ofstream file(filename);
    if(!file.is_open())
    {
        return false;
    }
    file << export_json(include_statistics);
    return true;
}

std::size_t FmhaRegistry::filter_by_arch(const std::string& gpu_arch)
{
    std::lock_guard<std::mutex> lock(mutex());

    std::vector<std::string> to_remove;
    for(const auto& [name, entry] : entries())
    {
        const auto& arch = entry.instance->get_key().gfx_arch;
        if(!arch.empty() && arch != gpu_arch)
        {
            to_remove.push_back(name);
        }
    }

    for(const auto& name : to_remove)
    {
        entries_mut().erase(name);
    }

    return to_remove.size();
}

std::size_t FmhaRegistry::filter_by_receipt(int receipt_id)
{
    std::lock_guard<std::mutex> lock(mutex());
    std::vector<std::string> to_remove;
    for(const auto& [name, entry] : entries())
    {
        if(entry.instance)
        {
            int r = entry.instance->get_key().signature.receipt;
            if(r >= 0 && r != receipt_id)
            {
                to_remove.push_back(name);
            }
        }
    }
    for(const auto& name : to_remove)
    {
        entries_mut().erase(name);
    }
    return to_remove.size();
}

std::vector<int> FmhaRegistry::available_receipts() const
{
    std::lock_guard<std::mutex> lock(mutex());
    std::set<int> receipts;
    for(const auto& [name, entry] : entries())
    {
        if(entry.instance)
        {
            int r = entry.instance->get_key().signature.receipt;
            if(r >= 0)
                receipts.insert(r);
        }
    }
    return {receipts.begin(), receipts.end()};
}

FmhaRegistry& FmhaRegistry::instance()
{
    static FmhaRegistry registry;
    return registry;
}

} // namespace dispatcher
} // namespace ck_tile
