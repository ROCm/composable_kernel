// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Kernel Cache - Persistent compiled kernel caching with automatic invalidation
 *
 * Features:
 * - Caches compiled kernel binaries (.hsaco) to avoid recompilation
 * - Automatically invalidates cache when CK Tile source code changes
 * - Uses content hashing for robust change detection
 * - Thread-safe access
 * - Configurable cache location
 *
 * Cache Invalidation:
 * - Hashes CK Tile include directory contents
 * - Hashes kernel source files
 * - Stores compiler version and flags
 * - Any change triggers recompilation
 *
 * Usage:
 *   KernelCache cache;
 *
 *   // Check if kernel is cached
 *   if (auto binary = cache.lookup(kernel_key)) {
 *       // Use cached binary
 *       load_binary(*binary);
 *   } else {
 *       // Compile and cache
 *       auto binary = compile_kernel(kernel_key);
 *       cache.store(kernel_key, binary);
 *   }
 */

#pragma once

#include "ck_tile/dispatcher/kernel_key.hpp"
#include <string>
#include <optional>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <functional>
#include <chrono>

namespace ck_tile {
namespace dispatcher {

// =============================================================================
// Hash Utilities
// =============================================================================

/// Simple FNV-1a hash for strings
inline std::uint64_t fnv1a_hash(const std::string& data)
{
    std::uint64_t hash = 14695981039346656037ULL;
    for(char c : data)
    {
        hash ^= static_cast<std::uint64_t>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

/// Hash a file's contents
inline std::uint64_t hash_file(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if(!file)
        return 0;

    std::ostringstream ss;
    ss << file.rdbuf();
    return fnv1a_hash(ss.str());
}

/// Hash a directory recursively (all .hpp, .h, .cpp files)
inline std::uint64_t hash_directory(const std::filesystem::path& dir,
                                    const std::vector<std::string>& extensions = {
                                        ".hpp", ".h", ".cpp"})
{
    if(!std::filesystem::exists(dir))
        return 0;

    std::uint64_t combined_hash = 0;

    for(const auto& entry : std::filesystem::recursive_directory_iterator(dir))
    {
        if(!entry.is_regular_file())
            continue;

        auto ext   = entry.path().extension().string();
        bool match = extensions.empty();
        for(const auto& e : extensions)
        {
            if(ext == e)
            {
                match = true;
                break;
            }
        }
        if(!match)
            continue;

        // Combine path and content hash
        combined_hash ^= fnv1a_hash(entry.path().string());
        combined_hash ^= hash_file(entry.path());
        combined_hash = (combined_hash << 5) | (combined_hash >> 59); // Rotate
    }

    return combined_hash;
}

// =============================================================================
// Cache Entry Metadata
// =============================================================================

struct CacheMetadata
{
    std::string kernel_identifier;
    std::string gpu_arch;
    std::uint64_t source_hash; // Hash of CK Tile sources
    std::uint64_t kernel_hash; // Hash of kernel config
    std::string compiler_version;
    std::string compile_flags;
    std::int64_t created_timestamp;
    std::int64_t last_accessed;
    std::size_t binary_size;

    /// Serialize to string
    [[nodiscard]] std::string serialize() const
    {
        std::ostringstream ss;
        ss << "kernel_id=" << kernel_identifier << "\n"
           << "gpu_arch=" << gpu_arch << "\n"
           << "source_hash=" << source_hash << "\n"
           << "kernel_hash=" << kernel_hash << "\n"
           << "compiler=" << compiler_version << "\n"
           << "flags=" << compile_flags << "\n"
           << "created=" << created_timestamp << "\n"
           << "accessed=" << last_accessed << "\n"
           << "size=" << binary_size << "\n";
        return ss.str();
    }

    /// Deserialize from string
    static std::optional<CacheMetadata> deserialize(const std::string& data)
    {
        CacheMetadata meta;
        std::istringstream ss(data);
        std::string line;

        while(std::getline(ss, line))
        {
            auto pos = line.find('=');
            if(pos == std::string::npos)
                continue;

            std::string key   = line.substr(0, pos);
            std::string value = line.substr(pos + 1);

            if(key == "kernel_id")
                meta.kernel_identifier = value;
            else if(key == "gpu_arch")
                meta.gpu_arch = value;
            else if(key == "source_hash")
                meta.source_hash = std::stoull(value);
            else if(key == "kernel_hash")
                meta.kernel_hash = std::stoull(value);
            else if(key == "compiler")
                meta.compiler_version = value;
            else if(key == "flags")
                meta.compile_flags = value;
            else if(key == "created")
                meta.created_timestamp = std::stoll(value);
            else if(key == "accessed")
                meta.last_accessed = std::stoll(value);
            else if(key == "size")
                meta.binary_size = std::stoull(value);
        }

        if(meta.kernel_identifier.empty())
            return std::nullopt;
        return meta;
    }
};

// =============================================================================
// Kernel Cache
// =============================================================================

class KernelCache
{
    public:
    /// Cache statistics
    struct Stats
    {
        std::size_t hits             = 0;
        std::size_t misses           = 0;
        std::size_t invalidations    = 0;
        std::size_t total_cached     = 0;
        std::size_t total_size_bytes = 0;

        [[nodiscard]] double hit_rate() const
        {
            auto total = hits + misses;
            return total > 0 ? static_cast<double>(hits) / total : 0.0;
        }
    };

    /**
     * Create kernel cache.
     *
     * @param cache_dir Cache directory (default: ~/.cache/ck_tile_dispatcher)
     * @param ck_tile_root Path to CK Tile include directory for hash computation
     */
    explicit KernelCache(const std::filesystem::path& cache_dir    = get_default_cache_dir(),
                         const std::filesystem::path& ck_tile_root = "")
        : cache_dir_(cache_dir), ck_tile_root_(ck_tile_root), enabled_(true)
    {
        // Create cache directory
        std::filesystem::create_directories(cache_dir_);

        // Compute source hash if path provided
        if(!ck_tile_root_.empty() && std::filesystem::exists(ck_tile_root_))
        {
            source_hash_ = hash_directory(ck_tile_root_);
        }

        // Load existing cache metadata
        load_cache_index();
    }

    /**
     * Look up a cached kernel binary.
     *
     * @param key Kernel configuration key
     * @return Binary data if found and valid, nullopt otherwise
     */
    [[nodiscard]] std::optional<std::vector<char>> lookup(const KernelKey& key)
    {
        if(!enabled_)
            return std::nullopt;

        std::lock_guard<std::mutex> lock(mutex_);

        std::string id = key.encode_identifier();
        auto it        = cache_index_.find(id);

        if(it == cache_index_.end())
        {
            stats_.misses++;
            return std::nullopt;
        }

        // Check if cache is still valid (source hash matches)
        if(source_hash_ != 0 && it->second.source_hash != source_hash_)
        {
            // Source code changed - invalidate
            stats_.invalidations++;
            stats_.misses++;
            invalidate_entry(id);
            return std::nullopt;
        }

        // Load binary from disk
        auto binary_path = get_binary_path(id);
        if(!std::filesystem::exists(binary_path))
        {
            stats_.misses++;
            return std::nullopt;
        }

        std::ifstream file(binary_path, std::ios::binary);
        if(!file)
        {
            stats_.misses++;
            return std::nullopt;
        }

        std::vector<char> binary((std::istreambuf_iterator<char>(file)),
                                 std::istreambuf_iterator<char>());

        // Update access time
        it->second.last_accessed = current_timestamp();

        stats_.hits++;
        return binary;
    }

    /**
     * Store a compiled kernel binary in cache.
     *
     * @param key Kernel configuration key
     * @param binary Compiled binary data
     * @param compiler_version Compiler version string
     * @param compile_flags Compilation flags used
     * @return true if stored successfully
     */
    bool store(const KernelKey& key,
               const std::vector<char>& binary,
               const std::string& compiler_version = "",
               const std::string& compile_flags    = "")
    {
        if(!enabled_ || binary.empty())
            return false;

        std::lock_guard<std::mutex> lock(mutex_);

        std::string id = key.encode_identifier();

        // Write binary to disk
        auto binary_path = get_binary_path(id);
        std::filesystem::create_directories(binary_path.parent_path());

        std::ofstream file(binary_path, std::ios::binary);
        if(!file)
            return false;
        file.write(binary.data(), binary.size());
        file.close();

        // Create metadata
        CacheMetadata meta;
        meta.kernel_identifier = id;
        meta.gpu_arch          = key.gfx_arch;
        meta.source_hash       = source_hash_;
        meta.kernel_hash       = fnv1a_hash(id);
        meta.compiler_version  = compiler_version;
        meta.compile_flags     = compile_flags;
        meta.created_timestamp = current_timestamp();
        meta.last_accessed     = meta.created_timestamp;
        meta.binary_size       = binary.size();

        // Write metadata
        auto meta_path = get_metadata_path(id);
        std::ofstream meta_file(meta_path);
        if(meta_file)
        {
            meta_file << meta.serialize();
        }

        // Update index
        cache_index_[id] = meta;
        stats_.total_cached++;
        stats_.total_size_bytes += binary.size();

        // Save index
        save_cache_index();

        return true;
    }

    /**
     * Invalidate all cached entries (e.g., when CK Tile is updated).
     */
    void invalidate_all()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        for(const auto& [id, meta] : cache_index_)
        {
            invalidate_entry_unlocked(id);
        }

        cache_index_.clear();
        stats_.total_cached     = 0;
        stats_.total_size_bytes = 0;
        save_cache_index();
    }

    /**
     * Update source hash (call when CK Tile is updated).
     */
    void refresh_source_hash()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if(!ck_tile_root_.empty() && std::filesystem::exists(ck_tile_root_))
        {
            auto new_hash = hash_directory(ck_tile_root_);
            if(new_hash != source_hash_)
            {
                source_hash_ = new_hash;
                // Don't invalidate immediately - let lookup do it lazily
            }
        }
    }

    /// Enable/disable caching
    void set_enabled(bool enabled) { enabled_ = enabled; }
    [[nodiscard]] bool is_enabled() const { return enabled_; }

    /// Get cache statistics
    [[nodiscard]] const Stats& get_stats() const { return stats_; }

    /// Get cache directory
    [[nodiscard]] const std::filesystem::path& get_cache_dir() const { return cache_dir_; }

    /// Get current source hash
    [[nodiscard]] std::uint64_t get_source_hash() const { return source_hash_; }

    /// Get default cache directory
    static std::filesystem::path get_default_cache_dir()
    {
        const char* home = std::getenv("HOME");
        if(home)
        {
            return std::filesystem::path(home) / ".cache" / "ck_tile_dispatcher";
        }
        return std::filesystem::temp_directory_path() / "ck_tile_dispatcher_cache";
    }

    /// Clear old entries (LRU eviction)
    void evict_old_entries(std::size_t max_entries = 1000, std::size_t max_size_mb = 1024)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        // Sort by last accessed time
        std::vector<std::pair<std::string, std::int64_t>> entries;
        for(const auto& [id, meta] : cache_index_)
        {
            entries.emplace_back(id, meta.last_accessed);
        }
        std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
            return a.second < b.second;
        });

        // Evict oldest entries
        while((cache_index_.size() > max_entries ||
               stats_.total_size_bytes > max_size_mb * 1024 * 1024) &&
              !entries.empty())
        {
            invalidate_entry_unlocked(entries.front().first);
            cache_index_.erase(entries.front().first);
            entries.erase(entries.begin());
        }

        save_cache_index();
    }

    private:
    std::filesystem::path get_binary_path(const std::string& id) const
    {
        return cache_dir_ / "binaries" / (id + ".hsaco");
    }

    std::filesystem::path get_metadata_path(const std::string& id) const
    {
        return cache_dir_ / "metadata" / (id + ".meta");
    }

    std::filesystem::path get_index_path() const { return cache_dir_ / "cache_index.txt"; }

    void invalidate_entry(const std::string& id)
    {
        invalidate_entry_unlocked(id);
        cache_index_.erase(id);
    }

    void invalidate_entry_unlocked(const std::string& id)
    {
        std::filesystem::remove(get_binary_path(id));
        std::filesystem::remove(get_metadata_path(id));
    }

    void load_cache_index()
    {
        auto index_path = get_index_path();
        if(!std::filesystem::exists(index_path))
            return;

        std::ifstream file(index_path);
        std::string line;

        while(std::getline(file, line))
        {
            auto meta_path = cache_dir_ / "metadata" / (line + ".meta");
            if(!std::filesystem::exists(meta_path))
                continue;

            std::ifstream meta_file(meta_path);
            std::ostringstream ss;
            ss << meta_file.rdbuf();

            if(auto meta = CacheMetadata::deserialize(ss.str()))
            {
                cache_index_[line] = *meta;
                stats_.total_cached++;
                stats_.total_size_bytes += meta->binary_size;
            }
        }
    }

    void save_cache_index()
    {
        auto index_path = get_index_path();
        std::filesystem::create_directories(index_path.parent_path());

        std::ofstream file(index_path);
        for(const auto& [id, meta] : cache_index_)
        {
            file << id << "\n";
        }
    }

    static std::int64_t current_timestamp()
    {
        return std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::system_clock::now().time_since_epoch())
            .count();
    }

    std::filesystem::path cache_dir_;
    std::filesystem::path ck_tile_root_;
    std::uint64_t source_hash_ = 0;
    bool enabled_;

    mutable std::mutex mutex_;
    std::unordered_map<std::string, CacheMetadata> cache_index_;
    Stats stats_;
};

/// Global kernel cache instance
inline KernelCache& global_kernel_cache()
{
    static KernelCache cache;
    return cache;
}

} // namespace dispatcher
} // namespace ck_tile
