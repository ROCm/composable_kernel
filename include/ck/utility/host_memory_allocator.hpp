// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/utility/env.hpp"
#include <map>
#include <queue>
#include <mutex>
#include <cstddef>
#include <limits>
#include <type_traits>
#include "unistd.h"

CK_DECLARE_ENV_VAR_BOOL(CK_USE_DYNAMIC_MEM_POOL)
CK_DECLARE_ENV_VAR_BOOL(CK_PREFER_NEW_PINNED_MEM_ALLOCATION)

namespace ck {
namespace memory {

    class IMemPool
    {
    public:
        virtual ~IMemPool() = default;
        virtual void* allocate(std::size_t sizeInBytes) = 0;
        virtual void deallocate(void* p, std::size_t sizeInBytes) = 0;
    };  

    class DynamicMemPool : public IMemPool
    {
    public:
        DynamicMemPool() : 
            enableLogging_(ck::EnvIsEnabled(CK_ENV(CK_LOGGING))),
            pid_(getpid())
        {
            if (enableLogging_)
                std::cout << "[ MemPool ] Created memory pool for process " << pid_ << std::endl;
        }

        ~DynamicMemPool() override
        {
            if (enableLogging_)
                std::cout << "[ MemPool ] Deleting pool for process " << pid_ << "..."<< std::endl;

            std::lock_guard<std::mutex> lock(mutex_);
            for (auto& [size, q] : memory_pool_)
            {
                clearMemoryPoolQueue(q);
            }

            if (enableLogging_)
                std::cout << "[ MemPool ] Deleted pool for process " << pid_ << std::endl;
        }

        void* allocate(std::size_t sizeInBytes) override
        {
            std::lock_guard<std::mutex> lock(mutex_);
            // If there is a memory pool for the requested size, return the memory from the pool.
            if (memory_pool_.find(sizeInBytes) != memory_pool_.end() && !memory_pool_[sizeInBytes].empty())
            {
                if (enableLogging_)
                {
                    std::cout << "[ MemPool ] Reusing memory from pool for size " << sizeInBytes << std::endl;
                }
                void* p = memory_pool_[sizeInBytes].front();
                memory_pool_[sizeInBytes].pop();
                memPoolSizeInBytes_ -= sizeInBytes;

                if (enableLogging_)
                {
                    std::cout << "[ MemPool ] Total memory in pool: " << memPoolSizeInBytes_ << std::endl;
                }
                return p;
            }

            if (enableLogging_)
            {
                std::cout << "[ MemPool ] Allocating new memory for size " << sizeInBytes << std::endl;
            }
            void* p;
            constexpr unsigned flags = hipDeviceScheduleYield; //hipDeviceScheduleSpin doesn not work, leads to freezing.
            hip_check_error(hipHostMalloc(&p, sizeInBytes, flags));
            return p;
        }

        void deallocate(void* p, std::size_t sizeInBytes) override
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (memory_pool_.find(sizeInBytes) != memory_pool_.end())
            {
                if (enableLogging_)
                {
                    std::cout << "[ MemPool ] Adding memory to pool for size " << sizeInBytes << std::endl;
                }
                auto& q = memory_pool_[sizeInBytes];
                q.push(p);
                memPoolSizeInBytes_ += sizeInBytes;
                // If the memory pool size exceeds the maximum size, free the memory.
                if (memPoolSizeInBytes_ > maxMemoryPoolSizeInBytes_)
                {
                    if (enableLogging_)
                    {
                        std::cout << "[ MemPool ] Clearing pool queue for size " << sizeInBytes << std::endl;
                    }
                    memPoolSizeInBytes_ -= sizeInBytes * q.size();
                    clearMemoryPoolQueue(q);
                }
            }
            else {
                if (enableLogging_)
                {
                    std::cout << "[ MemPool ] Creating new pool queue for size " << sizeInBytes << std::endl;
                }
                std::queue<void*> q;
                q.push(p);
                memory_pool_.insert(std::make_pair(sizeInBytes, std::move(q)));
                memPoolSizeInBytes_ += sizeInBytes;
            }
            if (enableLogging_)
            {
                std::cout << "[ MemPool ] Total memory in pool: " << memPoolSizeInBytes_ << std::endl;
            }
        }
    private:
        constexpr static size_t maxMemoryPoolSizeInBytes_ = 100 * 1024 * 1024; // 100MB

        static void clearMemoryPoolQueue(std::queue<void*>& q)
        {
            while (!q.empty())
            {
                void* p = q.front();
                q.pop();
                // This performs an implicit hipDeviceSynchronize().
                // Does this create a deadlock situation when grouped GEMM is used in distributed training with NCCL? 
                hip_check_error(hipHostFree(p));
            }
        }

        std::mutex mutex_; // Mutex to protect access to the memory pool.
        std::map<size_t, std::queue<void*>> memory_pool_{};
        size_t memPoolSizeInBytes_{0};
        bool enableLogging_{false};
        int pid_{-1};
    };

    class StaticMemPool : public IMemPool
    {
    public:
        StaticMemPool() : 
            enableLogging_(ck::EnvIsEnabled(CK_ENV(CK_LOGGING))),
            pid_(getpid()),
            offsetInBytes_(0)
        {
            hip_check_error(hipHostMalloc(&pinnedMemoryBaseAddress_, memoryPoolSizeInBytes_));
            if (enableLogging_)
            {
                std::cout << "[ MemPool ] Created memory pool with " << memoryPoolSizeInBytes_ << " bytes for process " << pid_ << std::endl;
            }   
        }

        ~StaticMemPool() override
        {
            hip_check_error(hipHostFree(pinnedMemoryBaseAddress_));
            if (enableLogging_) 
            {
                std::cout << "[ MemPool ] Deleted pool for process " << pid_ << std::endl;
            }
        }

        void* allocate(std::size_t sizeInBytes) override
        {
            std::lock_guard<std::mutex> lock(mutex_);

            if (offsetInBytes_ + sizeInBytes < memoryPoolSizeInBytes_)
            {
                // Return new memory from the preallocated block
                void* p = pinnedMemoryBaseAddress_ + offsetInBytes_;
                offsetInBytes_ += sizeInBytes;
                if (enableLogging_)
                {
                    const auto pct = 100.0f * static_cast<float>(offsetInBytes_) / memoryPoolSizeInBytes_;
                    std::cout << "[ MemPool ] Allocation: return new memory, pinned host memory usage: " << pct << "%." << std::endl;
                }
                return p;
            }
            else if (memory_pool_.find(sizeInBytes) != memory_pool_.end() && !memory_pool_[sizeInBytes].empty())
            {
                // If there is a memory pool for the requested size, return memory from the pool.
                if (enableLogging_)
                {
                    std::cout << "[ MemPool ] Allocation: reusing memory from pool for size " << sizeInBytes << std::endl;
                }
                void* p = memory_pool_[sizeInBytes].front();
                memory_pool_[sizeInBytes].pop();
                return p;
            }
            else 
            {
                // Try to find memory from the queue that is nearest in size.
                std::pair<size_t, std::queue<void*>> nearest_queue = {std::numeric_limits<size_t>::max(), std::queue<void*>()};
                for (auto& [size, q] : memory_pool_)
                {
                    if (size > sizeInBytes && !q.empty() && size < nearest_queue.first)
                    {
                        nearest_queue = {size, q};
                    }
                }

                if (nearest_queue.first != std::numeric_limits<size_t>::max())
                {
                    if (enableLogging_)
                    {
                        std::cout << "[ MemPool ] Allocation: reusing memory from pool for size " << nearest_queue.first << 
                            " to allocate " << sizeInBytes << "bytes" <<std::endl;
                    }
                    void* p = nearest_queue.second.front();
                    nearest_queue.second.pop();
                    return p;
                }

                // TODO: Fix the pinned host memory fragmentation. This should be fairly rare case.
                if (enableLogging_)
                {
                    std::cerr << "[ MemPool ] Internal error: memory pool exhausted." << std::endl;
                }
                throw std::runtime_error("Memory pool exhausted.");
            }       
        }

        void deallocate(void* p, std::size_t sizeInBytes) override
        {
            std::lock_guard<std::mutex> lock(mutex_);

            if (memory_pool_.find(sizeInBytes) != memory_pool_.end())
            {
                if (enableLogging_)
                {
                    std::cout << "[ MemPool ] Deallocate: Adding memory to pool for size " << sizeInBytes << std::endl;
                }
                auto& q = memory_pool_[sizeInBytes];
                q.push(p);
            }
            else {
                if (enableLogging_)
                {
                    std::cout << "[ MemPool ] Deallocate: Creating new pool queue for size " << sizeInBytes << std::endl;
                }
                std::queue<void*> q;
                q.push(p);
                memory_pool_.insert(std::make_pair(sizeInBytes, std::move(q)));
            }
        }
    private:
        constexpr static size_t memoryPoolSizeInBytes_ = 10 * 1024 * 1024; // 10MB
        std::mutex mutex_; // Mutex to protect access to the memory pool.
        std::map<size_t, std::queue<void*>> memory_pool_{};
        std::byte* pinnedMemoryBaseAddress_;
        bool enableLogging_;
        int pid_;
        int offsetInBytes_;
    };

    class PinnedHostMemoryAllocatorBase
    {
    protected:
        static IMemPool& get_memory_pool() {
            //static DynamicMemPool memory_pool;
            static StaticMemPool memory_pool;
            return memory_pool;
        }
    };

    template <typename T>
    class PinnedHostMemoryAllocator : public PinnedHostMemoryAllocatorBase
    { 
    public:
        using value_type = T;
        using pointer = T*;
        using const_pointer = const T*;
        using void_pointer = void*;
        using const_void_pointer = const void*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        template <typename U>
        struct rebind {
            using other = PinnedHostMemoryAllocator<U>;
        };

        PinnedHostMemoryAllocator() = default;

        template <typename U>
        PinnedHostMemoryAllocator(const PinnedHostMemoryAllocator<U>&)
        {}

        T* allocate(std::size_t n) {
            auto& memory_pool = get_memory_pool();
            const size_t sizeInBytes = n * sizeof(T);
            return static_cast<T*>(memory_pool.allocate(sizeInBytes));
        }

        void deallocate(T* p, std::size_t n) 
        {    
            if constexpr (std::is_destructible_v<T>) 
            {
                for (size_t i = 0; i < n; ++i) {
                    p[i].~T();
                }
            }

            auto& memory_pool = get_memory_pool();
            const size_t sizeInBytes = n * sizeof(T);
            memory_pool.deallocate(p, sizeInBytes);
        }

        template<typename U, typename... Args>
        void construct(U* p, Args&&... args) {
            new(p) U(std::forward<Args>(args)...);
        }

        template<typename U>
        void destroy(U* p) noexcept {
            p->~U();
        }
    };

    template <typename T, typename U>
    bool operator==(const PinnedHostMemoryAllocator<T>&, const PinnedHostMemoryAllocator<U>&) { return true; }

    template <typename T, typename U>
    bool operator!=(const PinnedHostMemoryAllocator<T>&, const PinnedHostMemoryAllocator<U>&) { return false; }
}
}
