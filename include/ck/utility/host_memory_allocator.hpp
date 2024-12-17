// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>
#include "ck/host_utility/hip_check_error.hpp"
#include <map>
#include <queue>
#include <mutex>
#include "unistd.h"

namespace ck {
namespace memory {

    class MemPool
    {
    public:
        MemPool() : 
            enableLogging_(ck::EnvIsEnabled(CK_ENV(CK_LOGGING))),
            pid_(getpid())
        {
            if (enableLogging_)
                std::cout << "[ MemPool ] Created memory pool for process " << pid_ << std::endl;
        }

        ~MemPool()
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

        void* allocate(std::size_t sizeInBytes)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            // If there is a memory pool for the requested size, return the memory from the pool.
            if (memory_pool_.find(sizeInBytes) != memory_pool_.end() && !memory_pool_[sizeInBytes].empty())
            {
                void* p = memory_pool_[sizeInBytes].front();
                memory_pool_[sizeInBytes].pop();
                memPoolSizeInBytes_ -= sizeInBytes;
                return p;
            }
            void* p;
            hip_check_error(hipHostMalloc(&p, sizeInBytes));
            return p;
        }

        void deallocate(void* p, std::size_t sizeInBytes)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (memory_pool_.find(sizeInBytes) != memory_pool_.end())
            {
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
                std::queue<void*> q;
                q.push(p);
                memory_pool_.insert(std::make_pair(sizeInBytes, std::move(q)));
                memPoolSizeInBytes_ += sizeInBytes;
            }
        }
    private:
        constexpr static size_t maxMemoryPoolSizeInBytes_ = 10 * 1024 * 1024; // 10MB

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

    class PinnedHostMemoryAllocatorBase
    {
    protected:
        static MemPool& get_memory_pool() {
            static MemPool memory_pool;
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
        PinnedHostMemoryAllocator(const PinnedHostMemoryAllocator<U>& other) : std::allocator<T>(other) 
        {}

        T* allocate(std::size_t n) {
            auto& memory_pool = get_memory_pool();
            const size_t sizeInBytes = n * sizeof(T);
            return static_cast<T*>(memory_pool.allocate(sizeInBytes));
        }

        void deallocate(T* p, std::size_t n) {
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
