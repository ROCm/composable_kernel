// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>
#include "ck/host_utility/hip_check_error.hpp"
#include <map>
#include <queue>
#include <mutex>

namespace ck {
namespace memory {

    class DebugStream
    {
    public:
        DebugStream(bool enable_output = true) : enable_output_(enable_output) {}

        template <typename T>
        DebugStream& operator<<(const T& value)
        {
            if (enable_output_)
            {
                std::cout << value;
            }
            return *this;
        }

        // Overload for std::ostream manipulators like std::endl
        using Manipulator = std::ostream& (*)(std::ostream&);
        DebugStream& operator<<(Manipulator manip)
        {
            if (enable_output_)
            {
                manip(std::cout);
            }
            return *this;
        }

        void enableOutput(bool enable)
        {
            enable_output_ = enable;
        }

    private:
        bool enable_output_;
    };

    template <typename T>
    class MemPool
    {
    public:
        MemPool() : debug_stream(true) {}

        ~MemPool()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            debug_stream << "Destroying memory pool of type " << typeid(T).name() << std::endl;
            for (auto& [size, q] : memory_pool_)
            {
                // Iterate through the queue and free the memory
                while (!q.empty())
                {
                    T* p = q.front();
                    q.pop();
                    hip_check_error(hipHostFree(p));
                }
            }
        }

        T* allocate(std::size_t n)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            debug_stream << "Allocating size " << n << " for type " << typeid(T).name() << std::endl;
            // If there is a memory pool for the requested size, return the memory from the pool.
            if (memory_pool_.find(n) != memory_pool_.end() && !memory_pool_[n].empty())
            {
                debug_stream << "\tReturning from memory pool" << std::endl;
                T* p = memory_pool_[n].front();
                memory_pool_[n].pop();
                return p;
            }
            debug_stream << "\tAllocating new memory" << std::endl;
            T* p;
            hip_check_error(hipHostMalloc(&p, n));
            return p;
        }

        void deallocate(T* p, std::size_t size)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (memory_pool_.find(size) != memory_pool_.end())
            {
                auto& q = memory_pool_[size];
                q.push(p);
                debug_stream << "Deallocating size " << size << " and type " << typeid(T).name() << " to memory pool." << std::endl;
                debug_stream << "\tPool size: " << q.size() << std::endl;

                // If the memory pool size exceeds the maximum size, free the memory.
                if (q.size() > maxMemoryPoolSize_)
                {
                    debug_stream << "Memory pool size exceeds the maximum size for type " << typeid(T).name()
                        << ". Freeing the memory." << std::endl;
                    while (!q.empty())
                    {
                        T* ptr = q.front();
                        q.pop();
                        hip_check_error(hipHostFree(ptr));
                    }
                }
            }
            else {
                debug_stream << "Creating new memory pool for size " << size << " and type " << typeid(T).name() << std::endl;
                std::queue<T*> q;
                q.push(p);
                memory_pool_.insert(std::make_pair(size, std::move(q)));
            }
        }
    private:
        constexpr static size_t maxMemoryPoolSizeInBytes_ = 1 << 20; // 1MB
        constexpr static size_t maxMemoryPoolSize_ = maxMemoryPoolSizeInBytes_ / sizeof(T);
        std::mutex mutex_; // Mutex to protect access to the memory pool.
        std::map<size_t, std::queue<T*>> memory_pool_{};
        DebugStream debug_stream;
    };

    template <typename T>
    class PinnedHostMemoryAllocator
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
            // T* p;
            // hip_check_error(hipHostMalloc(&p, n * sizeof(T)));
            // return p;
            auto& memory_pool = get_memory_pool();
            return memory_pool.allocate(n);
        }

        void deallocate(T* p, std::size_t size) {
            //hip_check_error(hipHostFree(p));
            auto& memory_pool = get_memory_pool();
            memory_pool.deallocate(p, size);
        }

        template<typename U, typename... Args>
        void construct(U* p, Args&&... args) {
            new(p) U(std::forward<Args>(args)...);
        }

        template<typename U>
        void destroy(U* p) noexcept {
            p->~U();
        }
    private:
        static MemPool<T>& get_memory_pool() {
            static MemPool<T> memory_pool_;
            return memory_pool_;
        }
    };

    template <typename T, typename U>
    bool operator==(const PinnedHostMemoryAllocator<T>&, const PinnedHostMemoryAllocator<U>&) { return true; }

    template <typename T, typename U>
    bool operator!=(const PinnedHostMemoryAllocator<T>&, const PinnedHostMemoryAllocator<U>&) { return false; }

}
}
