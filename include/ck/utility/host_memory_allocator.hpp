// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/utility/env.hpp"
#include <map>
#include <queue>
#include <stack>
#include <mutex>
#include <cstddef>
#include <limits>
#include <type_traits>
#include "unistd.h"

CK_DECLARE_ENV_VAR_BOOL(CK_USE_DYNAMIC_MEM_POOL)
CK_DECLARE_ENV_VAR_BOOL(CK_PREFER_RECYCLED_PINNED_MEM)
CK_DECLARE_ENV_VAR_UINT64(CK_PINNED_MEM_SIZE_KB)

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
        DynamicMemPool(size_t maxPoolSizeInBytes = defaultMaxMemoryPoolSizeInBytes_) : 
            enableLogging_(ck::EnvIsEnabled(CK_ENV(CK_LOGGING))),
            pid_(getpid()),
            maxPoolSizeInBytes_(maxPoolSizeInBytes)
        {
            if (enableLogging_)
            {
                std::cout << "[ DynamicMemPool ] Created memory pool for process " << pid_ << std::endl;
            }
        }

        ~DynamicMemPool() override
        {
            // Get keys of the map and clear the memory pool queue for each key.
            for (auto& [size, _] : memory_pool_)
            {
                clearMemoryPoolQueue(size);
            }

            if (enableLogging_)
            {
                std::cout << "[ DynamicMemPool ] Deleted pool for process " << pid_ << std::endl;
            }  
        }

        void* allocate(std::size_t sizeInBytes) override
        {
            std::lock_guard<std::mutex> lock(mutex_);
            // If there is a memory pool for the requested size, return the memory from the pool.
            if (memory_pool_.find(sizeInBytes) != memory_pool_.end() && !memory_pool_[sizeInBytes].empty())
            {

#ifdef ENABLE_MEM_POOL_LOGGING            
                if (enableLogging_)
                {
                    std::cout << "[ DynamicMemPool ] Reusing memory from pool for size " << sizeInBytes << std::endl;
                }
#endif
                
                void* p = memory_pool_[sizeInBytes].front();
                memory_pool_[sizeInBytes].pop();
                memPoolSizeInBytes_ -= sizeInBytes;

#ifdef ENABLE_MEM_POOL_LOGGING
                if (enableLogging_)
                {
                    std::cout << "[ DynamicMemPool ] Total memory in pool: " << memPoolSizeInBytes_ << std::endl;
                }
#endif
                return p;
            }

#ifdef ENABLE_MEM_POOL_LOGGING
            if (enableLogging_)
            {
                std::cout << "[ DynamicMemPool ] Allocating new memory for size " << sizeInBytes << std::endl;
            }
#endif

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
#ifdef ENABLE_MEM_POOL_LOGGING
                if (enableLogging_)
                {
                    std::cout << "[ DynamicMemPool ] Adding memory to pool for size " << sizeInBytes << std::endl;
                }
#endif
                memory_pool_[sizeInBytes].push(p);
                memPoolSizeInBytes_ += sizeInBytes;
                // If the memory pool size exceeds the maximum size, free the memory.
                if (memPoolSizeInBytes_ > maxPoolSizeInBytes_)
                {
                    if (enableLogging_)
                    {
                        std::cout << "[ DynamicMemPool ] Clearing pool queue for size " << sizeInBytes << std::endl;
                    }
                    memPoolSizeInBytes_ -= sizeInBytes * memory_pool_[sizeInBytes].size();
                    clearMemoryPoolQueue(sizeInBytes);
                }
            }
            else {
#ifdef ENABLE_MEM_POOL_LOGGING
                if (enableLogging_)
                {
                    std::cout << "[ DynamicMemPool ] Creating new pool queue for size " << sizeInBytes << std::endl;
                }
#endif
                std::queue<void*> q;
                q.push(p);
                memory_pool_.insert({sizeInBytes, std::move(q)});
                memPoolSizeInBytes_ += sizeInBytes;
            }
#ifdef ENABLE_MEM_POOL_LOGGING
            if (enableLogging_)
            {
                std::cout << "[ DynamicMemPool ] Total memory in pool: " << memPoolSizeInBytes_ << std::endl;
            }
#endif
        }
    private:
        constexpr static size_t defaultMaxMemoryPoolSizeInBytes_ = 1 * 1024 * 1024; // 1MB

        void clearMemoryPoolQueue(size_t sizeInBytes)
        {
            while (!memory_pool_[sizeInBytes].empty())
            {
                void* p = memory_pool_[sizeInBytes].front();
                memory_pool_[sizeInBytes].pop(); 
                hip_check_error(hipHostFree(p));
            }
        }

        std::mutex mutex_; // Mutex to protect access to the memory pool.
        std::map<size_t, std::queue<void*>> memory_pool_{};
        size_t memPoolSizeInBytes_{0};
        bool enableLogging_{false};
        int pid_{-1};
        size_t maxPoolSizeInBytes_;
    };

    class StaticMemPool : public IMemPool
    {
    public:
        StaticMemPool(size_t poolSizeInBytes = defaultMaxMemoryPoolSizeInBytes_) : 
            enableLogging_(ck::EnvIsEnabled(CK_ENV(CK_LOGGING))),
            pid_(getpid()),
            offsetInBytes_(0),
            preferRecycledMem_(ck::EnvIsEnabled(CK_ENV(CK_PREFER_RECYCLED_PINNED_MEM))),
            activeMemoryPoolSizeInBytes_(poolSizeInBytes)
        {
            if (!ck::EnvIsUnset(CK_ENV(CK_PINNED_MEM_SIZE_KB)))
            {
                // kB to bytes conversion
                constexpr size_t KB = 1024;
                activeMemoryPoolSizeInBytes_ = ck::EnvValue(CK_ENV(CK_PINNED_MEM_SIZE_KB)) * KB;
                std::cout << "[ StaticMemPool ] Override of default memory size to " << activeMemoryPoolSizeInBytes_ << " bytes." << std::endl;
            }
            allocateNewPinnedMemoryBlock(activeMemoryPoolSizeInBytes_); 
        }

        ~StaticMemPool() override
        {
            // Loop through all the pinned memory blocks and free them.
            while (!pinnedMemoryBaseAddress_.empty())
            {
                hip_check_error(hipHostFree(pinnedMemoryBaseAddress_.top()));
                pinnedMemoryBaseAddress_.pop();
            }
            if (enableLogging_) 
            {
                std::cout << "[ StaticMemPool ] Deleted pool for process " << pid_ << std::endl;
            }
        }

        void* allocate(std::size_t sizeInBytes) override
        {
            std::lock_guard<std::mutex> lock(mutex_);

            if (!preferRecycledMem_ && offsetInBytes_ + sizeInBytes - 1 < activeMemoryPoolSizeInBytes_)
            {
                return allocateNewMemory(sizeInBytes);
            }
            
            void* ptr = tryAllocateMemoryFromPool(sizeInBytes);
            if (ptr)
            {
                return ptr;
            }

            if (offsetInBytes_ + sizeInBytes - 1 < activeMemoryPoolSizeInBytes_)
            {
                return allocateNewMemory(sizeInBytes);
            }

            // Memory became too fragmented, reserve a new block.
            size_t requestedBlockSize = std::max(activeMemoryPoolSizeInBytes_, 2*sizeInBytes);    
            allocateNewPinnedMemoryBlock(requestedBlockSize);
            return allocateNewMemory(sizeInBytes);
        }

        void deallocate(void* p, std::size_t sizeInBytes) override
        {
            std::lock_guard<std::mutex> lock(mutex_);

            if (memory_pool_.find(sizeInBytes) != memory_pool_.end())
            {
                memory_pool_[sizeInBytes].push(p);
#ifdef ENABLE_MEM_POOL_LOGGING
                if (enableLogging_)
                {
                    std::cout << "[ StaticMemPool ] Deallocate: Added memory to back to pool for size " << sizeInBytes << 
                        ", pool has now " << memory_pool_[sizeInBytes].size() << " elements." << std::endl;
                }
#endif
            }
            else {
                std::queue<void*> q;
                q.push(p);
                memory_pool_.insert({sizeInBytes, std::move(q)});
#ifdef ENABLE_MEM_POOL_LOGGING
                if (enableLogging_)
                {
                    std::cout << "[ StaticMemPool ] Deallocate: Created new pool for size " << sizeInBytes << 
                        ", pool has now " << memory_pool_[sizeInBytes].size() << " elements." << std::endl;
                }
#endif
            }
        }

        size_t currentOffsetInBytes() const
        {
            return offsetInBytes_;
        }

        size_t numberOfPinnedMemoryBlocks() const
        {
            return pinnedMemoryBaseAddress_.size();
        }

        size_t memoryPoolSizeInBytes() const
        {
            return activeMemoryPoolSizeInBytes_;
        }

        const std::map<size_t, std::queue<void*>>& memoryPool() const
        {
            return memory_pool_;
        }

    private:
        constexpr static size_t defaultMaxMemoryPoolSizeInBytes_ = 10 * 1024 * 1024; // 10MB
        std::mutex mutex_;
        std::map<size_t, std::queue<void*>> memory_pool_{};
        std::stack<std::byte*> pinnedMemoryBaseAddress_;
        bool enableLogging_;
        int pid_;
        int offsetInBytes_;
        bool preferRecycledMem_;
        size_t activeMemoryPoolSizeInBytes_;

        void allocateNewPinnedMemoryBlock(size_t memoryPoolSizeInBytes)
        {
            activeMemoryPoolSizeInBytes_ = memoryPoolSizeInBytes;
            std::byte* pinnedMemoryBaseAddress;
            hip_check_error(hipHostMalloc(&pinnedMemoryBaseAddress, activeMemoryPoolSizeInBytes_));
            pinnedMemoryBaseAddress_.push(pinnedMemoryBaseAddress);
            offsetInBytes_ = 0;
            if (enableLogging_)
            {
                std::cout << "[ StaticMemPool ] Allocation: Created new pinned memory block of " << activeMemoryPoolSizeInBytes_ << " bytes." << std::endl;
            }
        }

        void* allocateNewMemory(size_t sizeInBytes)
        {
            // Return new memory from the preallocated block
            void* p = pinnedMemoryBaseAddress_.top() + offsetInBytes_;
            offsetInBytes_ += sizeInBytes;
#ifdef ENABLE_MEM_POOL_LOGGING
            if (enableLogging_)
            {
                const auto pct = 100.0f * static_cast<float>(offsetInBytes_) / activeMemoryPoolSizeInBytes_;
                std::cout << "[ StaticMemPool ] Allocation: Return new memory of " << sizeInBytes << 
                    " bytes, pinned host memory usage: " << pct << "%." << std::endl;
            }
#endif
            return p;
        }

        void* tryAllocateMemoryFromPool(size_t sizeInBytes)
        {
            if (memory_pool_.find(sizeInBytes) != memory_pool_.end() && !memory_pool_[sizeInBytes].empty())
            {
                // If there is a memory pool for the requested size, return memory from the pool.
                void* p = memory_pool_[sizeInBytes].front();
                memory_pool_[sizeInBytes].pop();
#ifdef ENABLE_MEM_POOL_LOGGING
                if (enableLogging_)
                {
                    std::cout << "[ StaticMemPool ] Allocation: Reusing memory from pool for size " << sizeInBytes << 
                        ", pool has now " << memory_pool_[sizeInBytes].size() << " elements." << std::endl;
                }
#endif
                return p;
            }
            
            // Try to find memory from the queue that is nearest in size.
            size_t nearest_queue_size = std::numeric_limits<size_t>::max();
            for (auto& [size, q] : memory_pool_)
            {
                if (size > sizeInBytes && !q.empty() && size < nearest_queue_size)
                {
                    nearest_queue_size = size;
                }
            }

            if (nearest_queue_size != std::numeric_limits<size_t>::max())
            {
                void* p = memory_pool_[nearest_queue_size].front();
                memory_pool_[nearest_queue_size].pop();
#ifdef ENABLE_MEM_POOL_LOGGING
                if (enableLogging_)
                {
                    std::cout << "[ StaticMemPool ] Allocation: Reusing memory from pool for size " << nearest_queue_size << 
                        " to allocate " << sizeInBytes << " bytes, pool has " << memory_pool_[nearest_queue_size].size() << " elements." <<
                        std::endl;
                }
#endif
                return p;
            }

            std::cerr << "[ StaticMemPool ] WARNING: Could not find memory from pool to allocate " << sizeInBytes << 
                " bytes." << std::endl;
            return nullptr;
        }
    };

    class PinnedHostMemoryAllocatorBase
    {
    public:
        IMemPool* get_memory_pool() {
            //static DynamicMemPool dynamic_memory_pool;
            static StaticMemPool static_memory_pool;
            //static bool use_dynamic_mem_pool = ck::EnvIsEnabled(CK_ENV(CK_USE_DYNAMIC_MEM_POOL));
            //return use_dynamic_mem_pool ? static_cast<IMemPool*>(&dynamic_memory_pool) : static_cast<IMemPool*>(&static_memory_pool);
            return &static_memory_pool;
        }
    };

    class MemoryCleanupThread 
    {
    public:
        MemoryCleanupThread(std::function<void()> cleanup_function) : cleanup_callback_(cleanup_function)
        {
            cleanup_thread_ = std::thread([this]() {
                while (!should_stop_) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    try 
                    {
                        cleanup_callback_();
                    }
                    catch (const std::exception& e) 
                    {
                        std::cerr << "Error in cleanup thread: " << e.what() << std::endl;
                    }
                    catch (...) 
                    {
                        std::cerr << "Error in cleanup thread." << std::endl;
                    }
                }
            });
        }

        ~MemoryCleanupThread() {
            should_stop_ = true;
            if(cleanup_thread_.joinable()) {
                cleanup_thread_.join();
            }
        }

        MemoryCleanupThread(const MemoryCleanupThread&) = delete;
        MemoryCleanupThread& operator=(const MemoryCleanupThread&) = delete;
        
        MemoryCleanupThread(MemoryCleanupThread&&) noexcept = default;
        MemoryCleanupThread& operator=(MemoryCleanupThread&&) noexcept = default;
    private:
        std::function<void()> cleanup_callback_;
        std::thread cleanup_thread_;
        bool should_stop_{false};
    };

    class PinnedHostMemoryDeallocator : public PinnedHostMemoryAllocatorBase
    {
    public:
        PinnedHostMemoryDeallocator() : cleanup_thread_([this]() { deallocate_all(); }) 
        {
        }

        void register_allocated_memory(void* p, size_t sizeInBytes)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            hipEvent_t event;
            hip_check_error(hipEventCreate(&event));
            device_destruct_events_.insert({p, event});
            allocated_memory_.insert({p, sizeInBytes});
            host_destruct_events_.insert({p, false});
        }

        void destruct_host(void* p /*, std::function<void()>&& destructor*/) 
        {
            std::lock_guard<std::mutex> lock(mutex_);
            host_destruct_events_[p] = true;
        }

        void destruct_device(const void* p, hipStream_t stream) 
        {
            std::lock_guard<std::mutex> lock(mutex_);
            hip_check_error(hipEventRecord(device_destruct_events_[const_cast<void*>(p)], stream));
        }

        void deallocate_all() 
        {
            std::lock_guard<std::mutex> lock(mutex_);
            std::vector<void*> keys;
            for (const auto& [p, _] : allocated_memory_) 
            {
                keys.push_back(p);
            }
            for (auto p : keys) 
            {
                if (canDeallocate(p))
                {
                    deallocate(p);
                }
            }
        }

        static PinnedHostMemoryDeallocator& instance() 
        {
            static PinnedHostMemoryDeallocator instance;
            return instance;
        }

    private:
        std::mutex mutex_;
        std::map<void*, std::size_t> allocated_memory_;
        std::map<void*, bool> host_destruct_events_;
        std::map<void*, hipEvent_t> device_destruct_events_;
        MemoryCleanupThread cleanup_thread_;

        void deallocate(void* p) 
        {   
            //destructors_[p]();
            host_destruct_events_.erase(p);
            auto* memory_pool = get_memory_pool();
            memory_pool->deallocate(p, allocated_memory_[p]);
            hip_check_error(hipEventDestroy(device_destruct_events_[p]));
            device_destruct_events_.erase(p);
            allocated_memory_.erase(p);
        }

        bool canDeallocate(void* p) 
        {
            bool can_deallocate_on_device = false;
            if (device_destruct_events_.find(p) != device_destruct_events_.end()) 
            {
                hipError_t state = hipEventQuery(device_destruct_events_[p]);
                if (state == hipSuccess) 
                {
                    can_deallocate_on_device =  true;
                }
                else if (state != hipErrorNotReady)
                {
                    throw std::runtime_error("Error querying event state: " + std::to_string(state));
                }
            }

            const bool can_deallocate_on_host = host_destruct_events_[p];
            return can_deallocate_on_device && can_deallocate_on_host;
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

        T* allocate(std::size_t n) 
        {
            auto* memory_pool = get_memory_pool();
            const size_t sizeInBytes = n * sizeof(T);
            T* p = static_cast<T*>(memory_pool->allocate(sizeInBytes));
            PinnedHostMemoryDeallocator::instance().register_allocated_memory(p, sizeInBytes);
            return p;
        }

        void deallocate(T* p, std::size_t) 
        {    
            // auto destructor = [&]() {
            //     if constexpr (std::is_destructible_v<T>) 
            //     {
            //         for (size_t i = 0; i < n; ++i) {
            //             p[i].~T();
            //         }
            //     }
            // };
            PinnedHostMemoryDeallocator::instance().destruct_host(p /*, std::move(destructor)*/);
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
