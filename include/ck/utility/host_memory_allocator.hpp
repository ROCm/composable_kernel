// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>
#include "ck/host_utility/hip_check_error.hpp"

namespace ck {
namespace memory {

    template <typename T>
    struct PinnedHostMemoryAllocator
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
            T* p;
            hip_check_error(hipHostMalloc(&p, n * sizeof(T)));
            return p;
        }

        void deallocate(T* p, std::size_t) {
            hip_check_error(hipHostFree(p));
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
