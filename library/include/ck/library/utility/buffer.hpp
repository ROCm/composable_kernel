// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <cstddef>
#include <type_traits>

namespace ck {
namespace utils {
struct mutable_buffer
{
    using pointer   = std::byte*;
    using size_type = std::size_t;

    friend mutable_buffer operator+(const mutable_buffer& buffer, size_type n) noexcept
    {
        mutable_buffer advanced(buffer);
        advanced += n;
        return advanced;
    }

    constexpr mutable_buffer() noexcept : mutable_buffer(nullptr, 0) {}

    constexpr mutable_buffer(void* data, size_type size = 0) noexcept
        : data_(static_cast<pointer>(data)), size_(size)
    {
    }

    constexpr mutable_buffer& operator+=(size_type n) noexcept
    {
        const size_type advance = std::min(size(), n);

        data_ = data() + advance;
        size_ = size() - advance;

        return *this;
    }

    constexpr pointer data() const noexcept { return data_; }

    constexpr size_type size() const noexcept { return size_; }

    constexpr operator void*() const noexcept { return data(); }

    constexpr operator const void*() const noexcept { return data(); }

    // this method only exists while T is complete type
    template <typename T, typename = std::void_t<decltype(sizeof(T))>>
    constexpr operator T*() const noexcept
    {
        if(size() % sizeof(T) != 0)
        {
            return nullptr;
        }

        return reinterpret_cast<T*>(data());
    }

    private:
    pointer data_;
    size_type size_;
};
} // namespace utils
} // namespace ck
