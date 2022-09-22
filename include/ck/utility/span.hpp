// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstddef>
#include <array>
#include <iterator>
#include <type_traits>

namespace ck {

template <typename T>
class span
{
    public:
    using element_type           = T;
    using value_type             = std::remove_cv_t<element_type>;
    using size_type              = std::size_t;
    using difference_type        = std::ptrdiff_t;
    using pointer                = element_type*;
    using const_pointer          = const element_type*;
    using reference              = element_type&;
    using const_reference        = const element_type&;
    using iterator               = pointer;
    using const_iterator         = pointer;
    using reverse_iterator       = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    constexpr span() : span(nullptr, size_type{0}) {}

    constexpr span(pointer first, size_type count) : ptr_(first), size_(count) {}

    constexpr span(pointer first, pointer last) : span(first, last - first) {}

    template <std::size_t N>
    constexpr span(element_type (&arr)[N]) noexcept : span(arr, N)
    {
    }

    template <std::size_t N>
    constexpr span(std::array<value_type, N>& arr) noexcept : span(arr.data(), N)
    {
    }

    template <typename Container>
    constexpr span(const Container& container) : span(container.data(), container.size())
    {
    }

    constexpr iterator begin() const noexcept { return ptr_; }
    constexpr const_iterator cbegin() const noexcept { return begin(); }

    constexpr iterator end() const noexcept { return begin() + size(); }
    constexpr const_iterator cend() const noexcept { return end(); }

    constexpr reverse_iterator rbegin() const noexcept { return std::make_reverse_iterator(end()); }
    constexpr const_reverse_iterator crbegin() const noexcept { return rbegin(); }

    constexpr reverse_iterator rend() const noexcept { return std::make_reverse_iterator(begin()); }
    constexpr const_reverse_iterator crend() const noexcept { return rend(); }

    constexpr reference front() const { return *begin(); }
    constexpr reference back() const { return *(--end()); }

    constexpr reference operator[](size_type idx) const { return *(begin() + idx); }
    constexpr pointer data() const noexcept { return ptr_; }

    constexpr size_type size() const noexcept { return size_; }

    constexpr bool empty() const noexcept { return size() == 0; }

    friend constexpr iterator begin(const span& s) noexcept { return s.begin(); }
    friend constexpr const_iterator cbegin(const span& s) noexcept { return s.begin(); }

    friend constexpr iterator end(const span& s) noexcept { return s.end(); }
    friend constexpr const_iterator cend(const span& s) noexcept { return s.end(); }

    friend constexpr reverse_iterator rbegin(const span& s) noexcept { return s.rbegin(); }
    friend constexpr const_reverse_iterator crbegin(const span& s) noexcept { return s.crbegin(); }

    friend constexpr reverse_iterator rend(const span& s) noexcept { return s.rend(); }
    friend constexpr const_reverse_iterator crend(const span& s) noexcept { return s.crend(); }

    friend constexpr reference front(const span& s) { return s.front(); }
    friend constexpr reference back(const span& s) { return s.back(); }

    friend constexpr pointer data(const span& s) noexcept { return s.data(); }

    friend constexpr size_type size(const span& s) noexcept { return s.size(); }

    friend constexpr bool empty(const span& s) noexcept { return s.empty(); }

    private:
    pointer ptr_;
    size_type size_;
};

} // namespace ck
