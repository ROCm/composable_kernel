// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <type_traits>
#include <utility>

namespace ck {

template <typename T>
struct AutoCastPtr
{
    using pointer = T*;

    AutoCastPtr(const AutoCastPtr&) = default;
    AutoCastPtr(AutoCastPtr&&)      = default;
    ~AutoCastPtr()                  = default;
    AutoCastPtr& operator=(const AutoCastPtr&) = default;
    AutoCastPtr& operator=(AutoCastPtr&&) = default;

    constexpr AutoCastPtr(pointer target = nullptr) noexcept : target_(target) {}

    // avoid base-to-derived casting
    template <
        typename U,
        typename = std::enable_if_t<
            (std::is_same_v<U, void> || (std::is_same_v<U, const void> && std::is_const_v<T>)) ||
            std::is_convertible_v<decltype(std::declval<AutoCastPtr<U>>().get()), pointer>>>
    constexpr AutoCastPtr(AutoCastPtr<U> other) noexcept
        : target_(static_cast<pointer>(other.get()))
    {
    }

    constexpr pointer get() const noexcept { return target_; }

    constexpr pointer operator->() const noexcept { return get(); }

    constexpr std::add_lvalue_reference_t<T> operator*() const
        noexcept(noexcept(*std::declval<pointer>()))
    {
        return *get();
    }

    template <typename U,
              typename =
                  std::enable_if_t<std::is_same_v<std::remove_const_t<T>, std::remove_const_t<U>> &&
                                   (std::is_const_v<U> || !std::is_const_v<T>)>>
    constexpr operator U*() const noexcept
    {
        return static_cast<U*>(get());
    }

    private:
    pointer target_;
};

template <>
struct AutoCastPtr<void>
{
    using pointer = void*;

    AutoCastPtr(const AutoCastPtr&) = default;
    AutoCastPtr(AutoCastPtr&&)      = default;
    ~AutoCastPtr()                  = default;
    AutoCastPtr& operator=(const AutoCastPtr&) = delete;
    AutoCastPtr& operator=(AutoCastPtr&&) = default;

    constexpr AutoCastPtr(pointer target = nullptr) noexcept : target_(target) {}

    template <typename T, typename = std::enable_if_t<!std::is_const_v<T>>>
    constexpr AutoCastPtr(T* target) noexcept : AutoCastPtr(target)
    {
    }

    template <typename T, typename = std::enable_if_t<!std::is_const_v<T>>>
    constexpr AutoCastPtr(AutoCastPtr<T> other) noexcept : target_(other.get())
    {
    }

    constexpr pointer get() const noexcept { return target_; }

    constexpr operator pointer() const noexcept { return get(); }

    template <typename T>
    constexpr operator T*() const noexcept
    {
        return static_cast<T*>(get());
    }

    private:
    pointer target_;
};

template <>
struct AutoCastPtr<const void>
{
    using pointer = const void*;

    AutoCastPtr(const AutoCastPtr&) = default;
    AutoCastPtr(AutoCastPtr&&)      = default;
    ~AutoCastPtr()                  = default;
    AutoCastPtr& operator=(const AutoCastPtr&) = default;
    AutoCastPtr& operator=(AutoCastPtr&&) = default;

    constexpr AutoCastPtr(pointer target = nullptr) noexcept : target_(target) {}

    template <typename T>
    constexpr AutoCastPtr(T* target) noexcept : AutoCastPtr(target)
    {
    }

    template <typename T>
    constexpr AutoCastPtr(AutoCastPtr<T> other) noexcept : target_(other.get())
    {
    }

    constexpr pointer get() const noexcept { return target_; }

    constexpr operator pointer() const noexcept { return get(); }

    template <typename T, typename = std::enable_if_t<std::is_const_v<T>>>
    constexpr operator T*() const noexcept
    {
        return static_cast<T*>(get());
    }

    private:
    pointer target_;
};

inline constexpr struct auto_cast_function_type_
{
    explicit auto_cast_function_type_() = default;

    template <typename T>
    inline constexpr auto operator()(T* target) const
        noexcept(std::is_nothrow_constructible_v<AutoCastPtr<T>, T*>)
    {
        return AutoCastPtr<T>(target);
    }

    inline constexpr auto operator()(std::nullptr_t) const
        noexcept(std::is_nothrow_constructible_v<AutoCastPtr<void>>)
    {
        return AutoCastPtr<void>(nullptr);
    }
} auto_cast{};
} // namespace ck
