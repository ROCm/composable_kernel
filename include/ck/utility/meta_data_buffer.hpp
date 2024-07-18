// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {

template <index_t MaxSize>
struct MetaDataBuffer
{
    __host__ __device__ constexpr MetaDataBuffer() : buffer_{}, size_{0} {}

    template <typename X, typename... Xs>
    __host__ __device__ constexpr MetaDataBuffer(const X& x, const Xs&... xs) : buffer_{}, size_{0}
    {
        Push(x, xs...);
    }

    template <typename T>
    __host__ __device__ constexpr void Push(const T& data)
    {
        if constexpr(!is_empty_v<T>)
        {
            constexpr index_t size = sizeof(T);

            auto tmp = bit_cast<Array<std::byte, size>>(data);

            for(int i = 0; i < size; i++)
            {
                buffer_(size_) = tmp[i];

                size_++;
            }
        }
    }

    template <typename X, typename... Xs>
    __host__ __device__ constexpr void Push(const X& x, const Xs&... xs)
    {
        Push(x);
        Push(xs...);
    }

    template <typename T>
    __host__ __device__ constexpr T Pop(index_t& pos) const
    {
        T data;

        if constexpr(!is_empty_v<T>)
        {
            constexpr index_t size = sizeof(T);

            Array<std::byte, size> tmp;

            for(int i = 0; i < size; i++)
            {
                tmp(i) = buffer_[pos];

                pos++;
            }

            data = bit_cast<T>(tmp);
        }

        return data;
    }

    template <typename T>
    __host__ __device__ constexpr T Get(index_t pos) const
    {
        constexpr index_t size = sizeof(T);

        Array<std::byte, size> tmp;

        for(int i = 0; i < size; i++)
        {
            tmp(i) = buffer_[pos];

            pos++;
        }

        auto data = bit_cast<T>(tmp);

        return data;
    }

    //
    Array<std::byte, MaxSize> buffer_;
    index_t size_ = 0;
};

} // namespace ck
