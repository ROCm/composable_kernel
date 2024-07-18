// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/array.hpp"
#include "ck/utility/tuple.hpp"

namespace ck {

// naive Map
template <typename Key, typename Data, index_t MaxSize = 128>
struct Map
{
    using Pair = Tuple<Key, Data>;
    using Impl = Array<Pair, MaxSize>;

    Impl impl_;
    index_t size_;

    struct Iterator
    {
        Impl& impl_;
        index_t pos_;

        __host__ __device__ constexpr Iterator(Impl& impl, index_t pos) : impl_{impl}, pos_{pos} {}

        __host__ __device__ constexpr Iterator& operator++()
        {
            pos_++;

            return *this;
        }

        __host__ __device__ constexpr bool operator!=(const Iterator& other) const
        {
            return other.pos_ != pos_;
        }

        __host__ __device__ constexpr Pair& operator*() { return impl_.At(pos_); }
    };

    struct ConstIterator
    {
        const Impl& impl_;
        index_t pos_;

        __host__ __device__ constexpr ConstIterator(const Impl& impl, index_t pos)
            : impl_{impl}, pos_{pos}
        {
        }

        __host__ __device__ constexpr ConstIterator& operator++()
        {
            pos_++;

            return *this;
        }

        __host__ __device__ constexpr bool operator!=(const ConstIterator& other) const
        {
            return other.pos_ != pos_;
        }

        __host__ __device__ constexpr const Pair& operator*() const { return impl_.At(pos_); }
    };

    __host__ __device__ constexpr Map() : impl_{}, size_{0} {}

    __host__ __device__ constexpr index_t Size() const { return size_; }

    __host__ __device__ void Clear() { size_ = 0; }

    __host__ __device__ constexpr index_t FindPosition(const Key& key) const
    {
        for(index_t i = 0; i < Size(); i++)
        {
            if(impl_[i].template At<0>() == key)
            {
                return i;
            }
        }

        return size_;
    }

    __host__ __device__ constexpr ConstIterator Find(const Key& key) const
    {
        return ConstIterator{impl_, FindPosition(key)};
    }

    __host__ __device__ constexpr Iterator Find(const Key& key)
    {
        return Iterator{impl_, FindPosition(key)};
    }

    __host__ __device__ constexpr const Data& operator[](const Key& key) const
    {
        const auto it = Find(key);

        // FIXME
        assert(pos < Size());

        return impl_[it.pos_].template At<1>();
    }

    __host__ __device__ constexpr Data& operator()(const Key& key)
    {
        auto it = Find(key);

        // if entry not found
        if(it.pos_ == Size())
        {
            impl_(it.pos_).template At<0>() = key;
            size_++;
        }

        // FIXME
        assert(size_ <= MaxSize);

        return impl_(it.pos_).template At<1>();
    }

    // WARNING: needed by compiler for C++ range-based for loop only, don't use this function!
    __host__ __device__ constexpr ConstIterator begin() const { return ConstIterator{impl_, 0}; }

    // WARNING: needed by compiler for C++ range-based for loop only, don't use this function!
    __host__ __device__ constexpr ConstIterator end() const { return ConstIterator{impl_, size_}; }

    // WARNING: needed by compiler for C++ range-based for loop only, don't use this function!
    __host__ __device__ constexpr Iterator begin() { return Iterator{impl_, 0}; }

    // WARNING: needed by compiler for C++ range-based for loop only, don't use this function!
    __host__ __device__ constexpr Iterator end() { return Iterator{impl_, size_}; }

    __host__ __device__ void Print() const
    {
        printf("Map{size_: %d, ", size_);
        //
        printf("impl_: [");
        //
        for(const auto& [key, data] : *this)
        {
            printf("{key: ");
            print(key);
            printf(", data: ");
            print(data);
            printf("}, ");
        }
        //
        printf("]");
        //
        printf("}");
    }
};

} // namespace ck
