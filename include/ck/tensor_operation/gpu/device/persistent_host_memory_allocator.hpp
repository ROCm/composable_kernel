// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <mutex>
#include <map>
#include <vector>

#include <ck/ck.hpp>

namespace ck {
namespace tensor_operation {
namespace device {

// ToDO: move the implementation to cpp file
// Allocator used for allocating persisent host memory buffers used as src/dst for
// H2D or D2H transfers, host memory persistency is required when hipGraph is used.
class PersistentHostMemoryAllocator
{
    private:
    static PersistentHostMemoryAllocator* singleton_;

    std::map<hipStream_t, std::vector<char*>> buffers_;
    std::mutex mtx_;

    protected:
    PersistentHostMemoryAllocator(){};

    public:
    void* allocate(size_t sizeInBytes, hipStream_t stream)
    {
        std::lock_guard<std::mutex> lck(mtx_);

        auto it = buffers_.find(stream);

        if(it != buffers_.end())
        {
            char* new_buf = new char[sizeInBytes];
            it->second.push_back(new_buf);

            return new_buf;
        }
        else
        {
            // allocate a buffer and keep it for the stream
            char* new_buf = new char[sizeInBytes];

            std::vector<char*> tmp_vec = {new_buf};

            buffers_.insert(std::make_pair(stream, tmp_vec));

            return new_buf;
        };
    };

    void releaseWithStream(hipStream_t stream)
    {
        std::lock_guard<std::mutex> lck(mtx_);

        auto it = buffers_.find(stream);

        if(it != buffers_.end())
        {
            for(auto buf : it->second)
                delete[] buf;

            it->second.clear();
        }
    };

    void releaseAll()
    {
        std::lock_guard<std::mutex> lck(mtx_);

        auto it = buffers_.begin();

        while(it != buffers_.end())
        {
            for(auto buf : it->second)
                delete[] buf;

            it->second.clear();

            ++it;
        }
    };

    static PersistentHostMemoryAllocator* getPersistentHostMemoryAllocatorPtr()
    {
        if(singleton_ == nullptr)
            singleton_ = new PersistentHostMemoryAllocator();

        return singleton_;
    };

    PersistentHostMemoryAllocator(const PersistentHostMemoryAllocator&) = delete;
    PersistentHostMemoryAllocator(PersistentHostMemoryAllocator&&)      = delete;
    PersistentHostMemoryAllocator& operator=(const PersistentHostMemoryAllocator&) = delete;
    PersistentHostMemoryAllocator& operator=(PersistentHostMemoryAllocator&&) = delete;
};

PersistentHostMemoryAllocator* PersistentHostMemoryAllocator::singleton_ = nullptr;

// ToDo:  move this to cpp file
static PersistentHostMemoryAllocator* getPersistentHostMemoryAllocatorPtr()
{
    return PersistentHostMemoryAllocator::getPersistentHostMemoryAllocatorPtr();
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
