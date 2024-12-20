// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
#include <vector>
#include <map>
#include <queue>
#include "gtest/gtest.h"
#include "ck/ck.hpp"
#include "ck/utility/host_memory_allocator.hpp"

using namespace ck::memory;

TEST(UtilityTests, PinnedHostMemoryAllocator_recycle_pinned_host_memory) 
{
    const size_t vSize = 10;
    int* ptr1;
    int* ptr2;
    {
        std::vector<int, PinnedHostMemoryAllocator<int>> v1(vSize);
        std::vector<int, PinnedHostMemoryAllocator<int>> v2(2*vSize);
        EXPECT_EQ(v1.size(), vSize);
        EXPECT_EQ(v2.size(), 2*vSize);
        EXPECT_TRUE(v1.data() != nullptr);
        EXPECT_TRUE(v2.data() != nullptr);
        EXPECT_TRUE(v1.data() != v2.data());

        ptr1 = v1.data();
        ptr2 = v2.data();
    }
    
    {
        // Check that for new vectors, the memory is reused.
        std::vector<int, PinnedHostMemoryAllocator<int>> v3(vSize);
        std::vector<int, PinnedHostMemoryAllocator<int>> v4(2*vSize);

        EXPECT_EQ(v3.data(), ptr1);
        EXPECT_EQ(v4.data(), ptr2);
    }
}

TEST(UtilityTests, PinnedHostMemoryAllocator_access_elements) 
{
    const size_t vSize = 10;
    {
        std::vector<int, PinnedHostMemoryAllocator<int>> v(vSize);
        for (size_t i = 0; i < vSize; ++i) {
            v[i] = i;
        }
        for (size_t i = 0; i < vSize; ++i) {
            EXPECT_EQ(v[i], i);
        }
    }

    {
        std::vector<int, PinnedHostMemoryAllocator<int>> v(vSize);
        for (size_t i = 0; i < vSize; ++i) {
            v[i] = 2*i;
        }
        for (size_t i = 0; i < vSize; ++i) {
            EXPECT_EQ(v[i], 2*i);
        }
    }
}

TEST(UtilityTests, PinnedHostMemoryAllocator_complex_object) 
{
    struct ComplexObject {
        int a;
        float b;
        double c;
        std::string d;
    };

    const size_t vSize = 10;
    {
        std::vector<ComplexObject, PinnedHostMemoryAllocator<ComplexObject>> v(vSize);
        for (int i = 0; i < vSize; ++i) {
            v[i] = ComplexObject{i, 2.0f*i, 3.0*i, "hello" + std::to_string(i)};
        }
        for (size_t i = 0; i < vSize; ++i) {
            EXPECT_EQ(v[i].a, i);
            EXPECT_EQ(v[i].b, 2.0f*i);
            EXPECT_EQ(v[i].c, 3.0*i);
            EXPECT_EQ(v[i].d, "hello"  + std::to_string(i));
        }
    }
}

TEST(UtilityTests, PinnedHostMemoryAllocator_nested_vector) 
{
    const size_t vSize = 10;
    using PinnedHostMemoryAllocatorInt = PinnedHostMemoryAllocator<int>;
    using PinnedHostMemoryAllocatorVectorInt = PinnedHostMemoryAllocator<std::vector<int, PinnedHostMemoryAllocatorInt>>;
    {
        std::vector<std::vector<int, PinnedHostMemoryAllocatorInt>, PinnedHostMemoryAllocatorVectorInt> v(vSize);
        for (size_t i = 0; i < vSize; ++i) {
            v[i].resize(i+1);
            for (size_t j = 0; j < i+1; ++j) {
                v[i][j] = i*j;
            }
        }
        for (size_t i = 0; i < vSize; ++i) {
            for (size_t j = 0; j < i+1; ++j) {
                EXPECT_EQ(v[i][j], i*j);
            }
        }
    }
}

TEST(UtilityTests, PinnedHostMemoryAllocator_multiple_threads_create_vector_of_same_size)
{
    const size_t vSize = 10;
    const size_t numThreads = 4;
    std::vector<std::thread> threads;
    for (size_t i = 0; i < numThreads; ++i) {
        threads.push_back(std::thread([vSize, i](){
            std::vector<int, PinnedHostMemoryAllocator<int>> v(vSize);
            for (size_t j = 0; j < vSize; ++j) {
                v[j] = i*j;
            }
        }));
    }
    for (size_t i = 0; i < numThreads; ++i) {
        threads[i].join();
    }
}

TEST(UtilityTests, PinnedHostMemoryAllocator_multiple_vectors_of_same_size_and_different_type)
{
    const size_t vSize = 10;
    {
        std::vector<int, PinnedHostMemoryAllocator<int>> v1(vSize);
        std::vector<float, PinnedHostMemoryAllocator<float>> v2(vSize);
        for (size_t i = 0; i < vSize; ++i) {
            v1[i] = i;
            v2[i] = 2.0f*i;
        }
        for (size_t i = 0; i < vSize; ++i) {
            EXPECT_EQ(v1[i], i);
            EXPECT_EQ(v2[i], 2.0f*i);
        }
    }
}