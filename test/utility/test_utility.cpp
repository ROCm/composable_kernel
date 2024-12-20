// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
#include <vector>
#include <map>
#include <queue>
#include "gtest/gtest.h"
#include "ck/ck.hpp"
#include "ck/utility/host_memory_allocator.hpp"

using namespace ck::memory;

namespace 
{

  class TestMemoryAllocator : public PinnedHostMemoryAllocator<std::byte>
  {
  public:
    TestMemoryAllocator() : PinnedHostMemoryAllocator()
    {
    }
  protected:
    IMemPool* get_memory_pool() override {
        static StaticMemPool pool(maxMemoryPoolSizeInBytes_);
        throw std::runtime_error("Static memory pool should not be used.");
        return &pool;
    }
  private:
    static constexpr size_t maxMemoryPoolSizeInBytes_ = 10;
  };
}

TEST(UtilityTests, StaticMemoryPool_test_memory_allocation) 
{
    const size_t size1 = 8;
    const size_t size2 = 2;
    std::byte *ptr1, *ptr2;
    StaticMemPool pool(size1 + size2);
    ptr1 = static_cast<std::byte*>(pool.allocate(size1));
    ptr2 = static_cast<std::byte*>(pool.allocate(size2));
    EXPECT_TRUE(ptr1 != nullptr);
    EXPECT_TRUE(ptr2 != nullptr);

    pool.deallocate(ptr1, size1);
    pool.deallocate(ptr2, size2);

    std::byte* ptr3 = static_cast<std::byte*>(pool.allocate(size2));
    std::byte* ptr4 = static_cast<std::byte*>(pool.allocate(size1));
    EXPECT_TRUE(ptr3 != nullptr);
    EXPECT_TRUE(ptr4 != nullptr);
    EXPECT_TRUE(ptr3 != ptr4);
    EXPECT_TRUE(ptr3 == ptr2);
    EXPECT_TRUE(ptr4 == ptr1);

    pool.deallocate(ptr3, size2);
    pool.deallocate(ptr4, size1);

    const size_t size3 = 6;
    const size_t size4 = 4;
    std::byte* ptr5 = static_cast<std::byte*>(pool.allocate(size3));
    std::byte* ptr6 = static_cast<std::byte*>(pool.allocate(size4));

    EXPECT_TRUE(ptr5 != nullptr);
    EXPECT_TRUE(ptr6 != nullptr);

    pool.deallocate(ptr5, size3);
    pool.deallocate(ptr6, size4);
}

TEST(UtilityTests, PinnedHostMemoryAllocator_new_memory_is_allocated) 
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

        EXPECT_TRUE(v3.data() != ptr1);
        EXPECT_TRUE(v4.data() != ptr2);
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