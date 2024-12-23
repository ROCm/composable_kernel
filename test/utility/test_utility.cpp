// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
#include <vector>
#include <map>
#include <queue>
#include <fstream>
#include <regex>
#include <filesystem>
#include "gtest/gtest.h"
#include "ck/ck.hpp"
#include "ck/utility/host_memory_allocator.hpp"

using namespace ck::memory;

namespace {

    enum class MemActionType
    {
        Allocate,
        Deallocate
    };

    struct MemAction
    {
        MemActionType type_;
        size_t size_;
        size_t index_;
    };

    std::vector<MemAction> getMemActions(const std::string filename)
    {
        std::vector<MemAction> actions;      
        std::ifstream file(filename);
        std::string line;
        std::cout << "Reading file: " << filename << std::endl;
        EXPECT_TRUE(file.is_open());

        size_t index = 1;
        while (std::getline(file, line))
        {
            std::regex allocation_regex(R"(Allocation: (\d+) bytes)");
            std::regex deallocation_regex(R"(De-allocation: (\d+) bytes)");
            std::smatch match;
            if (std::regex_search(line, match, allocation_regex) && match.size() > 1) 
            {
                actions.push_back({MemActionType::Allocate, std::stoul(match.str(1)), index++});
            } 
            else if (std::regex_search(line, match, deallocation_regex) && match.size() > 1) 
            {
                actions.push_back({MemActionType::Deallocate, std::stoul(match.str(1)), index++});
            }
            else 
            {
                std::cerr << "Could not parse line: " << line << std::endl;
            }
        }
        return actions;
    }

}

// Do not run automatically as this test requires test data and takes about a minute to run.
TEST(UtilityTests, DISABLED_StaticMemoryPool_stress_test) 
{
    std::filesystem::path currentDir = std::filesystem::current_path();
    std::filesystem::path dataPath = currentDir / "test_data" / "actions.log";
    const std::vector<MemAction> actions = getMemActions(dataPath.string());
    EXPECT_GT(actions.size(), 1);
    EXPECT_EQ(actions.size() % 2, 0);
    std::cout << "Running stress test for number of actions: " << actions.size() << std::endl;
    StaticMemPool pool;
    std::map<size_t, std::queue<void*>> allocated_ptrs;
    for (const MemAction& action : actions) 
    {
        if (action.type_ == MemActionType::Allocate) {
            allocated_ptrs[action.size_].push(pool.allocate(action.size_));
        } 
        else 
        {
            pool.deallocate(allocated_ptrs[action.size_].front(), action.size_);
            allocated_ptrs[action.size_].pop();
        }
    }

    for (auto& [size, q] : allocated_ptrs)
    {
        EXPECT_EQ(q.size(), 0);
    }

    EXPECT_EQ(pool.memoryPoolSizeInBytes(), 10 * 1024 * 1024);
    EXPECT_EQ(pool.numberOfPinnedMemoryBlocks(), 1);
    EXPECT_EQ(pool.currentOffsetInBytes(), pool.memoryPoolSizeInBytes());

    EXPECT_GT(pool.memoryPool().size(), 0);
    for (const auto& [size, q] : pool.memoryPool())
    {
        EXPECT_GT(q.size(), 0);
    }
}

TEST(UtilityTests, StaticMemoryPool_memory_has_correct_content)
{
    StaticMemPool pool(10);

    const size_t size1 = 4;
    const size_t size2 = 6;
    std::byte* ptr1 = static_cast<std::byte*>(pool.allocate(size1));
    std::byte* ptr2 = static_cast<std::byte*>(pool.allocate(size2));

    std::memcpy(ptr1, "abcd", size1);
    std::memcpy(ptr2, "efghij", size2);

    EXPECT_EQ(static_cast<const char>(ptr1[0]), 'a');
    EXPECT_EQ(static_cast<const char>(ptr1[1]), 'b');
    EXPECT_EQ(static_cast<const char>(ptr1[2]), 'c');
    EXPECT_EQ(static_cast<const char>(ptr1[3]), 'd');

    EXPECT_EQ(static_cast<const char>(ptr2[0]), 'e');
    EXPECT_EQ(static_cast<const char>(ptr2[1]), 'f');
    EXPECT_EQ(static_cast<const char>(ptr2[2]), 'g');
    EXPECT_EQ(static_cast<const char>(ptr2[3]), 'h');
    EXPECT_EQ(static_cast<const char>(ptr2[4]), 'i');
    EXPECT_EQ(static_cast<const char>(ptr2[5]), 'j');

    pool.deallocate(ptr1, size1);
    pool.deallocate(ptr2, size2);

    const size_t size3 = 3;
    std::byte* ptr3 = static_cast<std::byte*>(pool.allocate(size3));

    std::memcpy(ptr3, "klm", size1);
    EXPECT_EQ(static_cast<const char>(ptr3[0]), 'k');
    EXPECT_EQ(static_cast<const char>(ptr3[1]), 'l');
    EXPECT_EQ(static_cast<const char>(ptr3[2]), 'm');
} 

TEST(UtilityTests, StaticMemoryPool_repeated_memory_allocation) 
{
    const size_t size168 = 168;
    const size_t size368 = 368;
    const size_t size8 = 8;
    const size_t pool_size = 2*size8 + size168 + size368 + 1;
    StaticMemPool pool(pool_size);

    auto* ptr168 = pool.allocate(size168);
    pool.deallocate(ptr168, size168);

    auto* ptr368 = pool.allocate(size368);
    pool.deallocate(ptr368, size368);

    auto* ptr8 = pool.allocate(size8);
    pool.deallocate(ptr8, size8); 

    auto* ptr8_2 = pool.allocate(size8);
    pool.deallocate(ptr8_2, size8); 
  
    ptr8 = pool.allocate(size8);
    ptr8_2 = pool.allocate(size8);

    pool.deallocate(ptr8, size8); 
    pool.deallocate(ptr8_2, size8); 
    
    ptr368 = pool.allocate(size368);
    pool.deallocate(ptr368, size368);

    ptr168 = pool.allocate(size168);
    pool.deallocate(ptr168, size168);

    EXPECT_EQ(pool.numberOfPinnedMemoryBlocks(), 1);
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

    EXPECT_EQ(pool.numberOfPinnedMemoryBlocks(), 2);
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