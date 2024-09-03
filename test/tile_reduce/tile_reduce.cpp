// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <vector>
#include <iostream>
#include <numeric>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <time.h>
#include <unordered_set>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/reduce.hpp"

#ifndef TEST_TILE_REDUCE_VERBOSE
#define TEST_TILE_REDUCE_VERBOSE 1
#endif

#define HIP_CALL(call)                                                              \
    do                                                                              \
    {                                                                               \
        hipError_t err = call;                                                      \
        if(err != hipSuccess)                                                       \
        {                                                                           \
            printf("[hiperror](%d) fail to call %s", static_cast<int>(err), #call); \
            exit(0);                                                                \
        }                                                                           \
    } while(0)

#define BLOCK_SIZE 256

template <int Rows, int Cols, typename DataType, int BytesPerIssue = 16>
__global__ void reduce_row(DataType* p_src, DataType* p_dst)
{
    using namespace ck_tile;

    // some constexpr vars
    constexpr index_t vec = BytesPerIssue / sizeof(DataType);
    static_assert(Cols % vec == 0);
    constexpr index_t col_lanes = Cols / vec;
    constexpr index_t warp_size = ck_tile::get_warp_size();
    static_assert(warp_size % col_lanes == 0);
    constexpr index_t row_lanes = warp_size / col_lanes;
    constexpr index_t num_warps = BLOCK_SIZE / warp_size;
    static_assert(Rows % (num_warps * row_lanes) == 0);
    constexpr index_t row_repeat = Rows / (num_warps * row_lanes);

    auto src_tile = [&]() {
        constexpr auto src_dist = make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<row_repeat, num_warps, row_lanes>, sequence<col_lanes, vec>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 0>>,
                sequence<1, 2>,
                sequence<0, 1>>{});

        auto src_view =
            make_naive_tensor_view<address_space_enum::global>(p_src,
                                                               make_tuple(Rows, Cols),
                                                               make_tuple(Cols, 1),
                                                               number<vec>{}, // alignement
                                                               number<1>{});
        return make_tile_window(
            src_view, make_tuple(number<Rows>{}, number<Cols>{}), {0, 0}, src_dist);
    }();

    constexpr auto dst_dist = make_static_tile_distribution(
        tile_distribution_encoding<
            sequence<col_lanes>, // -> replicate here, hence we can figure out the offset
            tuple<sequence<row_repeat, num_warps, row_lanes>, sequence<1> /* only 1 per row*/>,
            tuple<sequence<1>, sequence<1, 0>>,
            tuple<sequence<1>, sequence<2, 0>>,
            sequence<1, 2>,
            sequence<0, 0>>{});

    auto dst_tile = [&]() {
        auto dst_view =
            make_naive_tensor_view<address_space_enum::global>(p_dst,
                                                               make_tuple(Rows, 1),
                                                               make_tuple(1, 1),
                                                               number<1>{}, // alignement
                                                               number<1>{});
        return make_tile_window(
            dst_view, make_tuple(number<Rows>{}, number<1>{}), {0, 0}, dst_dist);
    }();

    auto data = load_tile(src_tile);

    const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };

    // Note: the return type will fill the replicate dim
    //       usually is 2d. This is for the next block_tile_reduce_sync()
    //       in order to do further reduce.
    auto r =
        block_tile_reduce<DataType>(data, sequence<1>{}, f_max, -numeric<DataType>::infinity());

    // r.foo();

    // further reduce cross thread
    block_tile_reduce_sync(r, f_max, bool_constant<false>{});

    if(threadIdx.x % col_lanes == 0)
    {
        auto o                = make_static_distributed_tensor<DataType>(dst_dist);
        o.get_thread_buffer() = r.get_thread_buffer();
        store_tile(dst_tile, o);
    }
}

template <int Rows, int Cols, typename DataType, int BytesPerIssue = 16>
bool test_tile_reduce()
{
    std::srand(std::time(nullptr));
    DataType* src = reinterpret_cast<DataType*>(malloc(Rows * Cols * sizeof(DataType)));
    DataType* dst = reinterpret_cast<DataType*>(malloc(Rows * sizeof(DataType)));

    // const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };

    for(auto i = 0; i < Rows * Cols; i++)
    {
        float v = static_cast<float>(std::rand() % 2000 - 1000) / 1000.f;
        src[i]  = ck_tile::type_convert<DataType>(v);
    }

    void* dev_src;
    void* dev_dst;
    HIP_CALL(hipMalloc(&dev_src, Rows * Cols * sizeof(DataType)));
    HIP_CALL(hipMalloc(&dev_dst, Rows * sizeof(DataType)));

    HIP_CALL(hipMemcpy(dev_src, src, Rows * Cols * sizeof(DataType), hipMemcpyHostToDevice));

    constexpr int bdim = BLOCK_SIZE;
    int gdim           = 1;
    reduce_row<Rows, Cols, DataType, BytesPerIssue><<<gdim, bdim>>>(
        reinterpret_cast<DataType*>(dev_src), reinterpret_cast<DataType*>(dev_dst));

    HIP_CALL(hipMemcpy(dst, dev_dst, Rows * sizeof(DataType), hipMemcpyDeviceToHost));

    int err_cnt = 0;

    for(int i_r = 0; i_r < Rows; i_r++)
    {
        auto row_max = -ck_tile::numeric<float>::infinity();
        for(int i_c = 0; i_c < Cols; i_c++)
        {
            int idx = i_r * Cols + i_c;
            float v = ck_tile::type_convert<float>(src[idx]);
            row_max = row_max > v ? row_max : v;
#if TEST_TILE_REDUCE_VERBOSE
            printf("%.3f ", v);
#endif
        }
        {
            uint32_t ref = ck_tile::bit_cast<uint32_t>(row_max);
            uint32_t out = ck_tile::bit_cast<uint32_t>(dst[i_r]);
            if(ref != out)
                err_cnt++;
        }
#if TEST_TILE_REDUCE_VERBOSE
        printf(" -> %.3f (%.3f)\n", dst[i_r], row_max);
#endif
    }
#if TEST_TILE_REDUCE_VERBOSE
    printf("\n");
#endif

    free(src);
    free(dst);
    return err_cnt == 0 ? true : false;
}

int main()
{
    bool r = true;
    r &= test_tile_reduce<32, 64, float>();
    r &= test_tile_reduce<32, 16, float, 4>();

    return r ? 0 : -1;
}
