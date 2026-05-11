// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 14.15: static_array vs thread_buffer, side by side
 *
 * Two container types that look similar from 30,000ft but occupy
 * completely different niches in ck_tile. This tutorial has one
 * runnable example of each so you can see the distinction by doing.
 *
 *   PART A - static_array<T, N>
 *     Literal aggregate. No ctors. Usable inside constexpr lambdas.
 *     Niche: "use a normal for-loop in constexpr-land to build a
 *     lookup table, then unpack the table back into a sequence<...>".
 *     That is how ck_tile itself uses it in
 *       include/ck_tile/core/container/sequence.hpp
 *     (sequence_inclusive_scan_impl, sequence_map_inverse).
 *
 *     The two demos here mirror those two real usages:
 *       A1: compile-time prefix sum  sequence<...> -> sequence<...>
 *       A2: compile-time permutation inverse
 *
 *   PART B - thread_buffer<T, N>
 *     Runtime per-thread register pool. Has real ctors (zero init,
 *     broadcast init), specializes vector_traits, and supports
 *     reinterpret-cast type punning via get_as<Tx>() / set_as<Tx>(...).
 *     This is what static_distributed_tensor::thread_buf_ IS and is
 *     the glue between scalar storage and ISA-level vector ops
 *     (global_load_dwordx4, ds_read_b128, v_mfma_*).
 *
 *     The demos here run in a tiny 1-thread kernel:
 *       B1: broadcast ctor
 *       B2: reinterpret thread_buffer<uint16_t, 8> as thread_buffer<uint32_t, 4>
 *           -- show that writing a uint32_t slot changes 2 uint16_t slots
 *
 * Build:
 *   target is aa_tutorial_14_15_static_array_and_thread_buffer_intro
 */

#include "ck_tile/core.hpp"

#include <cstdio>
#include <hip/hip_runtime.h>
#include <array>

using namespace ck_tile;

// --------------------------------------------------------------------------
// PART A: static_array
//
// Pattern (used three times in sequence.hpp):
//   1. Build a static_array inside a constexpr lambda.
//   2. Mutate it with ordinary imperative code.
//   3. Return it from the lambda, then pack-expand arr[Js]...
//      back into a sequence<...> using a helper that takes sequence<Js...>.
//
// static_array is the ONLY container that lets step (2) work in constexpr,
// because it's a literal type (aggregate, no user-provided ctors).
// tuple is immutable in type-space. thread_buffer has ctors. sequence is
// type-level. static_array is the scratchpad.
// --------------------------------------------------------------------------

// --- A1: inclusive prefix sum ---------------------------------------------
namespace detail_a1 {

template <index_t... Is>
struct prefix_sum_impl
{
    static constexpr index_t N = sizeof...(Is);

    // Step 1 + 2: build an array in constexpr land
    static constexpr auto arr = []() {
        static_array<index_t, N> in  = {Is...};
        static_array<index_t, N> out = {0};
        out[0] = in[0];
        for(index_t i = 1; i < N; ++i)
            out[i] = out[i - 1] + in[i];
        return out;
    }();

    // Step 3: expand arr[Js]... back into a sequence<...>
    template <index_t... Js>
    static constexpr auto expand(sequence<Js...>)
    {
        return sequence<arr[Js]...>{};
    }

    using type = decltype(expand(make_index_sequence<N>{}));
};

} // namespace detail_a1

template <index_t... Is>
using prefix_sum_t = typename detail_a1::prefix_sum_impl<Is...>::type;

namespace detail_a1_std {

template <index_t... Is>
struct prefix_sum_impl_std
{
    static constexpr index_t N = sizeof...(Is);

    // Step 1 + 2: build an array in constexpr land
    static constexpr auto arr = []() {
        std::array<index_t, N> in  = {Is...};
        std::array<index_t, N> out = {0};
        out[0] = in[0];
        for(index_t i = 1; i < N; ++i)
            out[i] = out[i - 1] + in[i];
        return out;
    }();

    // Step 3: expand arr[Js]... back into a sequence<...>
    template <index_t... Js>
    static constexpr auto expand(sequence<Js...>)
    {
        return sequence<arr[Js]...>{};
    }

    using type = decltype(expand(make_index_sequence<N>{}));
};

} // namespace detail_a1

template <index_t... Is>
using prefix_sum_t = typename detail_a1::prefix_sum_impl<Is...>::type;

template <index_t... Is>
using prefix_sum_t_std = typename detail_a1_std::prefix_sum_impl_std<Is...>::type;

// --- A2: permutation inverse ----------------------------------------------
// If input = (3, 0, 2, 1)  (a permutation of {0,1,2,3})
// then inverse[input[pos]] = pos
// -> inverse[3]=0, inverse[0]=1, inverse[2]=2, inverse[1]=3
// -> inverse = (1, 3, 2, 0)
namespace detail_a2 {

template <index_t... Is>
struct invert_impl
{
    static constexpr index_t N = sizeof...(Is);

    static constexpr auto inv = []() {
        static_array<index_t, N> input  = {Is...};
        static_array<index_t, N> result = {0};
        for(index_t pos = 0; pos < N; ++pos)
            result[input[pos]] = pos;
        return result;
    }();

    template <index_t... Js>
    static constexpr auto expand(sequence<Js...>)
    {
        return sequence<inv[Js]...>{};
    }

    using type = decltype(expand(make_index_sequence<N>{}));
};

} // namespace detail_a2

template <index_t... Is>
using invert_permutation_t = typename detail_a2::invert_impl<Is...>::type;

// --------------------------------------------------------------------------
// PART B: thread_buffer
//
// Runs on the GPU with one thread so the result is trivially readable.
// Key things we exercise here that static_array CAN'T do:
//   - broadcast ctor thread_buffer<T, N>{ value } (not aggregate init)
//   - get_as<Tx>() reinterpret-cast view
//   - set_as<Tx>(i, x) reinterpret-cast write
// --------------------------------------------------------------------------

__global__ void thread_buffer_kernel()
{
    if(threadIdx.x != 0)
        return;

    // B1: broadcast ctor -- every slot gets 42.
    //     This is the constructor thread_buffer(const T&) that
    //     static_for-fills 'data'. static_array doesn't have one.
    thread_buffer<int, 4> bcast{42};
    printf("[B1] thread_buffer<int, 4>{42} = %d %d %d %d\n",
           bcast[0], bcast[1], bcast[2], bcast[3]);

    // B2: type punning. Same 16 bytes viewed as 8 halves or 4 ints.
    //     We use uint16_t + uint32_t so bit patterns are obvious.
    thread_buffer<uint16_t, 8> buf{}; // zero init
    for(index_t i = 0; i < 8; ++i)
        buf[i] = static_cast<uint16_t>(0x1000 + i);

    printf("[B2] buf as uint16_t x 8: ");
    for(index_t i = 0; i < 8; ++i)
        printf("0x%04x ", buf[i]);
    printf("\n");

    // get_as<uint32_t>() returns a REFERENCE to the same storage
    // reinterpreted as thread_buffer<uint32_t, 4>. No copy.
    //
    // On a little-endian device you should see each uint32_t slot
    // equal to  (buf[2i+1] << 16) | buf[2i]
    // e.g. slot 0  =  (0x1001 << 16) | 0x1000  =  0x10011000
    auto& as_u32 = buf.get_as<uint32_t>();
    printf("     buf as uint32_t x 4: ");
    for(index_t i = 0; i < 4; ++i)
        printf("0x%08x ", as_u32[i]);
    printf("\n");

    // Mutate slot 1 via the wider view. This is set_as<Tx>(i, x).
    // Writing a uint32_t to slot 1 overwrites uint16_t slots 2 and 3.
    buf.set_as<uint32_t>(1, 0xdeadbeef);

    printf("     after set_as<uint32_t>(1, 0xdeadbeef):\n");
    printf("     buf as uint16_t x 8: ");
    for(index_t i = 0; i < 8; ++i)
        printf("0x%04x ", buf[i]);
    printf("\n");
}

// --------------------------------------------------------------------------
// main: static_array work happens entirely at compile time on the host.
//       thread_buffer demos run in a tiny HIP kernel.
// --------------------------------------------------------------------------

int main()
{
    printf("=== Tutorial 14.15: static_array vs thread_buffer ===\n\n");

    printf("-- PART A: static_array (compile-time tables) --\n");

    // A1: prefix_sum of (1, 2, 3, 4, 5) -> (1, 3, 6, 10, 15)
    using PS = prefix_sum_t<1, 2, 3, 4, 5>;
    using PS_std = prefix_sum_t_std<1, 2, 3, 4, 5>;

    printf("[A1] prefix_sum(1,2,3,4,5) = %d %d %d %d %d\n",
           static_cast<int>(PS::at(number<0>{})),
           static_cast<int>(PS::at(number<1>{})),
           static_cast<int>(PS::at(number<2>{})),
           static_cast<int>(PS::at(number<3>{})),
           static_cast<int>(PS::at(number<4>{})));

    printf("[A1] prefix_sum_std(1,2,3,4,5) = %d %d %d %d %d\n",
           static_cast<int>(PS_std::at(number<0>{})),
           static_cast<int>(PS_std::at(number<1>{})),
           static_cast<int>(PS_std::at(number<2>{})),
           static_cast<int>(PS_std::at(number<3>{})),
           static_cast<int>(PS_std::at(number<4>{})));

    // A2: invert permutation (3, 0, 2, 1) -> (1, 3, 2, 0)
    using INV = invert_permutation_t<3, 0, 2, 1>;
    printf("[A2] invert_permutation(3,0,2,1) = %d %d %d %d\n",
           static_cast<int>(INV::at(number<0>{})),
           static_cast<int>(INV::at(number<1>{})),
           static_cast<int>(INV::at(number<2>{})),
           static_cast<int>(INV::at(number<3>{})));

    printf("\n-- PART B: thread_buffer (runtime register pool) --\n");

    hipLaunchKernelGGL(thread_buffer_kernel, dim3(1), dim3(1), 0, nullptr);
    auto err = hipDeviceSynchronize();
    if(err != hipSuccess)
    {
        fprintf(stderr, "hipDeviceSynchronize failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    return 0;
}
