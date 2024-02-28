// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#ifdef __HIPCC__
#define CK_TILE_HOST __host__
#define CK_TILE_DEVICE __device__
#define CK_TILE_HOST_DEVICE __host__ __device__
#else
#define CK_TILE_HOST inline
#define CK_TILE_DEVICE inline
#define CK_TILE_HOST_DEVICE inline
#endif

#define CK_TILE_FLOAT_TO_BFLOAT16_STANDARD 0
#define CK_TILE_FLOAT_TO_BFLOAT16_TRUNCATE_WITH_NAN 1
#define CK_TILE_FLOAT_TO_BFLOAT16_TRUNCATE 2

#ifndef CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT
#define CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT CK_TILE_FLOAT_TO_BFLOAT16_TRUNCATE
#endif

#define CK_TILE_FLOAT_TO_FP8_STANDARD 0
#define CK_TILE_FLOAT_TO_FP8_STOCHASTIC 1

#ifndef CK_TILE_FLOAT_TO_FP8_DEFAULT
#define CK_TILE_FLOAT_TO_FP8_DEFAULT CK_TILE_FLOAT_TO_FP8_STANDARD
#endif

#ifndef STATIC_ASSERT
#ifndef NDEBUG
#define STATIC_ASSERT(...) static_assert(__VA_ARGS__)
#else
#define STATIC_ASSERT(...)
#endif
#endif // #ifndef STATIC_ASSERT

// in the old rocm period, we have to use tuple array implementation to implement this
// so turn on the _USE_TUPLE if meet compiler error, otherwise _USE_ARRAY by default.
#define CK_TILE_STATICALLY_INDEXED_ARRAY_USE_ARRAY 0
#define CK_TILE_STATICALLY_INDEXED_ARRAY_USE_TUPLE 1
#ifndef CK_TILE_STATICALLY_INDEXED_ARRAY_DEFAULT
#define CK_TILE_STATICALLY_INDEXED_ARRAY_DEFAULT CK_TILE_STATICALLY_INDEXED_ARRAY_USE_ARRAY
#endif

#ifndef CK_TILE_USE_LAUNCH_BOUNDS
#define CK_TILE_USE_LAUNCH_BOUNDS 1
#endif

#ifndef CK_TILE_TIME_KERNEL
#define CK_TILE_TIME_KERNEL 1
#endif

#define CK_TILE_MAX_THREAD_PER_BLOCK 256
#define CK_TILE_MIN_BLOCK_PER_CU 2
