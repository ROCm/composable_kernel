// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue.hpp"

#include <gtest/gtest.h>

namespace {

using namespace ck_tile;

template <typename ODataType, index_t M, index_t N, bool PadN = false>
using Epilogue =
    LdsShuffle2DEpilogue<LdsShuffle2DEpilogueProblem<float, ODataType, 4, M, N, false, PadN>>;

// The LDS row pitch is 32 bytes off a 128-byte bank cycle, whatever the row size.
static_assert(Epilogue<bf16_t, 64, 128>::kNLds == 144);
static_assert(Epilogue<bf16_t, 64, 64>::kNLds == 80);
static_assert(Epilogue<bf16_t, 64, 96>::kNLds == 144);
static_assert(Epilogue<float, 64, 128>::kNLds == 136);

// 16-byte vectors; a padded N only guarantees hdim_v % 8 == 0.
static_assert(Epilogue<bf16_t, 64, 128>::kVector == 8);
static_assert(Epilogue<bf16_t, 64, 128, true>::kVector == 8);
static_assert(Epilogue<float, 64, 128>::kVector == 4);
static_assert(Epilogue<int8_t, 64, 128>::kVector == 16);
static_assert(Epilogue<int8_t, 64, 128, true>::kVector == 8);

// Rows of 8, 16 or 32 vectors tile a wave32 or wave64 wavefront and take the LDS
// shuffle; rows of 12, 20 or 24 vectors (d = 96, 160, 192) keep the direct store
// and need no LDS.
static_assert(Epilogue<bf16_t, 64, 128>::IsShuffleTileable());
static_assert(Epilogue<bf16_t, 128, 128>::IsShuffleTileable());
static_assert(Epilogue<bf16_t, 16, 128>::IsShuffleTileable());
static_assert(Epilogue<bf16_t, 64, 64>::IsShuffleTileable());
static_assert(Epilogue<bf16_t, 64, 256>::IsShuffleTileable());
static_assert(Epilogue<bf16_t, 64, 128>::GetSmemSize() == 64 * 144 * 2);
static_assert(Epilogue<bf16_t, 16, 128>::GetSmemSize() == 16 * 144 * 2);
static_assert(!Epilogue<bf16_t, 64, 96>::IsShuffleTileable());
static_assert(!Epilogue<bf16_t, 64, 160>::IsShuffleTileable());
static_assert(!Epilogue<bf16_t, 64, 192>::IsShuffleTileable());
static_assert(Epilogue<bf16_t, 64, 96>::GetSmemSize() == 0);
static_assert(Epilogue<bf16_t, 64, 192>::GetSmemSize() == 0);

// The kernel hands the epilogue its LDS through a 4th argument only if it says so.
static_assert(Epilogue<bf16_t, 64, 128>::kUsesSmem);

// A row longer than a wavefront of vectors (512 bf16 columns on a wave32 target, 1024 on
// either wave size: two or more vectors per thread and row pass) still stores whole
// 16-byte vectors: the last Y is the innermost column length, and it is a whole vector.
// The raw thread_raked pattern fails this on wave32 (its vector strides by the per-thread
// iteration count).
template <typename E>
using StoreEncoding =
    decltype(decltype(E::MakeODramTileDistribution())::get_static_tile_distribution_encoding());
template <typename E>
constexpr bool kStoresWholeVectors =
    StoreEncoding<E>::Ys2RHsMajor::back() == 2 &&
    StoreEncoding<E>::Ys2RHsMinor::back() ==
        typename StoreEncoding<E>::HsLengthss{}.at(number<1>{}).size() - 1 &&
    typename StoreEncoding<E>::HsLengthss{}.at(number<1>{}).back() == E::kVector;
// ... and in each store adjacent lanes write adjacent vectors: with a per-thread iteration
// length (three X lengths), the lanes take the middle one, between the iterations and the
// vector.
template <typename E>
constexpr bool kLanesContiguous =
    typename StoreEncoding<E>::HsLengthss{}.at(number<1>{}).size() == 3 &&
    typename StoreEncoding<E>::Ps2RHssMajor{}.at(number<1>{}).back() == 2 &&
    typename StoreEncoding<E>::Ps2RHssMinor{}.at(number<1>{}).back() == 1 &&
    StoreEncoding<E>::Ys2RHsMajor::at(number<1>{}) == 2 &&
    StoreEncoding<E>::Ys2RHsMinor::at(number<1>{}) == 0;
static_assert(Epilogue<bf16_t, 64, 512>::IsShuffleTileable());
static_assert(Epilogue<bf16_t, 64, 1024>::IsShuffleTileable());
static_assert(kStoresWholeVectors<Epilogue<bf16_t, 64, 512>>);
static_assert(kStoresWholeVectors<Epilogue<bf16_t, 64, 1024>>);
static_assert(kStoresWholeVectors<Epilogue<bf16_t, 64, 128>>);
static_assert(kLanesContiguous<Epilogue<bf16_t, 64, 1024>>);

TEST(LdsShuffle2DEpilogue, CompileTimeGeometry) { SUCCEED(); }

} // namespace
