// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

// Large-tensor RowColQuant coverage: exercises the 64-bit global load/store path
// (GemmConfigLargeTensor sets LargeTensors=true) for every A/B layout combination.
// RowColQuant requires a RowMajor C, so the combinations are A in {Row, Col} x B in
// {Row, Col} with C fixed to RowMajor.
//

// Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType,
// CDataType, QuantType, GemmConfig, QuantGroupSize>
// clang-format off
using RowColQuantLargeTensorTypes = ::testing::Types<
    std::tuple<RowMajor,    RowMajor,    RowMajor, RowMajor, FP8, FP8, float, Half, RowColQuant, GemmConfigLargeTensor, GroupSize1D_128>,
    std::tuple<RowMajor,    ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, RowColQuant, GemmConfigLargeTensor, GroupSize1D_128>,
    std::tuple<ColumnMajor, RowMajor,    RowMajor, RowMajor, FP8, FP8, float, Half, RowColQuant, GemmConfigLargeTensor, GroupSize1D_128>,
    std::tuple<ColumnMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, RowColQuant, GemmConfigLargeTensor, GroupSize1D_128>
>;
// clang-format on

TYPED_TEST_SUITE(TestCkTileGemmRowColQuant, RowColQuantLargeTensorTypes);

// Validated: builds the kernel with LargeTensors=true and checks numerics against the
// host reference for every layout combination at a modest size. This size is below the
// runtime large-tensor gate (2^31 bytes), so the executed branch is the normal
// small-tensor path -- the purpose here is to prove that enabling the compile-time
// LargeTensors flag (which widens offsets and compiles in the 64-bit branches) does not
// regress numerical correctness of the common path. The true >2GB 64-bit arithmetic is
// exercised (without a host reference) by LargeTensorPathLaunchOnly below; validating it
// numerically is infeasible because a host GEMM reference at that scale is prohibitive.
TYPED_TEST(TestCkTileGemmRowColQuant, LargeTensorPathValidated)
{
    this->run_test_with_validation(1024, 1024, 1024);
}

// GPU-reference validated: C is M * N * sizeof(Half) = 32768 * 32896 * 2 bytes ~= 2.006 GiB,
// strictly above the 2 GiB single-buffer limit (2^31) with a one-N-tile margin so the runtime
// large-tensor gate stays engaged for every combo even if its comparison is ever tightened.
// A=ColumnMajor combos are accepted via the global load/store branch; A=RowMajor combos are
// accepted via the M base-shift branch (with a globalized C store). N is kept a multiple of
// N_Tile (128) so no padding is required. The full output is validated against a GPU-computed
// reference GEMM. Requires enough free device memory for two ~2 GiB C buffers.
TYPED_TEST(TestCkTileGemmRowColQuant, LargeTensorPathGpuReference)
{
    this->run_test_with_gpu_reference(32768, 32896, 512);
}

// Non-tile-multiple coverage for the large-tensor path. GemmConfigLargeTensorPadded enables
// kPad* so IsSupportedArgument accepts remainder dimensions; with LargeTensors=true the
// kernel takes the 64-bit global path, so the pad-transform guards (not hardware buffer
// bounds-checking) must mask the out-of-range tiles for every A/B layout.
// clang-format off
using RowColQuantLargeTensorPaddedTypes = ::testing::Types<
    std::tuple<RowMajor,    RowMajor,    RowMajor, RowMajor, FP8, FP8, float, Half, RowColQuant, GemmConfigLargeTensorPadded, GroupSize1D_128>,
    std::tuple<RowMajor,    ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, RowColQuant, GemmConfigLargeTensorPadded, GroupSize1D_128>,
    std::tuple<ColumnMajor, RowMajor,    RowMajor, RowMajor, FP8, FP8, float, Half, RowColQuant, GemmConfigLargeTensorPadded, GroupSize1D_128>,
    std::tuple<ColumnMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, RowColQuant, GemmConfigLargeTensorPadded, GroupSize1D_128>
>;
// clang-format on

template <typename Tuple>
class TestCkTileGemmRowColQuantPadded : public TestCkTileGemmRowColQuant<Tuple>
{
};

TYPED_TEST_SUITE(TestCkTileGemmRowColQuantPadded, RowColQuantLargeTensorPaddedTypes);

// N=144 (=16*9, not a multiple of N_Tile 128) and K=272 (=16*17, not a multiple of K_Tile
// 256) carry remainder tiles across multiple M-tiles. M has no remainder: MPerBlock (16)
// equals the A vector size, so IsSupportedArgument requires M to be a multiple of 16
// regardless of kPadM.
TYPED_TEST(TestCkTileGemmRowColQuantPadded, NonMultiplePaddingValidated)
{
    this->run_test_with_validation(128, 144, 272);
}

// Additional large-tensor coverage derived from shapes.yaml (a corpus of 2199 hipblaslt
// bf16 matmul benchmark shapes) entries 433, 918, and 1303 -- the only 3 of those 2199
// shapes whose A, B, or C byte footprint (at this file's FP8/FP8/Half widths) crosses the
// 2^31-byte single-buffer limit this file exercises. M/N/K are shapes.yaml's raw values
// rounded down to the nearest multiple of 16 (the unconditional vector-load-width floor
// that IsSupportedArgument enforces regardless of kPad*); all three collapse to
// permutations of {6512, 640, 393216}. Each uses a single-tuple type list (rather than the
// full 4-combo sweep above) since only one A/B layout combination is relevant per shape.
// Each shape is validated numerically: GpuReference (full-output comparison against a GPU
// reference GEMM via run_test_with_gpu_reference) where reading the whole C output back is
// feasible, plus BoundaryCheck (spot-checks exact output values at the addressing boundary via
// run_test_boundary_check, which stays cheap regardless of C size). The large-C shape below
// (~4.8 GiB C) is validated by BoundaryCheck alone: a full GPU reference over its ~2.6e9
// outputs, plus the ~10 GiB host read-back of two C buffers, is impractical.

// shapes.yaml:433 -- large A (ColumnMajor A: 6512*393216*1B ~= 2.385 GiB). All of M, N, K
// land on exact tile multiples, so no padding is required.
template <typename Tuple>
class TestCkTileGemmRowColQuantLargeA : public TestCkTileGemmRowColQuant<Tuple>
{
};

// clang-format off
using RowColQuantLargeATypes = ::testing::Types<
    std::tuple<ColumnMajor, RowMajor, RowMajor, RowMajor, FP8, FP8, float, Half, RowColQuant, GemmConfigLargeTensor, GroupSize1D_128>
>;
// clang-format on

TYPED_TEST_SUITE(TestCkTileGemmRowColQuantLargeA, RowColQuantLargeATypes);

TYPED_TEST(TestCkTileGemmRowColQuantLargeA, GpuReference)
{
    this->run_test_with_gpu_reference(6512, 640, 393216);
}

// A's own addressing (ColumnMajor, stride_A=M=6512) crosses the 2^31-byte boundary at
// K-index ~329773 (2^31 / 6512); hot_k sits right at that column so every read of A during
// the main loop's K-tail is exercised at the dangerous offset. The spot-checked (m, n)
// coordinates cover a representative spread of C, since the risky address is in A's read,
// not C's write.
TYPED_TEST(TestCkTileGemmRowColQuantLargeA, BoundaryCheck)
{
    this->run_test_boundary_check(6512, 640, 393216, 329776, {{0, 0}, {3256, 320}, {6511, 639}});
}

// shapes.yaml:918 -- large C via huge N (RowMajor C: 6512*393216*2B ~= 4.769 GiB). K=640
// is not a multiple of K_Tile (256), so the padded config is required.
template <typename Tuple>
class TestCkTileGemmRowColQuantLargeC : public TestCkTileGemmRowColQuant<Tuple>
{
};

// clang-format off
using RowColQuantLargeCTypes = ::testing::Types<
    std::tuple<ColumnMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, RowColQuant, GemmConfigLargeTensorPadded, GroupSize1D_128>
>;
// clang-format on

TYPED_TEST_SUITE(TestCkTileGemmRowColQuantLargeC, RowColQuantLargeCTypes);

// Validated by BoundaryCheck only (C ~= 4.8 GiB): a full GPU reference over ~2.6e9 outputs plus
// a ~10 GiB two-buffer host read-back is impractical.
// C's own addressing (RowMajor, stride_C=N=393216) crosses the 2^31-byte boundary at
// M-index 2731 for n=196608 (the boundary M-index depends on n, since the linear offset is
// m*stride_C+n; naive division by stride_C alone ignores the n term). hot_k can be any
// valid K-index since K is small here and carries no addressing risk of its own.
TYPED_TEST(TestCkTileGemmRowColQuantLargeC, BoundaryCheck)
{
    this->run_test_boundary_check(6512, 393216, 640, 0, {{0, 0}, {2732, 196608}, {6511, 393215}});
}

// shapes.yaml:1303 -- large B via K*N (ColumnMajor B: 6512*393216*1B ~= 2.385 GiB). K=6512
// is not a multiple of K_Tile (256), so the padded config is required. RowMajor A here is
// small (640*6512*1B ~= 4 MiB) and is accepted through the pre-existing M base-shift path,
// not this PR's new global-load code -- this shape exercises the large-B path specifically.
template <typename Tuple>
class TestCkTileGemmRowColQuantLargeB : public TestCkTileGemmRowColQuant<Tuple>
{
};

// clang-format off
using RowColQuantLargeBTypes = ::testing::Types<
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, RowColQuant, GemmConfigLargeTensorPadded, GroupSize1D_128>
>;
// clang-format on

TYPED_TEST_SUITE(TestCkTileGemmRowColQuantLargeB, RowColQuantLargeBTypes);

TYPED_TEST(TestCkTileGemmRowColQuantLargeB, GpuReference)
{
    this->run_test_with_gpu_reference(640, 393216, 6512);
}

// B's own addressing (ColumnMajor, stride_B=K=6512) crosses the 2^31-byte boundary at
// N-index ~329773 (2^31 / 6512); the sparse B(hot_k, n) row spans every n, so this is
// exercised automatically -- the spot-check at n~329776 confirms the far-offset read
// landed on the correct value rather than merely not faulting.
TYPED_TEST(TestCkTileGemmRowColQuantLargeB, BoundaryCheck)
{
    this->run_test_boundary_check(640, 393216, 6512, 3256, {{0, 0}, {320, 329776}, {639, 393215}});
}

// Not from shapes.yaml: the largest M among the shapes above is 6512. This case pushes M
// itself to an extreme (1,000,000) with a comparatively small N/K, so A crosses the 2GiB
// threshold via M*K rather than via a large K (contrast LargeA, which uses a much larger K
// with a comparatively small M). ColumnMajor A means stride_A=M, so this specifically
// stresses the new global-load path's 64-bit offset arithmetic at the largest M scale
// tested in this file. M, N, and K are all exact tile multiples, so no padding is required.
template <typename Tuple>
class TestCkTileGemmRowColQuantLargeM : public TestCkTileGemmRowColQuant<Tuple>
{
};

// clang-format off
using RowColQuantLargeMTypes = ::testing::Types<
    std::tuple<ColumnMajor, RowMajor, RowMajor, RowMajor, FP8, FP8, float, Half, RowColQuant, GemmConfigLargeTensor, GroupSize1D_128>
>;
// clang-format on

TYPED_TEST_SUITE(TestCkTileGemmRowColQuantLargeM, RowColQuantLargeMTypes);

TYPED_TEST(TestCkTileGemmRowColQuantLargeM, GpuReference)
{
    this->run_test_with_gpu_reference(1000000, 128, 2560);
}

// A's own addressing (ColumnMajor, stride_A=M=1,000,000) crosses the 2^31-byte boundary at
// K-index 2148 (2^31 / 1,000,000); hot_k sits just past that column so every read of A
// during the main loop's K-tail is exercised at the dangerous offset, for every M-tile up
// to the far corner at M-1.
TYPED_TEST(TestCkTileGemmRowColQuantLargeM, BoundaryCheck)
{
    this->run_test_boundary_check(1000000, 128, 2560, 2150, {{0, 0}, {500000, 64}, {999999, 127}});
}

// --- Byte-extent boundary edge cases -------------------------------------------------------
// Per review: exercise the exact 2^31 / 2^32 byte crossings that were historically mishandled.
//
// NOTE on instruction selection: IsSupportedArgument's is_large_tensor() flags a tensor as
// "large" (routing it to the 64-bit global path) once its single-dimension byte extent is
// >= 2^31. Under that rule every case below takes the GLOBAL path. The review observes that a
// 4 GiB tensor whose maximum byte offset still fits a 32-bit *unsigned* index could in
// principle stay on buffer instructions; that would require raising the threshold (e.g. to
// 2^32 for uint-offset element types) and is left as a follow-up for the maintainers. These
// tests validate correctness at the boundary regardless of which instruction path is chosen.
// A=ColumnMajor forces the global load/store branch (RowMajor A would take the M base-shift
// path instead), so the large operand's 64-bit addressing is what is under test.

// fp8 B exactly 4 GiB (ColumnMajor B, K*N*1B = 256 * 16777216 = 2^32): the far-corner B byte
// offset reaches 2^32, so both the element index and the byte offset exceed 32 bits and the
// global (64-bit) path is mandatory. A (16x256) and C (16x16777216 Half ~= 512 MiB) stay
// small, isolating B's addressing. Validated in full against the GPU reference.
template <typename Tuple>
class TestCkTileGemmRowColQuantB4GiB : public TestCkTileGemmRowColQuant<Tuple>
{
};

// clang-format off
using RowColQuantB4GiBTypes = ::testing::Types<
    std::tuple<ColumnMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, RowColQuant, GemmConfigLargeTensor, GroupSize1D_128>
>;
// clang-format on

TYPED_TEST_SUITE(TestCkTileGemmRowColQuantB4GiB, RowColQuantB4GiBTypes);

TYPED_TEST(TestCkTileGemmRowColQuantB4GiB, GpuReference)
{
    this->run_test_with_gpu_reference(16, 16777216, 256);
}

// fp8 B exactly 2 GiB (ColumnMajor B, K*N*1B = 256 * 8388608 = 2^31): B lands exactly on the
// is_large_tensor threshold. Its maximum byte offset is 2^31 - 1 -- the largest a signed
// 32-bit index could still address -- yet the current >= 2^31 rule still routes it through the
// global path. Validated in full against the GPU reference.
template <typename Tuple>
class TestCkTileGemmRowColQuantB2GiB : public TestCkTileGemmRowColQuant<Tuple>
{
};

// clang-format off
using RowColQuantB2GiBTypes = ::testing::Types<
    std::tuple<ColumnMajor, ColumnMajor, RowMajor, RowMajor, FP8, FP8, float, Half, RowColQuant, GemmConfigLargeTensor, GroupSize1D_128>
>;
// clang-format on

TYPED_TEST_SUITE(TestCkTileGemmRowColQuantB2GiB, RowColQuantB2GiBTypes);

TYPED_TEST(TestCkTileGemmRowColQuantB2GiB, GpuReference)
{
    this->run_test_with_gpu_reference(16, 8388608, 256);
}

// fp16 (Half) C exactly 4 GiB / 2^31 elements (RowMajor C, M*N = 8192 * 262144 = 2^31): the C
// element index reaches 2^31 (overflows a signed 32-bit element index) while the byte offset
// reaches 2^32 (fits a 32-bit *unsigned* byte index) -- the "2 GB element / 4 GB byte" Half
// case from the review. A full GPU reference over 2^31 outputs is impractical, so C's global
// store is validated by exact boundary spot-checks (a single hot K column makes
// C(m, n) = (m % 8) * (n % 8)); the far-corner (M-1, N-1) exercises the 4 GiB-byte write.
template <typename Tuple>
class TestCkTileGemmRowColQuantCHalf4GiB : public TestCkTileGemmRowColQuant<Tuple>
{
};

// clang-format off
using RowColQuantCHalf4GiBTypes = ::testing::Types<
    std::tuple<ColumnMajor, RowMajor, RowMajor, RowMajor, FP8, FP8, float, Half, RowColQuant, GemmConfigLargeTensor, GroupSize1D_128>
>;
// clang-format on

TYPED_TEST_SUITE(TestCkTileGemmRowColQuantCHalf4GiB, RowColQuantCHalf4GiBTypes);

TYPED_TEST(TestCkTileGemmRowColQuantCHalf4GiB, BoundaryCheck)
{
    this->run_test_boundary_check(8192, 262144, 256, 128, {{0, 0}, {4096, 131072}, {8191, 262143}});
}

// --- Large-K RowMajor A (64-bit global load/store) -----------------------------------------
// A RowMajor A that is large purely because of K. It rides the M base-shift path
// (IsLargeTensorMOffsettingSupported()), which re-bases the A pointer per M-tile in 64-bit and
// clamps the per-block M to MPerBlock; its per-M-tile A view then spans the full K extent, whose
// far offset (MPerBlock-1)*stride_A + (K-1) = 16*K - 1 exceeds INT32_MAX. With LargeTensors=true
// that A view is built on the 64-bit global path (kAGlobalLoad), so the far element is addressed
// correctly instead of wrapping -- the RowMajor-A analog of the ColumnMajor / large-C 64-bit
// cases above. M=16 (=MPerBlock) and K=2^27+256 (K_Tile-aligned) make A large via K alone;
// hot_k=K-1 places the single non-zero A column at the far offset 16*K-1 = 2^31+4095, which only
// row M-1=15 reaches; the spot-checked C(15, n) would be wrong under the old 32-bit wrap, so it
// pins the 64-bit read. B (K*N*1B ~= 16 GiB) dominates the footprint, so this skips unless the
// device has room.
template <typename Tuple>
class TestCkTileGemmRowColQuantLargeKRowMajorA : public TestCkTileGemmRowColQuant<Tuple>
{
};

// clang-format off
using RowColQuantLargeKRowMajorATypes = ::testing::Types<
    std::tuple<RowMajor, RowMajor, RowMajor, RowMajor, FP8, FP8, float, Half, RowColQuant, GemmConfigLargeTensor, GroupSize1D_128>
>;
// clang-format on

TYPED_TEST_SUITE(TestCkTileGemmRowColQuantLargeKRowMajorA, RowColQuantLargeKRowMajorATypes);

TYPED_TEST(TestCkTileGemmRowColQuantLargeKRowMajorA, BoundaryCheck)
{
    this->run_test_boundary_check(
        16, 128, (1 << 27) + 256, (1 << 27) + 255, {{0, 0}, {15, 1}, {15, 127}});
}
