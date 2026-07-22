// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * MX-GEMM Dispatcher ctypes Library (TileEngine -> Dispatcher bridge).
 *
 * Provides a C API for Python ctypes integration. The kernel header is
 * force-included at compile time:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE mx_gemm_ctypes_lib.cpp
 *
 * Force-include defines (at global scope): SelectedKernel (with static constexpr
 * TileM/TileN/TileK, WarpPerBlock_{M,N,K}, WarpTile{M,N,K} and
 * static float launch(const MxGemmHostArgs&, const ck_tile::stream_config&)),
 * KERNEL_NAME, ADataType/BDataType/CDataType/AccDataType/ScaleType(=e8m0_t),
 * using MxGemmHostArgs = ck_tile::MxGemmHostArgs<1,1,0>, ALayout/BLayout/CLayout,
 * and #include "ck_tile/ops/gemm.hpp".
 *
 * Registry bypass: microscaling GEMM's launch takes ck_tile::MxGemmHostArgs
 * (with per-32-K e8m0 block scales that must be pre-shuffled for gfx950), which
 * the generic dispatcher backend cannot express. So this lib builds the HostArgs
 * from plain C arrays and calls SelectedKernel::launch() directly -- the same
 * direct-launch pattern used by the batched/multi-D bridges.
 *
 * Memory model: host-pointer in. The lib owns device allocation/copy/free via
 * ck_tile::HostTensor + ck_tile::DeviceMem (RAII), exactly like the Old-TE
 * profiler. A/B/C byte sizes come from HostTensor::get_element_space_size_in_bytes(),
 * which divides the logical element count by numeric_traits<T>::PackedSize -- so
 * fp8 (PackedSize==1) allocates M*K bytes while fp4 (pk_fp4_t PackedSize==2, two
 * e2m1 elements per byte) allocates M*K/2 bytes. The incoming A/B host buffers
 * are therefore expected PHYSICALLY packed ([M,K/PackedSize]/[N,K/PackedSize]
 * bytes); strides passed to the kernel stay in LOGICAL element units.
 * Layout is fixed by the compiled-in ALayout/BLayout/CLayout (rcr: A row-major
 * [M,K], B col-major [K,N] == [N,K] storage, C row-major [M,N]). K % 32 == 0.
 * v1 scope: k_batch == 1 (no split-K).
 *
 * The scale pre-shuffle mirrors mx_gemm_profiler.hpp exactly: pack params are
 * derived from SelectedKernel tile dims at compile time (not hardcoded).
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

// Kernel header force-included via -include. Brings in ck_tile core + the
// ck_tile::MxGemmHostArgs type, SelectedKernel/KERNEL_NAME, and (via
// ck_tile/ops/gemm.hpp) the mx pipeline. We additionally pull the host
// helpers explicitly for HostTensor / preShuffleScaleBuffer_gfx950.
#include "ck_tile/host.hpp"
// Old-TE common helpers: provides the free-function template is_row_major(Layout)
// -> ck_tile::bool_constant<...>, matching the mx_gemm profiler usage.
#include "common/utils.hpp"

#ifndef GFX_ARCH
#error \
    "GFX_ARCH must be defined at compile time (pass -DGFX_ARCH=<arch>); do not default to a specific GPU architecture."
#endif

// The MX (microscaling) block-scale pre-shuffle below uses ck_tile's gfx950-only
// preShuffleScaleBuffer_gfx950 host helper, so this bridge is inherently gfx950
// only. Make that scope explicit: fail the build clearly on any other arch
// instead of silently mis-calling the gfx950 helper.
static_assert(std::string_view(GFX_ARCH) == "gfx950",
              "mx_gemm dispatcher bridge is gfx950-only (uses preShuffleScaleBuffer_gfx950); "
              "build with -DGFX_ARCH=gfx950.");

static bool g_initialized = false;

namespace {

int env_int(const char* name, int fallback)
{
    const char* v = std::getenv(name);
    if(!v)
        return fallback;
    return std::atoi(v);
}

} // namespace

extern "C" {

int dispatcher_initialize()
{
    g_initialized = true;
    return 0;
}
int dispatcher_init() { return dispatcher_initialize(); }

const char* dispatcher_get_kernel_name()
{
#ifdef CK_TILE_SINGLE_KERNEL_INCLUDE
    return KERNEL_NAME;
#else
    // No kernel force-included (CMake no-kernel fallback): report an empty name.
    return "";
#endif
}
int dispatcher_get_kernel_count() { return 1; }

void dispatcher_cleanup() { g_initialized = false; }

/**
 * Run microscaling GEMM: C[M,N] = (A[M,K] * scale_a) . (B[K,N] * scale_b).
 *
 * A, B, C are host pointers. scale_a/scale_b are raw e8m0 bytes, unpacked and
 * unshuffled, shaped [M, K/32] and [N, K/32] row-major respectively. The lib
 * builds HostTensors from them, runs the SAME preShuffleScaleBuffer_gfx950 path
 * as the Old-TE profiler, uploads everything, and calls SelectedKernel::launch.
 *
 * Returns 0 ok, -1 HIP/bad-args, -2 unsupported shape (divisibility) or kernel rejects.
 */
int dispatcher_run_mx_gemm(const void* A,
                           const void* B,
                           void* C,
                           const uint8_t* scale_a,
                           const uint8_t* scale_b,
                           int M,
                           int N,
                           int K,
                           int k_batch,
                           float* time_ms)
{
#ifndef CK_TILE_SINGLE_KERNEL_INCLUDE
    // No kernel was force-included (see the CMake no-kernel fallback). The body
    // below needs SelectedKernel/MxGemmHostArgs/ScaleType/... which only exist
    // under -DCK_TILE_SINGLE_KERNEL_INCLUDE, so this build cannot run anything:
    // report "unsupported" (-2) instead of failing to compile/link.
    (void)A;
    (void)B;
    (void)C;
    (void)scale_a;
    (void)scale_b;
    (void)M;
    (void)N;
    (void)K;
    (void)k_batch;
    (void)time_ms;
    std::cerr << "dispatcher_run_mx_gemm: library built without a kernel; unsupported\n";
    return -2;
#else
    if(!g_initialized)
    {
        std::cerr << "dispatcher_run_mx_gemm: not initialized\n";
        return -1;
    }
    if(!A || !B || !C || !scale_a || !scale_b)
    {
        std::cerr << "dispatcher_run_mx_gemm: null pointer\n";
        return -1;
    }
    if(M <= 0 || N <= 0 || K <= 0)
    {
        std::cerr << "dispatcher_run_mx_gemm: M,N,K must be > 0\n";
        return -1;
    }
    if(K % 32 != 0)
    {
        std::cerr << "dispatcher_run_mx_gemm: MX GEMM requires K to be a multiple of 32, got K="
                  << K << "\n";
        return -1;
    }
    if(k_batch != 1)
    {
        // v1 scope: split-K (k_batch > 1) is not yet supported through this bridge.
        // This is a v1 feature limitation, not a bad-argument/HIP error, so report
        // -2 (unsupported) per the documented convention above -- the Python
        // wrapper then surfaces it as "unsupported" rather than a generic error.
        std::cerr << "dispatcher_run_mx_gemm: only k_batch==1 is supported in v1, got " << k_batch
                  << "\n";
        return -2;
    }

    // MX block-scale pre-shuffle is gfx950-only (preShuffleScaleBuffer_gfx950).
    // The compile-time static_assert already pins the build to gfx950; also guard
    // at runtime so a gfx950-built .so run on a non-gfx950 device fails clearly
    // instead of launching an arch-mismatched kernel.
    {
        int dev = 0;
        hipDeviceProp_t props{};
        if(hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&props, dev) != hipSuccess)
        {
            std::cerr << "dispatcher_run_mx_gemm: could not query device architecture\n";
            return -1;
        }
        if(std::string_view(props.gcnArchName).substr(0, 6) != "gfx950")
        {
            std::cerr << "dispatcher_run_mx_gemm: MX GEMM is gfx950-only; running device is "
                      << props.gcnArchName << "\n";
            return -1;
        }
    }

    const ALayout layout_a = ALayout{};
    const BLayout layout_b = BLayout{};
    const CLayout layout_c = CLayout{};

    const ck_tile::index_t m = static_cast<ck_tile::index_t>(M);
    const ck_tile::index_t n = static_cast<ck_tile::index_t>(N);
    const ck_tile::index_t k = static_cast<ck_tile::index_t>(K);

    // Packed default strides for the compiled-in layouts (rcr: A row-major,
    // B col-major, C row-major), matching the Old-TE profiler. get_default_stride
    // takes the row-major-ness as a compile-time bool_constant tag (4th arg).
    const ck_tile::index_t stride_a =
        static_cast<ck_tile::index_t>(ck_tile::get_default_stride(m, k, 0, is_row_major(layout_a)));
    const ck_tile::index_t stride_b =
        static_cast<ck_tile::index_t>(ck_tile::get_default_stride(k, n, 0, is_row_major(layout_b)));
    const ck_tile::index_t stride_c =
        static_cast<ck_tile::index_t>(ck_tile::get_default_stride(m, n, 0, is_row_major(layout_c)));

    const ck_tile::index_t scale_k_size = k / 32;

    // ---- Scale pre-shuffle pack params, derived from SelectedKernel (compile
    // time), exactly mirroring mx_gemm_profiler.hpp. ----
    constexpr ck_tile::index_t m_per_xdl = SelectedKernel::WarpTileM;
    constexpr ck_tile::index_t n_per_xdl = SelectedKernel::WarpTileN;
    constexpr ck_tile::index_t k_per_xdl = SelectedKernel::WarpTileK;
    constexpr ck_tile::index_t m_iter_per_warp =
        SelectedKernel::TileM / (SelectedKernel::WarpPerBlock_M * m_per_xdl);
    constexpr ck_tile::index_t n_iter_per_warp =
        SelectedKernel::TileN / (SelectedKernel::WarpPerBlock_N * n_per_xdl);
    constexpr ck_tile::index_t k_iter_per_warp = SelectedKernel::TileK / k_per_xdl;

    constexpr ck_tile::index_t m_xdl_pack =
        (m_iter_per_warp >= 2 && m_iter_per_warp % 2 == 0) ? 2 : 1;
    constexpr ck_tile::index_t n_xdl_pack =
        (n_iter_per_warp >= 2 && n_iter_per_warp % 2 == 0) ? 2 : 1;
    constexpr ck_tile::index_t k_xdl_pack =
        (k_iter_per_warp >= 2 && k_iter_per_warp % 2 == 0) ? 2 : 1;

    constexpr ck_tile::index_t xdl_mn_thread = SelectedKernel::WarpTileM;
    constexpr ck_tile::index_t xdl_k_thread  = 64 / xdl_mn_thread;

    // ---- Divisibility guard. The shuffled scale-buffer sizes below use integer
    // division by the xdl pack factors (m/m_xdl_pack, n/n_xdl_pack,
    // scale_k_size/k_xdl_pack). If M/N/scale_k are not exact multiples of their
    // pack factors, that division silently truncates the shuffled buffers, so the
    // pre-shuffle would read/write past valid data and the kernel would consume a
    // corrupt scale layout. Return -2 (unsupported shape for the selected warp
    // tile) rather than -1 (bad-args/HIP error), consistent with the IsSupportedArguments-
    // style rejects used elsewhere in the dispatcher ctypes layer. ----
    if(m % m_xdl_pack != 0 || n % n_xdl_pack != 0 || scale_k_size % k_xdl_pack != 0)
    {
        std::cerr << "dispatcher_run_mx_gemm: M, N, and scale_k (=K/32) must be divisible by the "
                     "xdl pack factors (m_xdl_pack="
                  << m_xdl_pack << ", n_xdl_pack=" << n_xdl_pack << ", k_xdl_pack=" << k_xdl_pack
                  << ") for the selected warp tile; got M=" << m << ", N=" << n
                  << ", scale_k=" << scale_k_size << "\n";
        return -2;
    }

    // ---- Build unshuffled scale HostTensors from the incoming raw e8m0 bytes.
    // scale_a: [M, K/32] row-major; scale_b: [N, K/32] row-major. ----
    ck_tile::HostTensor<ScaleType> scale_a_host(
        {static_cast<std::size_t>(m), static_cast<std::size_t>(scale_k_size)},
        {static_cast<std::size_t>(scale_k_size), static_cast<std::size_t>(1)});
    ck_tile::HostTensor<ScaleType> scale_b_host(
        {static_cast<std::size_t>(n), static_cast<std::size_t>(scale_k_size)},
        {static_cast<std::size_t>(scale_k_size), static_cast<std::size_t>(1)});

    const std::size_t scale_a_count = static_cast<std::size_t>(m) * scale_k_size;
    const std::size_t scale_b_count = static_cast<std::size_t>(n) * scale_k_size;
    // e8m0_t wraps a single uint8 (raw biased exponent); construct 1:1 from the
    // incoming raw byte via the explicit e8m0_t(raw_type) constructor.
    for(std::size_t i = 0; i < scale_a_count; ++i)
        scale_a_host.mData[i] = ScaleType(static_cast<typename ScaleType::raw_type>(scale_a[i]));
    for(std::size_t i = 0; i < scale_b_count; ++i)
        scale_b_host.mData[i] = ScaleType(static_cast<typename ScaleType::raw_type>(scale_b[i]));

    // ---- Shuffled scale buffers (same lengths as the profiler). ----
    ck_tile::HostTensor<ScaleType> scale_a_shuffled(
        {static_cast<std::size_t>(m / m_xdl_pack * 2),
         static_cast<std::size_t>(scale_k_size / k_xdl_pack * 2)},
        {static_cast<std::size_t>(scale_k_size / k_xdl_pack * 2), static_cast<std::size_t>(1)});
    ck_tile::HostTensor<ScaleType> scale_b_shuffled(
        {static_cast<std::size_t>(n / n_xdl_pack * 2),
         static_cast<std::size_t>(scale_k_size / k_xdl_pack * 2)},
        {static_cast<std::size_t>(scale_k_size / k_xdl_pack * 2), static_cast<std::size_t>(1)});

    ck_tile::preShuffleScaleBuffer_gfx950<m_xdl_pack, k_xdl_pack, xdl_mn_thread, xdl_k_thread>(
        scale_a_host.mData.data(), scale_a_shuffled.mData.data(), m, scale_k_size, true);
    ck_tile::preShuffleScaleBuffer_gfx950<n_xdl_pack, k_xdl_pack, xdl_mn_thread, xdl_k_thread>(
        scale_b_host.mData.data(), scale_b_shuffled.mData.data(), n, scale_k_size, true);

    // ---- Build A/B/C HostTensors with the SAME descriptors the Old-TE profiler
    // uses, then allocate/copy through ck_tile::DeviceMem. This makes the byte
    // accounting packing-correct for BOTH fp8 (PackedSize==1) and fp4
    // (pk_fp4_t::PackedSize==2, i.e. two logical e2m1 elements per byte):
    // HostTensor<T>::get_element_space_size_in_bytes() ==
    //   sizeof(T) * (logical_elems / numeric_traits<T>::PackedSize).
    // The incoming A/B host buffers are already PHYSICALLY packed
    // ([M, K/PackedSize] and [N, K/PackedSize] bytes for fp4; [M,K]/[N,K] for
    // fp8), so we byte-copy them into mData (which is sized in physical objects).
    // Strides remain in LOGICAL element units (get_default_stride above uses
    // logical M,N,K), which is what the kernel expects.
    ck_tile::HostTensor<ADataType> a_m_k(
        ck_tile::host_tensor_descriptor(m, k, stride_a, is_row_major(layout_a)));
    ck_tile::HostTensor<BDataType> b_k_n(
        ck_tile::host_tensor_descriptor(k, n, stride_b, is_row_major(layout_b)));
    ck_tile::HostTensor<CDataType> c_m_n(
        ck_tile::host_tensor_descriptor(m, n, stride_c, is_row_major(layout_c)));

    const std::size_t a_bytes = a_m_k.get_element_space_size_in_bytes();
    const std::size_t b_bytes = b_k_n.get_element_space_size_in_bytes();
    const std::size_t c_bytes = c_m_n.get_element_space_size_in_bytes();

    std::memcpy(a_m_k.mData.data(), A, a_bytes);
    std::memcpy(b_k_n.mData.data(), B, b_bytes);

    const std::size_t scale_a_bytes = scale_a_shuffled.get_element_space_size_in_bytes();
    const std::size_t scale_b_bytes = scale_b_shuffled.get_element_space_size_in_bytes();

    float exec_time = 0.0f;
    try
    {
        // Call-local device buffers, sized via HostTensor (packing-correct).
        // Each call allocates and (via RAII) frees its own DeviceMem -- the same
        // per-call lifetime model the batched/multi_abd bridges use. A/B/scales
        // are uploaded and C is zeroed every call, so per-shape data is fresh.
        ck_tile::DeviceMem a_dev(a_bytes);
        ck_tile::DeviceMem b_dev(b_bytes);
        ck_tile::DeviceMem c_dev(c_bytes);
        ck_tile::DeviceMem sa_dev(scale_a_bytes);
        ck_tile::DeviceMem sb_dev(scale_b_bytes);

        a_dev.ToDevice(a_m_k.data());
        b_dev.ToDevice(b_k_n.data());
        c_dev.SetZero();
        sa_dev.ToDevice(scale_a_shuffled.data());
        sb_dev.ToDevice(scale_b_shuffled.data());

        void* a_ptr       = a_dev.GetDeviceBuffer();
        void* b_ptr       = b_dev.GetDeviceBuffer();
        void* c_ptr       = c_dev.GetDeviceBuffer();
        void* scale_a_ptr = sa_dev.GetDeviceBuffer();
        void* scale_b_ptr = sb_dev.GetDeviceBuffer();

        // ---- Build MxGemmHostArgs in the exact profiler argument order:
        // a_ptr, scale_a, b_ptr, scale_b, ds{}, c_ptr, split_k, m, n, k,
        // {stride_a}, {stride_b}, ds_strides{}, stride_c. ----
        MxGemmHostArgs gemm_args({a_ptr},
                                 {scale_a_ptr},
                                 {b_ptr},
                                 {scale_b_ptr},
                                 {},
                                 c_ptr,
                                 static_cast<ck_tile::index_t>(k_batch),
                                 m,
                                 n,
                                 k,
                                 {stride_a},
                                 {stride_b},
                                 {},
                                 stride_c);

        const bool do_time = (time_ms != nullptr);
        // Defaults match the bridge parity-sweep convention (warmup 50 / repeat
        // 100) so bridge-vs-Old-TE timings are apples-to-apples out of the box;
        // both remain env-overridable for custom sweeps (set identical values on
        // both sides).
        const int warmup = do_time ? env_int("CK_TILE_BENCH_WARMUP", 50) : 0;
        const int repeat = do_time ? env_int("CK_TILE_BENCH_REPEAT", 100) : 1;
        // stream_config field order: stream_id, time_kernel, log_level, cold_niters
        // (warmup), nrepeat, is_gpu_timer, flush_cache, rotating_count.
        ck_tile::stream_config stream_cfg{nullptr, do_time, 0, warmup, repeat, false, false, 1};

        exec_time = SelectedKernel::launch(gemm_args, stream_cfg);
        if(exec_time < 0.0f)
        {
            std::cerr << "dispatcher_run_mx_gemm: kernel reports unsupported args\n";
            return -2;
        }

        // D2H from c_ptr into the host output tensor.
        if(hipMemcpy(c_m_n.data(), c_ptr, c_bytes, hipMemcpyDeviceToHost) != hipSuccess)
            throw std::runtime_error("hipMemcpy D2H failed");
    }
    catch(const std::exception& e)
    {
        std::cerr << "dispatcher_run_mx_gemm: launch threw: " << e.what() << "\n";
        return -1;
    }

    // C is always a non-packed output type (fp16/bf16), so c_bytes byte-copy back
    // to the caller buffer is a 1:1 element copy.
    std::memcpy(C, c_m_n.mData.data(), c_bytes);

    if(time_ms)
        *time_ms = exec_time;

    return 0;
#endif // CK_TILE_SINGLE_KERNEL_INCLUDE
}

} // extern "C"
