// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Role: meta -- GemmSpec structural NTTP descriptor and consteval factory.
//
// SHARED header: compiled in both host and device (--cuda-device-only) passes.
// Contains structural types, consteval makeSpec() factory, and named accessors.
// No runtime code.
//
// Wave tile validation and target properties live in arch_properties.hpp.
//
// Compilation boundary:
//   _spec.hpp (this) -- schema types + consteval factory (both passes)
//   _dev.hpp           -- CK Tile bridge + __device__ code (device pass only, #error on host)

#pragma once

#include <rocm_ck/arch_properties.hpp>
#include <rocm_ck/datatype.hpp>
#include <rocm_ck/gpu_target.hpp>
#include <rocm_ck/layout.hpp>
#include <rocm_ck/physical_tensor.hpp>
#include <rocm_ck/resolve.hpp>
#include <rocm_ck/resolved_tensor.hpp>

#include <optional>
#include <tuple>
#include <utility>

namespace rocm_ck {

// ============================================================================
// Epilogue operations (composable chain)
// ============================================================================

/// Epilogue operations applied after the GEMM matmul result.
/// NTTP projection of the Op variant alternatives that can appear in a GEMM
/// epilogue chain. Enum is structural (NTTP-compatible); std::variant is not.
///
/// Binary ops (Add, Mul) fold over D tensors via parameter pack:
///   Add -- result += D0 [+ D1]     (bias addition)
///   Mul -- result *= D0 [* D1]     (scaling)
///
/// Unary ops transform the accumulator in place:
///   Relu     -- max(0, x)
///   FastGelu -- approximate GELU: x * sigmoid(1.702 * x)
///   Gelu     -- exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
///   Silu     -- x * sigmoid(x)  (aka Swish with beta=1)
///   Sigmoid  -- 1 / (1 + exp(-x))
///
/// Operations compose as an ordered sequence in GemmSpec::epilogue_ops[].
/// The Signature's operator chain (AddOp -> ReluOp) maps directly to this
/// array, minus the string names (which aren't structural types for NTTP).
enum class EpilogueOp
{
    Add,
    Mul,
    Relu,
    FastGelu,
    Gelu,
    Silu,
    Sigmoid
};

/// Maximum epilogue ops in a chain. 4 covers combine + activation + future ops.
inline constexpr int kMaxEpilogueOps = 4;

// ============================================================================
// Tile geometry types
// ============================================================================

/// Pipeline implementation strategy for the GEMM kernel.
///
/// V1: Simple pipeline -- A/B from global memory, C in registers.
///     Uses GemmPipelineProblem + GemmPipelineAGmemBGmemCRegV1.
///
/// V3: Compute-optimized pipeline -- software-pipelined loads.
///     Uses UniversalGemmPipelineProblem + GemmPipelineAgBgCrCompV3.
///     Better compute utilization through overlapped memory/compute.
///
/// V4: Compute double-buffer -- ping-pong LDS layout.
///     Uses UniversalGemmPipelineProblem + GemmPipelineAgBgCrCompV4.
///     Better compute/memory overlap through dual LDS buffers.
///
/// Memory: Memory-optimized pipeline -- A/B from global memory through LDS.
///     Uses UniversalGemmPipelineProblem + GemmPipelineAgBgCrMem.
///     Supports both Intrawave and Interwave scheduling.
///
/// Preshuffle: Weight preshuffle pipeline -- B matrix pre-rearranged for
///     optimal LDS loads. Uses WeightPreshufflePipelineAGmemBGmemCRegV2.
///     Requires A=RowMajor, B=ColumnMajor. Host must call preshuffle on B
///     before kernel launch.
enum class Pipeline
{
    V1,
    V3,
    V4,
    Memory,
    Preshuffle
};

/// Instruction scheduling strategy within a wavefront.
///
/// Controls how MFMA/WMMA instructions are scheduled relative to memory
/// operations within each wave. This is instruction-level scheduling,
/// not spatial decomposition (which is TilePartitioner's concern).
///
/// Intrawave: Synchronous -- all waves in a workgroup synchronize after each
///     k-iteration. Memory loads and compute are interleaved within a single wave.
///     Two block_sync_lds() calls per iteration.
///
/// Interwave: Asynchronous -- waves proceed independently with minimal
///     synchronization. Only one block_sync_lds() per iteration. Overlaps
///     compute from one wave with memory loads from another.
///     Only valid with Pipeline::Memory.
enum class PipelineScheduler
{
    Intrawave,
    Interwave
};

/// Tile-to-workgroup distribution strategy.
///
/// Controls how GEMM output tiles are assigned to workgroups. This is spatial
/// decomposition, not instruction scheduling (which is Pipeline's concern:
/// Intrawave/Interwave).
///
/// Direct: 2D grid with direct blockIdx mapping.
///     Grid: (M/TileM) x (N/TileN) x k_batch.
///     Mapping: blockIdx.x -> M tiles, blockIdx.y -> N tiles.
///     Uses GemmKernel<GemmTile2DPartitioner, Pipeline, Epilogue>.
///
/// Linear: 1D grid with row-major linearized tile indexing (default).
///     Grid: (M/TileM * N/TileN) x 1 x k_batch.
///     Mapping: blockIdx.x linearized across M x N tile space.
///     Uses GemmKernel<GemmTile1DPartitioner, Pipeline, Epilogue>.
///
/// SpatiallyLocal: 1D grid with locality-aware grouping for multi-die GPUs.
///     Grid: (M/TileM * N/TileN) x 1 x k_batch.
///     Mapping: adjacent tiles grouped to same die for L2 cache locality.
///     Uses GemmKernel<GemmSpatiallyLocalTilePartitioner, Pipeline, Epilogue>.
///
/// StreamK: 1D grid with iteration-range streaming for work balancing.
///     Grid: device_CUs * occupancy (not tile-based).
///     Mapping: dynamic iteration range assignment across workgroups.
///     Uses StreamKKernel<StreamKTilePartitioner, Pipeline, Epilogue>.
enum class TilePartitioner
{
    Direct,
    Linear,
    SpatiallyLocal,
    StreamK
};

/// Store strategy for writing GEMM results to global memory.
///
/// CShuffle: Cross-wavefront shuffle through LDS for coalesced writes (default).
///     Supports all fused ops (Add, Mul, Relu) and D tensors.
///     Constraint: requires block_waves.k == 1.
///
/// Direct2D: Direct 2D thread-to-memory store without LDS shuffle.
///     Lower LDS pressure, simpler instruction pattern.
///     Limitation: no D tensor support yet (fused bias/scale).
enum class StoreStrategy
{
    CShuffle,
    Direct2D
};

/// M x N x K dimension triple for tile geometry specification.
struct Dim3
{
    int m, n, k;
};

/// Algorithm: describes HOW a GEMM executes (tile geometry, partitioning).
/// Independent of data types -- paired with Signature in makeSpec().
struct GemmAlgorithm
{
    Dim3 block_tile;       // Elements per workgroup {M, N, K}
    Dim3 block_waves;      // Wavefronts per workgroup {M, N, K}
    Dim3 wave_tile;        // Wave instruction tile {M, N, K} (MFMA on CDNA, WMMA on RDNA)
    int k_batch       = 1; // Split-K factor: partitions K across blockIdx.z
    Pipeline pipeline = Pipeline::V1; // Pipeline implementation strategy
    PipelineScheduler pipeline_scheduler =
        PipelineScheduler::Intrawave;                           // Instruction scheduling strategy
    TilePartitioner tile_partitioner = TilePartitioner::Linear; // Tile-to-workgroup distribution
    StoreStrategy store_strategy =
        StoreStrategy::CShuffle; // How results are written to global memory

    /// Padding flags control boundary handling for misaligned output dimensions.
    ///   pad_m: allow M not divisible by block_tile.m (masks A row / C row boundaries)
    ///   pad_n: allow N not divisible by block_tile.n (masks B col / C col boundaries)
    /// When false (default), kernel assumes aligned sizes for maximum performance.
    /// When true, kernel handles boundaries with bounds checks.
    ///
    /// Note: K must always be divisible by block_tile.k. CK Tile's kPadK flag only
    /// controls vector load width (scalar vs vectorized) -- it does NOT mask the K-tail.
    /// Passing non-aligned K produces silently wrong results.
    bool pad_m = false;
    bool pad_n = false;
};

// ============================================================================
// GemmSpec -- structural NTTP for template instantiation
// ============================================================================

/// Validated kernel descriptor with all types, layouts, and tile geometry resolved.
/// All members are structural types (enums, ints, aggregates) so this works as NTTP.
///
/// Physical tensor table layout (ordered by args_slot):
///   [0] = lhs (GEMM left operand -- name is user-chosen, e.g., "A", "Q")
///   [1] = rhs (GEMM right operand -- name is user-chosen, e.g., "B", "K")
///   [2] = output (final output -- name varies by epilogue chain)
///   [3] = D0 (optional -- first auxiliary epilogue tensor, e.g., "bias")
///   [4] = D1 (optional -- second auxiliary epilogue tensor)
///
/// "D tensor" is CK Tile's convention for auxiliary tensors that participate
/// in the epilogue (bias, scale, residual) but are not GEMM operands.
struct GemmSpec
{
    // Physical tensor table -- the kernel's view of Args::tensors[]
    int num_physical_tensors;
    std::array<PhysicalTensor, kMaxPhysicalTensors> physical_tensors;

    // Accumulator type (register-only, not a physical tensor)
    DataType acc_dtype;

    // Tile geometry
    Dim3 block_tile;
    Dim3 block_waves;
    Dim3 wave_tile;
    int workgroup_size;
    int k_batch; // Split-K factor (1 = no split)

    // Pipeline implementation strategy
    Pipeline pipeline;

    // Instruction scheduling strategy
    PipelineScheduler pipeline_scheduler;

    // Tile-to-workgroup distribution strategy
    TilePartitioner tile_partitioner;

    // Epilogue: composable op chain applied after matmul
    int num_epilogue_ops;
    std::array<EpilogueOp, kMaxEpilogueOps> epilogue_ops;

    // Store strategy (how results are written to global memory)
    StoreStrategy store_strategy;

    // Padding flags (boundary handling for misaligned output dimensions)
    bool pad_m;
    bool pad_n;

    // Quantization group size (0 = not quantized, >0 = elements per group along K)
    int group_size;

    /// Number of auxiliary D tensors (bias, etc.) -- excludes scale tensor.
    /// Derived from the physical tensor table: total slots minus lhs/rhs/output minus scale.
    constexpr int numDTensors() const
    {
        return num_physical_tensors - 3 - (group_size > 0 ? 1 : 0);
    }

    /// Check if the epilogue chain contains a specific op.
    constexpr bool hasEpilogueOp(EpilogueOp op) const
    {
        for(int i = 0; i < num_epilogue_ops; ++i)
            if(epilogue_ops[i] == op)
                return true;
        return false;
    }

    /// GEMM left operand (position 0 in the physical tensor table).
    constexpr PhysicalTensor lhs() const { return physical_tensors[0]; }

    /// GEMM right operand (position 1 in the physical tensor table).
    constexpr PhysicalTensor rhs() const { return physical_tensors[1]; }

    /// GEMM output tensor (position 2 in the physical tensor table).
    /// Name varies by epilogue chain: "C" (plain), "D" (with combine), "E" (with activation).
    constexpr PhysicalTensor output() const { return physical_tensors[2]; }

    /// First auxiliary tensor D0 (position 3 -- e.g., bias for AddOp).
    /// Only valid when num_physical_tensors > 3.
    constexpr PhysicalTensor d0() const { return physical_tensors[3]; }

    /// Second auxiliary tensor D1 (position 4 -- e.g., second bias/scale).
    /// Only valid when num_physical_tensors > 4.
    constexpr PhysicalTensor d1() const { return physical_tensors[4]; }

    /// Quantization scale tensor (last physical tensor when group_size > 0).
    /// Only valid when group_size > 0.
    constexpr PhysicalTensor scale() const { return physical_tensors[num_physical_tensors - 1]; }
};

// ============================================================================
// Named tensor accessors (consteval -- compile-time only)
// ============================================================================

/// Lookup a physical tensor by name. consteval -- compile-time only.
/// Used in static_asserts and consteval makeSpec() result inspection.
/// For runtime access, use GemmSpec::output() or physical_tensors[] directly.
consteval PhysicalTensor tensor(const GemmSpec& k, std::string_view name)
{
    for(int i = 0; i < k.num_physical_tensors; ++i)
        if(k.physical_tensors[i].name == name)
            return k.physical_tensors[i];
    throw "tensor is not a physical slot in this kernel";
}

/// Slot index lookup by name. consteval -- compile-time only.
consteval int slot(const GemmSpec& k, std::string_view name) { return tensor(k, name).args_slot; }

/// Dtype lookup by name. consteval -- compile-time only.
consteval DataType dtype(const GemmSpec& k, std::string_view name) { return tensor(k, name).dtype; }

/// Layout lookup by name. consteval -- compile-time only.
consteval Layout layout(const GemmSpec& k, std::string_view name) { return tensor(k, name).layout; }

// ============================================================================
// Epilogue op helpers
// ============================================================================

/// Parse a binary epilogue op (Add or Mul) from an Op.
/// Returns {EpilogueOp, rhs_name, output_name} or nullopt.
consteval std::optional<std::tuple<EpilogueOp, std::string_view, std::string_view>>
parseBinaryEpilogueOp(const Op& op)
{
    return visitOp(
        op,
        [](const auto& typed_op)
            -> std::optional<std::tuple<EpilogueOp, std::string_view, std::string_view>> {
            using T = std::remove_cvref_t<decltype(typed_op)>;
            if constexpr(std::is_same_v<T, AddOp>)
                return {{EpilogueOp::Add, typed_op.rhs, typed_op.out}};
            else if constexpr(std::is_same_v<T, MulOp>)
                return {{EpilogueOp::Mul, typed_op.rhs, typed_op.out}};
            else
                return std::nullopt;
        });
}

/// Parse a unary epilogue activation from an Op. Returns {EpilogueOp, output_name}
/// or nullopt if the op is not a unary activation.
/// Adding a new activation requires one line here + one EpilogueOp enum value.
consteval std::optional<std::pair<EpilogueOp, std::string_view>> parseUnaryEpilogueOp(const Op& op)
{
    return visitOp(
        op, [](const auto& typed_op) -> std::optional<std::pair<EpilogueOp, std::string_view>> {
            using T = std::remove_cvref_t<decltype(typed_op)>;
            if constexpr(std::is_same_v<T, ReluOp>)
                return {{EpilogueOp::Relu, typed_op.out}};
            else if constexpr(std::is_same_v<T, FastGeluOp>)
                return {{EpilogueOp::FastGelu, typed_op.out}};
            else if constexpr(std::is_same_v<T, GeluOp>)
                return {{EpilogueOp::Gelu, typed_op.out}};
            else if constexpr(std::is_same_v<T, SiluOp>)
                return {{EpilogueOp::Silu, typed_op.out}};
            else if constexpr(std::is_same_v<T, SigmoidOp>)
                return {{EpilogueOp::Sigmoid, typed_op.out}};
            else
                return std::nullopt;
        });
}

/// Extract dtype and layout for a D tensor (bias/scale operand in epilogue).
/// Auto layout defaults to Row for epilogue D tensors.
consteval std::pair<DataType, Layout> extractDTensorMeta(const ResolvedSignature& resolved,
                                                         std::string_view name)
{
    ResolvedTensor td = resolved.tensor(name);
    Layout l          = td.layout != Layout::Auto ? td.layout : Layout::Row;
    return {td.dtype, l};
}

// ============================================================================
// makeSpec: operator-centric Signature -> GemmSpec
// ============================================================================

/// Resolve and validate a GEMM using the operator-centric Signature.
///
/// Signature describes WHAT to compute (operator graph, dtypes, layouts).
/// Algorithm describes HOW to compute it (tile sizes, pipeline, partitioning).
///
/// Pattern-matches the ops array to build the epilogue_ops chain:
///   {GemmOp}                          -> epilogue_ops = {}
///   {GemmOp, AddOp}                   -> epilogue_ops = {Add}
///   {GemmOp, AddOp, ReluOp}           -> epilogue_ops = {Add, Relu}
///   {GemmOp, MulOp}                   -> epilogue_ops = {Mul}
///   {GemmOp, ReluOp}                  -> epilogue_ops = {Relu}
///
/// The D tensor is the rhs of the AddOp/MulOp (the "bias" or "scale" operand).
/// c_dtype comes from the GemmOp output tensor.
///
/// Validates:
///   - All tile dimensions are positive
///   - block_waves.k == 1 (CShuffleEpilogue requires waves_m x waves_n layout)
///   - Warp tile matches instruction table for the input dtype
///   - Block tile is divisible by (block_waves x wave_tile) in each dimension
///
/// Derives workgroup_size = block_waves.m x block_waves.n x block_waves.k x wavefront_size.
consteval GemmSpec
makeSpec(const Signature& sig, const GemmAlgorithm& algo, const TargetSet& targets)
{
    ResolvedSignature resolved = resolve(sig);

    // First op must be GemmOp
    if(!std::holds_alternative<GemmOp>(sig.ops[0]))
        throw "GEMM makeSpec requires GemmOp as first operator";
    const GemmOp& gemm = std::get<GemmOp>(sig.ops[0]);

    ResolvedTensor a_td = resolved.tensor(gemm.lhs);
    ResolvedTensor b_td = resolved.tensor(gemm.rhs);
    DataType acc        = gemm.acc_dtype;

    bool is_i8 = (a_td.dtype == DataType::I8 || b_td.dtype == DataType::I8);

    if(is_i8 && acc != DataType::I32)
        throw "INT8 GEMM requires I32 accumulator -- set GemmOp::acc_dtype = DataType::I32";

    if(is_i8 && targets.contains(GpuTarget::gfx90a))
        throw "INT8 GEMM requires gfx942+ -- gfx90a emulates int8 MFMA with float MFMA, "
              "producing corrupted output. Use TargetSet::family_gfx94() or exclude gfx90a.";

    // INT4 rhs requires .quantize (no unquantized INT4 path exists in CK Tile)
    if(b_td.dtype == DataType::I4 && !b_td.quantize.has_value())
        throw "rhs dtype is I4 but Tensor.quantize is not set -- "
              "INT4 requires quantization metadata (scale tensor and group_size)";

    // Build epilogue op chain from remaining ops after GemmOp.
    // Track the final output name (varies by epilogue chain) and D tensor names.
    int num_epi_ops = 0;
    std::array<EpilogueOp, kMaxEpilogueOps> epi_ops{};
    std::string_view final_output = gemm.out; // "C" by default
    int num_d_tensors             = 0;
    std::string_view d0_name;
    DataType d0_dtype = DataType::FP32;
    Layout d0_layout  = Layout::Row;
    std::string_view d1_name;
    DataType d1_dtype = DataType::FP32;
    Layout d1_layout  = Layout::Row;

    for(int next_op = 1; next_op < kMaxOps; ++next_op)
    {
        if(std::holds_alternative<std::monostate>(sig.ops[next_op]))
            break;

        if(num_epi_ops >= kMaxEpilogueOps)
            throw "too many epilogue operations (max 4)";

        if(auto bin = parseBinaryEpilogueOp(sig.ops[next_op]); bin.has_value())
        {
            auto [op, rhs, out] = *bin;

            if(num_d_tensors >= 2)
                throw "maximum 2 D tensors in epilogue chain";

            epi_ops[num_epi_ops++] = op;

            auto [dt, lo] = extractDTensorMeta(resolved, rhs);
            if(num_d_tensors == 0)
            {
                d0_name   = rhs;
                d0_dtype  = dt;
                d0_layout = lo;
            }
            else
            {
                d1_name   = rhs;
                d1_dtype  = dt;
                d1_layout = lo;
            }
            num_d_tensors++;
            final_output = out;
        }
        else if(auto epi = parseUnaryEpilogueOp(sig.ops[next_op]); epi.has_value())
        {
            epi_ops[num_epi_ops++] = epi->first;
            final_output           = epi->second;
        }
        else
        {
            throw "unrecognized operator in epilogue chain";
        }
    }

    // Direct2D epilogue does not support D tensors
    if(algo.store_strategy == StoreStrategy::Direct2D && num_d_tensors > 0)
        throw "Direct2D epilogue does not support D tensors -- use CShuffle or remove binary ops "
              "(Add/Mul)";

    // Tile validation
    if(algo.block_tile.m <= 0 || algo.block_tile.n <= 0 || algo.block_tile.k <= 0)
        throw "block_tile dimensions must be positive";
    if(algo.block_waves.m <= 0 || algo.block_waves.n <= 0 || algo.block_waves.k <= 0)
        throw "block_waves dimensions must be positive";
    if(algo.wave_tile.m <= 0 || algo.wave_tile.n <= 0 || algo.wave_tile.k <= 0)
        throw "wave_tile dimensions must be positive";

    if(algo.k_batch <= 0)
        throw "k_batch must be positive";

    if(algo.store_strategy == StoreStrategy::CShuffle && algo.block_waves.k != 1)
        throw "CShuffle epilogue requires block_waves.k == 1 (waves_m x waves_n layout)";

    // Pipeline-specific constraints
    if(is_i8 && algo.pipeline == Pipeline::V1)
        throw "INT8 GEMM requires V3/V4/Memory pipeline -- V1 does not support int8";

    if(algo.pipeline == Pipeline::Preshuffle)
    {
        if(a_td.layout != Layout::Row)
            throw "Preshuffle pipeline requires A layout = Row";
        if(b_td.layout != Layout::Col)
            throw "Preshuffle pipeline requires B layout = Col";
    }

    // Scheduling constraints
    if(algo.pipeline_scheduler == PipelineScheduler::Interwave && algo.pipeline != Pipeline::Memory)
        throw "Interwave scheduling requires Pipeline::Memory";

    // Tile partitioner constraints
    if(algo.tile_partitioner == TilePartitioner::StreamK && algo.k_batch > 1)
        throw "Stream-K tile partitioning is incompatible with split-K (k_batch > 1)";

    if(!isValidWaveTile(a_td.dtype, algo.wave_tile.m, algo.wave_tile.n, algo.wave_tile.k, targets))
        throw "wave_tile is not a valid instruction shape for this dtype and target set";

    if(algo.block_tile.m % (algo.block_waves.m * algo.wave_tile.m) != 0)
        throw "block_tile.m must be divisible by (block_waves.m * wave_tile.m)";
    if(algo.block_tile.n % (algo.block_waves.n * algo.wave_tile.n) != 0)
        throw "block_tile.n must be divisible by (block_waves.n * wave_tile.n)";
    if(algo.block_tile.k % (algo.block_waves.k * algo.wave_tile.k) != 0)
        throw "block_tile.k must be divisible by (block_waves.k * wave_tile.k)";

    int wf_size        = targets.wavefront_size();
    int workgroup_size = algo.block_waves.m * algo.block_waves.n * algo.block_waves.k * wf_size;

    // Build physical tensor table
    int num_phys = 3;
    std::array<PhysicalTensor, kMaxPhysicalTensors> phys{};
    phys[0]               = {gemm.lhs, a_td.dtype, a_td.layout, 0}; // A
    phys[1]               = {gemm.rhs, b_td.dtype, b_td.layout, 1}; // B
    ResolvedTensor out_td = resolved.tensor(final_output);
    phys[2]               = {final_output, out_td.dtype, out_td.layout, 2}; // output (C, D, or E)

    if(num_d_tensors >= 1)
    {
        phys[num_phys] = {d0_name, d0_dtype, d0_layout, num_phys};
        num_phys++;
    }
    if(num_d_tensors >= 2)
    {
        phys[num_phys] = {d1_name, d1_dtype, d1_layout, num_phys};
        num_phys++;
    }

    // Quantization: add scale tensor if rhs has .quantize
    int group_size = 0;
    if(b_td.quantize.has_value())
    {
        const auto& q           = *b_td.quantize;
        ResolvedTensor scale_td = resolved.tensor(q.scale_name);
        phys[num_phys]          = {q.scale_name, scale_td.dtype, scale_td.layout, num_phys};
        num_phys++;
        group_size = q.group_size;
    }

    return {num_phys,
            phys,
            acc,
            algo.block_tile,
            algo.block_waves,
            algo.wave_tile,
            workgroup_size,
            algo.k_batch,
            algo.pipeline,
            algo.pipeline_scheduler,
            algo.tile_partitioner,
            num_epi_ops,
            epi_ops,
            algo.store_strategy,
            algo.pad_m,
            algo.pad_n,
            group_size};
}

} // namespace rocm_ck
