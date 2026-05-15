# rocm_ck

A C++20 constexpr API for configuring and distributing
[CK Tile](../include/ck_tile/) GPU kernels across multiple architectures.

> **Status**: Early development. Foundation types are in place (DataType,
> Layout, Args, operators, FixedString, PhysicalTensor, ResolvedTensor).
> The schema engine (Signature, resolve(), Algorithm) and device bridge
> are under active development.

## Why rocm_ck exists

CK Tile kernels are C++ templates. A GEMM kernel's tile size, pipeline
strategy, data types, and epilogue are all template parameters — fixed at
compile time. This is excellent for performance (zero-overhead abstraction,
full inlining), but it creates a problem for multi-architecture distribution:
the host program must be compiled separately from device code, and the host
compiler must never see CK Tile headers.

rocm_ck solves this by introducing a **host-device boundary** built on
constexpr data rather than template parameters:

1. **On the host side**, kernel configurations are plain C++20 structs
   (`Signature`, `Algorithm`, `GemmSpec`). These are constexpr data —
   they describe *what* to compute and *how*, without instantiating any
   templates. Host code reasons about kernels using values, not types.

2. **On the device side**, a thin bridge layer lowers these constexpr
   descriptions into CK Tile template instantiations. Each `GemmSpec`
   maps to exactly one `ck_tile::GemmPipeline<...>` specialization.

3. **At the boundary**, pre-compiled kernels are packaged into
   [kpack archives](https://github.com/ROCm/TheRock/blob/main/docs/rfcs/RFC0008-Multi-Arch-Packaging.md) —
   self-describing, compressed, multi-architecture bundles. The host loads kernels at runtime
   by matching a `GemmSpec` against the kpack table of contents. No
   recompilation, no template instantiation on the host.

This separation is what makes CK Tile viable in
[TheRock](https://github.com/ROCm/TheRock)'s multi-arch build system,
where a single host binary must work with device code compiled for
many GPU targets (e.g. gfx90a, gfx942, gfx1151).

## The constexpr schema model

Traditional GPU kernel libraries select kernels through template
parameters or runtime enums. rocm_ck uses a third approach: **constexpr
structs that are validated at compile time and lowered to templates on
the device side.**

A kernel configuration has two axes:

- **Signature** — *what* the kernel computes: a directed graph of
  operators (`GemmOp`, `AddOp`, `ReluOp`, ...) connecting named tensor
  slots. Data types, layouts, and batch dimensions are part of the
  signature.

- **Algorithm** — *how* the kernel computes it: tile geometry, pipeline
  strategy, warp layout, padding, and scheduling. These are tuning
  parameters that don't change the mathematical result.

The `Signature` and `Algorithm` are plain aggregate structs with
designated initializers — no constructors, no inheritance, no runtime
polymorphism. Validation happens in `consteval` functions: invalid
configurations (unsupported tile size, incompatible data types, missing
tensor slots) fail at compile time with actionable error messages.

Here is a preview of the API direction (not yet implemented):

```cpp
// Host side — pure constexpr, any C++20 compiler, no CK headers
constexpr Signature sig = {
    .dtype = DataType::FP16,
    .ops = {
        GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
        AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
        ReluOp{.in = "D", .out = "E"},
    },
};

// Device side — make_kernel lowers to a CK Tile template instantiation.
// Compiled separately per architecture, packaged into .kpack archives.
```

## Directory layout

```text
rocm_ck/
├── CMakeLists.txt        # INTERFACE library, C++20, ck_tile_headers target
├── include/rocm_ck/      # Public headers — host-safe, no CK/HIP deps
├── src/                  # (planned) Device bridge, kpack loading
└── tests/
    ├── CMakeLists.txt    # Test tiers: ROCM_CK_SMOKE, ROCM_CK_KERNEL
    ├── unit/             # Fast host-only tests (< 1s, no GPU)
    └── kernel/           # (planned) GPU kernel tests
```

## Build

rocm_ck is a CK feature, gated by `CK_ENABLE_ROCM_CK`:

```bash
cd composablekernel
cmake -B build -S . -G Ninja \
    -DCK_ENABLE_ROCM_CK=ON \
    -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++

ninja -C build smoke-rocm-ck    # host-only smoke tests
ninja -C build check-rocm-ck    # all rocm_ck tests
ctest --test-dir build -L ROCM_CK_SMOKE --output-on-failure
```

Default CK builds (`CK_ENABLE_ROCM_CK=OFF`) are unaffected.
