# HSTU Attention Forward Operator

## Overview

HSTU (Hierarchical Sequential Transduction Unit) attention is a custom attention variant designed
for recommendation-system workloads. Unlike standard softmax attention, HSTU uses **SiLU
(Sigmoid Linear Unit)** as the non-linearity instead of softmax, together with a composite
functional masking scheme.

### Forward computation

Given inputs:
- `q : [batch, seqlen_q,  nhead, hdim_qk]`
- `k : [batch, seqlen_kv, nhead, hdim_qk]`
- `v : [batch, seqlen_kv, nhead, hdim_v]`

the forward pass performs:

1. **QK projection** — `s[b, h, i, j] = scale_s * q[b, i, h, :] @ k[b, j, h, :]`
2. **Functional masking** — set masked positions of `s` to 0 (see [Masking](#masking) below)
3. **Non-linearity** — `p = SiLU(s)` (element-wise, over the `seqlen_kv` dimension);
   alternatively `p = attn_scale * softmax(s)` when `--softmax=1`
4. **Output projection** — `o[b, i, h, :] = p[b, h, i, :] @ v[b, :, h, :]`

Output: `o : [batch, seqlen_q, nhead, hdim_v]`

---

## Masking

The functional mask is a composite of up to four components, all expressed in terms of the
**UIH (User Interaction History)** positions, i.e. positions that are not targets and not
contextual:

| Component | Parameter | Description |
|-----------|-----------|-------------|
| **Causal** | `--causal=1` | Standard lower-triangular causal mask |
| **Local (diagonal window)** | `--local_len=N` | Allow each query token to attend only to the `N` most recent keys (sliding-window causal) |
| **Contextual** | `--context_len=N` | The first `N` positions of the sequence are always visible to all queries |
| **Min-full attention** | `--minfull_len=N` | The last `N` UIH positions attend to the full UIH history (no local restriction) |
| **Targets** | `--targets=T[,T,...]` | Per-batch token counts at the *end* of the Q sequence that are excluded from attending |

Physical sequence length is computed as:
```
phy_seqlen_q  = uih_seqlen + num_targets + context_len
phy_seqlen_kv = uih_seqlen + num_targets + context_len   (self-attention)
phy_seqlen_kv = uih_seqlen_kv + context_len              (cross-attention)
```

---

## Modes of operation

### Batch modes

| Mode | Flag | Description |
|------|------|-------------|
| **Batched** | `-jagged=0` | All sequences in the batch share the same `seqlen`; tensors are `[B, S, H, D]` |
| **Jagged** | `-jagged=1` | Each sequence has its own length given by `seq_offsets`; tensors are flattened `[1, total_tokens, H, D]` |

### Attention variants

| Variant | Flag | Description |
|---------|------|-------------|
| **No-group HSTU** | `-g=1` (default) | All batches share the same masking parameters (`local_len`, `context_len`, `minfull_len`, `attn_scale`) |
| **Group HSTU** | `-g=G` (G > 1) | `num_batch` must be a multiple of `G`; each group has its own per-group masking parameters passed via `--g_*` flags |
| **Self-attention** | (default) | `seqlen_kv` == `seqlen_q` |
| **Cross-attention** | `--seqlens_kv=...` | Enabled implicitly when KV sequence lengths differ from Q |

### Non-linearity

| Mode | Flag | Description |
|------|------|-------------|
| **SiLU** | `-softmax=0` (default) | Element-wise SiLU; `attn_scale` applied afterwards |
| **Softmax** | `-softmax=1` | Standard scaled softmax; LSE saved when `-training=1` |

### Split-KV

A split-KV kernel path (`hstu_attention_fwd_splitkv_kernel.hpp` /
`hstu_attention_batched_forward_splitkv_dispatch.hpp` etc.) is also available, which splits the
KV dimension across multiple work-groups and uses a separate combine pass
(`hstu_attention_fwd_splitkv_combine_kernel.hpp`) to merge partial results.

---

## Data types

- **fp16** (`half_t`) and **bf16** (`bfloat16_t`) for Q/K/V/O
- Accumulation and computation use `float32`
- Supported head dimensions (`hdim_qk` / `hdim_v`): **64, 96, 128, 256**

---

## File structure

```
18_hstu_attention/
├── CMakeLists.txt                              # Build configuration
├── example_hstu_attention_fwd.cpp              # Driver / benchmark / validation harness
├── generate_instances.py                       # Python script to regenerate pre-compiled instances
│
│-- Core kernel headers --
├── hstu_attention_fwd_kernel.hpp               # Main forward kernel
├── hstu_attention_fwd_splitkv_kernel.hpp       # Split-KV forward kernel
├── hstu_attention_fwd_splitkv_combine_kernel.hpp  # Split-KV combine kernel
├── hstu_attention_no_softmax_fwd_pipeline.hpp  # SiLU pipeline (no softmax, batched/jagged)
├── hstu_attention_with_softmax_fwd_pipeline.hpp   # Softmax pipeline
├── hstu_attention_no_softmax_fwd_trload_pipeline.hpp   # SiLU pipeline (transposed load)
├── hstu_attention_with_softmax_fwd_trload_pipeline.hpp # Softmax pipeline (transposed load)
├── hstu_attention_no_softmax_fwd_splitkv_combine_pipeline.hpp
├── hstu_attention_with_softmax_fwd_splitkv_combine_pipeline.hpp
│
│-- Policy / settings --
├── hstu_attention_fwd_pipeline_policy.hpp      # Pipeline policy selection
├── hstu_attention_fwd_splitkv_combine_pipeline_policy.hpp
├── hstu_attention_fwd_setting.hpp              # Tile/warp sizes and dispatch settings
├── hstu_attention_fwd_splitkv_combine_setting.hpp
├── hstu_attention_fwd_type_config.hpp          # Type aliases (CompDataType, GemmAccDataType)
├── hstu_attention_pipeline_problem.hpp         # Problem descriptor passed to pipelines
├── hstu_attention_traits.hpp                   # Trait structs for padding/occupancy
│
│-- Masking --
├── hstu_block_masking.hpp                      # Block-level masking logic
│                                               # (HstuSelfAttentionBlockMaskWithLocal,
│                                               #  HstuCrossAttentionBlockMaskWithLocal)
├── hstu_attention_epilogue.hpp                 # Epilogue (output write-back + scaling)
│
│-- Dispatch layer --
├── hstu_attention_params.hpp                   # HstuAttentionNoGroupFwdParams /
│                                               # HstuAttentionGroupFwdParams structs
├── hstu_attention_api.hpp                      # Public C API declarations
├── hstu_attention_batched_forward_dispatch.hpp
├── hstu_attention_batched_forward_splitkv_dispatch.hpp
├── hstu_attention_jagged_forward_dispatch.hpp
├── hstu_attention_jagged_forward_splitkv_dispatch.hpp
├── hstu_attention_group_forward_dispatch.hpp
├── hstu_attention_group_forward_splitkv_dispatch.hpp
├── hstu_attention_no_group_forward_fp16.cpp    # fp16 entry points (no-group)
├── hstu_attention_no_group_forward_bf16.cpp    # bf16 entry points (no-group)
├── hstu_attention_group_forward_fp16.cpp       # fp16 entry points (group)
├── hstu_attention_group_forward_bf16.cpp       # bf16 entry points (group)
│
│-- Switch helpers --
├── hstu_attention_bool_switch.hpp              # BOOL_SWITCH macros
├── hstu_attention_hdim_switch.hpp              # Head-dim dispatch switch
├── hstu_attention_max_splits_switch.hpp        # Max-splits dispatch switch
├── hstu_attention_splitkv_helper.hpp           # SplitKV helper utilities
├── hstu_attention_tile_setting_define.hpp      # Tile-size macro definitions
│
│-- Utility / host --
├── hstu_attention_host_util.hpp                # Host-side helpers (offset computation, etc.)
├── hstu_attention_kernel_util.hpp              # Kernel-side helpers
├── reference_hstu_attention_fwd.hpp            # CPU reference implementation for validation
│
│-- Custom GEMM hacks --
├── block_gemm_areg_bsmem_creg_v2_hack_0.hpp
├── block_gemm_areg_bsmem_creg_v2_hack_1.hpp
├── block_gemm_areg_bsmem_trload_creg_v2_hack_1.hpp
│
├── instances/                                  # Pre-compiled kernel instances (auto-generated)
│   ├── hstu_attention_{batched,group,jagged}_forward_{fp16,bf16}_*.cpp
│   └── hstu_attention_{batched,group,jagged}_forward_{fp16,bf16}_instances_ref.hpp
│
├── scripts/                                    # Test and benchmark shell scripts
│   ├── test_hstu_attention.sh
│   ├── test_hstu_softmax_attention.sh
│   ├── test_group_hstu_attention.sh
│   ├── test_group_hstu_softmax_attention.sh
│   ├── test_hstu_cross_attention.sh
│   ├── test_hstu_attention_hdim96_hdim64.sh
│   ├── test_hstu_softmax_attention_hdim96_hdim64.sh
│   ├── test_ck_hstu_mask.sh
│   ├── test_cross_attention_with_sparsity.sh
│   ├── test_jagged_causal_mattn0_full0.sh
│   ├── test_jagged_causal_mattn256_full0.sh
│   ├── test_jagged_causal_mattn256_full256.sh
│   ├── bench_batched_causal.sh
│   ├── bench_jagged_causal.sh
│   ├── bench_jagged_causal_local.sh
│   ├── bench_jagged_causal_mattn0_full0.sh
│   ├── bench_jagged_causal_mattn256_full256.sh
│   ├── bench_jagged_causal_mattn256_full256_sparsity_90.sh
│   ├── bench_cross_attention_with_sparsity.sh
│   └── benchmark_hstu_attention.sh
│
├── test_pytorch_hstu_mask.py                   # PyTorch mask validation script
└── test_pytorch_hstu_mask_v2.py
```

---

## Build

```bash
mkdir build
cd build
../script/cmake-ck-dev.sh .. gfx942 -G Ninja   # use: rocminfo | grep "gfx" to check GPU arch
ninja tile_example_hstu_attention               # or: make -j tile_example_hstu_attention
```

The build target is `tile_example_hstu_attention` (excluded from `make all` by default).

**Optional compile-time flags:**

| Environment variable | Effect |
|---------------------|--------|
| `ASSUME_HIGHLY_VARIED_SEQLEN=1` | Schedules batch dimension as a non-leading grid dimension (trades occupancy for better load balance when sequence lengths vary widely) |

On `gfx950`-only builds (`-DBUILD_HSTU_FOR_GFX95_ONLY`), SLP vectorization is disabled to
improve pipeline performance.

---

## Test / Verify

### No-group HSTU (single set of masking parameters)

```bash
# Jagged batches, fp16/bf16, causal + local + context + targets
build/bin/tile_example_hstu_attention \
    -v=1 -prec=bf16 -b=10 -jagged=1 \
    -nhead=4 -hdim_qk=128 -hdim_v=128 \
    -seqlens=750,730,733,860,870,788,760,821,833,779 \
    -targets=5,5,6,6,5,6,5,6,4,6 \
    -causal=1 -local_len=5 -context_len=6 -minfull_len=6

# Run the full standard test suite
. example/ck_tile/18_hstu_attention/scripts/test_hstu_attention.sh

# Softmax variant
. example/ck_tile/18_hstu_attention/scripts/test_hstu_softmax_attention.sh

# Cross-attention
. example/ck_tile/18_hstu_attention/scripts/test_hstu_cross_attention.sh

# Asymmetric head dims (hdim_qk=96, hdim_v=64)
. example/ck_tile/18_hstu_attention/scripts/test_hstu_attention_hdim96_hdim64.sh
```

### Group HSTU (per-group masking parameters)

```bash
# 3 groups, 18 batches (6 per group), each group has distinct local/context/minfull/scale params
build/bin/tile_example_hstu_attention \
    -v=1 -prec=bf16 -b=18 -g=3 \
    -nhead=4 -hdim_qk=128 -hdim_v=128 \
    -seqlens=300,300,290,280,310,308,312 \
    -causal=1 -targets=8 \
    -g_max_seqlens=310,312,312 \
    -g_local_lens=5,5,5 \
    -g_context_lens=8,8,8 \
    -g_minfull_lens=7,7,7 \
    -g_attn_scales=0.0,0.1,0.0

# Run the full group test suite
. example/ck_tile/18_hstu_attention/scripts/test_group_hstu_attention.sh
```

---

## Command-line arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-v` | `1` | Run CPU validation (0 = disabled) |
| `-prec` | `fp16` | Data type: `fp16` or `bf16` |
| `-g` | `1` | Number of groups (>1 enables group-HSTU mode) |
| `-b` | `12` | Number of batches |
| `-jagged` | `0` | Jagged sequence mode (1 = enabled) |
| `-nhead` | `4` | Number of attention heads |
| `-hdim_qk` | `64` | Head dimension for Q and K |
| `-hdim_v` | `64` | Head dimension for V and O |
| `-seqlens` | `400` | UIH sequence length(s); comma-separated for jagged mode |
| `-seqlens_kv` | (same as Q) | KV UIH lengths; enables cross-attention when set |
| `-max_seqlen` | `0` | Override max UIH seqlen for Q (0 = auto) |
| `-targets` | (empty) | Per-batch target token counts appended to Q (and K/V for self-attn) |
| `-softmax` | `0` | Use softmax instead of SiLU (1 = enabled) |
| `-training` | `0` | Training mode; saves LSE when softmax is also enabled |
| `-causal` | `1` | Enable lower-triangular causal mask |
| `-local_len` | `5` | Diagonal window size (0 = no local mask) |
| `-context_len` | `6` | Contextual prefix length always visible to all queries |
| `-minfull_len` | `6` | Tail UIH length that receives full (non-local) attention |
| `-alpha` | `0` | QK scale factor (`0` = `1/sqrt(hdim_qk)`) |
| `-attn_scale` | `0` | Post-SiLU scale (`0` = `1/max_seqlen`) |
| `-seed` | `13579` | RNG seed for random initialization |
| `-norm_dist` | `0` | Use normal distribution for QKV init (0 = uniform) |
| `-init_qkv` | `0` | Load Q/K/V from binary files `q.dat`, `k.dat`, `v.dat` |
| `-perf` | `0` | Measure and report average execution time and TFLOPS |
| `-dump_output` | `0` | Dump device and reference outputs to binary files |
| `-save_mask` | `0` | Save the attention mask tensor to `ck_hstu_mask.dat` |

### Group-HSTU-specific arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-g_max_seqlens` | `0` | Per-group max UIH seqlens (comma-separated) |
| `-g_local_lens` | `5,` | Per-group local window sizes |
| `-g_context_lens` | `6,` | Per-group contextual prefix lengths |
| `-g_minfull_lens` | `6` | Per-group min-full-attention tail lengths |
| `-g_attn_scales` | `1.0,` | Per-group post-SiLU scale factors |

---

## Benchmark

```bash
# Batched causal
. example/ck_tile/18_hstu_attention/scripts/bench_batched_causal.sh

# Jagged causal
. example/ck_tile/18_hstu_attention/scripts/bench_jagged_causal.sh

# Jagged causal + local window
. example/ck_tile/18_hstu_attention/scripts/bench_jagged_causal_local.sh

# With -perf=1 flag directly:
build/bin/tile_example_hstu_attention -v=0 -perf=1 -prec=bf16 \
    -b=32 -jagged=1 -nhead=8 -hdim_qk=128 -hdim_v=128 \
    -seqlens=512 -causal=1 -local_len=5 -context_len=8 -minfull_len=8
```

Performance output reports average kernel execution time (ms) and estimated TFLOPS, counting
only the two GEMMs (QK and PV), ignoring masking, scaling, and SiLU overhead.

---

## Regenerating kernel instances

The `instances/` directory is auto-generated by `generate_instances.py`. To regenerate after
changing template parameters (dtypes, head dims, causal/softmax/bias/dropout combinations):

```bash
cd example/ck_tile/18_hstu_attention
python3 generate_instances.py
```
