# HSTU Attention Backward Pass Design

## 1. Mathematical Specification

See `reference_hstu_attention_bwd.hpp` for the full annotated reference implementation.
The summary below covers everything that directly drives kernel design decisions.

### Forward recap

```
S[sq, sk]  = alpha * (Q[sq] dot K[sk])             masked-in pairs; 0 (SiLU) or -inf (Softmax) elsewhere
P[sq, sk]  = silu(S[sq, sk]) * scale_p           (kUseSoftmax=false, SiLU path)
           = exp(S[sq, sk] - LSE[sq])             (kUseSoftmax=true, Softmax path)
O[sq, k]   = sum_sk  P[sq, sk] * V[sk, k]
LSE[sq]    = log(sum_sk exp(S[sq, sk]))           saved during forward only when kUseSoftmax=true
```

`alpha` maps to `HstuAttentionNoGroupFwdParams::scale_s` in the param struct.
`scale_p` maps to `attn_scale` (or `1/max_seqlen_q` when `attn_scale == 0`).

Note: masked-out positions use S=0 in the SiLU path (so silu(0)=0 and dS=0 naturally) and
S=-inf in the Softmax path (so exp(-inf - LSE)=0 naturally).

### Backward formulas -- common to both paths

```
dV[sk, k]  = sum_sq  P[sq, sk]  * dO[sq, k]       -- GEMM: P^T @ dO^T   (A=P^T[sk,sq], B=dO^T[hdim_v,sq])
dP[sq, sk] = sum_k   dO[sq, k]  * V[sk, k]         -- GEMM: dO @ V       (A=dO[sq,hdim_v], B=V[sk,hdim_v])
dQ[sq, k]  = alpha * sum_sk  dS[sq, sk] * K[sk, k]  -- GEMM: alpha * dS @ K^T   (A=dS[sq,sk], B=K^T[hdim_qk,sk])
dK[sk, k]  = alpha * sum_sq  dS[sq, sk] * Q[sq, k]  -- GEMM: alpha * dS^T @ Q^T (A=dS^T[sk,sq], B=Q^T[hdim_qk,sq])
```

### Path-specific: computing dS from dP

**SiLU path** (`kUseSoftmax=false`): S must be recomputed from Q and K (no saved value).
```
dsilu(x)  = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
dS[sq,sk] = dP[sq,sk] * scale_p * dsilu(S[sq,sk])    masked-in
           = 0                                          masked-out
```

**Softmax path** (`kUseSoftmax=true`): LSE from forward allows recovering P without a second pass.
```
D[sq]     = dO[sq] row(.) O[sq]    (equivalent to sum_sk dP[sq,sk]*P[sq,sk], proved by swapping summation:
                                    sum_sk dP*P = sum_sk (dO row(.) V[sk])*P[sk] = sum_k dO[k] * sum_sk P[sk]*V[sk,k]
                                               = sum_k dO[k] * O[k] = dO row(.) O)
dS[sq,sk] = P[sq,sk] * (dP[sq,sk] - D[sq])
```
Masked-out positions have P=0 naturally (exp(-inf - LSE) = 0), so no explicit masking is needed.

### Computation order and accumulation asymmetry

The key structural constraint driving the two-kernel split is:

- **dQ** is computed fresh for each `sq` row -- no inter-row accumulation needed.  
  Each thread block owns one or more `sq` rows and writes `dQ` directly.
- **dV** and **dK** both accumulate over **all** `sq` rows for a fixed `sk` column.  
  They must be kept in a high-precision accumulator across the entire `sq` loop,
  then written back once at the end (see `dk_acc`, `dv_acc` in reference).

This matches the standard Flash-Attention backward split: kernel 1 produces `dQ` (and
`D[sq]` for the softmax path), iterating over the K/V dimension; kernel 2 produces `dK`
and `dV`, iterating over the Q dimension.

---

## 2. New Data Structures

### 2.1 `hstu_attention_params.hpp` -- new backward param structs

#### `struct HstuAttentionNoGroupBwdParams`

Mirror of `HstuAttentionNoGroupFwdParams` with these changes:

| Field | Change vs. forward |
|---|---|
| `const void* do_ptr` | new -- upstream gradient |
| `const void* o_ptr` | new -- forward output (needed for D[sq] in softmax path) |
| `const void* lse_ptr` | repurposed as **input** (was output in forward) |
| `void* dq_ptr` | new output |
| `void* dk_ptr` | new output |
| `void* dv_ptr` | new output |
| `void* delta_ptr` | new output -- stores D[sq] = dO row(.) O (softmax path only; produced by kernel 1, consumed by kernel 2) |
| `void* lse_ptr` | **removed as output** -- backward only reads it |
| `is_training`, `p_drop`, `philox_*` | **removed** -- backward itself is not sampled |

New strides for dO (matching O layout):
- `seq_stride_do`, `nhead_stride_do`, `batch_stride_do`

New strides for dQ, dK, dV -- each set must be provided explicitly by the caller.
The gradient tensors are allocated by the upper-layer context (e.g. Python framework) and
their memory layout cannot be assumed to match Q, K, or V respectively:
- `seq_stride_dq`, `nhead_stride_dq`, `batch_stride_dq`
- `seq_stride_dk`, `nhead_stride_dk`, `batch_stride_dk`
- `seq_stride_dv`, `nhead_stride_dv`, `batch_stride_dv`

`batch_stride_dq`, `batch_stride_dk`, and `batch_stride_dv` are only meaningful in batched
mode (`kIsJagged == false`); callers using jagged/variable-length mode must set them to zero.

#### `struct HstuAttentionGroupBwdParams`

Same pattern as `HstuAttentionGroupFwdParams` -> `HstuAttentionNoGroupBwdParams` above,
plus the per-group arrays (`group_attn_scale_ptr`, `group_window_size_ptr`, etc.) carried over
unchanged from the forward group params.

### 2.2 `hstu_attention_pipeline_problem.hpp` -- new backward problem struct

#### `struct HstuAttentionBwdPipelineBaseProblem`

Analogous to `HstuAttentionFwdPipelineProblem`. Holds all template parameters that are
shared across both backward kernels, excluding tile-sizing policy. Template parameters:

```cpp
template <typename InOutDataType_,   // fp16 or bf16 -- Q, K, V, O, dO, dQ, dK, dV
          typename GemmAccDataType_, // float -- GEMM accumulator
          typename CompDataType_,    // float -- SiLU / softmax intermediate
          bool kIsCrossAttention_,
          bool kUseGroup_,
          bool kIsJagged_,
          bool kHasBias_,            // bias added to S (same semantics as forward)
          bool kHasCausal_,
          bool kUseSoftmax_>
struct HstuAttentionBwdPipelineBaseProblem { ... };
```

Differences from forward problem:
- No `kStoreLSE_` (bwd always reads LSE, never writes it).
- No `kHasDropout_` -- dropout mask replay can be added later if needed.
- `kHasBias_` is kept because `dBias` may be computed in a future extension, and the mask
  logic inside the pipeline still needs to know whether S had a bias added.
- `BwdTileSetting_` is split out into the per-kernel problem structs below.

#### `struct HstuAttentionBwdPipelineProblemForKernel1`

Composes `HstuAttentionBwdPipelineBaseProblem` with the tile-sizing policy for Kernel 1
(dQ / delta computation). Template parameters:

```cpp
template <typename PipelineBaseProblem,  // HstuAttentionBwdPipelineBaseProblem instance
          typename TileSetting>          // analogous to AttentionTileSetting_ in fwd
struct HstuAttentionBwdPipelineProblemForKernel1 { ... };
```

#### `struct HstuAttentionBwdPipelineProblemForKernel2`

Composes `HstuAttentionBwdPipelineBaseProblem` with the tile-sizing policy for Kernel 2
(dK / dV computation). Template parameters:

```cpp
template <typename PipelineBaseProblem,  // HstuAttentionBwdPipelineBaseProblem instance
          typename TileSetting>          // analogous to AttentionTileSetting_ in fwd
struct HstuAttentionBwdPipelineProblemForKernel2 { ... };
```

### 2.3 `hstu_attention_traits.hpp` -- new backward traits struct

#### `struct HstuAttentionBwdTraits`

Analogous to `HstuAttentionFwdTraits`:

```cpp
template <bool kPadSeqLenQ_,
          bool kPadSeqLenK_,
          bool kPadHeadDimQK_,
          bool kPadHeadDimV_,
          index_t kBlockPerCu_>
struct HstuAttentionBwdTraits { ... };
```

---

## 3. File Layout and Naming Conventions

The backward pass is split into two GPU kernels, each with its own pipeline hierarchy,
mirroring the forward structure. Files reused unchanged from forward are noted.

### 3.1 Tile settings

**`hstu_attention_bwd_setting.hpp`** -- a dedicated tile-setting file separate from the forward
`hstu_attention_fwd_setting.hpp`, with `HstuAttentionBwdBlockTile<MaxK>` specializations.

Two design decisions motivate this separation:

1. **MTile is fixed at 128 for all backward configurations.**  
   In HSTU training the input batch is intentionally prepared to be large enough to fully
   saturate the GPU. Under this assumption a larger Q-tile (MTile=128) is always preferred:
   for every workgroup that owns a distinct Q tile, the same K and V data must be re-loaded
   from global memory. A larger MTile reduces this re-loading ratio -- more Q rows are
   processed per K/V load, amortising the bandwidth cost. Therefore the MTile-64 vs.
   MTile-128 dispatch that exists in the forward settings is not needed here; only
   MTile=128 specializations are defined.

2. **Independent tuning of backward tile sizes.**  
   The backward GEMMs have transposed shapes relative to the forward (e.g. dQ: [M, N]x[N, K]
   vs. forward O: [M, K]x[K, N]). Keeping a separate file lets backward tile sizes be
   tuned independently without risking regressions in forward performance.

### 3.2 Pipeline policy

**`hstu_attention_bwd_pipeline_policy.hpp`**  
Analogous to `hstu_attention_fwd_pipeline_policy.hpp`. Defines:
- Smem layout for Q, K, V, dO tiles.
- The following `Get<XXX>BlockGemm` functions, named following the convention in
  `block_fmha_bwd_pipeline_default_policy.hpp`:

  | Function | Operation | Kernel |
  |---|---|---|
  | `GetQKBlockGemm` | S = alpha \* Q @ K (A=Q[sq,hdim\_qk], B=K[sk,hdim\_qk]) | Kernel 1 & 2 |
  | `GetOGradVBlockGemm` | dP = dO @ V (A=dO[sq,hdim\_v], B=V[sk,hdim\_v]) | Kernel 1 & 2 |
  | `GetPTOGradTBlockGemm` | dV += P^T @ dO^T (A=P^T[sk,sq], B=dO^T[hdim\_v,sq]) | Kernel 2 |
  | `GetSGradKTBlockGemm` | dQ += alpha \* dS @ K^T (A=dS[sq,sk], B=K^T[hdim\_qk,sk]) | Kernel 1 |
  | `GetSGradTQTBlockGemm` | dK += alpha \* dS^T @ Q^T (A=dS^T[sk,sq], B=Q^T[hdim\_qk,sq]) | Kernel 2 |

- Alignment helpers for dQ, dK, dV DRAM windows.

### 3.3 Pipeline implementations

Four pipelines, one per (softmax variant x kernel role):

| File | Struct name | Kernel role | Softmax variant |
|---|---|---|---|
| `hstu_attention_no_softmax_bwd_pipeline_dq.hpp` | `HstuAttentionNoSoftmaxBwdPipelineQRKSVS_dQ` | Kernel 1 -- computes dQ | SiLU path |
| `hstu_attention_with_softmax_bwd_pipeline_dq_delta.hpp` | `HstuAttentionWithSoftmaxBwdPipelineQRKSVS_dQ_D` | Kernel 1 -- computes dQ and D[sq] | Softmax path |
| `hstu_attention_bwd_pipeline_no_softmax_dk_dv.hpp` | `HstuAttentionNoSoftmaxBwdPipelineKRVRQS_dK_dV` | Kernel 2 -- computes dK, dV | SiLU path |
| `hstu_attention_bwd_pipeline_with_softmax_dk_dv.hpp` | `HstuAttentionWithSoftmaxBwdPipelineKRVRQS_dK_dV` | Kernel 2 -- computes dK, dV | Softmax path |

**Struct name encoding -- memory placement of each tensor:**

The suffix after `BwdPipeline` encodes, for each tensor, whether it is register-resident (`R`) or LDS-staged (`S`) before the BlockGemm that consumes it:

- **Kernel 1 names (`QRKSVS`):**  
  - `QR` -- Q and dO are each loaded once from device memory into **registers** before the main K/V loop. Because the Q tile is fixed for the entire block and dO is needed in every K/V iteration (for `dP = dO @ V`), keeping both in registers eliminates redundant global-memory traffic.  
  - `KS` -- K is loaded from device memory into **LDS** before the BlockGemm that computes `S = alpha * Q @ K`. K is consumed continuously along `seqlen_kv` in the major loop, so staging through LDS allows the BlockGemm to reuse data from the shared buffer. Each K sub-tile is also written into a separate KT LDS region in transposed layout for the subsequent `dQ += dS @ K^T` GEMM (Gemm4), so K is stored to LDS only once per iteration but read by two GEMMs.
  - `VS` -- V is likewise loaded from device memory into **LDS** before the BlockGemm that computes `dP = dO @ V`. V is streamed tile-by-tile alongside K in the same main loop.

- **Kernel 2 names (`KRVRQS`):**  
  - `KR` -- K is loaded directly from device memory into **registers**. Because K is fixed for the entire main loop (one K tile per block), it can remain register-resident.  
  - `VR` -- V is loaded directly from device memory into **registers** for the same reason (one V tile per block).  
  - `QS` -- Q is loaded from device memory into **LDS** before the BlockGemm that recomputes `S = alpha * Q @ K`. Q is consumed continuously along `seqlen_q` in the major loop, so staging through LDS is necessary for the BlockGemm.

- **Output suffix** (`_dQ`, `_dQ_D`, `_dK_dV`) -- the gradient tensors written by the pipeline. `_D` in the Softmax Kernel 1 name indicates that `D[sq] = dO row(.) O` is also computed and stored to `delta_ptr` for consumption by Kernel 2.

Each pipeline struct follows the same `operator()(dram_windows..., mask, scales, smem_ptr)`
pattern as the forward pipelines.

**Kernel 1 main loop** (one block per sq tile; iterates over the K/V dimension):
1. Load Q tile and dO tile once into registers outside the main loop; they stay register-resident for the lifetime of the block.
2. Softmax path only: load O tile once outside the main loop, compute `D[sq] = dO row(.) O`, and store it to `delta_ptr` -- all before the main K/V loop begins.
3. For each K/V block: load K tile -> LDS, load V tile -> LDS; GEMM: `S = alpha * Q @ K`; GEMM: `dP = dO @ V`. Apply mask (S=0 or S=-inf for masked-out). Softmax path: `P = exp(S - LSE)`.
4. Compute `dS`: SiLU path: `dP * scale_p * dsilu(S)`; Softmax path: `P * (dP - D[sq])`.
5. GEMM: `dQ += alpha * dS @ K^T`  *(A=dS[sq,sk], B=K^T[hdim\_qk,sk]; K is reused from step 3 via a transposed LDS region)*.
6. After all K/V blocks: write back `dQ`.

**Kernel 2 main loop** (one block per sk tile; iterates over the Q dimension):
1. Load K tile and V tile once into registers outside the main loop; they stay register-resident for the lifetime of the block.
2. For each Q block: load Q tile and dO tile from DRAM; recompute `S = alpha * Q @ K`; apply mask. Softmax path: `P = exp(S - LSE)`. SiLU path: also compute and stash `dsilu(S)` in the same elementwise pass so that `S` need not be kept alive after `P` is formed.
3. Softmax path only: load `D[sq]` from `delta_ptr` (written by kernel 1).
4. GEMM: `dV += P^T @ dO^T`  *(A=P^T[sk,sq], B=dO^T[hdim\_v,sq]; accumulates over all Q blocks)*.  `dO^T` LDS space is freed once this GEMM completes and is reused for `Q^T` in step 6.
5. Compute `dS`: SiLU path: `dP * scale_p * dsilu(S)` (using the stashed `dsilu(S)` from step 2); Softmax path: `P * (dP - D[sq])`.
6. GEMM: `dK += alpha * dS^T @ Q^T`  *(A=dS^T[sk,sq], B=Q^T[hdim\_qk,sq]; accumulates over all Q blocks)*.  `Q^T` is written into the LDS region freed by step 4.
7. After all Q blocks: write back `dK` and `dV`.

### 3.4 Kernels

**`hstu_attention_bwd_kernel_1.hpp`**  
`HstuAttentionBwdKernel1<Pipeline>` -- launches kernel 1 (dQ, optional D[sq]).  
Grid: one block per `(batch, head, sq_tile)` -- same as forward.

**`hstu_attention_bwd_kernel_2.hpp`**  
`HstuAttentionBwdKernel2<Pipeline>` -- launches kernel 2 (dK, dV).  
Grid: one block per `(batch, head, sk_tile)` -- columns instead of rows.

### 3.5 Dispatch headers

Following the forward pattern (`hstu_attention_batched_forward_dispatch.hpp`, etc.):

```
hstu_attention_batched_backward_dispatch.hpp
hstu_attention_jagged_backward_dispatch.hpp
hstu_attention_group_backward_dispatch.hpp
```

Each dispatch header chains the `kUseSoftmax`, `kHasCausal`, `kHasBias` switches,
selects the correct pipeline pair, and calls both kernel 1 and kernel 2 sequentially on
the same stream.

### 3.6 Instance `.cpp` files and `generate_instances.py`

The instance naming scheme follows the forward pattern:
```
hstu_attention_batched_backward_{dtype}_{causal}_{softmax}_{bias}_maxk_{K}.cpp
```
`generate_instances.py` will be extended to emit these.

### 3.7 Host API

**`hstu_attention_api.hpp`** -- add:
```cpp
extern void hstu_attention_no_group_backward_fp16(HstuAttentionNoGroupBwdParams&, hipStream_t);
extern void hstu_attention_no_group_backward_bf16(HstuAttentionNoGroupBwdParams&, hipStream_t);
extern void hstu_attention_group_backward_fp16(HstuAttentionGroupBwdParams&, hipStream_t);
extern void hstu_attention_group_backward_bf16(HstuAttentionGroupBwdParams&, hipStream_t);
```

### 3.8 Files reused unchanged from forward

| File | Role |
|---|---|
| `hstu_block_masking.hpp` | HSTU mask types and factories |
| `hstu_attention_bool_switch.hpp` | `BOOL_SWITCH`, `BOOL_SWITCH_2` macros |
| `hstu_attention_hdim_switch.hpp` | `hdim` dispatch switch |
| `hstu_attention_pipeline_problem.hpp` | forward problem only; backward adds its own struct |
| `hstu_attention_host_util.hpp` | host-side utilities |
| `hstu_attention_kernel_util.hpp` | grid/block launch helpers |

---

## 4. Open Questions

1. **Dropout mask replay.** The forward saves a Philox seed/offset for reproducibility.
   If dropout gradient is needed, kernel 1 must replay the same mask. Deferred for now.

2. **dBias output (deferred -- layout unconfirmed).** The correct treatment of dBias depends on
   the memory layout of the forward Bias tensor, which has not yet been confirmed by the HSTU
   project stake-owner. Known layouts used in frameworks such as xformers include:

   | Layout | Shape | dBias accumulation |
   |---|---|---|
   | Per-batch, per-head | `[batch, num_head, seqlen_q, seqlen_k]` | each (batch, head) pair writes its own dBias slice independently -- no atomics needed |
   | Per-batch, shared across heads | `[batch, seqlen_q, seqlen_k]` | partial dBias must be summed over `num_head` -- atomic-add across heads |
   | Shared across all | `[seqlen_q, seqlen_k]` | partial dBias must be summed over `batch x num_head` -- atomic-add across all work-groups covering that (sq, sk) tile |

   In the `[seqlen_q, seqlen_k]` case, each work-group computes `dS[sq, sk]` for its tile and
   issues an `atomic_add` into the shared dBias buffer. This accumulation can be placed in
   either kernel 1 or kernel 2 (both visit every (sq, sk) tile exactly once) without altering
   the general backward design.

   **Action:** defer implementation until the stake-owner explicitly confirms the Bias layout.

3. **Tile sizes for backward.** The dK/dV GEMM in kernel 2 has a different shape from the
   forward O-GEMM. Benchmark to find optimal `HstuAttentionBwdBlockTile` specializations.

4. **Load imbalance across batches (deferred).** In training the overall batch is always large
   enough for the grid to occupy all CUs at launch, so there is no general under-utilisation
   concern. A potential imbalance arises only when `seqlen_q` or `seqlen_kv` varies greatly
   across samples in the same batch (jagged mode): short-sequence work-groups finish early
   while long-sequence ones are still running, leaving CUs idle toward the end of the kernel.
   If this becomes a bottleneck, a split-kv scheme (for computing dQ) or split-q scheme (for
   computing dK and dV) -- where multiple blocks collaborate on a single sequence tile via
   atomics or a reduce-then-add pass -- could reduce the imbalance. This is deferred; the
   initial implementation uses one block per tile with no splitting.
