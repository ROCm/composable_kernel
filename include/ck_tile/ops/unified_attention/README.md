# CK Tile — Unified Attention kernel

FlashAttention-style fused attention forward for **paged and contiguous KV**,
covering **prefill and decode** in one operator, for **fp8 / bf16 / fp16** on
gfx950 (MI350/MI355). This is the kernel behind aiter's `unified_attention` op;
the test/perf/analysis tooling lives in `aiter/ua-test-scripts/` (see its README).

This document is the architecture reference for reading the kernel and for
porting it (e.g. to FlyDSL). The functional contract mirrors Triton's
`unified_attention` reference.

---

## File map

| file | role |
|---|---|
| `kernel/unified_attention_kernel.hpp` | Host `MakeKargs` / grid helpers + the device `operator()`: per-CTA index math (grid → kv_head/seq/q_block/split), Q/K/V/O DRAM window setup, mask construction, split-KV partitioning, the call into the pipeline, and the epilogue / split-KV workspace writes. |
| `pipeline/unified_attention_pipeline.hpp` | The core per-CTA attention pipeline (the hot loop). Two regimes share one file: the **FA4** 2-warp-group matrix‖softmax overlap (prefill) and the **single-warp-group serial deferred-PV** pipeline (decode). |
| `pipeline/unified_attention_pipeline_default_policy.hpp` | Compile-time policy: tile distributions, LDS descriptors, warp-gemm selection, load-width/alignment selection, smem sizing, async-ring depth, and tuning constants. |
| `pipeline/unified_attention_pipeline_problem.hpp` | Bundles dtypes + shape + traits + mask into the `Problem` type. |
| `pipeline/tile_unified_attention_shape.hpp` | Block/warp tile dims: `kBlockM`, `kBlockQ`, `kPageBlockSize`, `kHeadDim`. |
| `pipeline/tile_unified_attention_traits.hpp` | Padding + occupancy (`kBlockPerCu`) traits. |
| `pipeline/unified_attention_core_loop_scheduler.hpp` | Per-phase `sched_group_barrier` instruction-mix hints that enforce the FA4 phase overlap (kept in lock-step with the macros in the pipeline header). |
| `block/block_masking.hpp` | Causal / sliding-window (FA-style left/right) mask. |

Concrete shape/dtype **instances** (the JIT-compiled translation units) live in
`composable_kernel/example/ck_tile/42_unified_attention/instances/`, dispatched by
`unified_attention.cpp` there.

---

## Per-CTA work assignment (`kernel` `operator()`)

One CTA computes one `(kv_head, q_block, split)` tuple:

- **Decode grid** `dim3(g8, num_splits, num_head_groups * num_seqs)` (selected via
  the `kargs.use_decode_grid` flag), XCD-swizzled for balance + KV streaming:
  `x = head-in-group`, `y = split`, `z = head_group + num_head_groups * seq`, with
  `kv_head = head_group * g8 + x`. No binary search, no padding CTAs. The WG
  dispatcher round-robins blocks over `#XCD = 8` by `linear_blockIdx % 8`, and `x`
  is the fastest dim, so:
  - **`num_kv_heads % 8 == 0`** → `g8 = 8`, `num_head_groups = num_kv_heads / 8`.
    Each XCD owns exactly one head per group and sweeps all its `num_splits` (and
    the next group's head) before advancing — balanced for *any* `num_splits`, and
    every XCD streams a single head's KV in order (L2-friendly).
  - **otherwise** → `g8 = num_kv_heads`, `num_head_groups = 1` (head-in-x
    fallback). The dispatcher round-robins the GLOBAL linear workgroup id `% 8`,
    so balance needs the *total* grid `N = num_kv_heads * num_splits * num_seqs`
    to be `≡ 0 (mod 8)` (else `N mod 8` XCDs run one extra full block while the
    rest idle at the tail). `num_seqs` is a runtime batch value, so the host
    (`_pick_num_splits`) makes the batch-independent factor divisible:
    `num_splits` a multiple of `8 / gcd(num_kv_heads, 8)` ⇒ `num_kv_heads *
    num_splits ≡ 0 (mod 8)` ⇒ `N ≡ 0` for any batch. Verified per-XCD with
    rocprofv3 `SQ_WAVES` (see decode_pipeline_research_plan.md §15).
  `split` is second-fastest either way, so adjacent splits of a head co-schedule
  and share L2.
- **Prefill / 2D grid** `dim3(num_kv_heads * total_num_q_blocks, 1, num_splits)`:
  `blockIdx.x` is folded; a binary search over `query_start_len_ptr` recovers the
  sequence, and out-of-range q-blocks early-return. The split lives in `blockIdx.z`.

`num_queries_per_kv` (GQA ratio) is a **runtime** value: `kBlockQ_dyn = kBlockM /
num_queries_per_kv`, so one compiled binary serves MHA and any GQA-N that divides
`kBlockM`. The `kBlockM`-row MFMA tile packs `num_queries_per_kv` consecutive
query rows per KV token; when `kBlockM % num_queries_per_kv != 0` the last 1–2
rows spill into the next q-tile's first token (co-owned), which drives several
correctness invariants in the index math — see the comments around
`last_tile_row_q_off` and the split-KV partition.

---

## Math (online softmax)

Standard streaming softmax with running max `m`, running sum `l`, output
accumulator `o_acc`, over KV tiles of `kPageBlockSize` tokens:

- Scores `S = scale_s · (Q·Kᵀ)`, masked, then `P = exp2(S − m)` (base-2).
- `scale_s` is pre-fused on the host (`MakeKargs`): `sm_scale · q_descale ·
  k_descale · log2(e)`. The `log2(e)` lets the device use full/þrate `exp2`
  instead of `exp`; the fp8 Q/K per-tensor descales fold in so the inner loop
  carries a single scalar (matches Triton's `qk_scale = sm_scale·q_scale·k_scale`).
- The fp8 V per-tensor descale `v_descale` is **deferred** to the post-loop
  `o_acc · v_descale / l` step — exact, since V is a linear factor on the
  unnormalised output. Non-fp8 dtypes pass `1.0f` (free no-op).
- Output is `o_acc / l`; `lse = m + log(l)` (natural-log domain) is returned for
  split-KV combine.

**PV is deferred one tile** (double-buffered on parity): the sequence per tile is
`alu1/pack(prev) → PV(prev) → QK(cur) → alu0/rowmax(cur) → D_upd/rescale`. This
is the known-correct ordering shared by both regimes.

---

## Two pipeline regimes

### FA4 (prefill, 2 warp groups, `NumWarpGroups == 2`)
FlashAttention-4-style overlap. The deferred-PV sequence is cut into two phases:
- **MATRIX**  phase: `PV(k-1) + QK(k)` — matrix pipe only.
- **SOFTMAX** phase: `alu1/exp + alu0/rowmax + D_upd/rescale` — VALU/MUFU only.

The two warp groups are primed one phase apart (WG0 in MATRIX while WG1 in
SOFTMAX), so on each SIMD the matrix work of one wave hides the transcendental
work of its co-resident partner. K/V are prefetched a tile ahead into a shared
double buffer at the per-phase block barrier (issued cooperatively by all 8
warps). The `core_loop_scheduler` hints reserve the per-phase instruction mix.

For fp8 the QK-C and PV-A per-thread layouts diverge (PV is forced to
`WGAttrNumAccess::Single`), so after packing, `P` is round-tripped through an LDS
window in canonical (M,N) order and reloaded in the PV-A distribution.

### Serial (decode, single warp group, `kFA4 == false`)
The same deferred-PV pipeline run serially by one 4-warp group, with K/V
double-buffered in LDS. Decode is **HBM-bandwidth bound** at long context; see
`aiter/ua-test-scripts/decode_pipeline_*.md`.

---

## Paged KV

When `kIsPaged`, KV tokens are resolved through `block_tables` (per-sequence page
lists). Performance hinges on keeping page-index resolution off the critical
path, via tiers selected at compile time:

- **Constexpr page size** (`kPageSize_ > 0`, the `ps16/ps32/ps64/ps128`
  instances): strength-reduces every `/ % * page_size` to shifts and enables the
  exact tier gates below. `kPageSize_ == 0` is the runtime-page-size catch-all.
- **Scalar-promote / single-page SRD rebase**: when a wave's load lands within one
  page, fold the page base into the buffer SRD once per wave and drop the per-lane
  block-table path.
- **Tier-2 LDS-resident page-table cache** (`kPageTableLdsEntries = 4096`, 16 KiB):
  resolves the multi-page fallback's page indices from LDS instead of per-lane
  global reads. Coverage `≤ 4096 · page_size` tokens; beyond that the kernel
  traps (a runtime fallback was measured −30% from register pressure).

When `!kIsPaged` (contiguous/THD) the logical KV token index *is* its physical
row (per-sequence base folded into the K/V pointer), so all paging math compiles
out.

## Split-KV

`num_splits > 1` partitions the KV range across CTAs along the split grid dim
(decode grid: `blockIdx.y`, 2D grid: `blockIdx.z`); each writes fp32
`o_acc`/`lse` workspaces that a separate combine kernel merges. The
partition is computed over the **causal-independent full-sequence** block count
(not the per-tile causal horizon) so a token co-owned by adjacent q-tiles maps to
the same split in both — otherwise non-dividing-GQA + causal races on the shared
token. `num_splits == 1` skips this path entirely.

---

## Tuning knobs & failed experiments

Active defaults (do not change without re-validating correctness + perf):

| macro / policy | default | effect |
|---|---|---|
| `CONDITIONAL_RESCALE` | `1` | FA4-only: carry accumulators in a committed-max frame and skip the online rescale while shifted scores stay ≤ `τ` (`CONDITIONAL_RESCALE_TAU = 8`). Mathematically exact. |
| `UA_FA4_PREFETCH_IN_SOFTMAX` | `1` | bf16/fp16: issue next-tile K/V async prefetch from the SOFTMAX phase (keeps MATRIX pure-matrix). |
| `GetKVAlignmentBytes` | dwordx4 where it tiles | Widen fp8 decode K/V async loads to 16 B/lane (the narrow 4 B/lane default was the main fp8-slower-than-bf16 decode regression). |
| `kKFallbackLds` (`UA_K_FALLBACK_LDS`) | `1` | Resolve the multi-page K fallback (ps16/ps32) page indices via the LDS cache instead of per-lane global reads. |
| `UA_DECODE_STAGES` | `2` | Decode async-ring depth (deeper buffering was measured perf-neutral; decode is BW-bound). |

Experiments kept **OFF** (measured losers — retained as one-line gates so the
rationale isn't relitigated): `UA_FA4_PACKED_SHIFT`, `UA_FA4_PACKED_ALU1_RESCALE`
(together ~−3% on canonical fp8 prefill — softmax is hidden under the overlap),
`UA_FA4_PACKED_ROWSUM` (−13%, serial chain beats the log-depth tree),
`UA_DYNAMIC_SETPRIO`, `MOVE_FMHA_MASK_TO_COMPUTE` (fp8 +8.8% regression),
`MOVE_FMHA_MASK_TO_GEMM1`, `UA_FA4_PIN_PACK_IN_SOFTMAX`. `UA_FA4_EXP2_APPROX`
(Schraudolph 2^x) is an *approximation* — off by default, validate accuracy first.

---

## Building & testing

The kernel is consumed via aiter's JIT module `module_unified_attention`. After
editing any file here you **must** force a fresh build — the `.so` is not rebuilt
automatically:

```bash
# from the aiter repo root
rm -rf aiter/jit/build/module_unified_attention aiter/jit/module_unified_attention.so
AITER_REBUILD=1 HIP_VISIBLE_DEVICES=2 python3 op_tests/test_unified_attention_ck.py --full
```

See `aiter/ua-test-scripts/README.md` for correctness/perf, and
`aiter/ua-test-scripts/analysis/README.md` for ISA/VGPR/overlap-trace tooling
(including the JIT-free standalone driver, which stamps every build so a stale
binary can never be measured).
