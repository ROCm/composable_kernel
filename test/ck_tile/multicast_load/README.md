# Multicast Load Tests (gfx1250)

Tests for the `CLUSTER_LOAD` and `CLUSTER_LOAD_ASYNC_TO_LDS` instructions on gfx1250.

## Overview

These instructions broadcast global memory data to multiple workgroups within a cluster, reducing redundant memory traffic when multiple workgroups need the same data.

- **`CLUSTER_LOAD_B*`** — synchronous load; data lands in VGPRs. Tracked by `LOADcnt`; wait with `s_wait_loadcnt`.
- **`CLUSTER_LOAD_ASYNC_TO_LDS_B*`** — asynchronous load; data lands directly in LDS. Tracked by `ASYNCcnt`; wait with `s_wait_asynccnt`.

### How CLUSTER_LOAD_B* works

Each lane supplies its own global source address (VADDR). The hardware detects which lanes across the participating WGPs are loading the same cache line. Instead of each WGP issuing an independent memory fetch, the hardware coalesces those requests: the data is fetched once from global memory and the result is broadcast (multicast) to all requesting WGPs simultaneously. The loaded value is written into each lane's destination VGPR.

The instruction is synchronous from the wave's perspective — the issuing wave stalls until the data is available in VGPRs. No explicit barrier is needed between the load and subsequent use within the same wave.

### How CLUSTER_LOAD_ASYNC_TO_LDS_B* works

The async variant operates similarly at the cluster level — participating WGPs coordinate to fetch data once and broadcast — but differs in two important ways:

1. **LDS destination**: Data is written directly to LDS (Local Data Share) using the per-lane LDS address supplied in VDST, bypassing VGPRs entirely. This eliminates the VGPR → LDS copy that would otherwise be required.

2. **Asynchronous completion**: The issuing wave does not stall. The instruction returns immediately and increments `ASYNCcnt`. The wave must later issue `s_wait_asynccnt(0)` to ensure the LDS write has committed before reading from LDS.

The typical usage pattern is:

```
// Wave 0: issue async load to LDS (returns immediately)
cluster_multicast_load_async_to_lds(src + lane_id, lds_ptr, mask);

// All waves in the workgroup synchronize:
s_wait_asynccnt(0);       // Wave 0 waits for LDS write to complete
s_barrier_signal(-1);     // All waves signal they have reached the barrier
s_barrier_wait(-1);       // All waves wait for every other wave to arrive

// Now safe for all waves to read from LDS
dst[lane_id] = lds_buf[lane_id];
```

### Clusters

A cluster is a group of up to 16 Workgroup Processors (WGPs) that can share data via multicast. When multiple workgroups within a cluster request the same address, the hardware fetches the data once and broadcasts it to all requesters.

### Broadcasting

Broadcasting is controlled by the M0 register:
- Bits `M0[15:0]` form a bitmask indicating which WGPs should receive the data
- All waves requesting the same data must set identical M0 values
- If `M0[15:0] == 0`, the load behaves as a normal non-multicast load
- `M0[16]` is an early-timeout bit: when set, the instruction completes without waiting for all masked WGPs to participate, preventing deadlock when fewer WGPs are launched than the mask implies

### Variants

| Instruction | Data Size | Destination | Wait instruction |
|-------------|-----------|-------------|------------------|
| `CLUSTER_LOAD_B32` | 32-bit | VGPR | `s_wait_loadcnt` |
| `CLUSTER_LOAD_B64` | 64-bit | VGPR | `s_wait_loadcnt` |
| `CLUSTER_LOAD_B128` | 128-bit | VGPR | `s_wait_loadcnt` |
| `CLUSTER_LOAD_ASYNC_TO_LDS_B32` | 32-bit | LDS | `s_wait_asynccnt` |
| `CLUSTER_LOAD_ASYNC_TO_LDS_B64` | 64-bit | LDS | `s_wait_asynccnt` |
| `CLUSTER_LOAD_ASYNC_TO_LDS_B128` | 128-bit | LDS | `s_wait_asynccnt` |

### INST_OFFSET

For `CLUSTER_LOAD_ASYNC_TO_LDS_B*`, the compile-time `INST_OFFSET` immediate is applied to **both** the global source address (VADDR) and the LDS destination address (VDST), per ISA section 4.9.9.1:

```
LDS[VGPR[VDST][lane] + INST_OFFSET] = GLOBAL_MEMORY[VGPR[VADDR][lane] + INST_OFFSET]
```

To offset only the LDS write position, adjust VDST directly and keep `inst_offset=0`.

## Tests

### `test_cluster_load_multicast` — synchronous VGPR destination

| Group | Description |
|-------|-------------|
| `SingleWGP` | B32/B64/B128 correctness with a single WGP, mask=0x1 |
| `M0Mask` | mask=0x0 (non-multicast path) and mask=0x1 (single-WGP multicast) |
| `MultiWGP` | 2–6 WGP cluster broadcasts for B32, B64, B128 |
| `PartialBroadcast` | Non-contiguous mask (0x5): only WGPs 0 and 2 issue cluster load, others use a plain load |
| `ConcurrentGroups` | Two independent broadcast groups within the same 4-WGP cluster |
| `EarlyTimeout` | M0[16] early-timeout bit prevents deadlock when fewer WGPs are launched than the mask claims |

### `test_cluster_load_async_to_lds` — asynchronous LDS destination

| Group | Description |
|-------|-------------|
| 1 `AsyncLDS` | B32/B64/B128 single-WGP baseline; mask=0x0 zero-mask degradation |
| 2 `LDSVisibility` | Non-requesting waves read LDS correctly after `block_sync_lds_direct_load` |
| 3 `LDSAddressLayout` | Per-lane strided VDST addressing |
| 4 `MultiWGPBroadcast` | Async LDS delivery at cluster scale: 1D (2-WGP, 4-WGP) and 2D `dim3(2,2,1)` cluster dims |
| 5 `ASYNCcntOrdering` | `CLUSTER_LOAD_ASYNC_TO_LDS` and `GLOBAL_LOAD_ASYNC_TO_LDS` share one ASYNCcnt |
| 6 `PartialBroadcast` | Non-contiguous mask (0x5) with mixed instruction types |
| 8 `MultiWGPLDSVisibility` | Canonical GEMM tile-load pattern: one wave loads, all waves read |
| 10 `ConcurrentGroupsLDS` | LDS routing isolation between two independent broadcast groups |
| 11 `BufferViewAsyncGet` | `buffer_view::cluster_async_get()` interface; INST_OFFSET ISA behaviour |
