#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 17: AppendKV with RoPE Integration

Demonstrates:
1. KV cache append operation (new tokens added to existing cache)
2. RoPE (Rotary Position Embedding) integration:
   - Interleaved: pairs (x0,x1), (x2,x3), ... rotated together
   - Half-rotated: first half and second half rotated
3. Paged KV cache with page_block_size and cache_batch_idx
4. CPU reference for RoPE-transformed KV append

AppendKV is the first stage of a decode step: new K,V tokens are
RoPE-transformed and appended to the paged cache before attention.

Usage:
    python3 17_appendkv_fmha.py
    python3 17_appendkv_fmha.py --rope interleaved
    python3 17_appendkv_fmha.py --seqlen-new 4 --page-size 64
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from fmha_utils import (
    FmhaProblem,
    FmhaKernelConfig,
    detect_gpu_arch,
    setup_fmha_dispatcher,
)


def make_rotary_cos_sin(
    max_seqlen: int,
    hdim: int,
    base: float = 10000.0,
) -> tuple:
    """Generate RoPE cos/sin tables.

    Returns: (cos_table, sin_table) each of shape [max_seqlen, hdim//2]
    """
    half_dim = hdim // 2
    inv_freq = 1.0 / (base ** (np.arange(0, half_dim, dtype=np.float32) / half_dim))
    pos = np.arange(max_seqlen, dtype=np.float32)
    freqs = np.outer(pos, inv_freq)
    return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)


def apply_rope_interleaved(
    x: np.ndarray, cos: np.ndarray, sin: np.ndarray, start_pos: int
) -> np.ndarray:
    """Apply interleaved RoPE: pairs (x0,x1), (x2,x3), ... rotated together.

    x: [..., seqlen, hdim]
    cos, sin: [max_seqlen, hdim//2]
    """
    seqlen = x.shape[-2]
    hdim = x.shape[-1]
    half = hdim // 2

    cos_slice = cos[start_pos : start_pos + seqlen, :]
    sin_slice = sin[start_pos : start_pos + seqlen, :]

    cos_b = cos_slice.reshape((1,) * (x.ndim - 2) + (seqlen, half))
    sin_b = sin_slice.reshape((1,) * (x.ndim - 2) + (seqlen, half))

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    out = np.empty_like(x)
    out[..., 0::2] = x_even * cos_b - x_odd * sin_b
    out[..., 1::2] = x_odd * cos_b + x_even * sin_b
    return out


def apply_rope_half_rotated(
    x: np.ndarray, cos: np.ndarray, sin: np.ndarray, start_pos: int
) -> np.ndarray:
    """Apply half-rotated RoPE: first half and second half rotated.

    x: [..., seqlen, hdim]
    cos, sin: [max_seqlen, hdim//2]
    """
    seqlen = x.shape[-2]
    hdim = x.shape[-1]
    half = hdim // 2

    cos_slice = cos[start_pos : start_pos + seqlen, :]
    sin_slice = sin[start_pos : start_pos + seqlen, :]

    cos_b = cos_slice.reshape((1,) * (x.ndim - 2) + (seqlen, half))
    sin_b = sin_slice.reshape((1,) * (x.ndim - 2) + (seqlen, half))

    x1, x2 = x[..., :half], x[..., half:]

    out = np.empty_like(x)
    out[..., :half] = x1 * cos_b - x2 * sin_b
    out[..., half:] = x2 * cos_b + x1 * sin_b
    return out


def cpu_append_kv(
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    k_new: np.ndarray,
    v_new: np.ndarray,
    cache_seqlen: int,
    rope_fn,
    cos: np.ndarray,
    sin: np.ndarray,
) -> tuple:
    """CPU reference: append new KV tokens to cache with RoPE.

    k_cache/v_cache: [batch, nhead, max_seqlen, hdim]
    k_new/v_new: [batch, nhead, seqlen_new, hdim]

    Returns: (k_cache_updated, v_cache_updated)
    """
    seqlen_new = k_new.shape[2]

    if rope_fn is not None:
        k_rotated = rope_fn(k_new, cos, sin, cache_seqlen)
    else:
        k_rotated = k_new

    k_out = k_cache.copy()
    v_out = v_cache.copy()
    k_out[:, :, cache_seqlen : cache_seqlen + seqlen_new, :] = k_rotated
    v_out[:, :, cache_seqlen : cache_seqlen + seqlen_new, :] = v_new

    return k_out, v_out


def make_paged_cache(
    batch: int, nhead: int, total_pages: int, page_size: int, hdim: int
) -> tuple:
    """Create a paged KV cache layout.

    Returns: (k_pages, v_pages, page_table, cache_batch_idx)
    """
    k_pages = np.zeros((total_pages, nhead, page_size, hdim), dtype=np.float32)
    v_pages = np.zeros((total_pages, nhead, page_size, hdim), dtype=np.float32)

    pages_per_seq = total_pages // batch
    page_table = np.arange(total_pages, dtype=np.int32).reshape(batch, pages_per_seq)
    cache_batch_idx = np.arange(batch, dtype=np.int32)

    return k_pages, v_pages, page_table, cache_batch_idx


def main():
    parser = argparse.ArgumentParser(description="AppendKV with RoPE Integration")
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=16)
    parser.add_argument("--hdim", type=int, default=128)
    parser.add_argument(
        "--seqlen-new", type=int, default=1, help="New tokens to append"
    )
    parser.add_argument(
        "--cache-seqlen", type=int, default=512, help="Existing cache length"
    )
    parser.add_argument("--max-seqlen", type=int, default=2048)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument(
        "--rope", default="both", choices=["interleaved", "half", "none", "both"]
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Example 17: AppendKV with RoPE Integration")
    print("=" * 70)

    print(f"\n  Batch:        {args.batch}")
    print(f"  Heads:        {args.nhead}")
    print(f"  HDim:         {args.hdim}")
    print(f"  New tokens:   {args.seqlen_new}")
    print(f"  Cache len:    {args.cache_seqlen}")
    print(f"  Max seqlen:   {args.max_seqlen}")
    print(f"  Page size:    {args.page_size}")

    # --- Generate RoPE tables ---
    cos, sin = make_rotary_cos_sin(args.max_seqlen, args.hdim)
    print("\n  RoPE base:    10000.0")
    print(f"  Cos/Sin:      [{args.max_seqlen}, {args.hdim // 2}]")

    # --- Generate new KV data ---
    np.random.seed(42)
    k_new = (
        np.random.randn(args.batch, args.nhead, args.seqlen_new, args.hdim) * 0.1
    ).astype(np.float32)
    v_new = (
        np.random.randn(args.batch, args.nhead, args.seqlen_new, args.hdim) * 0.1
    ).astype(np.float32)

    # --- RoPE comparison ---
    rope_modes = []
    if args.rope in ("interleaved", "both"):
        rope_modes.append(("interleaved", apply_rope_interleaved))
    if args.rope in ("half", "both"):
        rope_modes.append(("half_rotated", apply_rope_half_rotated))
    if args.rope == "none":
        rope_modes.append(("none", None))

    print("\n--- RoPE Modes ---")
    print(f"\n  {'Mode':<16} {'K_new range':>20} {'K_rope range':>20} {'MaxDiff':>10}")
    print("  " + "-" * 70)

    for mode_name, rope_fn in rope_modes:
        if rope_fn is not None:
            k_roped = rope_fn(k_new, cos, sin, args.cache_seqlen)
        else:
            k_roped = k_new

        k_range = f"[{k_new.min():.4f}, {k_new.max():.4f}]"
        kr_range = f"[{k_roped.min():.4f}, {k_roped.max():.4f}]"
        diff = float(np.abs(k_roped - k_new).max())
        print(f"  {mode_name:<16} {k_range:>20} {kr_range:>20} {diff:>10.4f}")

    # --- KV Cache Append ---
    print("\n--- KV Cache Append ---")
    k_cache = np.zeros(
        (args.batch, args.nhead, args.max_seqlen, args.hdim), dtype=np.float32
    )
    v_cache = np.zeros(
        (args.batch, args.nhead, args.max_seqlen, args.hdim), dtype=np.float32
    )

    np.random.seed(0)
    k_cache[:, :, : args.cache_seqlen, :] = (
        np.random.randn(args.batch, args.nhead, args.cache_seqlen, args.hdim) * 0.1
    ).astype(np.float32)
    v_cache[:, :, : args.cache_seqlen, :] = (
        np.random.randn(args.batch, args.nhead, args.cache_seqlen, args.hdim) * 0.1
    ).astype(np.float32)

    for mode_name, rope_fn in rope_modes:
        k_up, v_up = cpu_append_kv(
            k_cache,
            v_cache,
            k_new,
            v_new,
            args.cache_seqlen,
            rope_fn,
            cos,
            sin,
        )
        new_len = args.cache_seqlen + args.seqlen_new
        k_appended = k_up[:, :, args.cache_seqlen : new_len, :]
        print(f"\n  {mode_name}:")
        print(f"    Cache after append: positions [0, {new_len})")
        print(f"    New K range: [{k_appended.min():.4f}, {k_appended.max():.4f}]")
        print(
            f"    Cache unchanged: {np.array_equal(k_up[:, :, : args.cache_seqlen, :], k_cache[:, :, : args.cache_seqlen, :])}"
        )

    # --- Paged KV Cache ---
    print("\n--- Paged KV Cache Layout ---")
    total_pages = (args.max_seqlen // args.page_size) * args.batch
    k_pages, v_pages, page_table, cache_batch_idx = make_paged_cache(
        args.batch,
        args.nhead,
        total_pages,
        args.page_size,
        args.hdim,
    )

    pages_per_seq = total_pages // args.batch
    print(f"  Total pages:     {total_pages}")
    print(f"  Pages per seq:   {pages_per_seq}")
    print(f"  Page size:       {args.page_size}")
    print(f"  K pages shape:   {k_pages.shape}")
    print(f"  Page table:      {page_table.shape}")
    print(f"  cache_batch_idx: {cache_batch_idx}")

    current_page = args.cache_seqlen // args.page_size
    offset_in_page = args.cache_seqlen % args.page_size
    print(f"\n  Append position: page={current_page}, offset={offset_in_page}")
    print(f"  Physical page idx (batch 0): {page_table[0, current_page]}")

    # --- GPU attempt ---
    print("\n--- GPU Execution ---")
    config = FmhaKernelConfig(
        data_type="fp16",
        hdim_q=args.hdim,
        hdim_v=args.hdim,
        gfx_arch=args.arch,
    )
    setup = setup_fmha_dispatcher(config)
    if not setup.success:
        print(f"  JIT build failed: {setup.error}")
    else:
        runner = setup.runner
        print(f"  JIT build: {setup.build_time_s:.1f}s")
        prob = FmhaProblem(
            batch=args.batch,
            nhead_q=args.nhead,
            nhead_k=args.nhead,
            seqlen_q=args.seqlen_new,
            seqlen_k=args.cache_seqlen + args.seqlen_new,
            hdim_q=args.hdim,
            hdim_v=args.hdim,
        )
        Q_fp16 = (np.random.randn(*prob.q_shape()) * 0.1).astype(np.float16)
        K_full = k_cache[:, :, : args.cache_seqlen + args.seqlen_new, :].astype(
            np.float16
        )
        V_full = v_cache[:, :, : args.cache_seqlen + args.seqlen_new, :].astype(
            np.float16
        )
        res = runner.run(Q_fp16, K_full, V_full, prob)
        if res.success:
            print(
                f"  Attention after append: {res.time_ms:.4f} ms, {res.tflops:.2f} TFLOPS"
            )
        else:
            print("  GPU: Kernel returned failure (appendkv not supported)")
    print("  Note: Prebuilt kernel does not support appendkv family")

    # --- RoPE position-dependency visualization ---
    print("\n--- RoPE Position Dependency ---")
    positions = [0, 128, 512, 1024]
    test_vec = np.ones((1, 1, 1, args.hdim), dtype=np.float32) * 0.1

    for rope_name, rope_fn in rope_modes:
        if rope_fn is None:
            continue
        print(f"\n  {rope_name} (first 4 dims of rotated unit vector):")
        print(f"  {'Position':>10}  {'dim0':>8} {'dim1':>8} {'dim2':>8} {'dim3':>8}")
        for pos in positions:
            if pos < args.max_seqlen:
                rotated = rope_fn(test_vec, cos, sin, pos)
                dims = rotated[0, 0, 0, :4]
                print(
                    f"  {pos:>10}  {dims[0]:>8.4f} {dims[1]:>8.4f} {dims[2]:>8.4f} {dims[3]:>8.4f}"
                )

    # --- Summary ---
    print("\n" + "=" * 70)
    print(
        f"  AppendKV:     Append {args.seqlen_new} new tokens at position {args.cache_seqlen}"
    )
    print(f"  RoPE modes:   {', '.join(m for m, _ in rope_modes)}")
    print(f"  Paged cache:  {total_pages} pages x {args.page_size} slots")
    print("  Pipeline:     appendkv -> fwd_pagedkv (2-stage decode)")
    print("  GPU:          Prebuilt supports fwd only (appendkv needs JIT)")
    print("  Status:       DEMO")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
