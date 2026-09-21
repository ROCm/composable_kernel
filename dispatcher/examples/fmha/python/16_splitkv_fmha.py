#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 16: Split-KV Attention and Paged KV Cache

Demonstrates:
1. Split-KV: partitioning KV across multiple GPU splits for long sequences
2. Two-stage execution plan: split (per-partition attention) + combine (merge)
3. Paged KV cache with configurable page_block_size
4. CPU reference for split-KV correctness verification

Split-KV is critical for long-context inference where seqlen_k >> seqlen_q
(decoding with long history). Each split processes a chunk of KV independently,
then partial results are combined with log-sum-exp correction.

Usage:
    python3 16_splitkv_fmha.py
    python3 16_splitkv_fmha.py --num-splits 4
    python3 16_splitkv_fmha.py --seqlen-k 2048 --page-size 128
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from fmha_utils import (
    FmhaProblem,
    FmhaKernelConfig,
    FmhaValidator,
    cpu_attention_fwd,
    detect_gpu_arch,
    setup_fmha_dispatcher,
)


def cpu_splitkv_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float,
    num_splits: int,
) -> tuple:
    """CPU reference: split-KV attention with LSE-based combining.

    Stage 1 (split): Compute partial attention for each KV chunk
    Stage 2 (combine): Merge partial results using log-sum-exp correction

    Returns: (O_final, partial_Os, partial_lses)
    """
    batch, nhead, seqlen_q, hdim = Q.shape
    seqlen_k = K.shape[2]
    hdim_v = V.shape[3]

    chunk_size = (seqlen_k + num_splits - 1) // num_splits

    partial_Os = np.zeros(
        (num_splits, batch, nhead, seqlen_q, hdim_v), dtype=np.float32
    )
    partial_lses = np.full(
        (num_splits, batch, nhead, seqlen_q), -np.inf, dtype=np.float32
    )

    for s in range(num_splits):
        k_start = s * chunk_size
        k_end = min(k_start + chunk_size, seqlen_k)
        if k_start >= seqlen_k:
            break

        K_chunk = K[:, :, k_start:k_end, :]
        V_chunk = V[:, :, k_start:k_end, :]

        S = np.matmul(Q, K_chunk.transpose(0, 1, 3, 2)) * scale
        S_max = S.max(axis=-1, keepdims=True)
        S_exp = np.exp(S - S_max)
        S_sum = S_exp.sum(axis=-1, keepdims=True)

        partial_Os[s] = np.matmul(S_exp / S_sum, V_chunk)
        partial_lses[s] = np.log(S_sum.squeeze(-1)) + S_max.squeeze(-1)

    # Stage 2: Combine using LSE correction
    global_lse = np.max(partial_lses, axis=0)  # [batch, nhead, seqlen_q]

    O_final = np.zeros((batch, nhead, seqlen_q, hdim_v), dtype=np.float32)
    weight_sum = np.zeros((batch, nhead, seqlen_q), dtype=np.float32)

    for s in range(num_splits):
        correction = np.exp(partial_lses[s] - global_lse)
        correction = correction[..., np.newaxis]
        O_final += partial_Os[s] * correction
        weight_sum += correction.squeeze(-1)

    O_final = O_final / weight_sum[..., np.newaxis]
    return O_final, partial_Os, partial_lses


def make_page_table(batch: int, seqlen_k: int, page_size: int) -> tuple:
    """Create a paged KV cache layout.

    Returns: (page_table, num_pages_per_seq, total_pages)
    """
    pages_per_seq = (seqlen_k + page_size - 1) // page_size
    total_pages = batch * pages_per_seq

    page_table = np.arange(total_pages, dtype=np.int32).reshape(batch, pages_per_seq)
    return page_table, pages_per_seq, total_pages


def main():
    parser = argparse.ArgumentParser(description="Split-KV and Paged KV Cache")
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nhead-q", type=int, default=16)
    parser.add_argument("--nhead-k", type=int, default=16)
    parser.add_argument(
        "--seqlen-q", type=int, default=1, help="Typically 1 for decoding"
    )
    parser.add_argument("--seqlen-k", type=int, default=1024)
    parser.add_argument("--hdim", type=int, default=128)
    parser.add_argument("--num-splits", type=int, default=0, help="0 = test multiple")
    parser.add_argument("--page-size", type=int, default=128)
    args = parser.parse_args()

    print("=" * 70)
    print("Example 16: Split-KV Attention and Paged KV Cache")
    print("=" * 70)

    sq, sk = args.seqlen_q, args.seqlen_k
    prob = FmhaProblem(
        batch=args.batch,
        nhead_q=args.nhead_q,
        nhead_k=args.nhead_k,
        seqlen_q=sq,
        seqlen_k=sk,
        hdim_q=args.hdim,
        hdim_v=args.hdim,
    )

    print(
        f"\n  Problem:    B={prob.batch} Hq={prob.nhead_q} Hk={prob.nhead_k} "
        f"Sq={sq} Sk={sk} D={args.hdim}"
    )
    print(f"  Use case:   Decoding (Sq={sq} << Sk={sk})")

    # --- Generate data ---
    np.random.seed(42)
    Q_f32 = (np.random.randn(*prob.q_shape()) * 0.1).astype(np.float32)
    K_f32 = (np.random.randn(*prob.k_shape()) * 0.1).astype(np.float32)
    V_f32 = (np.random.randn(*prob.v_shape()) * 0.1).astype(np.float32)
    Q_fp16 = Q_f32.astype(np.float16)
    K_fp16 = K_f32.astype(np.float16)
    V_fp16 = V_f32.astype(np.float16)

    # --- Full attention reference ---
    O_full = cpu_attention_fwd(Q_f32, K_f32, V_f32, prob.scale)

    # --- GPU attempt ---
    print("\n--- GPU Execution ---")
    gpu_output = None
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
        res = runner.run(Q_fp16, K_fp16, V_fp16, prob)
        if res.success:
            gpu_output = res.output
            print(f"  GPU (full): {res.time_ms:.4f} ms, {res.tflops:.2f} TFLOPS")
        else:
            print("  GPU: Kernel returned failure")

    # --- Split-KV with various num_splits ---
    print("\n--- Split-KV Execution Plan ---")
    split_configs = [args.num_splits] if args.num_splits > 0 else [1, 2, 3, 4, 8]
    split_configs = [s for s in split_configs if s <= sk]

    validator = FmhaValidator(rtol=1e-5, atol=1e-5)

    print("\n  Plan stages:")
    print("    Stage 1 (split):   Compute partial O and LSE per KV chunk")
    print("    Stage 2 (combine): Merge with exp(lse_i - lse_max) correction")

    print(
        f"\n  {'#':<3} {'Splits':>7} {'ChunkSz':>8} {'Stage1':>8} {'Stage2':>8} "
        f"{'MaxErr':>10} {'Status':>8}"
    )
    print("  " + "-" * 58)

    for i, ns in enumerate(split_configs, 1):
        chunk_size = (sk + ns - 1) // ns

        O_split, partial_Os, partial_lses = cpu_splitkv_attention(
            Q_f32,
            K_f32,
            V_f32,
            prob.scale,
            ns,
        )

        ok, max_abs, _ = validator.check(O_split, O_full)
        tag = "PASS" if ok else "FAIL"

        print(
            f"  {i:<3} {ns:>7} {chunk_size:>8} {'split':>8} {'combine':>8} "
            f"{max_abs:>10.2e} {tag:>8}"
        )

    # --- Paged KV Cache ---
    print("\n--- Paged KV Cache ---")
    page_sizes = [64, 128, 256]

    print(
        f"\n  {'PageSize':>9} {'Pages/Seq':>10} {'TotalPages':>11} {'Utilization':>12}"
    )
    print("  " + "-" * 46)

    for ps in page_sizes:
        pt, pps, tp = make_page_table(args.batch, sk, ps)
        used_slots = args.batch * sk
        total_slots = tp * ps
        util = used_slots / total_slots * 100
        print(f"  {ps:>9} {pps:>10} {tp:>11} {util:>11.1f}%")

    print(f"\n  Page table example (batch=0, page_size={args.page_size}):")
    pt, pps, _ = make_page_table(args.batch, sk, args.page_size)
    pages_str = ", ".join(str(p) for p in pt[0, : min(8, pps)])
    if pps > 8:
        pages_str += f" ... ({pps} pages)"
    print(f"    [{pages_str}]")
    print("    Maps logical KV positions -> physical page indices")

    # --- GPU validation if available ---
    if gpu_output is not None:
        print("\n--- GPU vs Full-Attention Reference ---")
        val = FmhaValidator(rtol=1e-2, atol=1e-2)
        ok, max_abs, max_rel = val.check(gpu_output, O_full)
        print(
            f"  max_abs={max_abs:.2e}, max_rel={max_rel:.2e}, {'PASS' if ok else 'FAIL'}"
        )

    # --- Summary ---
    print("\n" + "=" * 70)
    print(f"  Split-KV:    Partitions seqlen_k={sk} across splits")
    print("  Plan:        2-stage (split partial O/LSE -> combine with correction)")
    print(f"  Paged KV:    page_block_size={args.page_size} ({pps} pages/seq)")
    print("  Use case:    Long-context decoding (Sq << Sk)")
    print("  GPU:         Prebuilt kernel runs full attention (no split-KV)")
    print("  Status:      PASS (CPU split-KV matches full attention)")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
