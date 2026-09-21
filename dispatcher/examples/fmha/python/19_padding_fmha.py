#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 19: Batch Padding and Group Mode

Demonstrates:
1. Batch mode with effective lengths (q_eff_lens, kv_eff_lens)
   - Padded to max length but only effective positions contribute
2. Group mode with physical padding strides (s_qpad, s_kpad)
   - Variable-length sequences packed contiguously
   - seqstart pointers mark boundaries
3. Comparing batch vs group mode memory efficiency

In batch mode, each sequence in the batch is padded to the same max length.
In group mode, sequences are packed without padding using offset pointers,
saving memory for batches with high length variance.

Usage:
    python3 19_padding_fmha.py
    python3 19_padding_fmha.py --batch 8
    python3 19_padding_fmha.py --max-seqlen 512
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from fmha_utils import (
    FmhaProblem,
    FmhaKernelConfig,
    cpu_attention_fwd,
    detect_gpu_arch,
    setup_fmha_dispatcher,
)


def cpu_batch_padded_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float,
    q_eff_lens: np.ndarray,
    kv_eff_lens: np.ndarray,
) -> np.ndarray:
    """CPU reference: batch attention with effective lengths.

    Positions beyond effective length are masked out.
    Q: [batch, nhead, max_seqlen_q, hdim]
    """
    batch = Q.shape[0]
    nhead = Q.shape[1]
    max_sq = Q.shape[2]
    hdim_v = V.shape[3]

    out = np.zeros((batch, nhead, max_sq, hdim_v), dtype=np.float32)

    for b in range(batch):
        ql = q_eff_lens[b]
        kl = kv_eff_lens[b]

        Q_b = Q[b : b + 1, :, :ql, :]
        K_b = K[b : b + 1, :, :kl, :]
        V_b = V[b : b + 1, :, :kl, :]

        O_b = cpu_attention_fwd(Q_b, K_b, V_b, scale)
        out[b, :, :ql, :] = O_b[0]

    return out


def pack_group_mode(
    Q_batch: np.ndarray,
    K_batch: np.ndarray,
    V_batch: np.ndarray,
    q_lens: np.ndarray,
    kv_lens: np.ndarray,
) -> tuple:
    """Pack batch sequences into group mode (contiguous, no padding).

    Returns: (Q_packed, K_packed, V_packed, seqstart_q, seqstart_k)
    """
    batch = Q_batch.shape[0]
    nhead = Q_batch.shape[1]
    hdim_q = Q_batch.shape[3]
    hdim_v = V_batch.shape[3]

    total_q = int(q_lens.sum())
    total_k = int(kv_lens.sum())

    Q_packed = np.zeros((1, nhead, total_q, hdim_q), dtype=Q_batch.dtype)
    K_packed = np.zeros((1, nhead, total_k, hdim_q), dtype=K_batch.dtype)
    V_packed = np.zeros((1, nhead, total_k, hdim_v), dtype=V_batch.dtype)

    seqstart_q = np.zeros(batch + 1, dtype=np.int32)
    seqstart_k = np.zeros(batch + 1, dtype=np.int32)

    q_offset = 0
    k_offset = 0
    for b in range(batch):
        ql, kl = int(q_lens[b]), int(kv_lens[b])
        Q_packed[0, :, q_offset : q_offset + ql, :] = Q_batch[b, :, :ql, :]
        K_packed[0, :, k_offset : k_offset + kl, :] = K_batch[b, :, :kl, :]
        V_packed[0, :, k_offset : k_offset + kl, :] = V_batch[b, :, :kl, :]
        q_offset += ql
        k_offset += kl
        seqstart_q[b + 1] = q_offset
        seqstart_k[b + 1] = k_offset

    return Q_packed, K_packed, V_packed, seqstart_q, seqstart_k


def cpu_group_attention(
    Q_packed: np.ndarray,
    K_packed: np.ndarray,
    V_packed: np.ndarray,
    scale: float,
    seqstart_q: np.ndarray,
    seqstart_k: np.ndarray,
    batch: int,
) -> np.ndarray:
    """CPU reference: group mode attention on packed sequences.

    Q_packed: [1, nhead, total_q, hdim]
    """
    nhead = Q_packed.shape[1]
    total_q = Q_packed.shape[2]
    hdim_v = V_packed.shape[3]

    O_packed = np.zeros((1, nhead, total_q, hdim_v), dtype=np.float32)

    for b in range(batch):
        qs, qe = seqstart_q[b], seqstart_q[b + 1]
        ks, ke = seqstart_k[b], seqstart_k[b + 1]

        Q_b = Q_packed[:, :, qs:qe, :]
        K_b = K_packed[:, :, ks:ke, :]
        V_b = V_packed[:, :, ks:ke, :]

        O_b = cpu_attention_fwd(Q_b, K_b, V_b, scale)
        O_packed[0, :, qs:qe, :] = O_b[0]

    return O_packed


def main():
    parser = argparse.ArgumentParser(description="Batch Padding and Group Mode")
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--max-seqlen", type=int, default=256)
    parser.add_argument("--hdim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("Example 19: Batch Padding and Group Mode")
    print("=" * 70)

    batch = args.batch
    nhead = args.nhead
    max_sq = max_sk = args.max_seqlen
    hdim = args.hdim

    # --- Variable-length sequences ---
    np.random.seed(args.seed)
    q_eff_lens = np.sort(
        np.random.randint(32, max_sq + 1, size=batch).astype(np.int32)
    )[::-1]
    kv_eff_lens = np.sort(
        np.random.randint(32, max_sk + 1, size=batch).astype(np.int32)
    )[::-1]
    q_eff_lens = q_eff_lens.copy()
    kv_eff_lens = kv_eff_lens.copy()

    print(f"\n  Batch:       {batch}")
    print(f"  Max seqlen:  {max_sq}")
    print(f"  HDim:        {hdim}")
    print(f"\n  {'Seq#':<6} {'q_len':>8} {'kv_len':>8} {'q_pad%':>8} {'kv_pad%':>8}")
    print("  " + "-" * 42)
    for b in range(batch):
        q_pad = (1.0 - q_eff_lens[b] / max_sq) * 100
        kv_pad = (1.0 - kv_eff_lens[b] / max_sk) * 100
        print(
            f"  {b:<6} {q_eff_lens[b]:>8} {kv_eff_lens[b]:>8} {q_pad:>7.1f}% {kv_pad:>7.1f}%"
        )

    # --- Generate padded data ---
    Q_padded = (np.random.randn(batch, nhead, max_sq, hdim) * 0.1).astype(np.float32)
    K_padded = (np.random.randn(batch, nhead, max_sk, hdim) * 0.1).astype(np.float32)
    V_padded = (np.random.randn(batch, nhead, max_sk, hdim) * 0.1).astype(np.float32)

    # === BATCH MODE ===
    print("\n--- Batch Mode (padded) ---")
    O_batch = cpu_batch_padded_attention(
        Q_padded,
        K_padded,
        V_padded,
        1.0 / (hdim**0.5),
        q_eff_lens,
        kv_eff_lens,
    )

    batch_mem = batch * nhead * (max_sq + 2 * max_sk) * hdim * 4
    print(f"  Q/K/V layout: [{batch}, {nhead}, {max_sq}, {hdim}]")
    print(f"  Memory (Q+K+V): {batch_mem / 1024:.1f} KB")
    print(
        f"  Wasted (avg):   {(1.0 - q_eff_lens.mean() / max_sq) * 100:.1f}% (padding overhead)"
    )

    # === GROUP MODE ===
    print("\n--- Group Mode (packed) ---")
    Q_packed, K_packed, V_packed, seqstart_q, seqstart_k = pack_group_mode(
        Q_padded,
        K_padded,
        V_padded,
        q_eff_lens,
        kv_eff_lens,
    )

    total_q = int(q_eff_lens.sum())
    total_k = int(kv_eff_lens.sum())
    group_mem = nhead * (total_q + 2 * total_k) * hdim * 4

    print(f"  Q_packed:       [1, {nhead}, {total_q}, {hdim}]")
    print(f"  K_packed:       [1, {nhead}, {total_k}, {hdim}]")
    print(f"  seqstart_q:     {seqstart_q}")
    print(f"  seqstart_k:     {seqstart_k}")
    print(f"  Memory (Q+K+V): {group_mem / 1024:.1f} KB")
    print(f"  Saving vs batch: {(1.0 - group_mem / batch_mem) * 100:.1f}%")

    # Physical padding strides
    s_qpad = total_q
    s_kpad = total_k
    print("\n  Physical strides:")
    print(f"    s_qpad = {s_qpad} (total Q tokens)")
    print(f"    s_kpad = {s_kpad} (total KV tokens)")

    O_group = cpu_group_attention(
        Q_packed,
        K_packed,
        V_packed,
        1.0 / (hdim**0.5),
        seqstart_q,
        seqstart_k,
        batch,
    )

    # --- Cross-validate batch vs group ---
    print("\n--- Batch vs Group Validation ---")
    print(f"\n  {'Seq#':<6} {'q_len':>8} {'MaxErr':>10} {'Status':>8}")
    print("  " + "-" * 36)

    all_ok = True
    for b in range(batch):
        ql = q_eff_lens[b]
        qs = seqstart_q[b]
        O_b_batch = O_batch[b, :, :ql, :]
        O_b_group = O_group[0, :, qs : qs + ql, :]
        max_err = float(np.abs(O_b_batch - O_b_group).max())
        ok = max_err < 1e-5
        all_ok = all_ok and ok
        print(f"  {b:<6} {ql:>8} {max_err:>10.2e} {'PASS' if ok else 'FAIL':>8}")

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
            batch=batch,
            nhead_q=nhead,
            nhead_k=nhead,
            seqlen_q=max_sq,
            seqlen_k=max_sk,
            hdim_q=hdim,
            hdim_v=hdim,
        )
        Q_fp16 = Q_padded.astype(np.float16)
        K_fp16 = K_padded.astype(np.float16)
        V_fp16 = V_padded.astype(np.float16)
        res = runner.run(Q_fp16, K_fp16, V_fp16, prob)
        if res.success:
            print(f"  GPU (full padded): {res.time_ms:.4f} ms, {res.tflops:.2f} TFLOPS")
            print(
                "  Note: GPU runs full padded attention; effective-length masking needs kernel support"
            )
        else:
            print("  GPU: Kernel returned failure")

    # --- Memory analysis ---
    print("\n--- Memory Efficiency Analysis ---")
    print(f"\n  {'Metric':<24} {'Batch Mode':>14} {'Group Mode':>14} {'Ratio':>8}")
    print("  " + "-" * 64)

    batch_tokens_q = batch * max_sq
    group_tokens_q = total_q
    batch_tokens_k = batch * max_sk
    group_tokens_k = total_k

    print(
        f"  {'Q tokens':<24} {batch_tokens_q:>14} {group_tokens_q:>14} {group_tokens_q / batch_tokens_q:>7.2f}x"
    )
    print(
        f"  {'KV tokens':<24} {batch_tokens_k:>14} {group_tokens_k:>14} {group_tokens_k / batch_tokens_k:>7.2f}x"
    )
    print(
        f"  {'Memory (KB)':<24} {batch_mem / 1024:>14.1f} {group_mem / 1024:>14.1f} {group_mem / batch_mem:>7.2f}x"
    )
    print(
        f"  {'Compute (tokens)':<24} {batch_tokens_q * batch_tokens_k:>14} {sum(q_eff_lens[i] * kv_eff_lens[i] for i in range(batch)):>14} "
        f"{sum(q_eff_lens[i] * kv_eff_lens[i] for i in range(batch)) / (batch_tokens_q * batch_tokens_k):>7.2f}x"
    )

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  Batch mode:  Padded to max_seqlen, uses q_eff_lens/kv_eff_lens")
    print("  Group mode:  Packed contiguously, uses seqstart pointers")
    print(f"  Strides:     s_qpad={s_qpad}, s_kpad={s_kpad}")
    print(f"  Memory save: {(1.0 - group_mem / batch_mem) * 100:.1f}% with group mode")
    print(f"  Batch==Group: {'PASS' if all_ok else 'FAIL'} (identical results)")
    print("  GPU:          Prebuilt supports batch mode only")
    print(f"  Status:       {'PASS' if all_ok else 'FAIL'}")
    print("=" * 70)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
