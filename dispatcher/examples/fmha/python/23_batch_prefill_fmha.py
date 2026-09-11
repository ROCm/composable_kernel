#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 23: Batch Prefill FMHA for SGLang/vLLM

Demonstrates batch prefill with paged KV-cache, as used in serving
frameworks like SGLang and vLLM. Shows the KV page table configuration
(kv_indptr, kv_page_indices, kv_last_page_lens) for both:
  - SGLang: 1D page table with indirect page lookup
  - vLLM: 2D block table with per-sequence page arrays

This example builds the page table metadata on CPU and validates the
attention computation. The prebuilt library only supports the basic
forward kernel, so the page table logic is demonstrated via CPU reference.

Usage:
    python3 23_batch_prefill_fmha.py
    python3 23_batch_prefill_fmha.py --page-size 64
    python3 23_batch_prefill_fmha.py --num-seqs 8
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np

from fmha_utils import (
    FmhaKernelConfig,
    FmhaProblem,
    FmhaValidator,
    cpu_attention_fwd,
    detect_gpu_arch,
    setup_fmha_dispatcher,
)


def build_sglang_page_table(
    seq_lens_k: list,
    page_size: int,
    nhead_k: int,
    hdim: int,
) -> dict:
    """Build SGLang-style 1D page table for paged KV-cache.

    SGLang uses a flat 1D array of page indices. Each sequence's pages are
    stored contiguously in the page_indices array, with indptr marking
    boundaries.

    Returns dict with:
        kv_indptr:        [num_seqs + 1] cumulative page counts
        kv_page_indices:  [total_pages] global page IDs
        kv_last_page_lens: [num_seqs] tokens in last page of each seq
        num_total_pages:  total pages allocated
        kv_data_shape:    shape of the paged KV pool
    """
    num_seqs = len(seq_lens_k)
    kv_indptr = np.zeros(num_seqs + 1, dtype=np.int32)
    page_indices_list = []
    last_page_lens = np.zeros(num_seqs, dtype=np.int32)

    page_counter = 0
    for i, seqlen in enumerate(seq_lens_k):
        num_pages = (seqlen + page_size - 1) // page_size
        kv_indptr[i + 1] = kv_indptr[i] + num_pages
        page_indices_list.extend(range(page_counter, page_counter + num_pages))
        last_page_lens[i] = seqlen - (num_pages - 1) * page_size
        page_counter += num_pages

    kv_page_indices = np.array(page_indices_list, dtype=np.int32)
    total_pages = page_counter

    return {
        "kv_indptr": kv_indptr,
        "kv_page_indices": kv_page_indices,
        "kv_last_page_lens": last_page_lens,
        "num_total_pages": total_pages,
        "kv_data_shape": (total_pages, 2, nhead_k, page_size, hdim),
        "layout": "sglang_1d",
    }


def build_vllm_block_table(
    seq_lens_k: list,
    page_size: int,
    nhead_k: int,
    hdim: int,
) -> dict:
    """Build vLLM-style 2D block table for paged KV-cache.

    vLLM uses a 2D array [num_seqs, max_blocks_per_seq] where each entry
    is a block (page) index into the global KV pool.

    Returns dict with:
        block_table:      [num_seqs, max_blocks] page IDs (-1 = unused)
        kv_last_page_lens: [num_seqs] tokens in last page of each seq
        num_total_pages:  total pages allocated
        kv_data_shape:    shape of the paged KV pool
    """
    num_seqs = len(seq_lens_k)
    pages_per_seq = [(s + page_size - 1) // page_size for s in seq_lens_k]
    max_blocks = max(pages_per_seq)

    block_table = np.full((num_seqs, max_blocks), -1, dtype=np.int32)
    last_page_lens = np.zeros(num_seqs, dtype=np.int32)

    page_counter = 0
    for i, (seqlen, num_pages) in enumerate(zip(seq_lens_k, pages_per_seq)):
        for p in range(num_pages):
            block_table[i, p] = page_counter
            page_counter += 1
        last_page_lens[i] = seqlen - (num_pages - 1) * page_size

    return {
        "block_table": block_table,
        "kv_last_page_lens": last_page_lens,
        "num_total_pages": page_counter,
        "kv_data_shape": (page_counter, 2, nhead_k, page_size, hdim),
        "layout": "vllm_2d",
    }


def scatter_kv_to_pages(
    K: np.ndarray,
    V: np.ndarray,
    page_table: dict,
    page_size: int,
) -> np.ndarray:
    """Scatter contiguous K,V into paged KV pool using page table.

    Args:
        K: [nhead_k, seqlen_k, hdim]  float32  (single sequence)
        V: [nhead_k, seqlen_k, hdim]  float32
        page_table: page indices for this sequence
        page_size: tokens per page
    """
    nhead_k, seqlen_k, hdim = K.shape
    num_pages = (seqlen_k + page_size - 1) // page_size

    pages = np.zeros((num_pages, 2, nhead_k, page_size, hdim), dtype=np.float32)
    for p in range(num_pages):
        start = p * page_size
        end = min(start + page_size, seqlen_k)
        length = end - start
        pages[p, 0, :, :length, :] = K[:, start:end, :]
        pages[p, 1, :, :length, :] = V[:, start:end, :]

    return pages


def gather_kv_from_pages(
    kv_pool: np.ndarray,
    page_indices: np.ndarray,
    seqlen_k: int,
    page_size: int,
) -> tuple:
    """Gather K,V from paged KV pool back to contiguous arrays.

    Returns:
        K: [nhead_k, seqlen_k, hdim]
        V: [nhead_k, seqlen_k, hdim]
    """
    nhead_k = kv_pool.shape[2]
    hdim = kv_pool.shape[4]
    K = np.zeros((nhead_k, seqlen_k, hdim), dtype=np.float32)
    V = np.zeros((nhead_k, seqlen_k, hdim), dtype=np.float32)

    for p, page_idx in enumerate(page_indices):
        start = p * page_size
        end = min(start + page_size, seqlen_k)
        length = end - start
        K[:, start:end, :] = kv_pool[page_idx, 0, :, :length, :]
        V[:, start:end, :] = kv_pool[page_idx, 1, :, :length, :]

    return K, V


def main():
    parser = argparse.ArgumentParser(
        description="Batch Prefill FMHA for SGLang/vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--nhead-q", type=int, default=16)
    parser.add_argument("--nhead-k", type=int, default=4, help="KV heads (GQA)")
    parser.add_argument("--hdim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--num-seqs", type=int, default=4, help="Sequences in batch")
    args = parser.parse_args()

    print("=" * 70)
    print("Example 23: Batch Prefill FMHA (SGLang/vLLM)")
    print("=" * 70)

    seq_lens_q = [32, 64, 16, 48][: args.num_seqs]
    seq_lens_k = [256, 512, 128, 384][: args.num_seqs]

    # Step 1: SGLang page table
    print("\nStep 1: SGLang 1D Page Table")

    sglang_pt = build_sglang_page_table(
        seq_lens_k,
        args.page_size,
        args.nhead_k,
        args.hdim,
    )

    print(f"  Page size:       {args.page_size}")
    print(f"  Total pages:     {sglang_pt['num_total_pages']}")
    print(f"  KV pool shape:   {sglang_pt['kv_data_shape']}")
    print(f"  kv_indptr:       {sglang_pt['kv_indptr']}")
    print(
        f"  kv_page_indices: {sglang_pt['kv_page_indices'][:20]}{'...' if len(sglang_pt['kv_page_indices']) > 20 else ''}"
    )
    print(f"  last_page_lens:  {sglang_pt['kv_last_page_lens']}")

    print("\n  Per-sequence breakdown:")
    print(f"  {'Seq':>5} {'SeqQ':>6} {'SeqK':>6} {'Pages':>6} {'LastLen':>8}")
    print("  " + "-" * 35)
    for i in range(args.num_seqs):
        n_pages = sglang_pt["kv_indptr"][i + 1] - sglang_pt["kv_indptr"][i]
        print(
            f"  {i:>5} {seq_lens_q[i]:>6} {seq_lens_k[i]:>6} {n_pages:>6} {sglang_pt['kv_last_page_lens'][i]:>8}"
        )

    # Step 2: vLLM block table
    print("\nStep 2: vLLM 2D Block Table")

    vllm_pt = build_vllm_block_table(
        seq_lens_k,
        args.page_size,
        args.nhead_k,
        args.hdim,
    )

    print(f"  Block table shape: {vllm_pt['block_table'].shape}")
    print(f"  Total pages:       {vllm_pt['num_total_pages']}")
    for i in range(args.num_seqs):
        row = vllm_pt["block_table"][i]
        valid = row[row >= 0]
        print(f"  Seq {i}: pages={valid.tolist()}")

    # Step 3: Validate scatter/gather round-trip
    print("\nStep 3: KV Page Scatter/Gather Validation")

    np.random.seed(42)
    validator = FmhaValidator(rtol=1e-5, atol=1e-5)

    total_pages = sglang_pt["num_total_pages"]
    kv_pool = np.zeros(
        (total_pages, 2, args.nhead_k, args.page_size, args.hdim),
        dtype=np.float32,
    )

    all_Q, all_K, all_V, all_O_ref = [], [], [], []

    for i in range(args.num_seqs):
        sq, sk = seq_lens_q[i], seq_lens_k[i]
        Q_i = np.random.randn(args.nhead_q, sq, args.hdim).astype(np.float32) * 0.3
        K_i = np.random.randn(args.nhead_k, sk, args.hdim).astype(np.float32) * 0.3
        V_i = np.random.randn(args.nhead_k, sk, args.hdim).astype(np.float32) * 0.3

        start_page = sglang_pt["kv_indptr"][i]
        end_page = sglang_pt["kv_indptr"][i + 1]
        page_indices = sglang_pt["kv_page_indices"][start_page:end_page]

        pages = scatter_kv_to_pages(K_i, V_i, page_indices, args.page_size)
        for p_local, p_global in enumerate(page_indices):
            kv_pool[p_global] = pages[p_local]

        K_rt, V_rt = gather_kv_from_pages(kv_pool, page_indices, sk, args.page_size)

        k_ok = np.allclose(K_i, K_rt, atol=1e-7)
        v_ok = np.allclose(V_i, V_rt, atol=1e-7)
        print(
            f"  Seq {i}: K round-trip={'OK' if k_ok else 'FAIL'}  "
            f"V round-trip={'OK' if v_ok else 'FAIL'}"
        )

        all_Q.append(Q_i)
        all_K.append(K_i)
        all_V.append(V_i)

    # Step 4: CPU attention per-sequence
    print("\nStep 4: CPU Attention per Sequence (from Paged KV)")

    print(f"\n  {'Seq':>5} {'SeqQ':>6} {'SeqK':>6} {'OutRange':>22} {'Scale':>10}")
    print("  " + "-" * 50)

    for i in range(args.num_seqs):
        sq, sk = seq_lens_q[i], seq_lens_k[i]
        Q_i = all_Q[i][np.newaxis]  # [1, nhead_q, sq, hdim]
        K_i = all_K[i][np.newaxis]  # [1, nhead_k, sk, hdim]
        V_i = all_V[i][np.newaxis]  # [1, nhead_k, sk, hdim]

        if args.nhead_q != args.nhead_k:
            ratio = args.nhead_q // args.nhead_k
            K_i_exp = np.repeat(K_i, ratio, axis=1)
            V_i_exp = np.repeat(V_i, ratio, axis=1)
        else:
            K_i_exp, V_i_exp = K_i, V_i

        scale = 1.0 / (args.hdim**0.5)
        O_i = cpu_attention_fwd(Q_i, K_i_exp, V_i_exp, scale)
        all_O_ref.append(O_i)

        out_range = f"[{O_i.min():.4f}, {O_i.max():.4f}]"
        print(f"  {i:>5} {sq:>6} {sk:>6} {out_range:>22} {scale:>10.4f}")

    # Step 5: Memory layout comparison
    print("\nStep 5: Memory Layout Analysis")

    contiguous_bytes = sum(2 * args.nhead_k * sk * args.hdim * 4 for sk in seq_lens_k)
    paged_bytes = total_pages * 2 * args.nhead_k * args.page_size * args.hdim * 4
    overhead = (paged_bytes - contiguous_bytes) / contiguous_bytes * 100

    print(f"  Contiguous KV:  {contiguous_bytes / 1024:.1f} KB")
    print(f"  Paged KV pool:  {paged_bytes / 1024:.1f} KB")
    print(f"  Overhead:       {overhead:.1f}% (due to page padding)")
    print(f"  Pages used:     {total_pages}")
    print(f"  Avg tokens/seq: {sum(seq_lens_k) / args.num_seqs:.0f}")

    # Step 6: GPU API pattern
    print("\nStep 6: GPU Kernel Configuration")
    print("  NOTE: The prebuilt library uses basic forward kernels.")
    print("  For batch prefill, compile a kernel with:")
    print()
    print("    FmhaSignature()")
    print("        .family('batch_prefill')")
    print("        .mode('group')")
    print("        .paged_kv(true)")
    print("        .kv_cache('vectorized', 'sglang', page_size)")
    print("        .lse(true)")
    print()
    print("  FmhaKernelConfig codegen JSON:")
    print("    'family': 'batch_prefill',")
    print("    'mode': 'group',")
    print("    'paged_kv': true,")
    print("    'kv_memory_layout': 'vectorized',")
    print("    'kv_lookup_table': 'sglang' or 'vllm',")
    print(f"    'page_size': {args.page_size}")

    # Step 7: GPU baseline (contiguous, no paging)
    print("\nStep 7: GPU Baseline (contiguous KV, single sequence)")

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
            batch=1,
            nhead_q=args.nhead_q,
            nhead_k=args.nhead_k,
            seqlen_q=64,
            seqlen_k=256,
            hdim_q=args.hdim,
            hdim_v=args.hdim,
        )
        Q_gpu = (np.random.randn(*prob.q_shape()) * 0.3).astype(np.float16)
        K_gpu = (np.random.randn(*prob.k_shape()) * 0.3).astype(np.float16)
        V_gpu = (np.random.randn(*prob.v_shape()) * 0.3).astype(np.float16)

        result = runner.run(Q_gpu, K_gpu, V_gpu, prob)
        if result.success:
            O_ref = cpu_attention_fwd(
                Q_gpu.astype(np.float32),
                K_gpu.astype(np.float32),
                V_gpu.astype(np.float32),
                prob.scale,
            )
            ok, max_abs, _ = validator.check(result.output, O_ref)
            print(
                f"  GPU baseline: time={result.time_ms:.4f}ms  TFLOPS={result.tflops:.2f}  "
                f"max_err={max_abs:.2e}  {'PASS' if ok else 'FAIL'}"
            )
        else:
            print(f"  GPU error: {result.error}")

    # Summary
    print("\n" + "=" * 70)
    print("  Batch prefill: serves multiple prefill requests in a single kernel launch")
    print("  SGLang: 1D page table (kv_indptr + kv_page_indices)")
    print("  vLLM:   2D block table [num_seqs, max_blocks]")
    print(
        f"  Page size {args.page_size} -> {overhead:.1f}% memory overhead vs contiguous"
    )
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
