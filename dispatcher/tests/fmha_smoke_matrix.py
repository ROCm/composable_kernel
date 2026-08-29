#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
FMHA smoke test matrix generator.

Generates the same test cases as smoke_test_fwd.sh and smoke_test_bwd.sh
from the CK Tile 01_fmha example, for automated parity testing.
"""

from dataclasses import dataclass
from typing import List, Set, Tuple


@dataclass
class TestCase:
    name: str = ""
    direction: str = "fwd"
    prec: str = "fp16"
    mode: int = 0
    batch: int = 2
    nhead_q: int = 2
    nhead_k: int = -1
    hdim_q: int = 128
    hdim_v: int = -1
    seqlen_q: int = 128
    seqlen_k: int = 128
    bias: str = "n"
    mask: str = "0"
    lse: int = 0
    p_drop: float = 0.0
    perm: int = 1
    num_splits: int = 1
    page_block_size: int = 0
    cache_batch_idx: int = 0
    s_kpad: str = ""
    q_eff_lens: str = ""
    kv_eff_lens: str = ""
    s_qpad: str = ""
    rotary_dim: int = 0
    rotary_interleaved: int = 0
    deterministic: int = 0
    dbias: int = 0

    def effective_nhead_k(self):
        return self.nhead_k if self.nhead_k > 0 else self.nhead_q

    def effective_hdim_v(self):
        return self.hdim_v if self.hdim_v > 0 else self.hdim_q


def generate_fwd_fp16_bf16_matrix() -> List[TestCase]:
    """Generate the run_fp16_bf16_tests matrix from smoke_test_fwd.sh."""
    cases = []
    idx = 0
    for prec in ["fp16", "bf16"]:
        for mode in [1, 0]:
            for perm in [0, 1]:
                for hdim in [32, 64, 128, 256]:
                    for lse in [0, 1]:
                        for bias in ["n", "e", "a"]:
                            for p_drop in [0.0, 0.2]:
                                base = dict(
                                    prec=prec,
                                    mode=mode,
                                    perm=perm,
                                    lse=lse,
                                    bias=bias,
                                    p_drop=p_drop,
                                )
                                subcases = [
                                    dict(
                                        batch=2,
                                        nhead_q=2,
                                        nhead_k=1,
                                        hdim_q=16,
                                        hdim_v=hdim,
                                        seqlen_q=55,
                                        seqlen_k=256,
                                        mask="0",
                                    ),
                                    dict(
                                        batch=1,
                                        nhead_q=3,
                                        hdim_q=hdim,
                                        seqlen_q=100,
                                        seqlen_k=51,
                                        mask="0",
                                    ),
                                    dict(
                                        batch=2,
                                        nhead_q=1,
                                        hdim_q=16,
                                        hdim_v=hdim,
                                        seqlen_q=99,
                                        seqlen_k=256,
                                        mask="1",
                                    ),
                                    dict(
                                        batch=1,
                                        nhead_q=2,
                                        nhead_k=1,
                                        hdim_q=hdim,
                                        seqlen_q=1024,
                                        seqlen_k=256,
                                        mask="2",
                                    ),
                                    dict(
                                        batch=2,
                                        nhead_q=1,
                                        hdim_q=hdim,
                                        hdim_v=24,
                                        seqlen_q=3,
                                        seqlen_k=99,
                                        mask="2",
                                    ),
                                    dict(
                                        batch=3,
                                        nhead_q=2,
                                        nhead_k=1,
                                        hdim_q=hdim,
                                        seqlen_q=200,
                                        seqlen_k=520,
                                        mask="t:128,30",
                                    ),
                                    dict(
                                        batch=2,
                                        nhead_q=1,
                                        hdim_q=hdim,
                                        seqlen_q=99,
                                        seqlen_k=32,
                                        mask="b:4,35",
                                    ),
                                    dict(
                                        batch=1,
                                        nhead_q=2,
                                        nhead_k=1,
                                        hdim_q=hdim,
                                        seqlen_q=33,
                                        seqlen_k=0,
                                        mask="2",
                                    ),
                                    dict(
                                        batch=1,
                                        nhead_q=2,
                                        nhead_k=1,
                                        hdim_q=hdim,
                                        seqlen_q=1,
                                        seqlen_k=10,
                                        s_kpad="32",
                                        mask="2",
                                    ),
                                ]
                                for sc in subcases:
                                    idx += 1
                                    c = TestCase(
                                        name=f"fwd_{idx:04d}_{prec}_m{mode}_h{hdim}",
                                        direction="fwd",
                                        **base,
                                        **sc,
                                    )
                                    cases.append(c)
    return cases


def generate_bwd_matrix() -> List[TestCase]:
    """Generate the bwd smoke test matrix from smoke_test_bwd.sh."""
    cases = []
    idx = 0
    base_shapes = [
        dict(batch=1, nhead_q=4, nhead_k=2, seqlen_q=259, seqlen_k=259, mask="0"),
        dict(batch=2, nhead_q=2, seqlen_q=516, seqlen_k=253, mask="0"),
        dict(batch=1, nhead_q=4, nhead_k=1, seqlen_q=500, seqlen_k=251, mask="1"),
        dict(batch=1, nhead_q=2, seqlen_q=900, seqlen_k=258, mask="2"),
        dict(batch=2, nhead_q=1, seqlen_q=987, seqlen_k=219, mask="t:128,30"),
        dict(batch=2, nhead_q=3, nhead_k=1, seqlen_q=244, seqlen_k=499, mask="b:4,35"),
    ]
    for prec in ["fp16", "bf16"]:
        for perm in [0, 1]:
            for hdim in [32, 64, 128, 256]:
                for mode in [0, 1]:
                    for bias in ["n", "a"]:
                        for p_drop in [0.0, 0.2]:
                            for shape in base_shapes:
                                idx += 1
                                cases.append(
                                    TestCase(
                                        name=f"bwd_{idx:04d}_{prec}_h{hdim}",
                                        direction="bwd",
                                        prec=prec,
                                        mode=mode,
                                        perm=perm,
                                        hdim_q=hdim,
                                        hdim_v=hdim,
                                        bias=bias,
                                        p_drop=p_drop,
                                        lse=1,
                                        **shape,
                                    )
                                )
    return cases


def generate_splitkv_matrix() -> List[TestCase]:
    """Generate the splitkv smoke test matrix (same subcases as fwd, with num_splits > 1)."""
    cases = []
    idx = 0
    for prec in ["fp16", "bf16"]:
        for mode in [0]:  # splitkv only supports batch mode in smoke test
            for perm in [0, 1]:
                for hdim in [64, 128, 256]:
                    for num_splits in [2, 3]:
                        for bias in ["n"]:
                            subcases = [
                                dict(
                                    batch=2,
                                    nhead_q=2,
                                    nhead_k=1,
                                    seqlen_q=55,
                                    seqlen_k=256,
                                    mask="0",
                                ),
                                dict(
                                    batch=1,
                                    nhead_q=3,
                                    seqlen_q=100,
                                    seqlen_k=51,
                                    mask="0",
                                ),
                                dict(
                                    batch=1,
                                    nhead_q=2,
                                    nhead_k=1,
                                    seqlen_q=1024,
                                    seqlen_k=256,
                                    mask="2",
                                ),
                                dict(
                                    batch=3,
                                    nhead_q=2,
                                    nhead_k=1,
                                    seqlen_q=200,
                                    seqlen_k=520,
                                    mask="t:128,30",
                                ),
                            ]
                            for sc in subcases:
                                idx += 1
                                cases.append(
                                    TestCase(
                                        name=f"splitkv_{idx:04d}_{prec}_h{hdim}_s{num_splits}",
                                        direction="fwd_splitkv",
                                        prec=prec,
                                        mode=mode,
                                        perm=perm,
                                        hdim_q=hdim,
                                        hdim_v=hdim,
                                        lse=1,
                                        bias=bias,
                                        p_drop=0.0,
                                        num_splits=num_splits,
                                        page_block_size=128,
                                        cache_batch_idx=1,
                                        **sc,
                                    )
                                )
    return cases


def generate_padding_matrix() -> List[TestCase]:
    """Generate padding edge-case test cases."""
    cases = []
    idx = 0
    for prec in ["fp16"]:
        for hdim in [32, 64, 128]:
            edge_shapes = [
                dict(batch=1, nhead_q=1, seqlen_q=1, seqlen_k=1, mask="0"),
                dict(batch=1, nhead_q=1, seqlen_q=1, seqlen_k=256, mask="0"),
                dict(batch=1, nhead_q=1, seqlen_q=255, seqlen_k=1, mask="0"),
                dict(batch=1, nhead_q=2, seqlen_q=3, seqlen_k=5, mask="1"),
                dict(batch=2, nhead_q=1, seqlen_q=17, seqlen_k=33, mask="2"),
            ]
            for shape in edge_shapes:
                idx += 1
                cases.append(
                    TestCase(
                        name=f"pad_{idx:04d}_{prec}_h{hdim}",
                        direction="fwd",
                        prec=prec,
                        mode=0,
                        perm=1,
                        hdim_q=hdim,
                        hdim_v=hdim,
                        bias="n",
                        lse=0,
                        p_drop=0.0,
                        **shape,
                    )
                )
    return cases


def generate_fp8_matrix() -> List[TestCase]:
    """Generate fp8 smoke test cases (fp8bf16 and fp8fp32)."""
    cases = []
    idx = 0
    for prec in ["fp8bf16"]:
        for mode in [0]:
            for perm in [1]:
                for hdim in [64, 128]:
                    for mask in ["0", "2"]:
                        idx += 1
                        cases.append(
                            TestCase(
                                name=f"fp8_{idx:04d}_{prec}_h{hdim}",
                                direction="fwd",
                                prec=prec,
                                mode=mode,
                                perm=perm,
                                hdim_q=hdim,
                                hdim_v=hdim,
                                batch=2,
                                nhead_q=4,
                                nhead_k=4,
                                seqlen_q=128,
                                seqlen_k=128,
                                bias="n",
                                mask=mask,
                                lse=0,
                                p_drop=0.0,
                            )
                        )
    return cases


def unique_kernel_configs(cases: List[TestCase]) -> Set[Tuple]:
    """Extract unique kernel configs needed to run the test cases."""
    configs = set()
    for c in cases:
        dv = c.effective_hdim_v()
        mask_cat = (
            "no" if c.mask == "0" else ("causal" if c.mask in ["1", "2"] else "window")
        )
        bias_cat = c.bias
        configs.add(
            (
                c.direction,
                c.prec,
                c.hdim_q,
                dv,
                mask_cat,
                bias_cat,
                bool(c.lse),
                c.p_drop > 0,
            )
        )
    return configs


def to_ck_cli_args(case: TestCase) -> list:
    """Convert a TestCase to CK Tile CLI arguments."""
    nk = case.effective_nhead_k()
    dv = case.effective_hdim_v()
    args = [
        f"-prec={case.prec}",
        f"-mode={case.mode}",
        f"-b={case.batch}",
        f"-h={case.nhead_q}",
    ]
    if nk != case.nhead_q:
        args.append(f"-h_k={nk}")
    args += [f"-d={case.hdim_q}"]
    if dv != case.hdim_q:
        args.append(f"-d_v={dv}")
    args += [
        f"-s={case.seqlen_q}",
        f"-s_k={case.seqlen_k}",
        f"-bias={case.bias}",
        f"-mask={case.mask}",
        f"-lse={case.lse}",
        f"-p_drop={case.p_drop}",
        f"-iperm={case.perm}",
        f"-operm={case.perm}",
        "-v=1",
        "-warmup=0",
        "-repeat=1",
    ]
    if case.s_kpad:
        args.append(f"-s_kpad={case.s_kpad}")
    if case.num_splits > 1:
        args.append(f"-num_splits={case.num_splits}")
    if case.page_block_size > 0:
        args.append(f"-page_block_size={case.page_block_size}")
    if case.cache_batch_idx:
        args.append(f"-cache_batch_idx={case.cache_batch_idx}")
    return args


if __name__ == "__main__":
    fwd = generate_fwd_fp16_bf16_matrix()
    bwd = generate_bwd_matrix()
    skv = generate_splitkv_matrix()
    pad = generate_padding_matrix()
    fp8 = generate_fp8_matrix()

    all_cases = fwd + bwd + skv + pad + fp8
    all_configs = unique_kernel_configs(all_cases)

    print(f"Forward:   {len(fwd):5d} cases")
    print(f"Backward:  {len(bwd):5d} cases")
    print(f"SplitKV:   {len(skv):5d} cases")
    print(f"Padding:   {len(pad):5d} cases")
    print(f"FP8:       {len(fp8):5d} cases")
    print(f"Total:     {len(all_cases):5d} cases, {len(all_configs)} unique configs")
