#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
FMHA Dispatcher Python Utilities

Provides Python wrappers for FMHA dispatcher kernels via ctypes.
Mirrors ctypes_utils.py (GEMM) and grouped_conv_utils.py (Conv).

Usage:
    from fmha_utils import FmhaDispatcherLib, FmhaRunner, FmhaProblem, cpu_attention_fwd

    runner = FmhaRunner.from_prebuilt()
    result = runner.run(Q, K, V, problem)
"""

import ctypes
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# =============================================================================
# Utility helpers
# =============================================================================


try:
    from dispatcher_common import detect_gpu_arch, get_dispatcher_root
except ImportError:
    # Standalone usage without dispatcher_common on PYTHONPATH
    def get_dispatcher_root() -> Path:
        return Path(__file__).parent.parent

    def detect_gpu_arch(fallback: str = "gfx950") -> str:
        try:
            out = subprocess.check_output(
                ["rocminfo"], text=True, stderr=subprocess.DEVNULL
            )
            for line in out.splitlines():
                if "Name:" in line and "gfx" in line:
                    return line.split()[-1].strip()
        except Exception:
            pass
        return fallback


# =============================================================================
# Data types
# =============================================================================


@dataclass
class FmhaResult:
    success: bool
    output: Optional[np.ndarray] = None
    time_ms: float = 0.0
    tflops: float = 0.0
    error: str = ""


@dataclass
class FmhaProblem:
    batch: int = 2
    nhead_q: int = 8
    nhead_k: int = 8
    seqlen_q: int = 128
    seqlen_k: int = 128
    hdim_q: int = 128
    hdim_v: int = 128

    @property
    def scale(self) -> float:
        return 1.0 / (self.hdim_q**0.5)

    @property
    def num_ops(self) -> int:
        sq, sk = self.seqlen_q, self.seqlen_k
        return 2 * self.batch * self.nhead_q * sq * sk * (self.hdim_q + self.hdim_v)

    def q_shape(self):
        return (self.batch, self.nhead_q, self.seqlen_q, self.hdim_q)

    def k_shape(self):
        return (self.batch, self.nhead_k, self.seqlen_k, self.hdim_q)

    def v_shape(self):
        return (self.batch, self.nhead_k, self.seqlen_k, self.hdim_v)

    def o_shape(self):
        return (self.batch, self.nhead_q, self.seqlen_q, self.hdim_v)


@dataclass
class FmhaKernelConfig:
    """Complete kernel configuration for FMHA.

    All tile/wave/warp dimensions are explicitly named to match the
    GEMM pattern (tile_m, tile_n, tile_k) but extended for FMHA's
    two-stage computation (Q*K^T stage 0, Attn*V stage 1).
    """

    # -- Signature: what operation --
    family: str = "fwd"
    data_type: str = "fp16"
    mode: str = "batch"
    vlayout: str = "r"
    hdim_q: int = 128
    hdim_v: int = 128
    gfx_arch: str = "gfx950"

    # -- Algorithm: tile shape --
    # Stage 0 (Q * K^T): seqlen_q x seqlen_k x hdim_q
    tile_m0: int = 128  # seqlen_q tile
    tile_n0: int = 128  # seqlen_k tile
    tile_k0: int = 32  # hdim_q tile
    # Stage 1 (Attn * V): seqlen_q x hdim_v x seqlen_k
    tile_n1: int = 128  # hdim_v tile
    tile_k1: int = 32  # seqlen_k tile
    tile_k0max: int = 128  # max k0 (alignment)
    # BWD extra stages (9-element tile)
    tile_bwd6: int = 0
    tile_bwd7: int = 0
    tile_bwd8: int = 0

    # -- Algorithm: wave config (warps per block) --
    wave_m0: int = 4
    wave_n0: int = 1
    wave_k0: int = 1
    wave_m1: int = 4
    wave_n1: int = 1
    wave_k1: int = 1
    wave_m2: int = 1
    wave_n2: int = 1
    wave_k2: int = 1

    # -- Algorithm: warp tile (elements per warp) --
    warp_m0: int = 32
    warp_n0: int = 32
    warp_k0: int = 16
    warp_m1: int = 32
    warp_n1: int = 32
    warp_k1: int = 16
    warp_m2: int = 16
    warp_n2: int = 16
    warp_k2: int = 16

    # -- Algorithm: padding --
    # Values: 0=no pad, 1=pad, 8=pad with 8-byte alignment (BWD-specific)
    pad_s: int = 1
    pad_sk: int = 1
    pad_d: int = 1
    pad_dv: int = 1

    # -- Algorithm: pipeline --
    pipeline: str = "qr_async"
    block_per_cu: int = -1
    num_wave_groups: int = 1

    # -- Signature: features --
    mask: str = "no"
    bias: str = "no"
    lse: bool = False
    dropout: bool = False
    qscale: str = "no"
    rope: str = "none"
    logits: bool = False
    paged_kv: bool = False
    sink: bool = False
    skip_min_seqlen_q: bool = False
    page_size: int = 1
    kv_memory_layout: str = "vectorized"
    kv_lookup_table: str = "sglang"
    deterministic: bool = False
    dbias: bool = False
    dropout_variant: str = ""  # BWD: "no"/"dropout_wg16"/"dropout_wg16_storerandval"
    tile_tag: str = ""  # extra tile variant discriminator (e.g. "trload", "small")
    use_trload: bool = False  # BWD dq_dk_dv: use trload pipeline path

    @property
    def tile(self) -> Tuple[int, ...]:
        base = (
            self.tile_m0,
            self.tile_n0,
            self.tile_k0,
            self.tile_n1,
            self.tile_k1,
            self.tile_k0max,
        )
        if self.family == "bwd_dq_dk_dv" and self.tile_bwd6 > 0:
            return base + (self.tile_bwd6, self.tile_bwd7, self.tile_bwd8)
        return base

    @property
    def wave(self) -> Tuple[int, ...]:
        return (
            self.wave_m0,
            self.wave_n0,
            self.wave_k0,
            self.wave_m1,
            self.wave_n1,
            self.wave_k1,
            self.wave_m2,
            self.wave_n2,
            self.wave_k2,
        )

    @property
    def warp(self) -> Tuple[int, ...]:
        return (
            self.warp_m0,
            self.warp_n0,
            self.warp_k0,
            self.warp_m1,
            self.warp_n1,
            self.warp_k1,
            self.warp_m2,
            self.warp_n2,
            self.warp_k2,
        )

    @property
    def padding(self) -> Tuple[bool, ...]:
        return (self.pad_s, self.pad_sk, self.pad_d, self.pad_dv)

    @property
    def name(self) -> str:
        s = self.pad_s
        k = self.pad_sk
        d = self.pad_d
        v = self.pad_dv
        parts = [
            f"fmha_{self.family}_{self.data_type}",
            self.mode,
            f"h{self.hdim_q}x{self.hdim_v}"
            if self.hdim_q != self.hdim_v
            else f"h{self.hdim_q}",
            self.pipeline,
            f"t{self.tile_m0}x{self.tile_n0}x{self.tile_k0}x{self.tile_n1}x{self.tile_k1}x{self.tile_k0max}"
            + (f".{self.tile_tag}" if self.tile_tag else ""),
        ]
        # Always include warp class for uniform naming
        parts.append(f"w{self.warp_m0}x{self.warp_n0}x{self.warp_k0}")
        parts.extend(
            [
                f"pad{s}{k}{d}{v}",
                f"mask={self.mask}",
                f"bias={self.bias}",
            ]
        )
        if self.lse:
            parts.append("lse=1")
        if self.dropout:
            parts.append("drop=1")
        if self.logits:
            parts.append("logits=1")
        if self.sink:
            parts.append("sink=1")
        if self.skip_min_seqlen_q:
            parts.append("skip=1")
        if self.qscale != "no":
            parts.append(f"qs={self.qscale}")
        if self.paged_kv:
            parts.append("pkv=1")
        if self.rope != "none":
            parts.append(f"rope={self.rope}")
        if self.page_size != 1:
            parts.append(f"ps={self.page_size}")
        if self.kv_memory_layout != "vectorized":
            parts.append(f"kvl={self.kv_memory_layout}")
        if self.kv_lookup_table != "sglang":
            parts.append(f"kvt={self.kv_lookup_table}")
        if self.deterministic:
            parts.append("det=1")
        if self.dbias:
            parts.append("dbias=1")
        if self.dropout_variant and self.dropout_variant != "no":
            parts.append(f"drv={self.dropout_variant}")
        # Always include block_per_cu for uniform naming
        parts.append(f"bpc={self.block_per_cu}")
        return "_".join(parts)

    def to_codegen_json(self) -> str:
        return json.dumps(
            {
                "arch": self.gfx_arch,
                "signature": {
                    "family": self.family,
                    "data_type": self.data_type,
                    "mode": self.mode,
                    "vlayout": self.vlayout,
                    "hdim_q": self.hdim_q,
                    "hdim_v": self.hdim_v,
                    "mask": self.mask,
                    "bias": self.bias,
                    "lse": self.lse,
                    "dropout": self.dropout,
                    "qscale": self.qscale,
                    "rope": self.rope,
                    "logits": self.logits,
                    "paged_kv": self.paged_kv,
                    "fp8_static_quant": False,
                    "skip_min_seqlen_q": self.skip_min_seqlen_q,
                    "sink": self.sink,
                    "dbias": self.dbias,
                    "store_randval": "storerandval" in self.dropout_variant,
                    "deterministic": self.deterministic,
                    "dropout_variant": self.dropout_variant,
                    "kv_memory_layout": self.kv_memory_layout,
                    "kv_lookup_table": self.kv_lookup_table,
                    "page_size": self.page_size,
                },
                "algorithm": {
                    "pipeline": self.pipeline,
                    "tile": list(self.tile),
                    "wave": list(self.wave),
                    "warp": list(self.warp),
                    "padding": list(self.padding),
                    "block_per_cu": self.block_per_cu,
                    "num_wave_groups": self.num_wave_groups,
                    "max_splits_log2": 0,
                    "max_seq_len_q": 0,
                    "use_trload": self.use_trload,
                },
            }
        )


# =============================================================================
# CPU reference
# =============================================================================


def _float32_to_bf16(arr: np.ndarray) -> np.ndarray:
    """Convert float32 array to bf16 stored as uint16 (truncate lower 16 bits)."""
    return arr.astype(np.float32).view(np.uint32).__rshift__(16).astype(np.uint16)


def _bf16_to_float32(arr: np.ndarray) -> np.ndarray:
    """Convert bf16 (uint16) array back to float32."""
    return (arr.astype(np.uint32) << 16).view(np.float32)


def cpu_attention_fwd(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float,
    mask_type: int = 0,
) -> np.ndarray:
    """CPU reference: scaled dot-product attention (supports GQA and causal mask).

    Args:
        Q: [batch, nhead_q, seqlen_q, hdim_q]  float32
        K: [batch, nhead_k, seqlen_k, hdim_q]  float32
        V: [batch, nhead_k, seqlen_k, hdim_v]  float32
        mask_type: 0=no mask, 1=causal top-left, 2=causal bottom-right

    Returns:
        O: [batch, nhead_q, seqlen_q, hdim_v]  float32
    """
    nhead_q = Q.shape[1]
    nhead_k = K.shape[1]
    if nhead_q != nhead_k:
        ratio = nhead_q // nhead_k
        K = np.repeat(K, ratio, axis=1)
        V = np.repeat(V, ratio, axis=1)
    S = np.matmul(Q, K.transpose(0, 1, 3, 2)) * scale
    if mask_type in (1, 2):
        sq, sk = S.shape[-2], S.shape[-1]
        row = np.arange(sq).reshape(sq, 1)
        col = np.arange(sk).reshape(1, sk)
        if mask_type == 1:  # top-left causal
            causal_mask = col <= row
        else:  # bottom-right causal
            causal_mask = col <= (row + sk - sq)
        S = np.where(causal_mask, S, -1e9)
    S_max = S.max(axis=-1, keepdims=True)
    S_exp = np.exp(S - S_max)
    P = S_exp / S_exp.sum(axis=-1, keepdims=True)
    return np.matmul(P, V)


def cpu_attention_fwd_with_intermediates(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray, scale: float
) -> tuple:
    """CPU reference forward returning (output, P) for backward use.

    Same as cpu_attention_fwd but also returns the softmax probability matrix P.
    """
    nhead_q = Q.shape[1]
    nhead_k = K.shape[1]
    if nhead_q != nhead_k:
        ratio = nhead_q // nhead_k
        K = np.repeat(K, ratio, axis=1)
        V = np.repeat(V, ratio, axis=1)
    S = np.matmul(Q, K.transpose(0, 1, 3, 2)) * scale
    S_max = S.max(axis=-1, keepdims=True)
    S_exp = np.exp(S - S_max)
    P = S_exp / S_exp.sum(axis=-1, keepdims=True)
    out = np.matmul(P, V)
    return out, P


def cpu_attention_bwd(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    out: np.ndarray,
    dO: np.ndarray,
    P: np.ndarray,
    scale: float,
) -> tuple:
    """CPU reference backward. Returns (dQ, dK, dV).

    Args:
        Q, K, V: forward inputs [batch, heads, seq, dim]
        out: forward output
        dO: gradient of output
        P: softmax probabilities from forward
        scale: attention scale factor
    """
    D = (dO * out).sum(axis=-1, keepdims=True)
    dP = np.matmul(dO, V.transpose(0, 1, 3, 2))
    dS = P * (dP - D)
    dQ = np.matmul(dS, K) * scale
    dK = np.matmul(dS.transpose(0, 1, 3, 2), Q) * scale
    dV = np.matmul(P.transpose(0, 1, 3, 2), dO)
    return dQ, dK, dV


# =============================================================================
# Low-level ctypes wrapper
# =============================================================================


class FmhaDispatcherLib:
    """Wrapper for the FMHA dispatcher shared library (libdispatcher_fmha_lib.so)."""

    SEARCH_PATHS = [
        "build/examples/libdispatcher_fmha_lib.so",
        "build/libdispatcher_fmha_lib.so",
        "build/lib/libdispatcher_fmha_lib.so",
    ]

    def __init__(self, lib: ctypes.CDLL, path: Path):
        self._lib = lib
        self.path = path
        self._setup()

    def _setup(self):
        lib = self._lib
        lib.fmha_dispatcher_initialize.argtypes = [ctypes.c_char_p]
        lib.fmha_dispatcher_initialize.restype = ctypes.c_int
        lib.fmha_dispatcher_run_fwd.argtypes = [
            ctypes.c_void_p,  # q
            ctypes.c_void_p,  # k
            ctypes.c_void_p,  # v
            ctypes.c_void_p,  # o
            ctypes.c_int,  # batch
            ctypes.c_int,  # nhead_q
            ctypes.c_int,  # nhead_k
            ctypes.c_int,  # seqlen_q
            ctypes.c_int,  # seqlen_k
            ctypes.c_int,  # hdim_q
            ctypes.c_int,  # hdim_v
            ctypes.c_float,  # scale
            ctypes.c_int,  # mask_type
            ctypes.c_int,  # bias_type
            ctypes.c_int,  # has_lse
            ctypes.c_int,  # has_dropout
            ctypes.c_int,  # traits_hdim_q (0=same as hdim_q)
            ctypes.c_int,  # traits_hdim_v (0=same as hdim_v)
            ctypes.c_int,  # is_v_rowmajor (1=row, 0=col)
            ctypes.c_int,  # perm (1=BHSD, 0=BSHD)
            ctypes.c_char_p,  # data_type ("fp16", "bf16")
            ctypes.c_int,  # is_group_mode
            ctypes.c_int,  # window_left (-1=no window)
            ctypes.c_int,  # window_right (-1=no window, 0=causal)
            ctypes.c_int,  # has_logits
            ctypes.c_int,  # has_sink
            ctypes.c_int,  # has_skip
            ctypes.POINTER(ctypes.c_float),  # time_ms_out
        ]
        lib.fmha_dispatcher_run_fwd.restype = ctypes.c_int
        lib.fmha_dispatcher_run_bwd.argtypes = [
            ctypes.c_void_p,  # q
            ctypes.c_void_p,  # k
            ctypes.c_void_p,  # v
            ctypes.c_void_p,  # o
            ctypes.c_void_p,  # lse
            ctypes.c_void_p,  # do
            ctypes.c_void_p,  # dq
            ctypes.c_void_p,  # dk
            ctypes.c_void_p,  # dv
            ctypes.c_int,  # batch
            ctypes.c_int,  # nhead_q
            ctypes.c_int,  # nhead_k
            ctypes.c_int,  # seqlen_q
            ctypes.c_int,  # seqlen_k
            ctypes.c_int,  # hdim_q
            ctypes.c_int,  # hdim_v
            ctypes.c_float,  # scale
            ctypes.c_char_p,  # data_type_str
            ctypes.c_int,  # mask_type_int
            ctypes.c_int,  # bias_type_int
            ctypes.c_int,  # has_dropout
            ctypes.c_int,  # has_dbias
            ctypes.c_int,  # is_deterministic
            ctypes.c_int,  # is_group_mode
            ctypes.c_int,  # is_store_randval
            ctypes.c_int,  # tile_n0 (kN0 for nsplits computation)
            ctypes.POINTER(ctypes.c_float),  # time_ms_out
        ]
        lib.fmha_dispatcher_run_bwd.restype = ctypes.c_int

        # Split-KV forward
        lib.fmha_dispatcher_run_splitkv.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,  # mask_type
            ctypes.c_int,  # num_splits
            ctypes.c_int,  # is_v_rowmajor
            ctypes.c_char_p,
            ctypes.c_int,  # has_lse
            ctypes.c_int,  # is_group_mode
            ctypes.c_int,  # perm
            ctypes.c_int,  # has_logits
            ctypes.c_int,  # bias_type
            ctypes.c_int,  # has_sink
            ctypes.c_int,  # paged_kv
            ctypes.c_int,  # page_block_size
            ctypes.c_int,  # window_left
            ctypes.c_int,  # window_right
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.fmha_dispatcher_run_splitkv.restype = ctypes.c_int

        # Paged-KV forward
        lib.fmha_dispatcher_run_pagedkv.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,  # mask_type
            ctypes.c_int,  # page_block_size
            ctypes.c_int,  # is_v_rowmajor
            ctypes.c_char_p,
            ctypes.c_int,  # has_lse
            ctypes.c_int,  # has_logits
            ctypes.c_int,  # has_sink
            ctypes.c_int,  # skip_min_seqlen_q
            ctypes.c_int,  # bias_type
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.fmha_dispatcher_run_pagedkv.restype = ctypes.c_int

        # Append-KV
        lib.fmha_dispatcher_run_appendkv.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,  # is_v_rowmajor
            ctypes.c_int,  # rope_type
            ctypes.c_int,  # paged_kv
            ctypes.c_int,  # page_block_size
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.fmha_dispatcher_run_appendkv.restype = ctypes.c_int

        # Batch Prefill
        lib.fmha_dispatcher_run_batch_prefill.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,  # mask_type
            ctypes.c_int,  # bias_type
            ctypes.c_int,  # page_block_size
            ctypes.c_int,  # kv_layout_int
            ctypes.c_int,  # kv_lookup_int
            ctypes.c_int,  # is_v_rowmajor
            ctypes.c_char_p,
            ctypes.c_int,  # has_lse
            ctypes.c_int,  # has_dropout
            ctypes.c_int,  # has_logits
            ctypes.c_int,  # has_sink
            ctypes.c_int,  # skip_min_seqlen_q
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.fmha_dispatcher_run_batch_prefill.restype = ctypes.c_int

        lib.fmha_dispatcher_kernel_count.argtypes = []
        lib.fmha_dispatcher_kernel_count.restype = ctypes.c_int
        lib.fmha_dispatcher_cleanup.argtypes = []
        lib.fmha_dispatcher_cleanup.restype = None

    @classmethod
    def find(cls) -> Optional["FmhaDispatcherLib"]:
        root = get_dispatcher_root()
        for rel in cls.SEARCH_PATHS:
            path = root / rel
            if path.exists():
                try:
                    lib = ctypes.CDLL(str(path))
                    return cls(lib, path)
                except OSError:
                    continue
        return None

    @classmethod
    def load(cls, path: str) -> "FmhaDispatcherLib":
        lib = ctypes.CDLL(path)
        return cls(lib, Path(path))

    def initialize(self, arch: str = "gfx950") -> bool:
        return self._lib.fmha_dispatcher_initialize(arch.encode()) == 0

    def run_bwd(
        self,
        q: ctypes.c_void_p,
        k: ctypes.c_void_p,
        v: ctypes.c_void_p,
        o: ctypes.c_void_p,
        lse: ctypes.c_void_p,
        do_grad: ctypes.c_void_p,
        dq: ctypes.c_void_p,
        dk: ctypes.c_void_p,
        dv: ctypes.c_void_p,
        prob: FmhaProblem,
        data_type: str = "fp16",
        mask_type: int = 0,
        bias_type: int = 0,
        has_dropout: bool = False,
        has_dbias: bool = False,
        is_deterministic: bool = False,
        is_group_mode: bool = False,
        is_store_randval: bool = False,
        tile_n0: int = 128,
    ) -> Tuple[int, float]:
        time_ms = ctypes.c_float(0.0)
        rc = self._lib.fmha_dispatcher_run_bwd(
            q,
            k,
            v,
            o,
            lse,
            do_grad,
            dq,
            dk,
            dv,
            prob.batch,
            prob.nhead_q,
            prob.nhead_k,
            prob.seqlen_q,
            prob.seqlen_k,
            prob.hdim_q,
            prob.hdim_v,
            prob.scale,
            data_type.encode(),
            ctypes.c_int(mask_type),
            ctypes.c_int(bias_type),
            ctypes.c_int(int(has_dropout)),
            ctypes.c_int(int(has_dbias)),
            ctypes.c_int(int(is_deterministic)),
            ctypes.c_int(int(is_group_mode)),
            ctypes.c_int(int(is_store_randval)),
            ctypes.c_int(tile_n0),
            ctypes.byref(time_ms),
        )
        return rc, time_ms.value

    def kernel_count(self) -> int:
        return self._lib.fmha_dispatcher_kernel_count()

    def cleanup(self):
        self._lib.fmha_dispatcher_cleanup()


# =============================================================================
# High-level GPU runner (mirrors GpuGroupedConvRunner)
# =============================================================================


class FmhaRunner:
    """High-level FMHA runner with NumPy interface and HIP memory management."""

    HIP_MEMCPY_H2D = 1
    HIP_MEMCPY_D2H = 2

    def __init__(self, dispatch_lib: FmhaDispatcherLib, arch: str = "gfx950"):
        self._lib = dispatch_lib
        self._arch = arch
        self._hip = None
        self._load_hip()
        if not dispatch_lib.initialize(arch):
            raise RuntimeError("Failed to initialize FMHA dispatcher")

    def _load_hip(self):
        for name in ["libamdhip64.so", "libamdhip64.so.6"]:
            try:
                self._hip = ctypes.CDLL(name)
                self._hip.hipMalloc.argtypes = [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_size_t,
                ]
                self._hip.hipMalloc.restype = ctypes.c_int
                self._hip.hipFree.argtypes = [ctypes.c_void_p]
                self._hip.hipFree.restype = ctypes.c_int
                self._hip.hipMemcpy.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_int,
                ]
                self._hip.hipMemcpy.restype = ctypes.c_int
                self._hip.hipMemset.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_size_t,
                ]
                self._hip.hipMemset.restype = ctypes.c_int
                return
            except OSError:
                continue
        raise RuntimeError("Could not load libamdhip64.so")

    @classmethod
    def from_prebuilt(cls, arch: Optional[str] = None) -> "FmhaRunner":
        arch = arch or detect_gpu_arch()
        lib = FmhaDispatcherLib.find()
        if lib is None:
            raise RuntimeError(
                "FMHA dispatcher library not found. Build with:\n"
                "  cd dispatcher/build && cmake .. -DBUILD_DISPATCHER_EXAMPLES=ON && make dispatcher_fmha_lib"
            )
        return cls(lib, arch)

    @classmethod
    def from_library(cls, path: str, arch: Optional[str] = None) -> "FmhaRunner":
        arch = arch or detect_gpu_arch()
        return cls(FmhaDispatcherLib.load(path), arch)

    def run(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        prob: FmhaProblem,
        mask_type: int = 0,
        bias_type: int = 0,
        has_lse: int = 0,
        has_dropout: int = 0,
        has_logits: int = 0,
        has_sink: int = 0,
        has_skip: int = 0,
        api_family: str = "fwd",
        data_type: str = "fp16",
        **kwargs,
    ) -> "FmhaResult":
        """Run FMHA forward on GPU with automatic HIP memory management.

        Args:
            Q: [batch, nhead_q, seqlen_q, hdim_q]  float16
            K: [batch, nhead_k, seqlen_k, hdim_q]  float16
            V: [batch, nhead_k, seqlen_k, hdim_v]  float16

        Returns:
            FmhaResult with output array, timing, TFLOPS
        """
        # Map CK dtype to numpy dtype for buffer allocation.
        # bf16 is stored as uint16 (upper 16 bits of float32).
        # fp8 uses uint8 (1 byte per element).
        _NP_DTYPE = {
            "fp16": np.float16,
            "bf16": np.uint16,
            "fp32": np.float32,
            "fp8bf16": np.uint8,
            "fp8fp32": np.uint8,
            "bf8": np.uint8,
        }
        _NP_OUT_DTYPE = {
            "fp16": np.float16,
            "bf16": np.uint16,
            "fp32": np.float32,
            "fp8bf16": np.float16,
            "fp8fp32": np.float32,
            "bf8": np.uint8,
        }
        in_dt = _NP_DTYPE.get(data_type, np.float16)
        out_dt = _NP_OUT_DTYPE.get(data_type, np.float16)
        if data_type == "bf16":
            Q_c = _float32_to_bf16(np.ascontiguousarray(Q.astype(np.float32)))
            K_c = _float32_to_bf16(np.ascontiguousarray(K.astype(np.float32)))
            V_c = _float32_to_bf16(np.ascontiguousarray(V.astype(np.float32)))
        else:
            Q_c = np.ascontiguousarray(Q.astype(in_dt))
            K_c = np.ascontiguousarray(K.astype(in_dt))
            V_c = np.ascontiguousarray(V.astype(in_dt))
        O_c = np.zeros(prob.o_shape(), dtype=out_dt)

        d_q, d_k, d_v, d_o = (ctypes.c_void_p() for _ in range(4))

        try:
            self._hip.hipMalloc(ctypes.byref(d_q), Q_c.nbytes)
            self._hip.hipMalloc(ctypes.byref(d_k), K_c.nbytes)
            self._hip.hipMalloc(ctypes.byref(d_v), V_c.nbytes)
            self._hip.hipMalloc(ctypes.byref(d_o), O_c.nbytes)

            self._hip.hipMemcpy(d_q, Q_c.ctypes.data, Q_c.nbytes, self.HIP_MEMCPY_H2D)
            self._hip.hipMemcpy(d_k, K_c.ctypes.data, K_c.nbytes, self.HIP_MEMCPY_H2D)
            self._hip.hipMemcpy(d_v, V_c.ctypes.data, V_c.nbytes, self.HIP_MEMCPY_H2D)
            self._hip.hipMemset(d_o, 0, O_c.nbytes)

            time_ms = ctypes.c_float(0.0)
            lib = self._lib._lib

            is_v_rowmajor = kwargs.get("is_v_rowmajor", 1)
            is_group_mode = kwargs.get("is_group_mode", 0)
            perm = kwargs.get("perm", 1)
            window_left = kwargs.get("window_left", -1)
            window_right = kwargs.get("window_right", -1)
            num_splits = kwargs.get("num_splits", 4)
            page_size = kwargs.get("page_size", 64)
            kv_layout = kwargs.get("kv_layout", 0)
            kv_lookup = kwargs.get("kv_lookup", 0)
            traits_hdim_q = kwargs.get("traits_hdim_q", 0)
            traits_hdim_v = kwargs.get("traits_hdim_v", 0)

            if api_family == "splitkv":
                paged_kv = kwargs.get("paged_kv", 0)
                rc = lib.fmha_dispatcher_run_splitkv(
                    d_q,
                    d_k,
                    d_v,
                    d_o,
                    prob.batch,
                    prob.nhead_q,
                    prob.nhead_k,
                    prob.seqlen_q,
                    prob.seqlen_k,
                    prob.hdim_q,
                    prob.hdim_v,
                    prob.scale,
                    mask_type,
                    num_splits,
                    is_v_rowmajor,
                    data_type.encode(),
                    has_lse,
                    is_group_mode,
                    perm,
                    has_logits,
                    bias_type,
                    has_sink,
                    paged_kv,
                    page_size,
                    window_left,
                    window_right,
                    ctypes.byref(time_ms),
                )
            elif api_family == "pagedkv":
                rc = lib.fmha_dispatcher_run_pagedkv(
                    d_q,
                    d_k,
                    d_v,
                    d_o,
                    prob.batch,
                    prob.nhead_q,
                    prob.nhead_k,
                    prob.seqlen_q,
                    prob.seqlen_k,
                    prob.hdim_q,
                    prob.hdim_v,
                    prob.scale,
                    mask_type,
                    page_size,
                    is_v_rowmajor,
                    data_type.encode(),
                    has_lse,
                    has_logits,
                    has_sink,
                    has_skip,
                    bias_type,
                    ctypes.byref(time_ms),
                )
            elif api_family == "appendkv":
                seqlen_knew = kwargs.get("seqlen_knew", prob.seqlen_k)
                rc = lib.fmha_dispatcher_run_appendkv(
                    Q_c.ctypes.data,
                    K_c.ctypes.data,
                    V_c.ctypes.data,
                    prob.batch,
                    prob.nhead_q,
                    prob.nhead_k,
                    prob.seqlen_q,
                    seqlen_knew,
                    prob.hdim_q,
                    prob.hdim_v,
                    is_v_rowmajor,
                    kwargs.get("rope_type", 0),
                    kwargs.get("paged_kv", 0),
                    page_size,
                    data_type.encode(),
                    ctypes.byref(time_ms),
                )
            elif api_family == "batch_prefill":
                skip_min_sq = kwargs.get("skip_min_seqlen_q", 0)
                rc = lib.fmha_dispatcher_run_batch_prefill(
                    d_q,
                    d_k,
                    d_v,
                    d_o,
                    prob.batch,
                    prob.nhead_q,
                    prob.nhead_k,
                    prob.seqlen_q,
                    prob.seqlen_k,
                    prob.hdim_q,
                    prob.hdim_v,
                    prob.scale,
                    mask_type,
                    bias_type,
                    page_size,
                    kv_layout,
                    kv_lookup,
                    is_v_rowmajor,
                    data_type.encode(),
                    has_lse,
                    has_dropout,
                    has_logits,
                    has_sink,
                    skip_min_sq,
                    ctypes.byref(time_ms),
                )
            else:
                rc = lib.fmha_dispatcher_run_fwd(
                    d_q,
                    d_k,
                    d_v,
                    d_o,
                    prob.batch,
                    prob.nhead_q,
                    prob.nhead_k,
                    prob.seqlen_q,
                    prob.seqlen_k,
                    prob.hdim_q,
                    prob.hdim_v,
                    prob.scale,
                    mask_type,
                    bias_type,
                    has_lse,
                    has_dropout,
                    traits_hdim_q,
                    traits_hdim_v,
                    is_v_rowmajor,
                    perm,
                    data_type.encode(),
                    is_group_mode,
                    window_left,
                    window_right,
                    has_logits,
                    has_sink,
                    has_skip,
                    ctypes.byref(time_ms),
                )

            if rc != 0:
                return FmhaResult(success=False, error=f"Kernel failed (rc={rc})")

            self._hip.hipMemcpy(O_c.ctypes.data, d_o, O_c.nbytes, self.HIP_MEMCPY_D2H)

            # Convert bf16 output (uint16) back to float32 for comparison
            if data_type == "bf16":
                O_c = _bf16_to_float32(O_c)

            # appendkv is a memory op (KV cache copy), not compute -- no TFLOPS
            ops = 0 if api_family == "appendkv" else prob.num_ops
            tflops = (
                ops / (time_ms.value * 1e-3) / 1e12
                if time_ms.value > 0 and ops > 0
                else 0.0
            )
            return FmhaResult(
                success=True, output=O_c, time_ms=time_ms.value, tflops=tflops
            )

        finally:
            for d in [d_q, d_k, d_v, d_o]:
                if d.value:
                    self._hip.hipFree(d)

    def run_bwd(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        out: np.ndarray,
        LSE: np.ndarray,
        dO: np.ndarray,
        prob: FmhaProblem,
        data_type: str = "fp16",
        mask_type: int = 0,
        bias_type: int = 0,
        has_dropout: bool = False,
        has_dbias: bool = False,
        is_deterministic: bool = False,
        is_group_mode: bool = False,
        is_store_randval: bool = False,
        tile_n0: int = 128,
    ) -> "FmhaResult":
        """Run FMHA backward on GPU with automatic HIP memory management.

        Returns FmhaResult with dQ, dK, dV packed in output as a tuple.
        """
        _NP_DTYPE = {
            "fp16": np.float16,
            "bf16": np.float16,
            "fp32": np.float32,
            "fp8bf16": np.uint8,
            "fp8fp32": np.uint8,
            "bf8": np.uint8,
        }
        in_dt = _NP_DTYPE.get(data_type, np.float16)
        Q_c = np.ascontiguousarray(Q.astype(in_dt))
        K_c = np.ascontiguousarray(K.astype(in_dt))
        V_c = np.ascontiguousarray(V.astype(in_dt))
        O_c = np.ascontiguousarray(out.astype(in_dt))
        LSE_c = np.ascontiguousarray(LSE.astype(np.float32))
        dO_c = np.ascontiguousarray(dO.astype(in_dt))
        dQ_c = np.zeros_like(Q_c)
        dK_c = np.zeros_like(K_c)
        dV_c = np.zeros_like(V_c)

        ptrs = [ctypes.c_void_p() for _ in range(9)]
        d_q, d_k, d_v, d_o, d_lse, d_do, d_dq, d_dk, d_dv = ptrs

        try:
            for d, arr in zip(ptrs[:6], [Q_c, K_c, V_c, O_c, LSE_c, dO_c]):
                self._hip.hipMalloc(ctypes.byref(d), arr.nbytes)
                self._hip.hipMemcpy(d, arr.ctypes.data, arr.nbytes, self.HIP_MEMCPY_H2D)
            for d, arr in zip(ptrs[6:], [dQ_c, dK_c, dV_c]):
                self._hip.hipMalloc(ctypes.byref(d), arr.nbytes)
                self._hip.hipMemset(d, 0, arr.nbytes)

            rc, elapsed = self._lib.run_bwd(
                d_q,
                d_k,
                d_v,
                d_o,
                d_lse,
                d_do,
                d_dq,
                d_dk,
                d_dv,
                prob,
                data_type,
                mask_type=mask_type,
                bias_type=bias_type,
                has_dropout=has_dropout,
                has_dbias=has_dbias,
                is_deterministic=is_deterministic,
                is_group_mode=is_group_mode,
                is_store_randval=is_store_randval,
                tile_n0=tile_n0,
            )

            if rc != 0:
                return FmhaResult(success=False, error=f"BWD kernel failed (rc={rc})")

            for d, arr in zip(ptrs[6:], [dQ_c, dK_c, dV_c]):
                self._hip.hipMemcpy(arr.ctypes.data, d, arr.nbytes, self.HIP_MEMCPY_D2H)

            tflops = prob.num_ops / (elapsed * 1e-3) / 1e12 if elapsed > 0 else 0.0
            return FmhaResult(
                success=True,
                output=(dQ_c, dK_c, dV_c),
                time_ms=elapsed,
                tflops=tflops,
            )
        finally:
            for d in ptrs:
                if d.value:
                    self._hip.hipFree(d)

    @property
    def kernel_count(self) -> int:
        return self._lib.kernel_count()

    @property
    def library_path(self) -> str:
        return str(self._lib.path)

    def cleanup(self):
        self._lib.cleanup()


# =============================================================================
# JIT Build Support (mirrors setup_multiple_gemm_dispatchers)
# =============================================================================


@dataclass
class FmhaSetupResult:
    success: bool
    config: Optional[FmhaKernelConfig] = None
    runner: Optional[FmhaRunner] = None
    library_path: str = ""
    error: str = ""
    build_time_s: float = 0.0


def _build_static_lib(root: Path) -> Optional[Path]:
    """Build libck_tile_dispatcher.a via cmake if not already present."""
    build_dir = root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    hipcc = _find_hipcc()
    cmake_cmd = ["cmake", str(root), f"-DCMAKE_CXX_COMPILER={hipcc}"]
    r = subprocess.run(cmake_cmd, cwd=str(build_dir), capture_output=True, text=True)
    if r.returncode != 0:
        print(
            f"Warning: cmake failed for dispatcher lib: {r.stderr[:200]}",
            file=sys.stderr,
        )
        return None
    make_cmd = ["make", "ck_tile_dispatcher", f"-j{os.cpu_count() or 4}"]
    r = subprocess.run(make_cmd, cwd=str(build_dir), capture_output=True, text=True)
    if r.returncode != 0:
        print(
            f"Warning: make failed for dispatcher lib: {r.stderr[:200]}",
            file=sys.stderr,
        )
        return None
    lib_path = build_dir / "libck_tile_dispatcher.a"
    return lib_path if lib_path.exists() else None


def _find_static_lib() -> Optional[Path]:
    root = get_dispatcher_root()
    for rel in ["build/libck_tile_dispatcher.a", "build/lib/libck_tile_dispatcher.a"]:
        p = root / rel
        if p.exists():
            return p
    # Auto-build if not found
    print("  Building libck_tile_dispatcher.a (first time)...", file=sys.stderr)
    return _build_static_lib(root)


def _find_hipcc() -> str:
    for path in ["/opt/rocm/bin/hipcc", "/usr/bin/hipcc"]:
        if os.path.exists(path):
            return path
    return "hipcc"


def fmha_compile_flags(arch: str, hipcc: str = "", family: str = "") -> List[str]:
    """Base hipcc flags for compiling FMHA kernels. Shared by JIT and tile engine.

    Source: example/ck_tile/01_fmha/CMakeLists.txt — mirrors CK's own build
    flags to ensure parity.  Key defines:
    - CK_TILE_FMHA_FWD_FAST_EXP2: enables fast exp2 on gfx9 (CDNA)
    - CK_TILE_USE_OCP_FP8: uses OCP standard fp8 format
    - CK_GFX950_SUPPORT / CK_USE_GFX950: enables gfx950-specific code paths
    - CK_USE_XDL: enables MFMA (matrix fused multiply-add) instructions
    - CK_TILE_USE_WMMA: 0 for CDNA (uses MFMA instead)
    - CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3: BWD bf16 conversion mode
    """
    if not hipcc:
        hipcc = _find_hipcc()
    root = get_dispatcher_root()
    flags = [
        hipcc,
        "-c",
        "-fPIC",
        "-O3",
        "-DNDEBUG",
        f"--offload-arch={arch}",
        "-std=c++17",
        f"-I{root.parent / 'include'}",
        f"-I{root / 'include'}",
        f"-I{root.parent}",
        "-Wno-undefined-func-template",
        "-Wno-float-equal",
        "-fgpu-flush-denormals-to-zero",
        "-fno-offload-uniform-block",
        "-mllvm",
        "--lsr-drop-solution=1",
        "-mllvm",
        "-enable-post-misched=0",
        "-mllvm",
        "-amdgpu-early-inline-all=true",
        "-mllvm",
        "-amdgpu-function-calls=false",
    ]
    if arch.startswith("gfx9"):
        flags.append("-DCK_TILE_FMHA_FWD_FAST_EXP2=1")
        flags.append("-DCK_TILE_USE_OCP_FP8")
        flags.append("-DCK_GFX950_SUPPORT")
        flags.append("-DCK_USE_GFX950")
        flags.append("-DCK_USE_GFX94")
        flags.append("-DCK_USE_XDL")
        flags.append("-DCK_TILE_USE_WMMA=0")
    else:
        flags.append("-DCK_TILE_FMHA_FWD_FAST_EXP2=0")

    # API enablement flags (match CMakeLists.txt conditional defines)
    flags.append("-DCK_TILE_FMHA_FWD_SPLITKV_API=1")
    flags.append("-DCK_TILE_FMHA_FWD_APPENDKV_API=1")
    flags.append("-DCK_TILE_FMHA_FWD_PAGEDKV_API=1")

    # BWD-specific flags
    if family.startswith("bwd"):
        flags.append("-DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3")

    return flags


def _make_splitkv_combine_config(splitkv_cfg: FmhaKernelConfig) -> FmhaKernelConfig:
    """Create a matching fwd_splitkv_combine config for a fwd_splitkv config.

    Source: fmha_fwd.py splitkv_combine tile — fixed (32, hdim_v, 32, 32) tile.
    The combine_bn1=32 comes from specs.py load_arch_specs() splitkv_combine dict.
    The combine kernel merges partial results from the split stage into the
    final output.  Must be in the same .so as the split kernel for the
    2-stage splitkv pipeline.
    """
    import copy

    comb = copy.copy(splitkv_cfg)
    comb.family = "fwd_splitkv_combine"
    comb.pipeline = "splitkv_combine"
    hv = splitkv_cfg.hdim_v
    comb.hdim_q = hv
    comb.hdim_v = hv
    comb.tile_m0 = 32
    comb.tile_n0 = hv
    comb.tile_k0 = 32
    comb.tile_n1 = 32
    comb.tile_k1 = 0
    comb.tile_k0max = 0
    comb.pad_s = 1 if splitkv_cfg.mode == "group" else 0
    comb.pad_sk = 1
    comb.pad_d = 1
    comb.pad_dv = 1
    comb.lse = True
    # Combine doesn't use mask/bias/etc., but the dispatcher's supports() check
    # matches the combine kernel's signature against the problem traits.
    # Keep them from the split config so the signatures match.
    comb.dropout = False
    comb.skip_min_seqlen_q = False
    comb.qscale = "no"
    comb.rope = "none"
    return comb


def _make_bwd_dot_do_o_config(dq_cfg: FmhaKernelConfig) -> FmhaKernelConfig:
    """Create a matching bwd_dot_do_o config for a bwd_dq_dk_dv config.

    Source: fmha_bwd.py FmhaBwdDotDoOTileSize — fixed tile (64, max(hv,128), 32).
    Warp tile (32,32,16) with 4 waves in M = standard fp16/bf16 MFMA config.
    The dot_do_o kernel computes d = rowsum(O * dO) and must be in the same
    .so as the dq_dk_dv kernel for the 2-stage BWD pipeline.
    """
    import copy

    dot = copy.copy(dq_cfg)
    dot.family = "bwd_dot_do_o"
    dot.pipeline = "qr"
    hq, hv = dq_cfg.hdim_q, dq_cfg.hdim_v
    dot.tile_m0 = 64
    dot.tile_n0 = max(hv, 128)
    dot.tile_k0 = 32
    dot.tile_n1 = max(hv, 128)
    dot.tile_k1 = 32
    dot.tile_k0max = max(hq, 128)
    dot.wave_m0 = 4
    dot.wave_n0 = 1
    dot.wave_k0 = 1
    dot.wave_m1 = 4
    dot.wave_n1 = 1
    dot.wave_k1 = 1
    dot.warp_m0 = 32
    dot.warp_n0 = 32
    dot.warp_k0 = 16
    dot.warp_m1 = 32
    dot.warp_n1 = 32
    dot.warp_k1 = 16
    dot.use_trload = False
    # dot_do_o uses all-padded for maximum compatibility
    dot.pad_s = 1
    dot.pad_sk = 1
    dot.pad_d = 1
    dot.pad_dv = 1
    # BWD traits don't have logits/sink/skip/lse/paged_kv -- from_invocation
    # defaults them to false/0. The dot_do_o signature must match these defaults.
    dot.logits = False
    dot.sink = False
    dot.skip_min_seqlen_q = False
    dot.lse = False
    dot.paged_kv = False
    dot.qscale = "no"
    dot.rope = "no"
    # dot_do_o must match the problem's is_store_randval (from traits);
    # keep dropout_variant as-is so store_randval matches
    return dot


def setup_fmha_dispatcher(
    config: FmhaKernelConfig,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
) -> FmhaSetupResult:
    """JIT-compile a single FMHA kernel and return a runner.

    Cached: if the .so already exists, loads it directly (~1ms).
    Fresh build: codegen → parallel compile (kernel + ctypes) → link.
    """
    import time

    t0 = time.perf_counter()

    root = get_dispatcher_root()
    codegen_dir = root / "codegen"
    ctypes_src = root / "bindings" / "ctypes" / "fmha_ctypes_lib.cpp"
    static_lib = _find_static_lib()
    hipcc = _find_hipcc()

    if output_dir is None:
        output_dir = root / "build" / "examples" / f"fmha_jit_{config.name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    lib_name = f"libdispatcher_fmha_{config.name}.so"
    lib_path = output_dir / lib_name

    # Cache hit: .so already exists, just load
    if lib_path.exists():
        try:
            runner = FmhaRunner.from_library(str(lib_path), config.gfx_arch)
            return FmhaSetupResult(
                success=True,
                config=config,
                runner=runner,
                library_path=str(lib_path),
                build_time_s=time.perf_counter() - t0,
            )
        except Exception:
            pass  # stale .so, rebuild

    if not static_lib:
        return FmhaSetupResult(
            success=False, config=config, error="libck_tile_dispatcher.a not found"
        )
    if not ctypes_src.exists():
        return FmhaSetupResult(
            success=False, config=config, error="fmha_ctypes_lib.cpp not found"
        )

    # Step 1: Codegen
    # BWD dq_dk_dv needs a matching dot_do_o kernel in the same .so
    # BWD dq_dk_dv needs matching dot_do_o kernel for the 2-stage pipeline
    if config.family == "bwd_dq_dk_dv":
        dot_cfg = _make_bwd_dot_do_o_config(config)
        config_json_str = json.dumps(
            [
                json.loads(dot_cfg.to_codegen_json()),
                json.loads(config.to_codegen_json()),
            ]
        )
    else:
        config_json_str = config.to_codegen_json()
    gen_cmd = [
        sys.executable,
        str(codegen_dir / "fmha" / "generate_fallback.py"),
        "--output-dir",
        str(output_dir),
        "--gpu-target",
        config.gfx_arch,
        "--config-json",
        config_json_str,
    ]
    r = subprocess.run(gen_cmd, capture_output=True, text=True, cwd=str(codegen_dir))
    if r.returncode != 0:
        return FmhaSetupResult(
            success=False, config=config, error=f"Codegen failed: {r.stderr[:500]}"
        )

    dispatch_header = output_dir / "fmha_python_dispatch.hpp"
    if not dispatch_header.exists():
        return FmhaSetupResult(
            success=False, config=config, error="Dispatch header not generated"
        )

    # Step 2: Compile kernel .cpp AND ctypes in parallel
    kernel_cpps = list(output_dir.glob("fmha_*.cpp"))
    base_flags = fmha_compile_flags(config.gfx_arch, hipcc, family=config.family)

    compile_jobs = []
    for cpp in kernel_cpps:
        obj = cpp.with_suffix(".o")
        compile_jobs.append((base_flags + [str(cpp), "-o", str(obj)], obj, "kernel"))

    ctypes_obj = output_dir / "fmha_ctypes_lib.o"
    ctypes_cmd = base_flags + [
        f"-I{output_dir}",
        f"-I{output_dir / 'dispatcher_wrappers'}",
        f"-include{dispatch_header}",
        f'-DGFX_ARCH="{config.gfx_arch}"',
        str(ctypes_src),
        "-o",
        str(ctypes_obj),
    ]
    compile_jobs.append((ctypes_cmd, ctypes_obj, "ctypes"))

    def _run_compile(job):
        cmd, obj, label = job
        if obj.exists():
            return (True, obj, label, "")
        r = subprocess.run(cmd, capture_output=True, text=True)
        return (r.returncode == 0, obj, label, r.stderr[:500])

    with ThreadPoolExecutor(max_workers=len(compile_jobs)) as pool:
        results = list(pool.map(_run_compile, compile_jobs))

    kernel_objs = []
    for ok, obj, label, err in results:
        if not ok:
            return FmhaSetupResult(
                success=False,
                config=config,
                error=f"{label} compile failed: {err}",
            )
        if label == "kernel":
            kernel_objs.append(str(obj))

    # Step 3: Link
    link_cmd = [
        hipcc,
        "-shared",
        "-fPIC",
        str(ctypes_obj),
        *kernel_objs,
        str(static_lib),
        "-o",
        str(lib_path),
    ]
    r = subprocess.run(link_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return FmhaSetupResult(
            success=False, config=config, error=f"Link failed: {r.stderr[:500]}"
        )

    # Step 4: Load
    try:
        runner = FmhaRunner.from_library(str(lib_path), config.gfx_arch)
    except Exception as e:
        return FmhaSetupResult(success=False, config=config, error=f"Load failed: {e}")

    elapsed = time.perf_counter() - t0
    return FmhaSetupResult(
        success=True,
        config=config,
        runner=runner,
        library_path=str(lib_path),
        build_time_s=elapsed,
    )


def _run_compile_job(job):
    """Module-level compile worker -- no threads, uses file-based stderr."""
    cmd, obj_str, name, label = job
    if os.path.exists(obj_str):
        return (name, True, "")
    err_path = obj_str + ".err"
    with open(err_path, "w") as ef:
        rc = subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=ef)
    if rc != 0:
        try:
            err = open(err_path).read()[:200]
        except Exception:
            err = f"rc={rc}"
        return (name, False, err)
    try:
        os.unlink(err_path)
    except OSError:
        pass
    return (name, True, "")


def setup_multiple_fmha_dispatchers(
    configs: List[FmhaKernelConfig],
    output_dir: Optional[Path] = None,
    verbose: bool = False,
    max_workers: Optional[int] = None,
    executor=None,
    progress_callback=None,
) -> List[FmhaSetupResult]:
    """3-stage pipelined JIT: codegen(parallel) -> compile(parallel) -> link+load(parallel).

    Faster than calling setup_fmha_dispatcher() per-kernel because all hipcc
    compile jobs (kernel + ctypes from ALL kernels) share one thread pool.
    """
    if not configs:
        return []

    root = get_dispatcher_root()
    codegen_dir = root / "codegen"
    ctypes_src = root / "bindings" / "ctypes" / "fmha_ctypes_lib.cpp"
    static_lib = _find_static_lib()
    hipcc = _find_hipcc()
    arch = configs[0].gfx_arch

    if output_dir is None:
        output_dir = root / "build" / "examples"

    results: dict[str, FmhaSetupResult] = {}

    # --- Stage 1: Codegen (sequential, skip cached) ---
    def _codegen(cfg):
        out = output_dir / f"fmha_jit_{cfg.name}"
        lib_path = out / f"libdispatcher_fmha_{cfg.name}.so"
        # Fast path: .so exists, register result and skip
        if lib_path.exists():
            results[cfg.name] = FmhaSetupResult(
                success=True, config=cfg, library_path=str(lib_path)
            )
            return (cfg.name, cfg, out, True)
        # Fast path: previous codegen already failed (no .hpp generated)
        if out.exists() and not (out / "fmha_python_dispatch.hpp").exists():
            err_file = out / "_codegen_err.txt"
            if err_file.exists():
                results[cfg.name] = FmhaSetupResult(
                    success=False, config=cfg, error="Codegen failed (cached)"
                )
                return (cfg.name, cfg, out, False)
        out.mkdir(parents=True, exist_ok=True)
        # Check if codegen was already done (has .hpp but no .so yet)
        if (out / "fmha_python_dispatch.hpp").exists():
            return (cfg.name, cfg, out, True)
        if cfg.family == "bwd_dq_dk_dv":
            dot = _make_bwd_dot_do_o_config(cfg)
            config_json_str = json.dumps(
                [
                    json.loads(dot.to_codegen_json()),
                    json.loads(cfg.to_codegen_json()),
                ]
            )
        elif cfg.family == "fwd_splitkv":
            comb = _make_splitkv_combine_config(cfg)
            config_json_str = json.dumps(
                [
                    json.loads(cfg.to_codegen_json()),
                    json.loads(comb.to_codegen_json()),
                ]
            )
        else:
            config_json_str = cfg.to_codegen_json()
        err_file = out / "_codegen_err.txt"
        with open(err_file, "w") as ef:
            rc = subprocess.call(
                [
                    sys.executable,
                    str(codegen_dir / "fmha" / "generate_fallback.py"),
                    "--output-dir",
                    str(out),
                    "--gpu-target",
                    cfg.gfx_arch,
                    "--config-json",
                    config_json_str,
                ],
                stdout=subprocess.DEVNULL,
                stderr=ef,
                cwd=str(codegen_dir),
            )
        ok = rc == 0 and (out / "fmha_python_dispatch.hpp").exists()
        if not ok:
            err_msg = err_file.read_text()[:200] if err_file.exists() else "unknown"
            results[cfg.name] = FmhaSetupResult(
                success=False, config=cfg, error=f"Codegen failed: {err_msg}"
            )
        return (cfg.name, cfg, out, ok)

    codegen_results = []
    for i, cfg in enumerate(configs):
        codegen_results.append(_codegen(cfg))
        if progress_callback:
            progress_callback("codegen", i + 1, len(configs))

    # --- Stage 2: Collect ALL compile jobs, run in one pool ---
    # Use bwd family flag to get the superset of all flags (includes BWD-specific defines)
    base_flags = fmha_compile_flags(arch, hipcc, family="bwd")
    compile_jobs = []  # (cmd, obj_path, kernel_name, label)

    config_dirs: dict[str, tuple[FmhaKernelConfig, Path]] = {}
    for name, cfg, out, ok in codegen_results:
        if not ok or name in results:
            continue
        config_dirs[name] = (cfg, out)
        for cpp in out.glob("fmha_*.cpp"):
            obj = cpp.with_suffix(".o")
            if not obj.exists():
                compile_jobs.append(
                    (base_flags + [str(cpp), "-o", str(obj)], str(obj), name, "kernel")
                )
        ctypes_obj = out / "fmha_ctypes_lib.o"
        if not ctypes_obj.exists():
            dispatch = out / "fmha_python_dispatch.hpp"
            compile_jobs.append(
                (
                    base_flags
                    + [
                        f"-I{out}",
                        f"-I{out / 'dispatcher_wrappers'}",
                        f"-include{dispatch}",
                        f'-DGFX_ARCH="{arch}"',
                        str(ctypes_src),
                        "-o",
                        str(ctypes_obj),
                    ],
                    str(ctypes_obj),
                    name,
                    "ctypes",
                )
            )

    failed_names: set = set()

    if compile_jobs:
        _own_pool = None
        _pool = executor
        if _pool is None:
            workers = max_workers or min(len(compile_jobs), os.cpu_count() or 4)
            _own_pool = ProcessPoolExecutor(max_workers=workers)
            _pool = _own_pool
        try:
            done_count = 0
            total_jobs = len(compile_jobs)
            for name, ok, err in _pool.map(_run_compile_job, compile_jobs):
                done_count += 1
                if progress_callback:
                    progress_callback("compile", done_count, total_jobs)
                if not ok:
                    failed_names.add(name)
                    if name not in results:
                        cfg, _ = config_dirs[name]
                        results[name] = FmhaSetupResult(
                            success=False, config=cfg, error=f"Compile: {err}"
                        )
        finally:
            if _own_pool is not None:
                _own_pool.shutdown(wait=True)

    # --- Stage 3: Link (no GPU access -- runner loading deferred to caller) ---
    def _link(item):
        name, (cfg, out) = item
        if name in failed_names or name in results:
            return
        objs = list(out.glob("*.o"))
        lib_path = out / f"libdispatcher_fmha_{name}.so"
        if not lib_path.exists():
            r = subprocess.run(
                [
                    hipcc,
                    "-shared",
                    "-fPIC",
                    *[str(o) for o in objs],
                    str(static_lib),
                    "-o",
                    str(lib_path),
                ],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                results[name] = FmhaSetupResult(
                    success=False, config=cfg, error=f"Link: {r.stderr[:200]}"
                )
                return
        results[name] = FmhaSetupResult(
            success=True, config=cfg, library_path=str(lib_path)
        )

    for item in config_dirs.items():
        _link(item)

    # Return in original order
    return [
        results.get(c.name, FmhaSetupResult(success=False, config=c, error="skipped"))
        for c in configs
    ]


# =============================================================================
# Registry (mirrors ctypes_utils.Registry)
# =============================================================================


class FmhaRegistry:
    """Kernel registry with parallel JIT build support."""

    def __init__(self, name: str = "fmha"):
        self._name = name
        self._kernels: List[FmhaKernelConfig] = []

    def register_kernel(self, config: FmhaKernelConfig):
        self._kernels.append(config)

    def __len__(self):
        return len(self._kernels)

    def build(
        self,
        verbose: bool = False,
        max_workers: Optional[int] = None,
    ) -> List[FmhaSetupResult]:
        return setup_multiple_fmha_dispatchers(
            self._kernels,
            verbose=verbose,
            max_workers=max_workers,
        )


# =============================================================================
# Validator (mirrors ctypes_utils.Validator)
# =============================================================================


class FmhaValidator:
    """Validates FMHA GPU output against a reference.

    Usage:
        validator = FmhaValidator(rtol=1e-2, atol=1e-2)
        ok, max_abs, max_rel = validator.check(gpu_output, cpu_reference)
    """

    def __init__(self, rtol: float = 1e-2, atol: float = 1e-2):
        self.rtol = rtol
        self.atol = atol

    def check(
        self, output: np.ndarray, reference: np.ndarray
    ) -> Tuple[bool, float, float]:
        """Check output against reference.

        Returns:
            (is_valid, max_abs_error, max_rel_error)
        """
        out_f32 = output.astype(np.float32)
        ref_f32 = reference.astype(np.float32)
        diff = np.abs(out_f32 - ref_f32)
        max_abs = float(diff.max())
        max_rel = float((diff / (np.abs(ref_f32) + 1e-6)).max())
        ok = bool(np.allclose(out_f32, ref_f32, atol=self.atol, rtol=self.rtol))
        return ok, max_abs, max_rel


# =============================================================================
# KernelSpec + spec_to_config (mirrors ctypes_utils.KernelSpec)
# =============================================================================


@dataclass
class FmhaKernelSpec:
    """High-level kernel specification for easy declaration.

    Mirrors GEMM's KernelSpec: specify name + key dimensions, get a
    full FmhaKernelConfig via spec_to_config().
    """

    name: str
    hdim: int = 128
    pipeline: str = "qr_async"
    # Stage 0 tile (Q*K^T)
    tile_m0: int = 128
    tile_n0: int = 128
    tile_k0: int = 32


def spec_to_config(
    spec: FmhaKernelSpec, dtype: str = "fp16", arch: str = "gfx950"
) -> FmhaKernelConfig:
    """Convert a high-level FmhaKernelSpec to a full FmhaKernelConfig."""
    hdim = spec.hdim
    return FmhaKernelConfig(
        data_type=dtype,
        hdim_q=hdim,
        hdim_v=hdim,
        pipeline=spec.pipeline,
        tile_m0=spec.tile_m0,
        tile_n0=spec.tile_n0,
        tile_k0=spec.tile_k0,
        tile_n1=hdim,
        tile_k1=spec.tile_k0,
        tile_k0max=hdim,
        gfx_arch=arch,
    )


# =============================================================================
# Split-K heuristic (from fmhaarch.md Section 9.5)
# =============================================================================
