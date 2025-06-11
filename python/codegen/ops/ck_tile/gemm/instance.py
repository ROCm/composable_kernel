# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass, asdict
from typing import Dict, Tuple


@dataclass
class GEMM:
    layout_a: str
    layout_b: str
    layout_c: str

    datatype_a: str
    datatype_b: str
    datatype_c: str

    tile_m: int
    tile_n: int
    tile_k: int

    warp_m: int
    warp_n: int
    warp_k: int

    warp_tile_m: int
    warp_tile_n: int
    warp_tile_k: int

    m_is_padded: str
    n_is_padded: str
    k_is_padded: str

    pipeline: str
    scheduler: str
    epilogue: str

    def layout_repr(self):
        return f"{self.layout_a[0]}{self.layout_b[0]}{self.layout_c[0]}"

    def dtype_repr(self):
        return f"{self.datatype_a}{self.datatype_b}{self.datatype_c}"

    def tile_sizes(self):
        return "_".join(
            [
                f"{self.tile_m}{self.tile_n}{self.tile_k}",
                f"{self.warp_m}{self.warp_n}{self.warp_k}",
                f"{self.warp_tile_m}{self.warp_tile_n}{self.warp_tile_k}",
            ]
        )

    def name(self):
        return "ck_tile_gemm_universal_" + "_".join(
            [
                f"{self.layout_repr()}",
                f"{self.dtype_repr()}",
                f"{self.tile_sizes()}",
                f"{self.pipeline}",
                f"{self.scheduler}",
                f"{self.epilogue}",
            ]
        )

    def dict_items(self):
        return asdict(self).items()


_test_instance = GEMM(
    layout_a="Row",
    layout_b="Col",
    layout_c="Row",
    datatype_a="BF16",
    datatype_b="BF16",
    datatype_c="BF16",
    tile_m=256,
    tile_n=256,
    tile_k=64,
    warp_m=2,
    warp_n=2,
    warp_k=1,
    warp_tile_m=32,
    warp_tile_n=32,
    warp_tile_k=16,
    m_is_padded="false",
    n_is_padded="false",
    k_is_padded="false",
    pipeline="CompV3",
    scheduler="Intrawave",
    epilogue="Default",
)
