# SPDX-License-Identifier: MIT
# Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import asdict, dataclass
from typing import Optional, Tuple


@dataclass
class GEMM:
    """
    A python dataclass storing the template parameters of a CK Universal Gemm template instance
    """

    a_layout: str
    b_layout: str
    ds_layouts: Tuple[str]  # addmm specific
    c_layout: str

    a_element_dtype: str
    b_element_dtype: str
    ds_element_dtypes: Tuple[str]  # addmm specific
    c_element_dtype: str

    acc_dtype: str
    c_shuffle_dtype: str

    a_elementwise_op: str
    b_elementwise_op: str
    c_elementwise_op: str

    gemm_specialization: str

    block_size: int

    m_per_block: int
    n_per_block: int
    k_per_block: int

    a_k1: int
    b_k1: int

    m_per_xdl: int
    n_per_xdl: int

    m_xdl_per_wave: int
    n_xdl_per_wave: int

    a_block_transfer_thread_cluster_lengths_ak0_m_ak1: Tuple[int, int, int]
    a_block_transfer_thread_cluster_arrange_order: Tuple[int, int, int]
    a_block_transfer_src_access_order: Tuple[int, int, int]
    a_block_transfer_src_vector_dim: int
    a_block_transfer_src_scalar_per_vector: int
    a_block_transfer_dst_scalar_per_vector_ak1: int
    a_block_lds_extra_m: bool

    b_block_transfer_thread_cluster_lengths_bk0_n_bk1: Tuple[int, int, int]
    b_block_transfer_thread_cluster_arrange_order: Tuple[int, int, int]
    b_block_transfer_src_access_order: Tuple[int, int, int]

    b_block_transfer_src_vector_dim: int
    b_block_transfer_src_scalar_per_vector: int
    b_block_transfer_dst_scalar_per_vector_bk1: int
    b_block_lds_extra_n: bool

    c_shuffle_m_xdl_per_wave_per_shuffle: int
    c_shuffle_n_xdl_per_wave_per_shuffle: int

    c_shuffle_block_transfer_cluster_lengths_m_block_m_per_block_n_block_n_per_block: (
        Tuple[int, int, int, int]
    )
    c_shuffle_block_transfer_scalar_per_vector_n_per_block: int
    block_gemm_pipeline_scheduler: str
    block_gemm_pipeline_version: str

    a_compute_dtype: Optional[str] = None
    b_compute_dtype: Optional[str] = None

    def name(self):
        # cpp alias for template instance
        return f"ck_devicegemm_multid_xdl_cshuffle_v3_{self.key_name()}"

    def layout(self):
        return "".join([l[0] for l in (self.a_layout, self.b_layout, self.c_layout, *self.ds_layouts)])

    def dtype(self):
        return "".join([t for t in (self.a_element_dtype, self.b_element_dtype, self.c_element_dtype, *self.ds_element_dtypes)])

    def tiles(self):
        return "_".join([
            "block",
            "x".join(map(str, [self.m_per_block, self.n_per_block, self.k_per_block])),
            "warp",
            "x".join(map(str, [self.m_xdl_per_wave, self.n_xdl_per_wave])),
            "core",
            "x".join(map(str, [self.m_per_xdl, self.n_per_xdl])),
            "ks",
            "x".join(map(str, [self.a_k1, self.b_k1, *self.a_block_transfer_thread_cluster_lengths_ak0_m_ak1]))
        ])

    def short_name(self):
        return f"ck_gemm_{self.layout().lower()}_{self.dtype().lower()}_{self.tiles()}"

    def key_name(self):
        # TBD; must be unique per instance. Intended to use as dict key
        return "_".join(
            [
                "K"
                + field_name.replace("_", "").lower()
                + "V"
                + (
                    "x".join(map(str, iter(field_value)))
                    if isinstance(field_value, tuple)
                    else str(field_value).replace(":", "")
                )
                for field_name, field_value in self.dict_items()
            ]
        )

    def dict_items(self):
        return asdict(self).items()


_test_instance = GEMM(
    a_layout="Row",
    b_layout="Col",
    c_layout="Row",
    ds_element_dtypes=tuple(),
    ds_layouts=tuple(),
    a_element_dtype="F16",
    b_element_dtype="F16",
    c_element_dtype="F16",
    acc_dtype="F32",
    c_shuffle_dtype="F16",
    a_elementwise_op="PassThrough",
    b_elementwise_op="PassThrough",
    c_elementwise_op="PassThrough",
    k_per_block=64,
    a_k1=8,
    b_k1=8,
    a_block_transfer_thread_cluster_arrange_order=(1, 0, 2),
    a_block_transfer_src_access_order=(1, 0, 2),
    a_block_transfer_src_vector_dim=2,
    a_block_transfer_src_scalar_per_vector=8,
    a_block_transfer_dst_scalar_per_vector_ak1=8,
    a_block_lds_extra_m=0,
    b_block_transfer_thread_cluster_arrange_order=(1, 0, 2),
    b_block_transfer_src_access_order=(1, 0, 2),
    b_block_transfer_src_vector_dim=2,
    b_block_transfer_src_scalar_per_vector=8,
    b_block_transfer_dst_scalar_per_vector_bk1=8,
    b_block_lds_extra_n=0,
    a_compute_dtype="F16",
    b_compute_dtype="F16",
    gemm_specialization="GemmSpecialization::MNKPadding",
    m_per_block=224,
    n_per_block=256,
    m_per_xdl=16,
    n_per_xdl=16,
    m_xdl_per_wave=7,
    n_xdl_per_wave=8,
    c_shuffle_m_xdl_per_wave_per_shuffle=1,
    c_shuffle_n_xdl_per_wave_per_shuffle=2,
    block_gemm_pipeline_scheduler="BlockGemmPipelineScheduler::Intrawave",
    block_gemm_pipeline_version="BlockGemmPipelineVersion::v3",
    block_size=256,
    a_block_transfer_thread_cluster_lengths_ak0_m_ak1=(8, 32, 1),
    b_block_transfer_thread_cluster_lengths_bk0_n_bk1=(8, 32, 1),
    c_shuffle_block_transfer_cluster_lengths_m_block_m_per_block_n_block_n_per_block=(
        1,
        32,
        1,
        8,
    ),
    c_shuffle_block_transfer_scalar_per_vector_n_per_block=8,
)