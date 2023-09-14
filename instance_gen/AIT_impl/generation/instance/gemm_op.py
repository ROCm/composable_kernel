# take in input for gemm from user, send it to example template
# the structure for constructing this gemm op was taken from AIT's 
# implementation of creating a gemm op
import enum
import ck_types
from copy import deepcopy
from dataclasses import dataclass
from enum import auto
from typing import List
from ck_types import *


@dataclass
class TileDesc:
    block_size: int
    m_per_block: int
    n_per_block: int
    k_per_block: int
    ak1: int
    bk1: int
    m_per_XDL: int
    n_per_XDL: int
    m_Xdl_per_wave: int
    n_Xdl_per_wave: int
    num_gemmk_prefetch_stage: int

    def __str__(self) -> str:
        values = list(self.__dict__.values())

@dataclass
class BlockTransferDesc:
    thread_cluster_length: str
    thread_cluster_arrange_order: str
    src_access_order: str
    src_vec_dim: int
    src_scalar_per_vector: int
    dst_scalar_per_vector_k1: int
    lds_add_extra_dim: int

    def __str__(self) -> str:
        args = deepcopy(self.__dict__)

@dataclass
class CShuffleDesc:
    m_Xdl_per_wave_per_shuffle: int
    n_Xdl_per_wave_per_shuffle: int

    def __str__(self) -> str:
        args = deepcopy(self.__dict__)

@dataclass
class CBlockTransferDesc:
    cluster_lengths_m_block_m_wave_m_per_Xdl_n_block_n_wave_n_per_Xdl: str
    scalar_per_vector_n_wave_n_per_Xdl: int

    def __str__(self) -> str:
        args = deepcopy(self.__dict__)


@dataclass
class GemmOperation:
    A: TensorDesc
    B: TensorDesc
    acc: DataType
    cs_type: DataType
    Ds: TensorDesc
    E: TensorDesc
    a_elem_op: TensorOperation
    b_elem_op: TensorOperation
    cde_elem_op: TensorOperation
    gemm_specialization: GemmType #GemmSpecialization
    tile_desc: TileDesc
    a_block_transfer: BlockTransferDesc
    b_block_transfer: BlockTransferDesc
    cshuffle: CShuffleDesc
    b1_block_transfer: BlockTransferDesc = None
    c_block_transfer: CBlockTransferDesc = None


    def __str__(self) -> str:
        io_name = "{gemm_kind}_{gemm_specialization}_{a_dtype}{b_dtype}{c_dtype}_{a_layout}{b_layout}{c_layout}".format(
            #gemm_kind=library.GemmKindNames[self.operation_kind],
            gemm_specialization=self.gemm_specialization.value,
            a_dtype=[self.A.element],
            b_dtype=[self.B.element],
            a_layout=[self.A.layout],
            b_layout=[self.B.layout],
        ) 
 
