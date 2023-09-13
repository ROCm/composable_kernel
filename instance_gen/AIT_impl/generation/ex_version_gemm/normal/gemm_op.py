#take in input for gemm from user, send it to example template
import enum
import ck_types
from copy import deepcopy
from dataclasses import dataclass
from enum import auto
from typing import List
from ck_types import *

class GemmType():
    GemmDefault = "ck::tensor_operation::device::GemmSpecialization::Default"

# class GemmSpecialization(enum.Enum):
#     GemmDefault = auto()
#     MNKPadding = auto()
#     MNPadding = auto()
#     MNOPadding = auto()
#     MNKOPadding = auto()


# GemmSpecializationTag = {
#     GemmSpecialization.GemmDefault: "ck::tensor_operation::device::GemmSpecialization::Default",
#     GemmSpecialization.MNKPadding: "ck::tensor_operation::device::GemmSpecialization::MNKPadding",
#     GemmSpecialization.MNPadding: "ck::tensor_operation::device::GemmSpecialization::MNPadding",
#     GemmSpecialization.MNOPadding: "ck::tensor_operation::device::GemmSpecialization::MNOPadding",
#     GemmSpecialization.MNKOPadding: "ck::tensor_operation::device::GemmSpecialization::MNKOPadding",
# }

@dataclass
class TileDesc:
    block_size: int
    m_per_block: int
    n_per_block: int
    k_per_block: int
    k1: int
    m_per_thread: int
    n_per_thread: int
    k_per_thread: int
    m1n1_thcluster_m1xs: str
    m1n1_thcluster_n1xs: str

    def __str__(self) -> str:
        values = list(self.__dict__.values())
        return "_".join([str(x) for x in values])
        return template.render(param=args)

@dataclass
class BlockTransferDesc:
    thread_slice_length: str
    thread_cluster_length: str
    thread_cluster_arrange_order: str
    src_access_order: str
    src_vec_tensor_lengths: str
    src_vec_tensor_cont_dim_order: str
    dst_vec_tensor_lengths: str

    def __str__(self) -> str:
        args = deepcopy(self.__dict__)
        args["thread_cluster_length"] = [str(x) for x in self.thread_cluster_length]
        args["thread_cluster_arrange_order"] = [
            str(x) for x in self.thread_cluster_arrange_order
        ]
        args["src_access_order"] = [str(x) for x in self.src_access_order]

@dataclass
class CBlockTransferDesc:
    src_dst_access_order: str
    src_dst_vec_dim: int
    dst_scalar_per_vector: int

    def __str__(self) -> str:
        args = deepcopy(self.__dict__)
        #args["m_n_block_wave_per_xdl"] = [str(x) for x in self.m_n_block_wave_per_xdl]


@dataclass
class GemmOperation:
    A: TensorDesc
    B: TensorDesc
    C: TensorDesc
    a_elem_op: TensorOperation
    b_elem_op: TensorOperation
    epilogue_functor: TensorOperation
    gemm_specialization: GemmType #GemmSpecialization
    tile_desc: TileDesc
    a_block_transfer: BlockTransferDesc
    b_block_transfer: BlockTransferDesc
    b1_block_transfer: BlockTransferDesc = None
    c_block_transfer: CBlockTransferDesc = None


    def __str__(self) -> str:
        io_name = "{gemm_kind}_{gemm_specialization}_{a_dtype}{b_dtype}{c_dtype}_{a_layout}{b_layout}{c_layout}".format(
            #gemm_kind=library.GemmKindNames[self.operation_kind],
            gemm_specialization=self.gemm_specialization.value,
            a_dtype=[self.A.element],
            b_dtype=[self.B.element],
            c_dtype=[self.C.element],
            a_layout=[self.A.layout],
            b_layout=[self.B.layout],
            c_layout=[self.C.layout],
        )
        extra_tile = ""
        if self.c_block_transfer is not None:
            if self.c_block_transfer.scalar_per_vector == 4:
                extra_tile = "_C4"
            elif self.c_block_transfer.scalar_per_vector == 1:
                extra_tile = "_C1"

        tile_name = str(self.tile_desc) + extra_tile
        
        return "{io_name}_{tile_name}_{epilogue_functor}".format(
            io_name=io_name,
            tile_name=tile_name,
            epilogue_functor=[self.epilogue_functor],
        )

    def accumulator_type(self):
        return DataType.f16 #f.32?

if __name__ == "__main__":
    A = TensorDesc(DataType.f16, Layout.RowMajor)
    B = TensorDesc(DataType.f16, Layout.ColumnMajor)
    C = TensorDesc(DataType.f16, Layout.RowMajor)
    GemmOp = GemmOperation(
        A=A,
        B=B,
        C=C,
        a_elem_op=TensorOperation.PassThrough,
        b_elem_op=TensorOperation.PassThrough,
        epilogue_functor=TensorOperation.PassThrough,
        gemm_specialization=GemmType.GemmDefault,
        tile_desc=TileDesc(256, 256, 128, 32, 8, 2, 32, 32, 4, 2),
        a_block_transfer=BlockTransferDesc(
            [4, 64, 1], [1, 0, 2], [1, 0, 2], 2, 8, 8, 1, True
        ),
        b_block_transfer=BlockTransferDesc(
            [8, 32, 1], [0, 2, 1], [0, 2, 1], 1, 4, 1, 0, True
        ),
        c_block_transfer=CBlockTransferDesc(1, 1, [1, 32, 1, 8], 8),
        #ds_dtype=[DataType.f16],
    )
    print(GemmOp.a_elem_op)

