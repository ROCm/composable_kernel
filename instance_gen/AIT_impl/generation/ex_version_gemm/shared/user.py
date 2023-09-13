# the structure for creating a list of instances for an op 
# was taken from Meta's AIT library 
import gemm_op as gemm
import enum
from dataclasses import dataclass
from enum import auto
import ck_types
from ck_types import *

def CreateGemmOperator():
    #operation_kind = library.GemmKind.Gemm
    a_element_desc = TensorDesc(
       DataType.f16, Layout.ColumnMajor
    )
    b_element_desc = TensorDesc(
        DataType.f16, Layout.RowMajor
    )
    c_element_desc = TensorDesc(
        DataType.f16,Layout.RowMajor
    )
    element_op = TensorOperation.PassThrough

    tile_descriptions = [
        gemm.TileDesc(256, 128, 128, 16, 2, 4, 4, 1, "S<8, 2>", "S<8, 2>"),
        gemm.TileDesc(256, 128, 128, 8, 2, 4, 4, 1, "S<8, 2>", "S<8, 2>"),
        gemm.TileDesc(128, 64, 128, 8, 2, 4, 4, 1, "S<4, 2>", "S<8, 2>"),
        gemm.TileDesc(128, 128, 64, 8, 2, 4, 4, 1, "S<8, 2>", "S<4, 2>"),
        gemm.TileDesc(256, 64, 128, 8, 2, 2, 4, 1, "S<8, 2>", "S<8, 2>"),
        gemm.TileDesc(256, 128, 64, 8, 2, 4, 2, 1, "S<8, 2>", "S<8, 2>"),
        gemm.TileDesc(128, 32, 128, 8, 2, 2, 4, 1, "S<4, 2>", "S<8, 2>"),
        gemm.TileDesc(128, 128, 32, 8, 2, 4, 2, 1, "S<8, 2>", "S<4, 2>"),
        gemm.TileDesc(128, 32, 64, 8, 2, 2, 2, 1, "S<4, 2>", "S<8, 2>"),
        gemm.TileDesc(128, 64, 32, 8, 2, 2, 2, 1, "S<8, 2>", "S<4, 2>"),
    ]

    a_block_descriptions = [
        gemm.BlockTransferDesc("S<2, 1, 4, 2>", "S<8, 1, 32, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 4, 1>", "S<0, 3, 1, 2>", "S<1, 1, 4, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 4, 2>", "S<8, 1, 32, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 4, 1>", "S<0, 3, 1, 2>", "S<1, 1, 4, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 4, 2>", "S<8, 1, 16, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 4, 1>", "S<0, 3, 1, 2>", "S<1, 1, 4, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 8, 2>", "S<8, 1, 16, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 4, 1>", "S<0, 3, 1, 2>", "S<1, 1, 4, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 2, 2>", "S<8, 1, 32, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 2, 1>", "S<0, 3, 1, 2>", "S<1, 1, 2, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 4, 2>", "S<8, 1, 32, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 4, 1>", "S<0, 3, 1, 2>", "S<1, 1, 4, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 2, 2>", "S<8, 1, 16, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 2, 1>", "S<0, 3, 1, 2>", "S<1, 1, 2, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 8, 2>", "S<8, 1, 16, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 4, 1>", "S<0, 3, 1, 2>", "S<1, 1, 4, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 2, 2>", "S<8, 1, 16, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 2, 1>", "S<0, 3, 1, 2>", "S<1, 1, 2, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 4, 2>", "S<8, 1, 16, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 4, 1>", "S<0, 3, 1, 2>", "S<1, 1, 4, 2>"),
        
    ]

    b_block_descriptions = [
        gemm.BlockTransferDesc("S<2, 1, 4, 2>", "S<8, 1, 32, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 4, 1>", "S<0, 3, 1, 2>", "S<1, 1, 4, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 4, 2>", "S<8, 1, 32, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 4, 1>", "S<0, 3, 1, 2>", "S<1, 1, 4, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 8, 2>", "S<8, 1, 16, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 4, 1>", "S<0, 3, 1, 2>", "S<1, 1, 4, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 4, 2>", "S<8, 1, 16, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 4, 1>", "S<0, 3, 1, 2>", "S<1, 1, 4, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 4, 2>", "S<8, 1, 32, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 4, 1>", "S<0, 3, 1, 2>", "S<1, 1, 4, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 2, 2>", "S<8, 1, 32, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 2, 1>", "S<0, 3, 1, 2>", "S<1, 1, 2, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 8, 2>", "S<8, 1, 16, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 4, 1>", "S<0, 3, 1, 2>", "S<1, 1, 4, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 2, 2>", "S<8, 1, 16, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 2, 1>", "S<0, 3, 1, 2>", "S<1, 1, 2, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 4, 2>", "S<8, 1, 16, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 4, 1>", "S<0, 3, 1, 2>", "S<1, 1, 4, 2>"),
        gemm.BlockTransferDesc("S<1, 1, 2, 2>", "S<8, 1, 16, 1>", "S<0, 3, 1, 2>", "S<0, 3, 1, 2>", "S<1, 1, 2, 1>", "S<0, 3, 1, 2>", "S<1, 1, 2, 2>"),
    ]

    c_block_descriptions = [
        gemm.CBlockTransferDesc("S<0, 1, 2, 3, 4, 5>", 5, 4),
        gemm.CBlockTransferDesc("S<0, 1, 2, 3, 4, 5>", 5, 4),
        gemm.CBlockTransferDesc("S<0, 1, 2, 3, 4, 5>", 5, 4),
        gemm.CBlockTransferDesc("S<0, 1, 2, 3, 4, 5>", 5, 4),
        gemm.CBlockTransferDesc("S<0, 1, 2, 3, 4, 5>", 5, 4),
        gemm.CBlockTransferDesc("S<0, 1, 2, 3, 4, 5>", 5, 2),
        gemm.CBlockTransferDesc("S<0, 1, 2, 3, 4, 5>", 5, 4),
        gemm.CBlockTransferDesc("S<0, 1, 2, 3, 4, 5>", 5, 2),
        gemm.CBlockTransferDesc("S<0, 1, 2, 3, 4, 5>", 5, 4),
        gemm.CBlockTransferDesc("S<0, 1, 2, 3, 4, 5>", 5, 2),
    ]
    #a_block_descriptions = b_block_descriptions
    #c_block_descriptions = []
    # AIT logic, adapt later
    # for t in tile_descriptions:
    #     a_block_transfer = -1
    #     c_block_transfer = -1
    #     if t.block_size == 256:
    #         a_block_transfer = [4, 64, 1]
    #         c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 8], 8)
    #     if t.block_size == 128:
    #         a_block_transfer = [4, 32, 1]
    #         if t.n_per_block == 128:
    #             c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 16, 1, 8], 8)
    #         if t.n_per_block == 64:
    #             c_block_transfer = gemm.CBlockTransferDesc(1, 1, [1, 32, 1, 4], 8)

    #     assert (
    #         a_block_transfer != -1
    #         and c_block_transfer != -1
    #         and "Cannot determine block_transfer_size with block_size "
    #         + str(t.block_size)
    #     )
    #     a_block_descriptions.append(
    #         gemm.BlockTransferDesc(a_block_transfer, [1, 0, 2], [1, 0, 2], 2, 8, 8, 1)
    #     )
    #     c_block_descriptions.append(c_block_transfer)

    gemm_specialization = [
        gemm.GemmType.GemmDefault
    ]
    operations = []
    for gemm_spec in gemm_specialization:
        for tile_desc, a_block_desc, b_block_desc, c_block_desc in zip(
            tile_descriptions,
            a_block_descriptions,
            b_block_descriptions,
            c_block_descriptions,
        ):
            new_operation = gemm.GemmOperation(
                #operation_kind=operation_kind,
                A=a_element_desc,
                B=b_element_desc,
                C=c_element_desc,
                a_elem_op=element_op,
                b_elem_op=element_op,
                epilogue_functor=element_op,
                gemm_specialization=gemm_spec,
                tile_desc=tile_desc,
                a_block_transfer=a_block_desc,
                b_block_transfer=b_block_desc,
                c_block_transfer=c_block_desc,
            )
            #manifest.append(new_operation)
            operations.append(new_operation)
    return operations

    print (operations[0].tile_desc)

