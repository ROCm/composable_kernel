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
    ds_element_desc = TensorDesc(
        DataType.f16_tuple,Layout.Row_Tuple
    )
    e_element_desc = TensorDesc(
        DataType.f16,Layout.RowMajor
    )
    a_element_op = TensorOperation.PassThrough
    b_element_op = TensorOperation.PassThrough
    cde_element_op = TensorOperation.Bilinear

    acc_type = DataType.f16
    cshuffle_type = DataType.f32

    tile_descriptions = [
        gemm.TileDesc(256, 256, 128, 32, 8, 8, 32, 32, 4, 2, 1), 
        gemm.TileDesc(256, 128, 256, 32, 8, 8, 32, 32, 2, 4, 1), 
        gemm.TileDesc(128, 128, 128, 32, 8, 8, 32, 32, 4, 2, 1),  
        gemm.TileDesc(256, 128, 128, 32, 8, 8, 32, 32, 2, 2, 1), 
        gemm.TileDesc(128, 128, 64, 32, 8, 8, 32, 32, 2, 2, 1), 
        gemm.TileDesc(128, 64, 128, 32, 8, 8, 32, 32, 2, 2, 1), 
        gemm.TileDesc(64, 64, 64, 32, 8, 8, 32, 32, 2, 2, 1), 
        gemm.TileDesc(256, 128, 64, 32, 8, 8, 32, 32, 2, 1, 1), 
        gemm.TileDesc(256, 64, 128, 32, 8, 8, 32, 32, 1, 2, 1), 
        gemm.TileDesc(128, 128, 32, 32, 8, 8, 32, 32, 2, 1, 1), 
        gemm.TileDesc(128, 32, 128, 32, 8, 8, 32, 32, 1, 2, 1), 
        gemm.TileDesc(64, 64, 32, 32, 8, 8, 32, 32, 2, 1, 1), 
        gemm.TileDesc(64, 32, 64, 32, 8, 8, 32, 32, 1, 2, 1), 
    ]

    a_block_descriptions = [
        gemm.BlockTransferDesc("S<4, 64, 1>", "S<1, 0, 2>", "S<1, 0, 2>", 2, 8, 8, 1), 
        gemm.BlockTransferDesc("S<4, 64, 1>", "S<1, 0, 2>", "S<1, 0, 2>", 2, 8, 8, 1), 
        gemm.BlockTransferDesc("S<4, 32, 1>", "S<1, 0, 2>", "S<1, 0, 2>", 2, 8, 8, 1), 
        gemm.BlockTransferDesc("S<4, 64, 1>", "S<1, 0, 2>", "S<1, 0, 2>", 2, 8, 8, 1), 
        gemm.BlockTransferDesc("S<4, 32, 1>", "S<1, 0, 2>", "S<1, 0, 2>", 2, 8, 8, 1),
        gemm.BlockTransferDesc("S<4, 32, 1>", "S<1, 0, 2>", "S<1, 0, 2>", 2, 8, 8, 1), 
        gemm.BlockTransferDesc("S<4, 16, 1>", "S<1, 0, 2>", "S<1, 0, 2>", 2, 8, 8, 1), 
        gemm.BlockTransferDesc("S<4, 64, 1>", "S<1, 0, 2>", "S<1, 0, 2>", 2, 8, 8, 1),
        gemm.BlockTransferDesc("S<4, 64, 1>", "S<1, 0, 2>", "S<1, 0, 2>", 2, 8, 8, 1), 
        gemm.BlockTransferDesc("S<4, 32, 1>", "S<1, 0, 2>", "S<1, 0, 2>", 2, 8, 8, 1), 
        gemm.BlockTransferDesc("S<4, 32, 1>", "S<1, 0, 2>", "S<1, 0, 2>", 2, 8, 8, 1), 
        gemm.BlockTransferDesc("S<4, 16, 1>", "S<1, 0, 2>", "S<1, 0, 2>", 2, 8, 8, 1), 
        gemm.BlockTransferDesc("S<4, 16, 1>", "S<1, 0, 2>", "S<1, 0, 2>", 2, 8, 8, 1), 
    ]

    b_block_descriptions = [
        gemm.BlockTransferDesc("S<4, 64, 1>", "S<1, 0, 2>", "S<1, 0, 2>", 2, 8, 8, 1), 
        gemm.BlockTransferDesc("S<4, 64, 1>", "S<1, 0, 2>", "S<1, 0, 2>", 2, 8, 8, 1), 
        gemm.BlockTransferDesc("S<4, 32, 1>", "S<1, 0, 2>", "S<1, 0, 2>", 2, 8, 8, 1), 
        gemm.BlockTransferDesc("S<4, 64, 1>", "S<1, 0, 2>", "S<1, 0, 2>", 2, 8, 8, 1), 
        gemm.BlockTransferDesc("S<4, 32, 1>", "S<1, 0, 2>", "S<1, 0, 2>", 2, 8, 8, 1),  
        gemm.BlockTransferDesc("S<4, 32, 1>", "S<1, 0, 2>", "S<1, 0, 2>", 2, 8, 8, 1), 
        gemm.BlockTransferDesc("S<4, 16, 1>", "S<1, 0, 2>", "S<1, 0, 2>", 2, 8, 8, 1),
        gemm.BlockTransferDesc("S<4, 64, 1>", "S<1, 0, 2>", "S<1, 0, 2>", 2, 8, 8, 1), 
        gemm.BlockTransferDesc("S<4, 64, 1>", "S<1, 0, 2>", "S<1, 0, 2>", 2, 8, 8, 1), 
        gemm.BlockTransferDesc("S<4, 32, 1>", "S<1, 0, 2>", "S<1, 0, 2>", 2, 8, 8, 1), 
        gemm.BlockTransferDesc("S<4, 32, 1>", "S<1, 0, 2>", "S<1, 0, 2>", 2, 8, 8, 1), 
        gemm.BlockTransferDesc("S<4, 16, 1>", "S<1, 0, 2>", "S<1, 0, 2>", 2, 8, 8, 1), 
        gemm.BlockTransferDesc("S<4, 16, 1>", "S<1, 0, 2>", "S<1, 0, 2>", 2, 8, 8, 1), 
    ]

    cshuffle_descriptions = [
        gemm.CShuffleDesc(1,1),
        gemm.CShuffleDesc(1,1),     
        gemm.CShuffleDesc(1,1),
        gemm.CShuffleDesc(1,1),
        gemm.CShuffleDesc(1,1),
        gemm.CShuffleDesc(1,1),
        gemm.CShuffleDesc(1,1),
        gemm.CShuffleDesc(1,1),
        gemm.CShuffleDesc(1,1),
        gemm.CShuffleDesc(1,1),
        gemm.CShuffleDesc(1,1),
        gemm.CShuffleDesc(1,1),
    ]

    c_block_descriptions = [
        gemm.CBlockTransferDesc("S<1, 32, 1, 8>", 8), 
        gemm.CBlockTransferDesc("S<1, 32, 1, 8>", 8), 
        gemm.CBlockTransferDesc("S<1, 16, 1, 8>", 8), 
        gemm.CBlockTransferDesc("S<1, 32, 1, 8>", 8), 
        gemm.CBlockTransferDesc("S<1, 32, 1, 4>", 8), 
        gemm.CBlockTransferDesc("S<1, 16, 1, 8>", 8), 
        gemm.CBlockTransferDesc("S<1, 16, 1, 4>", 8), 
        gemm.CBlockTransferDesc("S<1, 32, 1, 8>", 8), 
        gemm.CBlockTransferDesc("S<1, 32, 1, 8>", 8), 
        gemm.CBlockTransferDesc("S<1, 32, 1, 4>", 8), 
        gemm.CBlockTransferDesc("S<1, 16, 1, 8>", 8), 
        gemm.CBlockTransferDesc("S<1, 16, 1, 4>", 8), 
        gemm.CBlockTransferDesc("S<1, 16, 1, 4>", 8), 
    ]
    #a_block_descriptions = b_block_descriptions

    gemm_specialization = [
        gemm.GemmType.GemmDefault
    ]
    operations = []
    for gemm_spec in gemm_specialization:
        for tile_desc, a_block_desc, b_block_desc, cshuffle_desc, c_block_desc in zip(
            tile_descriptions,
            a_block_descriptions,
            b_block_descriptions,
            cshuffle_descriptions,
            c_block_descriptions,
        ):
            new_operation = gemm.GemmOperation(
                #operation_kind=operation_kind,
                A=a_element_desc,
                B=b_element_desc,
                acc = acc_type,
                cs_type = cshuffle_type,
                Ds=ds_element_desc,
                E=e_element_desc,
                a_elem_op = a_element_op,
                b_elem_op=b_element_op,
                cde_elem_op=cde_element_op,
                gemm_specialization=gemm_spec,
                tile_desc=tile_desc,
                a_block_transfer=a_block_desc,
                b_block_transfer=b_block_desc,
                cshuffle = cshuffle_desc,
                c_block_transfer=c_block_desc,
            )
            #manifest.append(new_operation)
            operations.append(new_operation)
    return operations

