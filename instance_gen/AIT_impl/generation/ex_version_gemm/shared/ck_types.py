from dataclasses import dataclass

class DataType:
    f16 = "ck::half_t"

class Layout:
    ColumnMajor = "ck::tensor_layout::gemm::ColumnMajor"
    RowMajor = "ck::tensor_layout::gemm::RowMajor"

class TensorOperation:
    PassThrough = "ck::tensor_operation::element_wise::PassThrough"

@dataclass
class TensorDesc: #set up and import properly
    element: DataType
    layout: Layout

