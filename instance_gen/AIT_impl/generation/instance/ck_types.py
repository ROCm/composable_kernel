from dataclasses import dataclass

class DataType:
    f16 = "F16"
    f32 = "F32"
    f16_tuple = "F16_Tuple"

class Layout:
    ColumnMajor = "Col"
    RowMajor = "Row"
    Row_Tuple = "Row_Tuple"

class TensorOperation:
    PassThrough = "PassThrough"
    Bilinear = "Bilinear"

@dataclass
class TensorDesc: #set up and import properly
    element: DataType
    layout: Layout

