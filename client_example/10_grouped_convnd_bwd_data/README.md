[Back to supported operations](../../../include/ck/README.md)
# Composable Kernel Grouped Convolution

## Grouped Convolution Backward Data

Grouped convolution operation for 1D, 2D or 3D spatial dimensions. Convolution utilizes GEMM kernel after tensor coordinate transform. In CK Grouped Convolution Backward Data operation is called as `DeviceGroupedConvBwdDataMultipleD` and requires following types as template parameters:

* **NumDimSpatial** - number of spatial dimensions (1D, 2D, 3D).
* **ALayout** - output layout (NHWGK, GNHWK, NGKHW).
* **BLayout** - weight layout (GKYXC).
* **DsLayout** - layouts for additional tensors for fused operations.
* **ELayout** - input layout (NHWGC, GNHWC, NGCHW).
* **ADataType** - output data type.
* **BDataType** - weight data type.
* **DsDataType** - data types for additional tensors for fused operations.
* **EDataType** - input data type.
* **AElementwiseOperation** - fused operation on tensor A (output).
* **BElementwiseOperation** - fused operation on tensor B (weight).
* **CDEElementwiseOperation** - fused operation on tensor C (input).
* **AComputeType** - compute data type of tensor A for mfma instruction (ADataType by default).
* **BComputeType** - compute data type of tensor B for mfma instruction (AComputeType by default).

Grouped convolution backward data supports tensors larger than 2GB (except when image is larger than 2GB).

List of the device operations for grouped convolution backward data in CK:

* **DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1** - Device operation with XDL instructions and support of fused operations to input.
* **DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle** - Device operation with WMMA instructions.

Table of supported cases by instance factory with XDL instruction:

|       |NHWGC/GKYXC/NHWGK|NGCHW/GKYXC/NGKHW|GNHWC/GKYXC/GNHWK|
|-------|---|---|---|
|bf16|2D, 3D|2D, 3D|2D, 3D|
|fp16 |2D, 3D|2D, 3D|2D, 3D|
|fp32  |2D, 3D|2D, 3D|2D, 3D|

Table of supported cases by instance factory with WMMA instruction:

|       |NHWGC/GKYXC/NHWGK|NGCHW/GKYXC/NGKHW|GNHWC/GKYXC/GNHWK|
|-------|---|---|---|
|fp16 |2D, 3D|&cross;|2D, 3D|
|int8 |2D, 3D|&cross;|2D, 3D|

Table of supported cases by instance factory with fused elementwise operation:

* **Bilinear** - 3D, NHWGC, bf16/fp16/fp32
* **Scale** - 3D, NHWGC, bf16/fp16/fp32
