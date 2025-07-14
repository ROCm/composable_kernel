[Back to supported operations](../../../include/ck/README.md)
# Composable Kernel Grouped Convolution

## Grouped Convolution Backward Weight

Grouped convolution operation for 1D, 2D or 3D spatial dimensions. Convolution utilizes GEMM kernel after tensor coordinate transform. Backward weight version uses splitK feature (due to large GEMM K dimension). In CK Grouped Convolution Backward Weight operation is called as `DeviceGroupedConvBwdWeight` and requires following types as template parameters:

* **NumDimSpatial** - number of spatial dimensions (1D, 2D, 3D).
* **InLayout** - input layout (NHWGC, GNHWC, NGCHW).
* **WeiLayout** - weight layout (GKYXC).
* **OutLayout** - output layout (NHWGK, GNHWK, NGKHW).
* **InDataType** - input data type.
* **WeiDataType** - weight data type.
* **OutDataType** - output data type.
* **InElementwiseOperation** - fused operation on tensor input.
* **WeiElementwiseOperation** - fused operation on tensor weight.
* **OutElementwiseOperation** - fused operation on tensor output.
* **ComputeTypeA** - compute data type of tensor A for mfma instruction (ADataType by default).
* **ComputeTypeB** - compute data type of tensor B for mfma instruction (ComputeTypeA by default).

For fused operations with additional tensor there is `DeviceGroupedConvBwdWeightMultipleD` operation which requires following parameters:
* **DsLayout** - layouts for additional tensors for fused operations.
* **DsDataType** - data types for additional tensors for fused operations.

Grouped convolution backward weight doesn't supports tensors larger than 2GB.

List of the device operations for grouped convolution backward weight in CK:

* **DeviceGroupedConvBwdWeight_Xdl_CShuffle** - Device operation with XDL instructions.
* **DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle** - Device operation with XDL instructions. Optimized for small C or K.
* **DeviceGroupedConvBwdWeight_Wmma_CShuffle** - Device operation with WMMA instructions.
* **DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle** - Device operation with XDL instructions and support of fused operations to output.
* **DeviceGroupedConvBwdWeight_Dl** - Device operation with DL instructions.

Table of supported cases by instance factory with XDL instruction:

|       |NHWGC/GKYXC/NHWGK|NGCHW/GKYXC/NGKHW|NGCHW/GKCYX/NGKHW|GNHWC/GKYXC/GNHWK|
|-------|---|---|---|---|
|bf16|2D, 3D|2D, 3D|2D, 3D|&cross;|
|bf16(fp32 for weight)|2D, 3D|&cross;|&cross;|1D, 2D, 3D|
|fp16 |2D, 3D|2D, 3D|2D, 3D|1D, 2D, 3D|
|fp32  |2D, 3D|2D, 3D|2D, 3D|1D, 2D, 3D|

Table of supported cases by instance factory with WMMA instruction:

|       |NHWGC/GKYXC/NHWGK|NGCHW/GKYXC/NGKHW|GNHWC/GKYXC/GNHWK|
|-------|---|---|---|
|fp16 |3D|&cross;|3D|
|int8 |3D|&cross;|3D|

Table of supported cases by instance factory with DL instruction:

|       |NHWGC/GKYXC/NHWGK|NGCHW/GKYXC/NGKHW|GNHWC/GKYXC/GNHWK|
|-------|---|---|---|
|bf16(fp32 for weight)|1D, 2D, 3D|&cross;|1D, 2D, 3D|
|fp16 |1D, 2D, 3D|&cross;|1D, 2D, 3D|
|fp32  |1D, 2D, 3D|&cross;|1D, 2D, 3D|

Table of supported cases by instance factory with fused elementwise operation:

* **Bilinear** - 3D, NHWGC, bf16(fp32 for weight)/fp16/fp32
* **Scale** - 3D, NHWGC, bf16(fp32 for weight)/fp16/fp32
