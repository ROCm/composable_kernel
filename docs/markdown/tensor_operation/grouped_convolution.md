[Back to supported operations](../../../include/ck/README.md)
# Composable Kernel Grouped Convolution

## Grouped Convolution Forward
Grouped convolution operation for 1D, 2D or 3D spatial dimensions. Convolution utilizes GEMM kernel after tensor coordinate transform. In CK Grouped Convolution Forward operation is called as `DeviceGroupedConvFwdMultipleABD` and requires following types as template parameters:

* **NumDimSpatial** - number of spatial dimensions (1D, 2D, 3D).
* **InLayout** - input layout (NHWGC, GNHWC, NGCHW).
* **WeiLayout** - weight layout (GKYXC).
* **DsLayout** - layouts for additional tensors for fused operations.
* **OutLayout** - output layout (NHWGK, GNHWK, NGKHW).
* **ADataType** - input data type. Pass tuple if there is fused operation with input.
* **BDataType** - weight data type. Pass tuple if there is fused operation with weight.
* **DsDataType** - data types for additional tensors for fused operations.
* **EDataType** - Output data type.
* **AElementwiseOperation** - fused operation on tensor A (input).
* **BElementwiseOperation** - fused operation on tensor B (weight).
* **CDEElementwiseOperation** - fused operation on tensor C (output).
* **AComputeType** - compute data type of tensor A for mfma instruction (ADataType by default).
* **BComputeType** - compute data type of tensor B for mfma instruction (AComputeType by default).

Grouped convolution forward support tensors larger than 2GB.

List of the device operations for grouped convolution forward in CK:

* **DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3** - Device operation with XDL instructions. Optimized for AMD Instinct MI300 series.
* **DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle** - Device operation with XDL instructions and support of fused operations to input, weight and output.
* **DeviceGroupedConvFwdMultipleD_Wmma_CShuffle** - Device operation with WMMA instructions.
* **DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK** - Device operation with DL instructions.

Table of supported cases by instance factory with XDL instruction:

|       |NHWGC/GKYXC/NHWGK|NGCHW/GKYXC/NGKHW|GNHWC/GKYXC/GNHWK|
|-------|---|---|---|
|bhalf_t|2D, 3D|2D|1D, 2D, 3D|
|half_t |2D, 3D|2D|1D, 2D, 3D|
|float  |2D, 3D|2D|1D, 2D, 3D|
|int8_t |2D, 3D|2D|1D, 3D|
|fp8_t  |3D|&cross;|&cross;|
|bf8_t  |3D|&cross;|&cross;|

Table of supported cases by instance factory with WMMA instruction:

|       |NHWGC/GKYXC/NHWGK|NGCHW/GKYXC/NGKHW|GNHWC/GKYXC/GNHWK|
|-------|---|---|---|
|half_t |2D, 3D|&cross;|2D, 3D|
|int8_t |2D, 3D|&cross;|2D, 3D|

Table of supported cases by instance factory with DL instruction:

|       |NHWGC/GKYXC/NHWGK|NGCHW/GKYXC/NGKHW|GNHWC/GKYXC/GNHWK|
|-------|---|---|---|
|bhalf_t|&cross;|&cross;|2D|
|half_t |&cross;|&cross;|2D|
|float  |&cross;|&cross;|2D|
|int8_t |&cross;|&cross;|2D|

Table of supported cases by instance factory with fused elementwise operation:

* **Dynamic elementwise operation** - 2D/3D, NHWGC, bhalf_t/half_t/float/int8_t
* **Bilinear** - 3D, NHWGC, bhalf_t/half_t/float/int8_t
* **ConvInvScale** - 3D, NHWGC, fp8_t
* **ConvScale** - 3D, NHWGC, fp8_t/bf8_t
* **ConvScale + Add** - 3D, NHWGC, fp8_t
* **ConvScale + Relu** - 3D, NHWGC, fp8_t
* **Scale** - 3D, NHWGC, bhalf_t/half_t/float/int8_t
* **Scale + Add (for A and B)** - 3D, NHWGC, bhalf_t/half_t/float/int8_t
* **Scale + Add + Scale + Add + Relu** - 3D, NHWGC, bhalf_t/half_t/float/int8_t

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

|       |NHWGC/GKYXC/NHWGK|NGCHW/GKYXC/NGKHW|GNHWC/GKYXC/GNHWK|
|-------|---|---|---|
|bhalf_t|2D, 3D|&cross;|&cross;|
|bhalf_t(float for weight)|2D, 3D|&cross;|1D, 2D, 3D|
|half_t |2D, 3D|&cross;|1D, 2D, 3D|
|float  |2D, 3D|&cross;|1D, 2D, 3D|

Table of supported cases by instance factory with WMMA instruction:

|       |NHWGC/GKYXC/NHWGK|NGCHW/GKYXC/NGKHW|GNHWC/GKYXC/GNHWK|
|-------|---|---|---|
|half_t |3D|&cross;|3D|
|int8_t |3D|&cross;|3D|

Table of supported cases by instance factory with DL instruction:

|       |NHWGC/GKYXC/NHWGK|NGCHW/GKYXC/NGKHW|GNHWC/GKYXC/GNHWK|
|-------|---|---|---|
|bhalf_t(float for weight)|1D, 2D, 3D|&cross;|1D, 2D, 3D|
|half_t |1D, 2D, 3D|&cross;|1D, 2D, 3D|
|float  |1D, 2D, 3D|&cross;|1D, 2D, 3D|

Table of supported cases by instance factory with fused elementwise operation:

* **Bilinear** - 3D, NHWGC, bhalf_t(float for weight)/half_t/float
* **Scale** - 3D, NHWGC, bhalf_t(float for weight)/half_t/float

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
|bhalf_t|2D, 3D|&cross;|2D, 3D|
|half_t |2D, 3D|&cross;|2D, 3D|
|float  |2D, 3D|&cross;|2D, 3D|

Table of supported cases by instance factory with WMMA instruction:

|       |NHWGC/GKYXC/NHWGK|NGCHW/GKYXC/NGKHW|GNHWC/GKYXC/GNHWK|
|-------|---|---|---|
|half_t |2D, 3D|&cross;|2D, 3D|
|int8_t |2D, 3D|&cross;|2D, 3D|

Table of supported cases by instance factory with fused elementwise operation:

* **Bilinear** - 3D, NHWGC, bhalf_t/half_t/float
* **Scale** - 3D, NHWGC, bhalf_t/half_t/float
