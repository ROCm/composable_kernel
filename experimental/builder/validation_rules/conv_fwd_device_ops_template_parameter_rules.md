# Template Parameter Constraint Rules for Forward Convolution Device Operations

This document lists all static_assert rules and runtime validation checks that constrain template parameter selection for the five forward convolution device operations in Composable Kernel.

## Device Operations Analyzed

1. **DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3**
2. **DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle**
3. **DeviceGroupedConvFwdMultipleD_Wmma_CShuffle**
4. **DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor**
5. **DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK**

---

## 1. DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3

### Gridwise Implementation
Uses: `GridwiseGemmMultiD_xdl_cshuffle_v3`

### Compile-Time Static Asserts (Gridwise Level)

#### Block and Wave Tiling Constraints
```cpp
static_assert((MPerBlock % (MPerXdl * MXdlPerWave) == 0) &&
              (NPerBlock % (NXdlPerWave * NPerXdl)) == 0,
              "Invalid tuning param!");
```
- **Rule**: MPerBlock must be divisible by (MPerXdl × MXdlPerWave)
- **Rule**: NPerBlock must be divisible by (NXdlPerWave × NPerXdl)

#### Shuffle Constraints
```cpp
static_assert(MXdlPerWave % CShuffleMXdlPerWavePerShuffle == 0 &&
              NXdlPerWave % CShuffleNXdlPerWavePerShuffle == 0,
              "wrong!");
```
- **Rule**: MXdlPerWave must be divisible by CShuffleMXdlPerWavePerShuffle
- **Rule**: NXdlPerWave must be divisible by CShuffleNXdlPerWavePerShuffle

### Compile-Time Static Asserts (Blockwise Level)

From `BlockwiseGemmXdlops`:
```cpp
static_assert(MPerBlock % (MPerXDL * MRepeat) == 0 && 
              NPerBlock % (NPerXDL * NRepeat) == 0,
              "wrong!");

static_assert(KPerThread % KPack == 0,
              "Wrong KPack setting; try increasing KPerThread or decreasing KPack");

static_assert(ThisThreadBlock::GetNumOfThread() == MWaves * NWaves * WaveSize,
              "ThisThreadBlock::GetNumOfThread() != MWaves * NWaves * WaveSize\n");
```
- **Rule**: MPerBlock must be divisible by (MPerXDL × MRepeat)
- **Rule**: NPerBlock must be divisible by (NPerXDL × NRepeat)
- **Rule**: KPerThread must be divisible by KPack
- **Rule**: BlockSize must equal MWaves × NWaves × WaveSize

### Runtime Validation Checks

#### Vector Access for A (Input) Tensor
For layouts G_NW_C, G_NHW_C, G_NDHW_C, GNWC, GNHWC, GNDHWC, NWGC, NHWGC, NDHWGC, NGCW, NGCHW, NGCDHW:
```cpp
C % ABlockTransferSrcScalarPerVector == 0
```
- **Rule**: C (input channels) must be divisible by ABlockTransferSrcScalarPerVector when ABlockTransferSrcVectorDim == 2

#### Vector Access for B (Weight) Tensor
For layouts G_K_X_C, G_K_YX_C, G_K_ZYX_C, GKXC, GKYXC, GKZYXC, KXGC, KYXGC, KZYXGC, GKCX, GKCYX, GKCZYX:
```cpp
C % BBlockTransferSrcScalarPerVector == 0
```
- **Rule**: C (input channels) must be divisible by BBlockTransferSrcScalarPerVector when BBlockTransferSrcVectorDim == 2

#### Vector Access for E (Output) Tensor
For layouts G_NW_K, G_NHW_K, G_NDHW_K, GNWK, GNHWK, GNDHWK, NWGK, NHWGK, NDHWGK, NGKW, NGKHW, NGKDHW:
```cpp
K % CDEBlockTransferScalarPerVector_NPerBlock == 0
```
- **Rule**: K (output channels) must be divisible by CDEBlockTransferScalarPerVector_NPerBlock

#### Special NGCHW/NGCDHW Layout Constraints
For NGCHW/NGCDHW layouts requiring transpose:
```cpp
(G * C) % CDEBlockTransferScalarPerVector_NPerBlock == 0
(G * K) % CDEBlockTransferScalarPerVector_NPerBlock == 0
input_spatial_acum % CDEBlockTransferScalarPerVector_NPerBlock == 0
output_spatial_acum % CDEBlockTransferScalarPerVector_NPerBlock == 0
```
- **Rule**: G×C must be divisible by CDEBlockTransferScalarPerVector_NPerBlock
- **Rule**: G×K must be divisible by CDEBlockTransferScalarPerVector_NPerBlock
- **Rule**: Product of input spatial dimensions must be divisible by CDEBlockTransferScalarPerVector_NPerBlock
- **Rule**: Product of output spatial dimensions must be divisible by CDEBlockTransferScalarPerVector_NPerBlock

#### Descriptor Size Constraints
```cpp
a_grid_desc.GetElementSpaceSize() * sizeof(ADataType) <= 2GB
b_grid_desc.GetElementSpaceSize() * sizeof(BDataType) <= 2GB
c_grid_desc.GetElementSpaceSize() * sizeof(CDataType) <= 2GB
```
- **Rule**: Each tensor descriptor must represent less than 2GB of data

#### Device-Specific Constraints
- On **gfx908**: AccDataType must be `float` or `int32_t`
- **DirectLoad** mode: Only supported on gfx950

---

## 2. DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle

### Gridwise Implementation
Uses: 
- `GridwiseGemmMultipleABD_xdl_cshuffle` (when isMultiA || isMultiB)
- `GridwiseGemmMultipleD_xdl_cshuffle` (otherwise)

### Compile-Time Static Asserts (Gridwise Level)

Same as V3 version:
```cpp
static_assert((MPerBlock % (MPerXdl * MXdlPerWave) == 0) &&
              (NPerBlock % (NXdlPerWave * NPerXdl)) == 0,
              "Invalid tuning param!");

static_assert(KPerBlock % AK1Value == 0 && KPerBlock % BK1Value == 0,
              "KPerBlock must be divisible by AK1Value and BK1Value!");

static_assert(MXdlPerWave % CShuffleMXdlPerWavePerShuffle == 0 &&
              NXdlPerWave % CShuffleNXdlPerWavePerShuffle == 0,
              "wrong!");
```
- **Rule**: MPerBlock must be divisible by (MPerXdl × MXdlPerWave)
- **Rule**: NPerBlock must be divisible by (NXdlPerWave × NPerXdl)
- **Rule**: KPerBlock must be divisible by both AK1Value and BK1Value
- **Rule**: MXdlPerWave must be divisible by CShuffleMXdlPerWavePerShuffle
- **Rule**: NXdlPerWave must be divisible by CShuffleNXdlPerWavePerShuffle

### Runtime Validation Checks

#### Vector Access for A (Input) Tensor
For standard layouts (G_NW_C, G_NHW_C, etc.):
```cpp
C % ABlockTransferSrcScalarPerVector == 0  // When ABlockTransferSrcVectorDim == 2
```
- **Rule**: C must be divisible by ABlockTransferSrcScalarPerVector

Alternative for grouped layouts with C==1 or NumGroupsToMerge==1:
```cpp
G % ABlockTransferSrcScalarPerVector == 0  // When ABlockTransferSrcVectorDim == 1
```
- **Rule**: G must be divisible by ABlockTransferSrcScalarPerVector when accessing per G dimension

For NGCHW/NGCDHW layouts without transpose:
```cpp
input_spatial_acum % ABlockTransferSrcScalarPerVector == 0  // When ABlockTransferSrcVectorDim == 1
```
- **Rule**: Product of input spatial dimensions must be divisible by ABlockTransferSrcScalarPerVector

#### Vector Access for B (Weight) Tensor
```cpp
C % BBlockTransferSrcScalarPerVector == 0  // When BBlockTransferSrcVectorDim == 2
```
- **Rule**: C must be divisible by BBlockTransferSrcScalarPerVector

#### Vector Access for D Tensors
For each D tensor with layouts G_NW_K, G_NHW_K, etc.:
```cpp
K % CDEBlockTransferScalarPerVector_NPerBlock == 0
```
- **Rule**: K must be divisible by CDEBlockTransferScalarPerVector_NPerBlock
- **Rule**: D and E tensors must have identical shapes (all dimensions must match)

#### Vector Access for E (Output) Tensor
For standard layouts:
```cpp
K % CDEBlockTransferScalarPerVector_NPerBlock == 0  // When CTranspose == false
```
For transposed layouts:
```cpp
output_spatial_acum % CDEBlockTransferScalarPerVector_NPerBlock == 0
```

#### Transpose Kernel Requirements
For NGCHW/NGCDHW layouts with transpose:
```cpp
(G * C) % CDEBlockTransferScalarPerVector_NPerBlock == 0
(G * K) % CDEBlockTransferScalarPerVector_NPerBlock == 0
input_spatial_acum % CDEBlockTransferScalarPerVector_NPerBlock == 0
output_spatial_acum % CDEBlockTransferScalarPerVector_NPerBlock == 0
```
- Workspace pointer must be allocated

#### NumGroupsToMerge Constraints
When NumGroupsToMerge > 1:
```cpp
C == 1
G % NumGroupsToMerge == 0
```
- **Rule**: C must equal 1
- **Rule**: G must be divisible by NumGroupsToMerge

#### Tensor Size Constraints
```cpp
a_grid_desc.GetElementSpaceSize() * sizeof(ADataType) <= 2GB
b_grid_desc.GetElementSpaceSize() * sizeof(BDataType) <= 2GB
e_grid_desc.GetElementSpaceSize() * sizeof(EDataType) <= 2GB
```

---

## 3. DeviceGroupedConvFwdMultipleD_Wmma_CShuffle

### Gridwise Implementation
Uses: `GridwiseGemmMultipleD_Wmma`

### Compile-Time Static Asserts

```cpp
static_assert((MPerBlock % (MPerWmma * MRepeat) == 0) &&
              (NPerBlock % (NRepeat * NPerWmma)) == 0,
              "Invalid tuning param!");

static_assert(KPack % (A_K1 * A_KRow) == 0, "wrong!");
static_assert(KPack % (B_K1 * B_KRow) == 0, "wrong!");

static_assert(ThisThreadBlock::GetNumOfThread() == MWaves * NWaves * WaveSize,
              "ThisThreadBlock::GetNumOfThread() != MWaves * NWaves * WaveSize");
```
- **Rule**: MPerBlock must be divisible by (MPerWmma × MRepeat)
- **Rule**: NPerBlock must be divisible by (NRepeat × NPerWmma)
- **Rule**: KPack must be divisible by (A_K1 × A_KRow) where A_KRow = 2
- **Rule**: KPack must be divisible by (B_K1 × B_KRow) where B_KRow = 2
- **Rule**: BlockSize must equal MWaves × NWaves × WaveSize
  - Where: MWaves = MPerBlock / (MRepeat × MPerWmma)
  - Where: NWaves = NPerBlock / (NRepeat × NPerWmma)

### Derived Constraints
```cpp
K % K1 == 0  // Asserted in MakeAGridDescriptor and MakeBGridDescriptor
KPack = math::integer_least_multiple(K1, WmmaK)  // Where WmmaK = 16
```
- **Rule**: K must be divisible by K1
- **Rule**: KPack must be at least lcm(K1, 16)

### Runtime Validation Checks

#### Device Support
```cpp
ck::is_gfx11_supported() || ck::is_gfx12_supported()
```
- **Rule**: Only supports gfx11 and gfx12 architectures
- **Rule**: On these devices, AccDataType must be `float` or `int32_t`

#### Vector Access for A
For layouts G_NW_C, G_NHW_C, G_NDHW_C, GNWC, GNHWC, GNDHWC, NWGC, NHWGC, NDHWGC:
```cpp
C % ABlockTransferSrcScalarPerVector == 0  // When ABlockTransferSrcVectorDim == 2
```

#### Vector Access for B
For layouts G_K_X_C, G_K_YX_C, G_K_ZYX_C, GKXC, GKYXC, GKZYXC, KXGC, KYXGC, KZYXGC:
```cpp
C % BBlockTransferSrcScalarPerVector == 0  // When BBlockTransferSrcVectorDim == 2
```

#### Vector Access for D and E
For all D tensors and E tensor:
```cpp
K % CDEShuffleBlockTransferScalarPerVector_NPerBlock == 0
```

---

## 4. DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor

### Gridwise Implementation
Uses: `GridwiseGemmMultipleD_xdl_cshuffle`

### Compile-Time Static Asserts

Same as other XDL-based operations:
```cpp
static_assert((MPerBlock % (MPerXdl * MXdlPerWave) == 0) &&
              (NPerBlock % (NXdlPerWave * NPerXdl)) == 0,
              "Invalid tuning param!");

static_assert(KPerBlock % AK1Value == 0 && KPerBlock % BK1Value == 0,
              "KPerBlock must be divisible by AK1Value and BK1Value!");

static_assert(MXdlPerWave % CShuffleMXdlPerWavePerShuffle == 0 &&
              NXdlPerWave % CShuffleNXdlPerWavePerShuffle == 0,
              "wrong!");
```

### Runtime Validation Checks

#### Tensor Splitting Validation
This operation splits large tensors that exceed 2GB:
```cpp
is_split_valid_ && gemms_count_ == valid_gemms_count_
```
- **Rule**: The tensor splitting algorithm must successfully partition the problem into sub-problems < 2GB

#### D and E Tensor Matching
```cpp
ds_g_n_k_wos_strides_[i] == e_g_n_k_wos_strides_
ds_g_n_k_wos_lengths_[i] == e_g_n_k_wos_lengths_
```
- **Rule**: All D tensors must have identical strides and lengths to E tensor

#### Vector Access Constraints
Same as standard XDL operations:
```cpp
C % ABlockTransferSrcScalarPerVector == 0
C % BBlockTransferSrcScalarPerVector == 0
K % CDEBlockTransferScalarPerVector_NPerBlock == 0
```

---

## 5. DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK

### Gridwise Implementation
Uses: `GridwiseGemmDlMultipleD_km_kn_mn`

### Compile-Time Static Asserts

From `BlockwiseGemmDl_v2r3`:
```cpp
static_assert(BM % BM1 == 0 && BN % BN1 == 0, "wrong!");
static_assert(BM0 == 2 && BN0 == 2, "wrong");
static_assert(BlockSize == BM101 * BM100 * BN101 * BN100,
              "wrong! blocksize and cluster size not consistent");
```
- **Rule**: BM (MPerBlock) must be divisible by BM1
- **Rule**: BN (NPerBlock) must be divisible by BN1  
- **Rule**: BM0 must equal 2
- **Rule**: BN0 must equal 2
- **Rule**: BlockSize must equal the product of thread cluster dimensions

### Runtime Validation Checks

#### Device Support
```cpp
ck::get_device_name() == "gfx906" || ck::is_xdl_supported() || 
ck::is_gfx103_supported() || ck::is_gfx11_supported() || ck::is_gfx12_supported()
```
- **Rule**: Must be one of: gfx906, gfx103, gfx11, gfx12, or support XDL instructions

#### Vector Transfer Constraints for A
```cpp
srcVectorLengths[I1] == 1 && srcVectorLengths[I2] == 1
K1 % srcVectorLengths[I3] == 0
K0PerBlock % srcVectorLengths[I0] == 0
C % (srcVectorLengths[I0] * srcVectorLengths[I3]) == 0
```
- **Rule**: Vector lengths for M dimensions must be 1
- **Rule**: K1 must be divisible by K1 vector length
- **Rule**: K0PerBlock must be divisible by K0 vector length
- **Rule**: C must be divisible by the product of K0 and K1 vector lengths

#### Vector Transfer Constraints for B
Same structure as A:
```cpp
srcVectorLengths[I1] == 1 && srcVectorLengths[I2] == 1
K1 % srcVectorLengths[I3] == 0
K0PerBlock % srcVectorLengths[I0] == 0
C % (srcVectorLengths[I0] * srcVectorLengths[I3]) == 0
```

#### Vector Access for E (Output)
```cpp
K % CThreadTransferDstScalarPerVector == 0
CThreadTransferSrcDstVectorDim == 5
```
- **Rule**: K must be divisible by CThreadTransferDstScalarPerVector
- **Rule**: Vector dimension must be 5 (the K dimension)

#### Tile Size Constraints
```cpp
M % MPerBlock == 0
N % NPerBlock == 0
K0 % K0PerBlock == 0
```
- **Rule**: M must be divisible by MPerBlock
- **Rule**: N must be divisible by NPerBlock
- **Rule**: K0 must be divisible by K0PerBlock

---

## Common Rules Across All Operations

### Specialization Requirements

For **Filter1x1Stride1Pad0** specialization:
```cpp
FilterSpatialDim == 1
ConvStride == 1
LeftPad == 0
RightPad == 0
```
- Must be true for all spatial dimensions

For **Filter1x1Pad0** specialization:
```cpp
FilterSpatialDim == 1
LeftPad == 0
RightPad == 0
```
- Must be true for all spatial dimensions

For **Filter3x3** specialization:
```cpp
C == 1
FilterSpatialDim == 3
```
- Must be true for all spatial dimensions

### Pipeline Stage Constraints

For non-v1 pipeline versions:
```cpp
num_k_loop > PrefetchStages
```
- **Rule**: Number of K-blocks must exceed the number of prefetch stages

### TF32 Support Constraints
```cpp
is_same_v<AComputeDataType, BComputeDataType>  // When using TF32
is_tf32_supported()  // Device must support TF32
```
- **Rule**: When using TF32, A and B compute data types must match
- **Rule**: Device must have TF32 support

### XDL/WMMA Support Validation
```cpp
ck::is_xdl_wmma_supported<AComputeDataType, BComputeDataType, MPerXdl, NPerXdl>()
```
- **Rule**: The combination of data types and XDL/WMMA tile sizes must be supported by the device

---

## Summary of Key Parameter Relationships

### Block-Level Tiling
```
MPerBlock = MPerXdl × MXdlPerWave × MWaves
NPerBlock = NPerXdl × NXdlPerWave × NWaves
BlockSize = MWaves × NWaves × WaveSize
```

For WMMA:
```
MPerBlock = MPerWmma × MRepeat × MWaves
NPerBlock = NPerWmma × NRepeat × NWaves
```

### K-Dimension Decomposition
```
K = AK0 × AK1 = BK0 × BK1
KPerBlock = AK0PerBlock × AK1 = BK0PerBlock × BK1
```

### Shuffle Constraints
```
MXdlPerWave = N × CShuffleMXdlPerWavePerShuffle  (N is integer)
NXdlPerWave = M × CShuffleNXdlPerWavePerShuffle  (M is integer)
```

### Vector Access Hierarchy
1. **Data must be aligned** to vector access size
2. **Dimensions accessed vectorially** must be divisible by ScalarPerVector
3. **Different layouts** have different vectorizable dimensions:
   - Row-major A: vectorize K dimension
   - Column-major A: vectorize M dimension
   - Row-major B: vectorize N dimension  
   - Column-major B: vectorize K dimension

### LDS Padding
```
ABlockLdsExtraM: Padding for A matrix in LDS to avoid bank conflicts
BBlockLdsExtraN: Padding for B matrix in LDS to avoid bank conflicts
```
- Often set to 1 on gfx950 to reduce bank conflicts

---

## Implementation Notes

1. **Hierarchy**: Device ops compose gridwise ops, which compose blockwise ops, which compose threadwise ops
2. **Memory Flow**: Global → LDS → Register (VGPR) → Compute → Register → LDS → Global
3. **Direct Load**: Some implementations support direct global-to-register load (bypassing LDS) on gfx950
4. **Pipeline Versions**: Different pipeline versions (v1, v2, v3, v4, v5) have different prefetch and scheduling strategies
5. **Multi-AB Support**: Some operations support multiple A/B input tensors (tuples)
6. **Transpose Support**: Some layouts require intermediate transpose operations with workspace allocation

---

## Validation Checklist for Template Parameter Selection

When selecting template parameters, verify:

- [ ] MPerBlock % (MPerXdl × MXdlPerWave) == 0
- [ ] NPerBlock % (NPerXdl × NXdlPerWave) == 0  
- [ ] KPerBlock % AK1 == 0 and KPerBlock % BK1 == 0
- [ ] BlockSize == computed_from_waves_and_wave_size
- [ ] All channel/spatial dimensions divisible by respective ScalarPerVector values
- [ ] Tensor descriptors < 2GB each
- [ ] Correct device architecture and data type support
- [ ] Specialization requirements met (filter size, stride, padding)
- [ ] Shuffle parameters properly divide wave parameters
- [ ] Pipeline stage requirements met for chosen version
- [ ] Workspace allocated if using transpose kernels
