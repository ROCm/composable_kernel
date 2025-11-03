# Template Parameter Validation Guide for Forward Convolution Device Operations

This guide maps each template parameter to its constraint rules, enabling upstream validation before instantiation.

## Quick Reference: Template Parameter Names

### Common XDL-based Parameters
- **BlockSize**: Number of threads per block
- **MPerBlock, NPerBlock, KPerBlock**: Block tile sizes for GEMM dimensions
- **AK1, BK1**: K-dimension decomposition (K = K0 × K1)
- **MPerXDL, NPerXDL**: XDL/MFMA instruction tile size
- **MXdlPerWave, NXdlPerWave**: Number of XDL tiles per wave
- **ABlockTransferSrcScalarPerVector**: Vector width for loading A matrix
- **BBlockTransferSrcScalarPerVector**: Vector width for loading B matrix
- **ABlockTransferDstScalarPerVector_AK1**: Vector width for storing A to LDS
- **BBlockTransferDstScalarPerVector_BK1**: Vector width for storing B to LDS
- **CShuffleMXdlPerWavePerShuffle, CShuffleNXdlPerWavePerShuffle**: C-shuffle granularity
- **CDEBlockTransferScalarPerVector_NPerBlock**: Vector width for C/D/E transfers
- **ABlockLdsExtraM, BBlockLdsExtraN**: LDS padding to avoid bank conflicts

### WMMA-specific Parameters
- **MPerWmma, NPerWmma**: WMMA instruction tile size (typically 16×16)
- **K1**: K-dimension granularity
- **MRepeat, NRepeat**: Number of WMMA tiles per wave
- **CShuffleMRepeatPerShuffle, CShuffleNRepeatPerShuffle**: C-shuffle granularity

### DL-specific Parameters
- **K0PerBlock**: K0 dimension of block tile
- **K1**: K1 value (typically 4 or 8)
- **M1PerThread, N1PerThread**: Thread tile size
- **KPerThread**: K dimension per thread

---

## 1. DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3

### Template Declaration
```cpp
template <
  index_t NDimSpatial,
  typename ALayout, typename BLayout, typename DsLayout, typename ELayout,
  typename ADataType, typename BDataType, typename AccDataType,
  typename CShuffleDataType, typename DsDataType, typename EDataType,
  typename AElementwiseOperation, typename BElementwiseOperation,
  typename CDEElementwiseOperation,
  ConvolutionForwardSpecialization ConvForwardSpecialization,
  GemmSpecialization GemmSpec,
  index_t BlockSize,
  index_t MPerBlock, index_t NPerBlock, index_t KPerBlock,
  index_t AK1, index_t BK1,
  index_t MPerXDL, index_t NPerXDL,
  index_t MXdlPerWave, index_t NXdlPerWave,
  typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
  typename ABlockTransferThreadClusterArrangeOrder,
  typename ABlockTransferSrcAccessOrder,
  index_t ABlockTransferSrcVectorDim,
  index_t ABlockTransferSrcScalarPerVector,
  index_t ABlockTransferDstScalarPerVector_AK1,
  index_t ABlockLdsExtraM,
  typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
  typename BBlockTransferThreadClusterArrangeOrder,
  typename BBlockTransferSrcAccessOrder,
  index_t BBlockTransferSrcVectorDim,
  index_t BBlockTransferSrcScalarPerVector,
  index_t BBlockTransferDstScalarPerVector_BK1,
  index_t BBlockLdsExtraN,
  index_t CShuffleMXdlPerWavePerShuffle,
  index_t CShuffleNXdlPerWavePerShuffle,
  typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
  index_t CDEBlockTransferScalarPerVector_NPerBlock,
  BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
  BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
  typename AComputeDataType = ...,
  typename BComputeDataType = AComputeDataType,
  bool DirectLoad = false
>
```

### Compile-Time Constraints (Can Check Before Instantiation)

#### Rule 1: Block Tiling Divisibility
```cpp
CONSTRAINT: MPerBlock % (MPerXDL * MXdlPerWave) == 0
PARAMETERS: MPerBlock, MPerXDL, MXdlPerWave
```

#### Rule 2: Block Tiling Divisibility (N dimension)
```cpp
CONSTRAINT: NPerBlock % (NXdlPerWave * NPerXDL) == 0
PARAMETERS: NPerBlock, NPerXDL, NXdlPerWave
```

#### Rule 3: K-dimension Decomposition
```cpp
CONSTRAINT: KPerBlock % AK1 == 0
CONSTRAINT: KPerBlock % BK1 == 0
PARAMETERS: KPerBlock, AK1, BK1
```

#### Rule 4: Shuffle Granularity (M dimension)
```cpp
CONSTRAINT: MXdlPerWave % CShuffleMXdlPerWavePerShuffle == 0
PARAMETERS: MXdlPerWave, CShuffleMXdlPerWavePerShuffle
```

#### Rule 5: Shuffle Granularity (N dimension)
```cpp
CONSTRAINT: NXdlPerWave % CShuffleNXdlPerWavePerShuffle == 0
PARAMETERS: NXdlPerWave, CShuffleNXdlPerWavePerShuffle
```

#### Rule 6: Derived - Wave Count and BlockSize
```cpp
DERIVED: MWaves = MPerBlock / (MPerXDL * MXdlPerWave)
DERIVED: NWaves = NPerBlock / (NPerXDL * NXdlPerWave)
DERIVED: WaveSize = 64 (or 32 for some architectures)
CONSTRAINT: BlockSize == MWaves * NWaves * WaveSize
PARAMETERS: BlockSize, MPerBlock, NPerBlock, MPerXDL, NPerXDL, MXdlPerWave, NXdlPerWave
```

#### Rule 7: Derived - KPerThread and KPack
```cpp
DERIVED: KPack = max(lcm(AK1, BK1), MFMA_k_per_blk)
DERIVED: KPerThread = KPerBlock / (KPack)
CONSTRAINT: KPerThread % KPack == 0
PARAMETERS: KPerBlock, AK1, BK1, MPerXDL, NPerXDL (MFMA size affects KPack)
```

#### Rule 8: XDL Support
```cpp
CONSTRAINT: is_xdl_wmma_supported<AComputeDataType, BComputeDataType, MPerXDL, NPerXDL>()
PARAMETERS: AComputeDataType, BComputeDataType, MPerXDL, NPerXDL
```

#### Rule 9: TF32 Constraints
```cpp
CONSTRAINT: IF (AComputeDataType == tf32 OR BComputeDataType == tf32)
            THEN AComputeDataType == BComputeDataType
PARAMETERS: AComputeDataType, BComputeDataType
```

#### Rule 10: DirectLoad Limitation
```cpp
CONSTRAINT: IF DirectLoad == true THEN device == "gfx950"
CONSTRAINT: IF DirectLoad == true THEN 
            AElementwiseOperation == PassThrough AND 
            BElementwiseOperation == PassThrough
PARAMETERS: DirectLoad, AElementwiseOperation, BElementwiseOperation
```

### Upstream Validation Function Template

```cpp
template<typename DeviceOp>
struct TemplateParameterValidator {
    static constexpr bool IsValid() {
        // Rule 1: MPerBlock divisibility
        if constexpr(DeviceOp::MPerBlock % 
                     (DeviceOp::MPerXDL * DeviceOp::MXdlPerWave) != 0)
            return false;
        
        // Rule 2: NPerBlock divisibility
        if constexpr(DeviceOp::NPerBlock % 
                     (DeviceOp::NXdlPerWave * DeviceOp::NPerXDL) != 0)
            return false;
        
        // Rule 3: KPerBlock divisibility
        if constexpr(DeviceOp::KPerBlock % DeviceOp::AK1 != 0 ||
                     DeviceOp::KPerBlock % DeviceOp::BK1 != 0)
            return false;
        
        // Rule 4-5: Shuffle constraints
        if constexpr(DeviceOp::MXdlPerWave % DeviceOp::CShuffleMXdlPerWavePerShuffle != 0)
            return false;
        if constexpr(DeviceOp::NXdlPerWave % DeviceOp::CShuffleNXdlPerWavePerShuffle != 0)
            return false;
        
        // Rule 6: BlockSize validation
        constexpr auto MWaves = DeviceOp::MPerBlock / 
                               (DeviceOp::MPerXDL * DeviceOp::MXdlPerWave);
        constexpr auto NWaves = DeviceOp::NPerBlock / 
                               (DeviceOp::NPerXDL * DeviceOp::NXdlPerWave);
        constexpr auto WaveSize = 64; // Adjust based on architecture
        if constexpr(DeviceOp::BlockSize != MWaves * NWaves * WaveSize)
            return false;
        
        // Additional checks...
        return true;
    }
};
```

---

## 2. DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle

### Template Declaration
```cpp
template <
  index_t NDimSpatial,
  typename ALayout, typename BLayout, typename DsLayout, typename ELayout,
  typename ADataType, typename BDataType, typename AccDataType,
  typename CShuffleDataType, typename DsDataType, typename EDataType,
  typename AElementwiseOperation, typename BElementwiseOperation,
  typename CDEElementwiseOperation,
  ConvolutionForwardSpecialization ConvForwardSpecialization,
  GemmSpecialization GemmSpec,
  index_t NumGemmKPrefetchStage,
  index_t BlockSize,
  index_t MPerBlock, index_t NPerBlock, index_t KPerBlock,
  index_t AK1, index_t BK1,
  index_t MPerXDL, index_t NPerXDL,
  index_t MXdlPerWave, index_t NXdlPerWave,
  typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
  typename ABlockTransferThreadClusterArrangeOrder,
  typename ABlockTransferSrcAccessOrder,
  index_t ABlockTransferSrcVectorDim,
  index_t ABlockTransferSrcScalarPerVector,
  index_t ABlockTransferDstScalarPerVector_AK1,
  index_t ABlockLdsExtraM,
  typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
  typename BBlockTransferThreadClusterArrangeOrder,
  typename BBlockTransferSrcAccessOrder,
  index_t BBlockTransferSrcVectorDim,
  index_t BBlockTransferSrcScalarPerVector,
  index_t BBlockTransferDstScalarPerVector_BK1,
  index_t BBlockLdsExtraN,
  index_t CShuffleMXdlPerWavePerShuffle,
  index_t CShuffleNXdlPerWavePerShuffle,
  typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
  index_t CDEBlockTransferScalarPerVector_NPerBlock,
  typename AComputeDataType = ...,
  typename BComputeDataType = AComputeDataType,
  LoopScheduler LoopSched = make_default_loop_scheduler(),
  index_t NumGroupsToMerge = 1
>
```

### Compile-Time Constraints

Same as V3 (Rules 1-9 above) plus:

#### Rule 11: NumGroupsToMerge (compile-time checkable when C=1 is known)
```cpp
CONSTRAINT: IF NumGroupsToMerge > 1 THEN must use specific layouts
PARAMETERS: NumGroupsToMerge, ALayout, BLayout, ELayout
REQUIRED_LAYOUTS: NSpatialGC_GKSpatial_NSpatialGK OR 
                  NGCSpatial_GKSpatial_NGKSpatial OR
                  NGCHW_NGKHW OR NGCDHW_NGKDHW
```

---

## 3. DeviceGroupedConvFwdMultipleD_Wmma_CShuffle

### Template Declaration
```cpp
template <
  index_t NDimSpatial,
  typename ALayout, typename BLayout, typename DsLayout, typename ELayout,
  typename ADataType, typename BDataType, typename AccDataType,
  typename CShuffleDataType, typename DsDataType, typename EDataType,
  typename AElementwiseOperation, typename BElementwiseOperation,
  typename CDEElementwiseOperation,
  ConvolutionForwardSpecialization ConvForwardSpecialization,
  GemmSpecialization GemmSpec,
  index_t NumGemmKPrefetchStage,
  index_t BlockSize,
  index_t MPerBlock, index_t NPerBlock, index_t KPerBlock,
  index_t K1,
  index_t MPerWmma, index_t NPerWmma,
  index_t MRepeat, index_t NRepeat,
  typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
  typename ABlockTransferThreadClusterArrangeOrder,
  typename ABlockTransferSrcAccessOrder,
  index_t ABlockTransferSrcVectorDim,
  index_t ABlockTransferSrcScalarPerVector,
  index_t ABlockTransferDstScalarPerVector_AK1,
  bool ABlockLdsExtraM,
  typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
  typename BBlockTransferThreadClusterArrangeOrder,
  typename BBlockTransferSrcAccessOrder,
  index_t BBlockTransferSrcVectorDim,
  index_t BBlockTransferSrcScalarPerVector,
  index_t BBlockTransferDstScalarPerVector_BK1,
  bool BBlockLdsExtraN,
  index_t CShuffleMRepeatPerShuffle,
  index_t CShuffleNRepeatPerShuffle,
  typename CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
  index_t CDEShuffleBlockTransferScalarPerVector_NPerBlock,
  LoopScheduler LoopSched = make_default_loop_scheduler(),
  PipelineVersion PipelineVer = PipelineVersion::v1
>
```

### Compile-Time Constraints

#### Rule 12: WMMA Block Tiling (M dimension)
```cpp
CONSTRAINT: MPerBlock % (MPerWmma * MRepeat) == 0
PARAMETERS: MPerBlock, MPerWmma, MRepeat
```

#### Rule 13: WMMA Block Tiling (N dimension)
```cpp
CONSTRAINT: NPerBlock % (NPerWmma * NRepeat) == 0
PARAMETERS: NPerBlock, NPerWmma, NRepeat
```

#### Rule 14: WMMA Wave Count and BlockSize
```cpp
DERIVED: MWaves = MPerBlock / (MPerWmma * MRepeat)
DERIVED: NWaves = NPerBlock / (NPerWmma * NRepeat)
DERIVED: WaveSize = 32 (for WMMA architectures)
CONSTRAINT: BlockSize == MWaves * NWaves * WaveSize
PARAMETERS: BlockSize, MPerBlock, NPerBlock, MPerWmma, NPerWmma, MRepeat, NRepeat
```

#### Rule 15: WMMA KPack Constraints
```cpp
DERIVED: WmmaK = (K1 == 16) ? 32 : 16
DERIVED: KPack = math::integer_least_multiple(K1, WmmaK)
CONSTRAINT: KPack % (K1 * 2) == 0  // A_KRow = 2
CONSTRAINT: KPack % (K1 * 2) == 0  // B_KRow = 2
PARAMETERS: K1, KPerBlock
NOTE: KPerBlock should be chosen such that resulting KPack satisfies these
```

#### Rule 16: Device Architecture for WMMA
```cpp
CONSTRAINT: is_gfx11_supported() OR is_gfx12_supported()
CONSTRAINT: AccDataType == float OR AccDataType == int32_t
PARAMETERS: AccDataType
DEVICE: Must be gfx11 or gfx12
```

---

## 4. DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor

### Template Declaration
Same as standard XDL version (similar to #2 but without NumGroupsToMerge)

### Compile-Time Constraints
Rules 1-10 apply (same as DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle)

### Special Consideration
This operation handles tensors > 2GB by splitting, so descriptor size constraints are managed internally through splitting algorithm.

---

## 5. DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK

### Template Declaration
```cpp
template <
  index_t NDimSpatial,
  typename ADataType, typename BDataType,
  typename DsDataType, typename EDataType, typename AccDataType,
  typename ALayout, typename BLayout, typename DsLayout, typename ELayout,
  typename AElementwiseOperation, typename BElementwiseOperation,
  typename CDEElementwiseOperation,
  ConvolutionForwardSpecialization ConvForwardSpecialization,
  GemmSpecialization GemmSpec,
  index_t BlockSize,
  index_t MPerBlock, index_t NPerBlock,
  index_t K0PerBlock, index_t K1,
  index_t M1PerThread, index_t N1PerThread, index_t KPerThread,
  typename M1N1ThreadClusterM1Xs,
  typename M1N1ThreadClusterN1Xs,
  typename ABlockTransferThreadSliceLengths_K0_M0_M1_K1,
  typename ABlockTransferThreadClusterLengths_K0_M0_M1_K1,
  typename ABlockTransferThreadClusterArrangeOrder,
  typename ABlockTransferSrcAccessOrder,
  typename ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1,
  typename ABlockTransferSrcVectorTensorContiguousDimOrder,
  typename ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1,
  typename BBlockTransferThreadSliceLengths_K0_N0_N1_K1,
  typename BBlockTransferThreadClusterLengths_K0_N0_N1_K1,
  typename BBlockTransferThreadClusterArrangeOrder,
  typename BBlockTransferSrcAccessOrder,
  typename BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1,
  typename BBlockTransferSrcVectorTensorContiguousDimOrder,
  typename BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1,
  typename CThreadTransferSrcDstAccessOrder,
  index_t CThreadTransferSrcDstVectorDim,
  index_t CThreadTransferDstScalarPerVector
>
```

### Compile-Time Constraints

#### Rule 17: DL Thread Cluster Constraints
```cpp
DERIVED: BM1 = M1PerThread
DERIVED: BN1 = N1PerThread
DERIVED: BM = MPerBlock
DERIVED: BN = NPerBlock
CONSTRAINT: BM % BM1 == 0
CONSTRAINT: BN % BN1 == 0
PARAMETERS: MPerBlock, NPerBlock, M1PerThread, N1PerThread
```

#### Rule 18: DL Grid Decomposition
```cpp
CONSTRAINT: BM0 == 2
CONSTRAINT: BN0 == 2
DERIVED: BM0, BN0 are derived from thread cluster configuration
PARAMETERS: M1N1ThreadClusterM1Xs, M1N1ThreadClusterN1Xs
NOTE: Thread cluster must result in BM0=2, BN0=2
```

#### Rule 19: DL BlockSize Validation
```cpp
DERIVED: BM101, BM100, BN101, BN100 from thread cluster
CONSTRAINT: BlockSize == BM101 * BM100 * BN101 * BN100
PARAMETERS: BlockSize, M1N1ThreadClusterM1Xs, M1N1ThreadClusterN1Xs
```

#### Rule 20: DL Vector Transfer Dimensions
```cpp
CONSTRAINT: ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1[I1] == 1
CONSTRAINT: ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1[I2] == 1
CONSTRAINT: BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1[I1] == 1
CONSTRAINT: BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1[I2] == 1
PARAMETERS: ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1,
            BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1
```

#### Rule 21: DL Output Vector Dimension
```cpp
CONSTRAINT: CThreadTransferSrcDstVectorDim == 5
PARAMETERS: CThreadTransferSrcDstVectorDim
```

---

## Consolidated Validation Matrix

### For XDL-based Operations (V3, Standard, Large_Tensor)

| Parameter | Constraint | Depends On | Rule # |
|-----------|------------|------------|--------|
| MPerBlock | % (MPerXDL × MXdlPerWave) == 0 | MPerXDL, MXdlPerWave | 1 |
| NPerBlock | % (NXdlPerWave × NPerXDL) == 0 | NPerXDL, NXdlPerWave | 2 |
| KPerBlock | % AK1 == 0 AND % BK1 == 0 | AK1, BK1 | 3 |
| MXdlPerWave | % CShuffleMXdlPerWavePerShuffle == 0 | CShuffleMXdlPerWavePerShuffle | 4 |
| NXdlPerWave | % CShuffleNXdlPerWavePerShuffle == 0 | CShuffleNXdlPerWavePerShuffle | 5 |
| BlockSize | == MWaves × NWaves × WaveSize | All M/N tiling params | 6 |
| KPerThread | % KPack == 0 | KPerBlock, AK1, BK1 | 7 |
| MPerXDL, NPerXDL | is_xdl_supported(...) | AComputeDataType, BComputeDataType | 8 |
| AComputeDataType | == BComputeDataType (if TF32) | BComputeDataType | 9 |
| DirectLoad | Requires gfx950 & PassThrough ops | Device, ElementwiseOps | 10 |

### For WMMA-based Operations

| Parameter | Constraint | Depends On | Rule # |
|-----------|------------|------------|--------|
| MPerBlock | % (MPerWmma × MRepeat) == 0 | MPerWmma, MRepeat | 12 |
| NPerBlock | % (NPerWmma × NRepeat) == 0 | NPerWmma, NRepeat | 13 |
| BlockSize | == MWaves × NWaves × WaveSize | All M/N tiling params | 14 |
| KPack | % (K1 × 2) == 0 (for A and B) | K1 | 15 |
| AccDataType | == float OR == int32_t | - | 16 |
| Device | gfx11 or gfx12 only | - | 16 |

### For DL-based Operations

| Parameter | Constraint | Depends On | Rule # |
|-----------|------------|------------|--------|
| MPerBlock | % M1PerThread == 0 | M1PerThread | 17 |
| NPerBlock | % N1PerThread == 0 | N1PerThread | 17 |
| Thread Cluster | Must result in BM0=2, BN0=2 | M1N1ThreadClusterM1Xs, M1N1ThreadClusterN1Xs | 18 |
| BlockSize | == BM101 × BM100 × BN101 × BN100 | Thread cluster config | 19 |
| Vector Lengths | M/N dimensions == 1 | ABlockTransferSrcVectorTensorLengths | 20 |
| CThreadTransferSrcDstVectorDim | == 5 | - | 21 |

---

## Upstream Validation Strategy

### Phase 1: Compile-Time Template Validation

Create a validation struct that can be checked at compile-time:

```cpp
template<typename DeviceOp>
struct DeviceOpTemplateValidator {
    // Extract template parameters as constexpr values
    static constexpr auto BlockSize = DeviceOp::BlockSize;
    static constexpr auto MPerBlock = DeviceOp::MPerBlock;
    static constexpr auto NPerBlock = DeviceOp::NPerBlock;
    static constexpr auto KPerBlock = DeviceOp::KPerBlock;
    // ... extract all other parameters
    
    // Validate each rule
    static constexpr bool ValidateRule1() {
        if constexpr(is_xdl_based) {
            return MPerBlock % (MPerXDL * MXdlPerWave) == 0;
        }
        return true; // N/A for non-XDL
    }
    
    static constexpr bool ValidateRule2() {
        if constexpr(is_xdl_based) {
            return NPerBlock % (NXdlPerWave * NPerXDL) == 0;
        }
        return true;
    }
    
    // ... all rules
    
    static constexpr bool IsValid() {
        return ValidateRule1() && ValidateRule2() && /* ... all rules */;
    }
    
    // Optional: Generate error messages
    static constexpr const char* GetErrorMessage() {
        if constexpr(!ValidateRule1())
            return "MPerBlock not divisible by (MPerXDL * MXdlPerWave)";
        if constexpr(!ValidateRule2())
            return "NPerBlock not divisible by (NXdlPerWave * NPerXDL)";
        // ... etc
        return "Valid";
    }
};

// Usage:
static_assert(DeviceOpTemplateValidator<MyDeviceOp>::IsValid(),
              DeviceOpTemplateValidator<MyDeviceOp>::GetErrorMessage());
```

### Phase 2: Runtime Problem-Size Validation

After template instantiation, validate against actual problem dimensions:

```cpp
struct RuntimeValidator {
    // Check vector access divisibility
    static bool CheckVectorAccess(
        index_t C, index_t K, index_t G,
        index_t ABlockTransferSrcScalarPerVector,
        index_t BBlockTransferSrcScalarPerVector,
        index_t CDEBlockTransferScalarPerVector_NPerBlock,
        const Layout& layouts)
    {
        // Check C divisibility for A and B
        if (layouts.AUsesChannelVectorization())
            if (C % ABlockTransferSrcScalarPerVector != 0)
                return false;
        
        if (layouts.BUsesChannelVectorization())
            if (C % BBlockTransferSrcScalarPerVector != 0)
                return false;
        
        // Check K divisibility for E
        if (K % CDEBlockTransferScalarPerVector_NPerBlock != 0)
            return false;
        
        return true;
    }
    
    // Check tile size divisibility
    static bool CheckTileSizes(
        index_t M, index_t N, index_t K,
        index_t MPerBlock, index_t NPerBlock, index_t KPerBlock)
    {
        return (M % MPerBlock == 0) && 
               (N % NPerBlock == 0) && 
               (K % KPerBlock == 0);
    }
    
    // Check descriptor size limits
    static bool CheckDescriptorSizes(
        index_t M, index_t N, index_t K,
        size_t ADataTypeSize, size_t BDataTypeSize, size_t EDataTypeSize)
    {
        constexpr long_index_t TwoGB = (1L << 31);
        return (M * K * ADataTypeSize <= TwoGB) &&
               (N * K * BDataTypeSize <= TwoGB) &&
               (M * N * EDataTypeSize <= TwoGB);
    }
};
```

---

## Complete Validation Code Template

```cpp
#pragma once
#include <type_traits>

namespace ck {
namespace validation {

// Trait to detect operation type
template<typename T> struct is_xdl_based : std::false_type {};
template<typename T> struct is_wmma_based : std::false_type {};
template<typename T> struct is_dl_based : std::false_type {};

// Specialize for each operation type...

template<typename DeviceOp>
struct TemplateParameterValidator {
    // Extract parameters
    static constexpr auto BlockSize = DeviceOp::BlockSize;
    static constexpr auto MPerBlock = DeviceOp::MPerBlock;
    static constexpr auto NPerBlock = DeviceOp::NPerBlock;
    static constexpr auto KPerBlock = DeviceOp::KPerBlock;
    
    // XDL-specific validation
    template<typename T = DeviceOp>
    static constexpr bool ValidateXDL() {
        if constexpr(is_xdl_based<T>::value) {
            constexpr auto MPerXDL = T::MPerXDL;
            constexpr auto NPerXDL = T::NPerXDL;
            constexpr auto MXdlPerWave = T::MXdlPerWave;
            constexpr auto NXdlPerWave = T::NXdlPerWave;
            constexpr auto AK1 = T::AK1;
            constexpr auto BK1 = T::BK1;
            constexpr auto CShuffleMXdlPerWavePerShuffle = T::CShuffleMXdlPerWavePerShuffle;
            constexpr auto CShuffleNXdlPerWavePerShuffle = T::CShuffleNXdlPerWavePerShuffle;
            
            // Rule 1
            if constexpr(MPerBlock % (MPerXDL * MXdlPerWave) != 0)
                return false;
            
            // Rule 2
            if constexpr(NPerBlock % (NXdlPerWave * NPerXDL) != 0)
                return false;
            
            // Rule 3
            if constexpr(KPerBlock % AK1 != 0 || KPerBlock % BK1 != 0)
                return false;
            
            // Rule 4
            if constexpr(MXdlPerWave % CShuffleMXdlPerWavePerShuffle != 0)
                return false;
            
            // Rule 5
            if constexpr(NXdlPerWave % CShuffleNXdlPerWavePerShuffle != 0)
                return false;
            
            // Rule 6
            constexpr auto MWaves = MPerBlock / (MPerXDL * MXdlPerWave);
            constexpr auto NWaves = NPerBlock / (NPerXDL * NXdlPerWave);
            constexpr auto WaveSize = 64; // Adjust based on arch
            if constexpr(BlockSize != MWaves * NWaves * WaveSize)
                return false;
            
            return true;
        }
        return true; // N/A
    }
    
    // WMMA-specific validation
    template<typename T = DeviceOp>
    static constexpr bool ValidateWMMA() {
        if constexpr(is_wmma_based<T>::value) {
            constexpr auto MPerWmma = T::MPerWmma;
            constexpr auto NPerWmma = T::NPerWmma;
            constexpr auto MRepeat = T::MRepeat;
            constexpr auto NRepeat = T::NRepeat;
            constexpr auto K1 = T::K1;
            
            // Rule 12
            if constexpr(MPerBlock % (MPerWmma * MRepeat) != 0)
                return false;
            
            // Rule 13
            if constexpr(NPerBlock % (NPerWmma * NRepeat) != 0)
                return false;
            
            // Rule 14
            constexpr auto MWaves = MPerBlock / (MPerWmma * MRepeat);
            constexpr auto NWaves = NPerBlock / (NPerWmma * NRepeat);
            constexpr auto WaveSize = 32;
            if constexpr(BlockSize != MWaves * NWaves * WaveSize)
                return false;
            
            // Rule 15 - KPack constraints
            constexpr auto WmmaK = (K1 == 16) ? 32 : 16;
            // KPack computation is complex, may need runtime check
            
            return true;
        }
        return true; // N/A
    }
    
    // DL-specific validation  
    template<typename T = DeviceOp>
    static constexpr bool ValidateDL() {
        if constexpr(is_dl_based<T>::value) {
            constexpr auto M1PerThread = T::M1PerThread;
            constexpr auto N1PerThread = T::N1PerThread;
            
            // Rule 17
            if constexpr(MPerBlock % M1PerThread != 0)
                return false;
            if constexpr(NPerBlock % N1PerThread != 0)
                return false;
            
            // Rules 18-19 require analyzing thread cluster which is complex
            // May need partial compile-time, partial runtime validation
            
            return true;
        }
        return true; // N/A
    }
    
    // Master validation
    static constexpr bool IsValid() {
        return ValidateXDL<DeviceOp>() && 
               ValidateWMMA<DeviceOp>() && 
               ValidateDL<DeviceOp>();
    }
};

// Runtime validation for problem-dependent constraints
template<typename DeviceOp>
struct RuntimeParameterValidator {
    static bool Validate(
        index_t G, index_t N, index_t C, index_t K,
        const std::array<index_t, DeviceOp::NDimSpatial>& spatial_dims_in,
        const std::array<index_t, DeviceOp::NDimSpatial>& spatial_dims_out)
    {
        // Calculate spatial products
        index_t input_spatial_acum = 1;
        index_t output_spatial_acum = 1;
        for(index_t i = 0; i < DeviceOp::NDimSpatial; ++i) {
            input_spatial_acum *= spatial_dims_in[i];
            output_spatial_acum *= spatial_dims_out[i];
        }
        
        // Check vector access based on layout
        if constexpr(DeviceOp::UsesChannelVectorForA()) {
            if(C % DeviceOp::ABlockTransferSrcScalarPerVector != 0)
                return false;
        }
        
        if constexpr(DeviceOp::UsesChannelVectorForB()) {
            if(C % DeviceOp::BBlockTransferSrcScalarPerVector != 0)
                return false;
        }
        
        if(K % DeviceOp::CDEBlockTransferScalarPerVector_NPerBlock != 0)
            return false;
        
        // Check tile divisibility
        const auto MPadded = DeviceOp::CalculateMPadded(/* M calculated from problem */);
        const auto NPadded = DeviceOp::CalculateNPadded(/* N calculated from problem */);
        const auto KPadded = DeviceOp::CalculateKPadded(/* K calculated from problem */);
        
        // Additional checks based on specialization...
        
        return true;
    }
};

} // namespace validation
} // namespace ck
```

---

## Usage Example

```cpp
// At compile-time (before instantiation)
using MyDeviceOp = DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
    2,  // NDimSpatial
    /* ... layouts ... */
    256,  // BlockSize
    128,  // MPerBlock
    128,  // NPerBlock
    16,   // KPerBlock
    8,    // AK1
    8,    // BK1
    32,   // MPerXDL
    32,   // NPerXDL
    2,    // MXdlPerWave
    2,    // NXdlPerWave
    /* ... other params ... */
>;

// Compile-time validation
static_assert(ck::validation::TemplateParameterValidator<MyDeviceOp>::IsValid(),
              "Invalid template parameters for device operation!");

// At runtime (when problem sizes are known)
bool is_valid = ck::validation::RuntimeParameterValidator<MyDeviceOp>::Validate(
    G, N, C, K, input_spatial_dims, output_spatial_dims);
```

---

## Parameter Selection Guidelines

### Step 1: Choose Architecture-Specific Base Values
- **XDL**: MPerXDL=32, NPerXDL=32 (for fp16/bf16) or MPerXDL=16, NPerXDL=16 (for int8/fp8)
- **WMMA**: MPerWmma=16, NPerWmma=16

### Step 2: Determine Wave Configuration
- Calculate desired waves: MWaves × NWaves
- Ensure BlockSize = MWaves × NWaves × WaveSize

### Step 3: Calculate Block Tile Sizes
- **XDL**: MPerBlock = MPerXDL × MXdlPerWave × MWaves
- **WMMA**: MPerBlock = MPerWmma × MRepeat × MWaves

### Step 4: Choose K Decomposition
- Select AK1, BK1 (typically 4, 8, or 16)
- Ensure KPerBlock % AK1 == 0 and KPerBlock % BK1 == 0

### Step 5: Choose Shuffle Parameters
- CShuffleMXdlPerWavePerShuffle ≤ MXdlPerWave
- Ensure MXdlPerWave % CShuffleMXdlPerWavePerShuffle == 0

### Step 6: Select Vector Transfer Widths
- Match to memory access pattern and data type size
- Ensure alignment with channel/output dimensions

### Step 7: Validate Against All Rules
- Run compile-time validator
- Test with representative problem sizes
