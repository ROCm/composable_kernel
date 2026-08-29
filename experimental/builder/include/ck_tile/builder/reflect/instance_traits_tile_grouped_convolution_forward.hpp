// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// InstanceTraits specialization for GroupedConvolutionForwardKernel
//
// CRITICAL MAINTENANCE NOTE:
// This InstanceTraits file MUST be kept strictly in sync with the device implementation header:
//   ck_tile/ops/grouped_convolution/kernel/grouped_convolution_forward_kernel.hpp
// "In sync" means that the template parameter order, names, and types in the declaration below
// MUST EXACTLY MATCH those in the device implementation. If these diverge, you may encounter
// compilation errors, subtle template instantiation mismatches, or silent runtime bugs that are
// difficult to diagnose. Always update both files together and review changes carefully.

#pragma once

#include "instance_traits.hpp"
#include "instance_traits_util.hpp"

// Forward declaration to avoid circular dependency.
namespace ck_tile {

template <typename GroupedConvTraitsType_,
          typename TilePartitioner_,
          typename GemmPipeline_,
          typename EpiloguePipeline_>
struct GroupedConvolutionForwardKernel;

} // namespace ck_tile

namespace ck_tile {
namespace reflect {

namespace detail {

// Guards access to TilePartitioner members - primary template is depthwise (void partitioner).
template <typename TilePartitioner, bool IsDepthwise>
struct TilePartitionerFields
{
    static constexpr int kMPerBlock = 0;
    static constexpr int kNPerBlock = 0;
    static constexpr int kKPerBlock = 0;
    static constexpr int kMWarp     = 0;
    static constexpr int kNWarp     = 0;
    static constexpr int kKWarp     = 0;
    static constexpr int kMWarpTile = 0;
    static constexpr int kNWarpTile = 0;
    static constexpr int kKWarpTile = 0;
};

template <typename TilePartitioner>
struct TilePartitionerFields<TilePartitioner, false>
{
    static constexpr int kMPerBlock = TilePartitioner::MPerBlock;
    static constexpr int kNPerBlock = TilePartitioner::NPerBlock;
    static constexpr int kKPerBlock = TilePartitioner::KPerBlock;

    static constexpr int kMWarp = TilePartitioner::BlockGemmShape::BlockWarps::at(number<0>{});
    static constexpr int kNWarp = TilePartitioner::BlockGemmShape::BlockWarps::at(number<1>{});
    static constexpr int kKWarp = TilePartitioner::BlockGemmShape::BlockWarps::at(number<2>{});

    static constexpr int kMWarpTile = TilePartitioner::BlockGemmShape::WarpTile::at(number<0>{});
    static constexpr int kNWarpTile = TilePartitioner::BlockGemmShape::WarpTile::at(number<1>{});
    static constexpr int kKWarpTile = TilePartitioner::BlockGemmShape::WarpTile::at(number<2>{});
};

// Guards access to GemmPipeline scheduling members - primary template is depthwise.
template <typename GemmPipeline, bool IsDepthwise>
struct GemmPipelineFields
{
    static constexpr ck_tile::GemmPipelineScheduler kPipelineScheduler =
        ck_tile::GemmPipelineScheduler::Default;
    static constexpr bool kDoubleSmemBuffer = false;
    static constexpr int kNumWaveGroups     = 1;
};

template <typename GemmPipeline>
struct GemmPipelineFields<GemmPipeline, false>
{
    static constexpr ck_tile::GemmPipelineScheduler kPipelineScheduler = GemmPipeline::Scheduler;
    static constexpr bool kDoubleSmemBuffer = GemmPipeline::DoubleSmemBuffer;
    static constexpr int kNumWaveGroups     = GemmPipeline::NumWaveGroups;
};

} // namespace detail

// Specialization for GroupedConvolutionForwardKernel
template <typename GroupedConvTraitsType_,
          typename TilePartitioner_,
          typename GemmPipeline_,
          typename EpiloguePipeline_>
struct InstanceTraits<ck_tile::GroupedConvolutionForwardKernel<GroupedConvTraitsType_,
                                                               TilePartitioner_,
                                                               GemmPipeline_,
                                                               EpiloguePipeline_>>
{
    static constexpr bool kIsDepthwise = GroupedConvTraitsType_::IsDepthwise;

    // CK Tile Conv Traits
    static constexpr int kSpatialDim = GroupedConvTraitsType_::NDimSpatial;
    static constexpr ck_tile::ConvolutionSpecialization ConvSpecialization =
        GroupedConvTraitsType_::ConvSpecialization;

    // Layout types (void for depthwise - access guarded in instance_string())
    using InLayout  = typename GroupedConvTraitsType_::InLayout;
    using WeiLayout = typename GroupedConvTraitsType_::WeiLayout;
    using DsLayout  = typename GroupedConvTraitsType_::DsLayout;
    using OutLayout = typename GroupedConvTraitsType_::OutLayout;

    static constexpr int kVectorSizeA       = GroupedConvTraitsType_::VectorSizeA;
    static constexpr int kVectorSizeB       = GroupedConvTraitsType_::VectorSizeB;
    static constexpr int kVectorSizeC       = GroupedConvTraitsType_::VectorSizeC;
    static constexpr int kNumGroupsToMerge  = GroupedConvTraitsType_::NumGroupsToMerge;
    static constexpr bool kEnableSplitImage = GroupedConvTraitsType_::EnableSplitImage;
    static constexpr int kExplicitGemm      = GroupedConvTraitsType_::ExplicitGemm;

    // TilePartitioner fields - safe for both GEMM and depthwise (void) partitioners
    using TPF                       = detail::TilePartitionerFields<TilePartitioner_, kIsDepthwise>;
    static constexpr int kMPerBlock = TPF::kMPerBlock;
    static constexpr int kNPerBlock = TPF::kNPerBlock;
    static constexpr int kKPerBlock = TPF::kKPerBlock;
    static constexpr int kMWarp     = TPF::kMWarp;
    static constexpr int kNWarp     = TPF::kNWarp;
    static constexpr int kKWarp     = TPF::kKWarp;
    static constexpr int kMWarpTile = TPF::kMWarpTile;
    static constexpr int kNWarpTile = TPF::kNWarpTile;
    static constexpr int kKWarpTile = TPF::kKWarpTile;

    // Data types (both GEMM and depthwise pipelines expose ADataType / BDataType)
    using ADataType = typename GemmPipeline_::ADataType;
    using BDataType = typename GemmPipeline_::BDataType;

    // GemmPipeline scheduling fields - safe for both paths
    using GPF          = detail::GemmPipelineFields<GemmPipeline_, kIsDepthwise>;
    using GemmPipeline = GemmPipeline_;
    static constexpr ck_tile::GemmPipelineScheduler kPipelineScheduler = GPF::kPipelineScheduler;
    static constexpr bool kDoubleSmemBuffer                            = GPF::kDoubleSmemBuffer;
    static constexpr int kNumWaveGroups                                = GPF::kNumWaveGroups;

    // Epilogue Pipeline
    using AccDataType             = typename EpiloguePipeline_::AccDataType;
    using EDataType               = typename EpiloguePipeline_::ODataType;
    using DsDataType              = typename EpiloguePipeline_::DsDataType;
    using CDEElementwiseOperation = typename EpiloguePipeline_::CDElementwise;

    static std::string instance_string()
    {
        std::ostringstream oss;

        if constexpr(kIsDepthwise)
        {
            oss << "GroupedConvolutionForwardKernel";
            oss << "<" << kSpatialDim; // 1. NDimSpatial
            oss << ","
                << ck_tile::getConvSpecializationString(
                       ConvSpecialization);                 // 2. ConvSpecialization
            oss << "," << "Depthwise";                      // 3. Layout tag
            oss << "," << kVectorSizeA;                     // 4. InVecSize
            oss << "," << kVectorSizeB;                     // 5. WeiVecSize
            oss << "," << kVectorSizeC;                     // 6. OutVecSize
            oss << "," << GemmPipeline_::BlockSize;         // 7. BlockSize
            oss << "," << GemmPipeline_::TileOutH;          // 8. TileH
            oss << "," << GemmPipeline_::TileOutW;          // 9. TileW
            oss << "," << GemmPipeline_::FilterH;           // 10. FilterH
            oss << "," << GemmPipeline_::FilterW;           // 11. FilterW
            oss << "," << GemmPipeline_::StrideH;           // 12. StrideH
            oss << "," << GemmPipeline_::StrideW;           // 13. StrideW
            oss << "," << GemmPipeline_::NBatch;            // 14. NBatch
            oss << "," << GemmPipeline_::SubTileH;          // 15. SubTileH
            oss << "," << GemmPipeline_::SubTileW;          // 16. SubTileW
            oss << "," << detail::type_name<ADataType>();   // 17. InDataType
            oss << "," << detail::type_name<BDataType>();   // 18. WeiDataType
            oss << "," << detail::type_name<AccDataType>(); // 19. AccDataType
            oss << "," << detail::type_name<EDataType>();   // 20. OutDataType
            oss << ">";
        }
        else
        {
            oss << "GroupedConvolutionForwardKernel";
            oss << "<" << kSpatialDim; // 1. NDimSpatial
            oss << ","
                << ck_tile::getConvSpecializationString(
                       ConvSpecialization);                 // 2. ConvSpecialization
            oss << "," << detail::layout_name<InLayout>();  // 3. InLayout
            oss << "," << detail::layout_name<WeiLayout>(); // 4. WeiLayout
            oss << "," << detail::tuple_name<DsLayout>();   // 5. DsLayout
            oss << "," << detail::layout_name<OutLayout>(); // 6. OutLayout
            oss << "," << kVectorSizeA;                     // 7. VectorSizeA
            oss << "," << kVectorSizeB;                     // 8. VectorSizeB
            oss << "," << kVectorSizeC;                     // 9. VectorSizeC
            oss << "," << kNumGroupsToMerge;                // 10. NumGroupsToMerge
            oss << "," << kEnableSplitImage;                // 11. EnableSplitImage
            oss << "," << kExplicitGemm;                    // 12. ExplicitGemm
            oss << "," << kMPerBlock;                       // 13. MPerBlock
            oss << "," << kNPerBlock;                       // 14. NPerBlock
            oss << "," << kKPerBlock;                       // 15. KPerBlock
            oss << "," << kMWarp;                           // 16. MWarp
            oss << "," << kNWarp;                           // 17. NWarp
            oss << "," << kKWarp;                           // 18. KWarp
            oss << "," << kMWarpTile;                       // 19. MWarpTile
            oss << "," << kNWarpTile;                       // 20. NWarpTile
            oss << "," << kKWarpTile;                       // 21. KWarpTile
            oss << "," << detail::type_name<ADataType>();   // 22. ADataType
            oss << "," << detail::type_name<BDataType>();   // 23. BDataType
            oss << "," << GemmPipeline::GetPipelineName();  // 24. BlkGemmPipelineVer
            oss << ","
                << detail::pipeline_scheduler_name(kPipelineScheduler); // 25. BlkGemmPipeSched
            oss << "," << kDoubleSmemBuffer;                            // 26. DoubleSmemBuffer
            oss << "," << kNumWaveGroups;                               // 27. NumWaveGroups
            oss << "," << detail::type_name<AccDataType>();             // 28. AccDataType
            oss << "," << detail::type_name<EDataType>();               // 29. EDataType
            oss << "," << detail::tuple_name<DsDataType>();             // 30. DsDataType
            oss << ","
                << detail::elementwise_op_name<
                       CDEElementwiseOperation>(); // 31. CDEElementwiseOperation
            oss << ">";
        }

        return oss.str();
    }
};

} // namespace reflect
} // namespace ck_tile
