// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

// InstanceTraits specialization for GroupedConvolutionForwardKernel
//
// CRITICAL MAINTENANCE NOTE:
// This InstanceTraits file MUST be kept strictly in sync with the device implementation header:
//   ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp
// "In sync" means that the template parameter order, names, and types in the declaration below
// MUST EXACTLY MATCH those in the device implementation. If these diverge, you may encounter
// compilation errors, subtle template instantiation mismatches, or silent runtime bugs that are
// difficult to diagnose. Always update both files together and review changes carefully.

#pragma once

#include "instance_traits.hpp"

// Forward declaration to avoid circular dependency.
namespace ck::tensor_operation::device {

template <typename GroupedConvTraitsType_,
          typename TilePartitioner_,
          typename GemmPipeline_,
          typename EpiloguePipeline_>
struct GroupedConvolutionForwardKernel;

} // namespace ck::tensor_operation::device

namespace ck_tile::reflect {

// Specialization for GroupedConvolutionForwardKernel
template <typename GroupedConvTraitsType_,
          typename TilePartitioner_,
          typename GemmPipeline_,
          typename EpiloguePipeline_>
struct InstanceTraits<ck::tensor_operation::device::GroupedConvolutionForwardKernel<
            GroupedConvTraitsType_,
            TilePartitioner_,
            GemmPipeline_,
            EpiloguePipeline_>>
{
    // CK Tile Conv Traits
    // Spatial dimension
    static constexpr int kSpatialDim = GroupedConvTraitsType_::NDimSpatial;
    // Specialization
    static constexpr ck_tile::ConvolutionSpecialization
        kConvForwardSpecialization = GroupedConvTraitsType_::ConvForwardSpecialization;
    // Layout types
    using InLayout  = typename GroupedConvTraitsType_::InLayout;
    using WeiLayout  = typename GroupedConvTraitsType_::WeiLayout;
    using DsLayout = typename GroupedConvTraitsType_::DsLayout;
    using OutLayout  = typename GroupedConvTraitsType_::OutLayout;
    // Vector size
    static constexpr int kVectorSizeA = GroupedConvTraitsType_::VectorSizeA;
    static constexpr int kVectorSizeB = GroupedConvTraitsType_::VectorSizeB;
    static constexpr int kVectorSizeC = GroupedConvTraitsType_::VectorSizeC;
    // Num Groups To Merge
    static constexpr int kNumGroupsToMerge = GroupedConvTraitsType_::NumGroupsToMerge;
    // Split image (large tensors)
    static constexpr bool kEnableSplitImage = GroupedConvTraitsType_::EnableSplitImage;
    
    // TilePartitioner
    // Block configuration
    static constexpr int kMPerBlock = TilePartitioner_::MPerBlock;
    static constexpr int kNPerBlock = TilePartitioner_::NPerBlock;
    static constexpr int kKPerBlock = TilePartitioner_::KPerBlock;

    static constexpr int kMWarp = TilePartitioner_::BlockGemmShape::BlockWarps::at(number<0>{});
    static constexpr int kNWarp = TilePartitioner_::BlockGemmShape::BlockWarps::at(number<1>{});
    static constexpr int kKWarp = TilePartitioner_::BlockGemmShape::BlockWarps::at(number<2>{});

    static constexpr int kMWarpTile = TilePartitioner_::BlockGemmShape::WarpTile::at(number<0>{});
    static constexpr int kNWarpTile = TilePartitioner_::BlockGemmShape::WarpTile::at(number<1>{});
    static constexpr int kKWarpTile = TilePartitioner_::BlockGemmShape::WarpTile::at(number<2>{});

    // Gemm Pipeline
    using GemmPipeline = GemmPipeline_;
    static constexpr ck_tile::BlockGemmPipelineScheduler kPipelineScheduler = GemmPipeline_::Scheduler;
    static constexpr bool HasHotLoop = GemmPipeline_::HasHotLoop;
    static constexpr ck_tile::TailNumber = GemmPipeline_::TailNum;
    static constexpr bool kDoubleSmemBuffer = GemmPipeline_::DoubleSmemBuffer;
    using AGemmLayout = typename GemmPipeline_::ALayout;
    using BGemmLayout = typename GemmPipeline_::BLayout;
    using CGemmLayout = typename GemmPipeline_::CLayout;
    static constexpr int kNumWaveGroups = GemmPipeline_::NumWaveGroups;
    
    // Epilogue Pipeline
    using CDEElementwiseOperation = typename EpiloguePipeline_::CDElementwise;
    static constexpr ck_tile::memory_operation_enum kMemoryOperation = EpiloguePipeline_::MemoryOperation;

    // Static member function to generate instance string
    static std::string instance_string()
    {
        std::ostringstream oss;

        // Kernel type name
        oss << "GroupedConvolutionForwardKernel";

        // Template parameters in exact order matching InstanceTraits member order
        oss << "<" << kSpatialDim;                                  // 1. NDimSpatial
        oss << "," << ck_tile::getConvSpecializationString(kConvForwardSpecialization); // 2. ConvForwardSpecialization
        oss << "," << detail::layout_name<InLayout>();               // 3. InLayout
        oss << "," << detail::layout_name<WeiLayout>();               // 4. BLayout
        oss << "," << detail::tuple_name<DsLayout>();               // 5. DsLayout
        oss << "," << detail::layout_name<OutLayout>();               // 6. ELayout
        oss << "," << kVectorSizeA                               // 7. VectorSizeA
        oss << "," << kVectorSizeB                               // 8. VectorSizeB
        oss << "," << kVectorSizeC                               // 9. VectorSizeC
        oss << "," << kNumGroupsToMerge;                           // 10. NumGroupsToMerge
        oss << "," << kEnableSplitImage;                            // 11. EnableSplitImage
        oss << "," << kMPerBlock;                                // 12. MPerBlock
        oss << "," << kNPerBlock;                                // 13. NPerBlock
        oss << "," << kKPerBlock;                                // 14. KPerBlock
        oss << "," << kMWarp;                                    // 15. MWarp
        oss << "," << kNWarp;                                    // 16. NWarp
        oss << "," << kKWarp;                                    // 17. KWarp
        oss << "," << kMWarpTile;                                // 18. MWarpTile
        oss << "," << kNWarpTile;                                // 19. NWarpTile
        oss << "," << kKWarpTile;                                // 20. KWarpTile
        oss << "," << GemmPipeline::GetPipelineName();     // 21. BlkGemmPipelineVer
        oss << "," << detail::pipeline_scheduler_name(kPipelineScheduler); // 22. BlkGemmPipeSched
        oss << "," << HasHotLoop;
        oss << "," << detail::tail_number_name(kTailNumber);
        oss << "," << detail::layout_name<AGemmLayout>();               // 23. AGemmLayout
        oss << "," << detail::layout_name<BGemmLayout>();               // 24. BGemmLayout
        oss << "," << detail::layout_name<CGemmLayout>();               // 25. CGemmLayout
        oss << "," << kNumWaveGroups;                                // 26. NumWaveGroups
        oss << ","
            << detail::elementwise_op_name<CDEElementwiseOperation>(); // 27.
                                                                       // CDEElementwiseOperation
        oss << ","<< ck_tile::mem_op_string<kMemoryOperation>();
        oss << ">";

        return oss.str();
    }
};

} // namespace ck_tile::reflect
