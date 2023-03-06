// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <limits>
#include <initializer_list>
#include <cstdlib>
#include <getopt.h>

#include "ck/ck.hpp"
#include "ck/utility/reduction_enums.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_elementwise_normalization_impl.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_common_util.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

#include "device/device_gemm_xdl_splitk_c_shuffle_simplified.hpp"
#include "ck/library/utility/literals.hpp"

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

// static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto MNPadding = ck::tensor_operation::device::GemmSpecialization::MNPadding;

using PT = ck::tensor_operation::element_wise::PassThrough;

using DTA = F16;
using DTB = F16;
using DTC = F16;
using DTQ = F32;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

using AOP = PT;
using BOP = PT;
using COP = PT;

using ReferenceGemmInstance =
    ck::tensor_operation::host::ReferenceGemm<DTA, DTB, DTC, DTQ, AOP, BOP, COP>;

namespace ck::tensor_operation::device::instance {

using device_simplified_gemm_split_k_rcr_instance = std::tuple<
    // clang-format off
        //#########################          |AData| BData| CData| AccData| ALayout| BLayout| CLayout|   A|   B|   C|          GEMM| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|     CBlockTransferClusterLengths|  CBlockTransfer|
        //#########################          | Type|  Type|  Type|    Type|        |        |        |    |    |    |Specialization|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave| _MBlock_MXdlPerWave_MWaveMPerXdl| ScalarPerVector|
        //#########################          |     |      |      |        |        |        |        |    |    |    |              |      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle| _NBlock_NXdlPerWave_NWaveNPerXdl|   _NWaveNPerXdl|
        //#########################          |     |      |      |        |        |        |        |    |    |    |              |      |      |      |      |   |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                                 |                |
// DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   GemmDefault,   256,   256,   128,     4,  8,   32,   32,    4,    2,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 64, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 32, 1, 8>,               8>,
// DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   GemmDefault,   256,   128,   256,     4,  8,   32,   32,    2,    4,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 64, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 32, 1, 8>,               8>,
// DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   GemmDefault,   128,   128,   128,     4,  8,   32,   32,    4,    2,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 16, 1, 8>,               8>,
// DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   GemmDefault,   256,   128,   128,     4,  8,   32,   32,    2,    2,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 64, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 32, 1, 8>,               8>,
// DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   GemmDefault,   128,   128,    64,     4,  8,   32,   32,    2,    2,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 32, 1, 4>,               8>,
// DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   GemmDefault,   128,    64,   128,     4,  8,   32,   32,    2,    2,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 16, 1, 8>,               8>,
// DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   GemmDefault,    64,    64,    64,     4,  8,   32,   32,    2,    2,  S<1, 4, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 16, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 16, 1, 4>,               8>,
// DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   GemmDefault,   256,   128,    64,     4,  8,   32,   32,    2,    1,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 64, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 32, 1, 8>,               8>,
// DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   GemmDefault,   256,    64,   128,     4,  8,   32,   32,    1,    2,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 64, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 32, 1, 8>,               8>,
// DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   GemmDefault,   128,   128,    32,     4,  8,   32,   32,    2,    1,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 32, 1, 4>,               8>,
// DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   GemmDefault,   128,    32,   128,     4,  8,   32,   32,    1,    2,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 16, 1, 8>,               8>,
// DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   GemmDefault,    64,    64,    32,     4,  8,   32,   32,    2,    1,  S<1, 4, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 16, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 16, 1, 4>,               8>,
// DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   GemmDefault,    64,    32,    64,     4,  8,   32,   32,    1,    2,  S<1, 4, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 16, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 16, 1, 4>,               8>,

DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   MNPadding,   256,   256,   128,     4,  8,   32,   32,    4,    2,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 64, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 32, 1, 8>,               8>,
DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   MNPadding,   256,   128,   256,     4,  8,   32,   32,    2,    4,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 64, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 32, 1, 8>,               8>,
DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   MNPadding,   128,   128,   128,     4,  8,   32,   32,    4,    2,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 16, 1, 8>,               8>,
DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   MNPadding,   256,   128,   128,     4,  8,   32,   32,    2,    2,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 64, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 32, 1, 8>,               8>,
DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   MNPadding,   128,   128,    64,     4,  8,   32,   32,    2,    2,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 32, 1, 4>,               8>,
DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   MNPadding,   128,    64,   128,     4,  8,   32,   32,    2,    2,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 16, 1, 8>,               8>,
DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   MNPadding,    64,    64,    64,     4,  8,   32,   32,    2,    2,  S<1, 4, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 16, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 16, 1, 4>,               8>,
DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   MNPadding,   256,   128,    64,     4,  8,   32,   32,    2,    1,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 64, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 32, 1, 8>,               8>,
DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   MNPadding,   256,    64,   128,     4,  8,   32,   32,    1,    2,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 64, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 32, 1, 8>,               8>,
DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   MNPadding,   128,   128,    32,     4,  8,   32,   32,    2,    1,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 32, 1, 4>,               8>,
DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   MNPadding,   128,    32,   128,     4,  8,   32,   32,    1,    2,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 16, 1, 8>,               8>,
DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   MNPadding,    64,    64,    32,     4,  8,   32,   32,    2,    1,  S<1, 4, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 16, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 16, 1, 4>,               8>,
DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   MNPadding,    64,    32,    64,     4,  8,   32,   32,    1,    2,  S<1, 4, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 16, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 16, 1, 4>,               8>,

DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   MNPadding,   256,   256,   128,     4,  8,   32,   32,    4,    2,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 64, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 32, 1, 8>,               2>,
DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   MNPadding,   256,   128,   256,     4,  8,   32,   32,    2,    4,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 64, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 32, 1, 8>,               2>,
DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   MNPadding,   128,   128,   128,     4,  8,   32,   32,    4,    2,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 16, 1, 8>,               2>,
DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   MNPadding,   256,   128,   128,     4,  8,   32,   32,    2,    2,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 64, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 32, 1, 8>,               2>,
DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   MNPadding,   128,   128,    64,     4,  8,   32,   32,    2,    2,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 32, 1, 4>,               2>,
DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   MNPadding,   128,    64,   128,     4,  8,   32,   32,    2,    2,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 16, 1, 8>,               2>,
DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   MNPadding,    64,    64,    64,     4,  8,   32,   32,    2,    2,  S<1, 4, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 16, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 16, 1, 4>,               2>,
DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   MNPadding,   256,   128,    64,     4,  8,   32,   32,    2,    1,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 64, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 32, 1, 8>,               2>,
DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   MNPadding,   256,    64,   128,     4,  8,   32,   32,    1,    2,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 64, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 32, 1, 8>,               2>,
DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   MNPadding,   128,   128,    32,     4,  8,   32,   32,    2,    1,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 32, 1, 4>,               2>,
DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   MNPadding,   128,    32,   128,     4,  8,   32,   32,    1,    2,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 16, 1, 8>,               2>,
DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   MNPadding,    64,    64,    32,     4,  8,   32,   32,    2,    1,  S<1, 4, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 16, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 16, 1, 4>,               2>,
DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   MNPadding,    64,    32,    64,     4,  8,   32,   32,    1,    2,  S<1, 4, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 16, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 16, 1, 4>,               2>
>;

// using device_simplified_gemm_split_k_rcr_instance = std::tuple<
// DeviceGemmXdlSplitKCShuffleSimplified< DTA,  DTB,  DTC,    DTQ,     Row,     Col,     Row, PT, PT, PT,   MNPadding,   256,   256,   128,     4,  8,   32,   32,    4,    2,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 4, 64, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,             3,              8,              8,      true,           1,           1,                   S<1, 32, 1, 8>,               8>
// >;

void add_device_gemm_xdl_splitk_rcr_simplified_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<ALayout, BLayout, CLayout, DTA, DTB, DTC, AOP, BOP, COP>>>&
        instances)
{
    add_device_operation_instances(instances,
                                   device_simplified_gemm_split_k_rcr_instance{});
}

}



struct ProblemSize
{
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA = 4096;
    ck::index_t StrideB = 4096;
    ck::index_t StrideC = 4096;
    ck::index_t KBatch = 1;
};

struct ExecutionConfig
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
};

template<typename Layout>
struct GetStride;

template<>
struct GetStride<Row>{
    static ck::index_t Get(ck::index_t /*Rows_*/, ck::index_t Cols_, ck::index_t Stride_)
    {
        if(Stride_ <= 0){
            return Cols_;
        }
        else
            return Stride_;
    }
};

template<>
struct GetStride<Col>{
    static ck::index_t Get(ck::index_t Rows_, ck::index_t /*Cols_*/, ck::index_t Stride_)
    {
        if(Stride_ <= 0){
            return Rows_;
        }
        else
            return Stride_;
    }
};

inline bool
parse_cmd_args(int argc, char* argv[], ProblemSize& problem_size, ExecutionConfig& config)
{
    if(argc < 4 || argc > 11){
        std::cerr << "arg1: verification (0=no, 1=yes)" << std::endl
                  << "arg2: initialization (0=no init, 1=integer value, 2=decimal value, 3=all one)"
                  << std::endl
                  << "arg3: time kernel (0=no, 1=yes)" << std::endl
                  << "arg4 to 9: M (256x), N(128x), K(32x), StrideA(0), StrideB(0), StrideC(0)" << std::endl;
        return false;
    }
    if(argc >= 4)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
    }
    if(argc >= 10)
    {
        problem_size.M = std::stoi(argv[4]);
        problem_size.N = std::stoi(argv[5]);
        problem_size.K = std::stoi(argv[6]);

        problem_size.StrideA = GetStride<ALayout>::Get(problem_size.M, problem_size.K, std::stoi(argv[7]));
        problem_size.StrideB = GetStride<BLayout>::Get(problem_size.K, problem_size.N, std::stoi(argv[8]));
        problem_size.StrideC = GetStride<CLayout>::Get(problem_size.M, problem_size.N, std::stoi(argv[9]));
    }
    if(argc >= 11)
    {
        problem_size.KBatch = std::stoi(argv[10]);
    }
    return true;
}

template<typename BasePtr>
bool run_gemm(const std::vector<std::unique_ptr<BasePtr>> & op_ptrs, const ProblemSize& problem_size, const ExecutionConfig& config)
{
#if defined(BUILD_INT4_EXAMPLE) && defined(CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4)
    static_assert(sizeof(ck::int4_t) == sizeof(int8_t));
#endif

    using namespace ck::literals;

    auto& [M, N, K, StrideA, StrideB, StrideC, KBatch] = problem_size;

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    Tensor<DTA> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<DTB> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));

    switch(config.init_method)
    {
    case 0: break;
    case 1:
        ck::utils::FillUniformDistributionIntegerValue<DTA>{-5.f, 5.f}(a_m_k);
        ck::utils::FillUniformDistributionIntegerValue<DTB>{-5.f, 5.f}(b_k_n);
        break;
    case 2:
        ck::utils::FillUniformDistribution<DTA>{-1.f, 1.f}(a_m_k);
        ck::utils::FillUniformDistribution<DTB>{-1.f, 1.f}(b_k_n);
        break;
    case 3:
        ck::utils::FillConstant<DTA>{1}(a_m_k.begin(), a_m_k.end());
        ck::utils::FillConstant<DTB>{1}(b_k_n.begin(), b_k_n.end());
        break;
    }

    Tensor<DTC> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<DTC> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;

    DeviceMem a_m_k_device_buf(sizeof(DTA) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_device_buf(sizeof(DTB) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(DTC) * c_m_n_device_result.mDesc.GetElementSpaceSize());

    a_m_k_device_buf.ToDevice(a_m_k.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n.mData.data());

    auto a_element_op = AOP{};
    auto b_element_op = BOP{};
    auto c_element_op = COP{};

    float fastest_time = std::numeric_limits<float>::max();
    float fastest_gb_per_sec = 0;
    float fastest_tflops = 0;
    int fastest_id = -1;

    for(int i = 0; i < static_cast<int>(op_ptrs.size()); i++) {
        auto& op_ptr = op_ptrs[i];

        // do GEMM
        auto argument_ptr = op_ptr->MakeArgumentPointer(
            static_cast<DTA*>(a_m_k_device_buf.GetDeviceBuffer()),
            static_cast<DTB*>(b_k_n_device_buf.GetDeviceBuffer()),
            static_cast<DTC*>(c_m_n_device_buf.GetDeviceBuffer()),
            M,
            N,
            K,
            StrideA,
            StrideB,
            StrideC,
            a_element_op,
            b_element_op,
            c_element_op,
            KBatch
            );
        auto invoker_ptr = op_ptr->MakeInvokerPointer();
        std::string op_name = op_ptr->GetTypeString();

        std::cout << op_name <<":" << std::flush;

        if(!op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            std::cout << " not supported" << std::endl;
            continue;
        }

        float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, config.time_kernel, 0});

        std::size_t flop = static_cast<std::size_t>(2) * M * N * K;
        std::size_t num_btype =
            sizeof(DTA) * M * K + sizeof(DTB) * K * N + sizeof(DTC) * M * N;

        float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

        float gb_per_sec = num_btype / 1.E6 / ave_time;

        std::cout << " " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s" <<  std::flush;

        if(ave_time < fastest_time){
            fastest_time = ave_time;
            fastest_id = i;
            fastest_gb_per_sec = gb_per_sec;
            fastest_tflops = tflops;
        }

        if(config.do_verification)
        {
            bool valid = false;
            auto ref_gemm    = ReferenceGemmInstance{};
            auto ref_invoker = ref_gemm.MakeInvoker();

            auto ref_argument = ref_gemm.MakeArgument(
                a_m_k, b_k_n, c_m_n_host_result, a_element_op, b_element_op, c_element_op);

            ref_invoker.Run(ref_argument);
            c_m_n_device_buf.FromDevice(c_m_n_device_result.mData.data());

            if(std::is_same<DTC, F16>::value)
            {
                valid = ck::utils::check_err(c_m_n_device_result, c_m_n_host_result, "fp16 incorrect result", 3e-3, 1e-3);
            }
            else{
                valid = ck::utils::check_err(c_m_n_device_result, c_m_n_host_result);
            }
            std::cout<< ", Valid:" << (valid ? "y":"n");
        }
        std::cout << std::endl;
    }

    if(config.time_kernel && fastest_id >= 0){
        std::cout << "fastest: " << op_ptrs[fastest_id]->GetTypeString() << ", " << fastest_time << " ms, "  << fastest_tflops << " TFlops, " << fastest_gb_per_sec << " GB/s"
         << std::endl;
    }

    return true;
}

int main(int argc, char* argv[])
{
    ProblemSize problem_size;
    ExecutionConfig config;

    parse_cmd_args(argc, argv, problem_size, config);

    using DeviceGemmSplitK = ck::tensor_operation::device::DeviceGemmSplitK<ALayout, BLayout, CLayout, DTA, DTB, DTC, AOP, BOP, COP>;

    std::vector<std::unique_ptr<DeviceGemmSplitK>>
        split_k_ptrs;
    ck::tensor_operation::device::instance::add_device_gemm_xdl_splitk_rcr_simplified_instances(split_k_ptrs);

    run_gemm<DeviceGemmSplitK>(split_k_ptrs, problem_size, config);

    return 0;
}
