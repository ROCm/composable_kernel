// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cassert>
#include <iostream>

#include "ck_tile/dispatcher.hpp"

using namespace ck_tile::dispatcher;

DECL_FMHA_KERNEL_SET(decl_test_fmha_kernels,
                     .add(FmhaSignature()
                              .family("fwd")
                              .dtype("fp16")
                              .mode("batch")
                              .vlayout("r")
                              .hdim(128)
                              .mask("no")
                              .bias("no"),
                          FmhaAlgorithm().pipeline("qr_async").tile(128, 128, 32, 128, 32, 128),
                          "gfx942")
                         .add(FmhaSignature()
                                  .family("bwd_dq_dk_dv")
                                  .dtype("fp16")
                                  .mode("batch")
                                  .hdim(128)
                                  .mask("no")
                                  .bias("no"),
                              FmhaAlgorithm().pipeline("qr").tile(128, 128, 32, 128, 32, 128),
                              "gfx942"));

int main()
{
    const auto& set = FmhaKernelSetRegistry::instance().get("decl_test_fmha_kernels");
    assert(set.size() == 2);
    std::cout << "FMHA decl registry contains " << set.size() << " entries\n";
    return 0;
}
