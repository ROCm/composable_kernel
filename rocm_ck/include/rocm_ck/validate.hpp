// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Role: host -- debug-only runtime validation. No CK deps.
//
// Validates Args against a spec's physical tensor table before kernel launch.
// Catches "forgot to fill a tensor slot" at launch time instead of silent GPU
// corruption.
//
// Compiles to nothing in release builds (#ifndef NDEBUG).
// Works with any spec type that has num_physical_tensors / physical_tensors[]
// (GemmSpec, future spec types).

#pragma once

#include <rocm_ck/args.hpp>
#include <rocm_ck/physical_tensor.hpp>

#ifndef NDEBUG
#include <cstdio>
#include <cstdlib>
#endif

namespace rocm_ck {

/// Validate that all required tensor slots in Args are filled (non-null pointer).
///
/// Walks the spec's physical_tensors[] table and checks that each corresponding
/// Args::tensors[slot].ptr is non-null. On failure, prints the missing tensor
/// name to stderr and aborts.
///
/// Compiles to nothing in release builds.
///
/// Usage:
///   rocm_ck::validate(kernel_args, spec);
///   // launch kernel ...
template <typename Spec>
void validate(const Args& args, const Spec& spec)
{
#ifndef NDEBUG
    for(int i = 0; i < spec.num_physical_tensors; ++i)
    {
        int slot = spec.physical_tensors[i].args_slot;
        if(args.tensors[slot].ptr == nullptr)
        {
            std::fprintf(stderr,
                         "rocm_ck::validate: tensor \"%.*s\" (slot %d) has null pointer\n",
                         spec.physical_tensors[i].name.len,
                         spec.physical_tensors[i].name.data,
                         slot);
            std::abort();
        }
    }
#else
    (void)args;
    (void)spec;
#endif
}

} // namespace rocm_ck
