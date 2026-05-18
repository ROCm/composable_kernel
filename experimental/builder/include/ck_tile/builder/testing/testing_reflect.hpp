// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>

#include "ck_tile/builder/testing/testing.hpp"

/// testing.hpp requires developers of a type of SIGNATURE to implement
/// quite a lot of functionality for each SIGNATURE. For example, next
/// to `Args`, `Inputs`, `Outputs`, `run`, they also have to define
/// `UniqueInputs`, `UniqueOutputs`, `alloc_inputs`, `alloc_outputs`,
/// and `validate`. The implementation of these latter few functions
/// is usually quite straight forward and adds a bunch of copy-paste
/// overhead. The functionality in this file offers an alternative
/// route: By implementing some reflection functionality in `Inputs`
/// and `Outputs`, we can automatically derive most of the functionality.

namespace ck_tile::builder::test {

/// @brief Check whether an `Input` or `Output` struct can be reflected.
///
/// In order to avoid having to manually redefine a bunch of types related to
/// each `Inputs`/`Outputs` structure, those structures can also provide some
/// "reflection" functionality. To this end, they should implement
/// `static void reflect(const Args<SIGNATURE> args&, auto inspect)`, where `inspect`
/// is called with information about each field in the struct. In more detail,
/// the signature of the `inspect` function is as follows:
///
///     void inspect(
///          // A human-readable name for the tensor
///          std::string_view name,
///          // Descriptor for the tensor in memory, usually obtained via `args`.
///          const TensorDescriptor<DT, RANK>& desc,
///          // Member pointer to a field of `T`, which is a GPU-memory pointer
///          // to the relevant tensor memory.
///          void* T::* ptr);
///
/// Here, `T` is `Inputs<SIGNATURE>` or `Outputs<SIGNATURE>`.
///
/// @see Inputs
/// @see Outputs
template <typename T, auto SIGNATURE>
concept TensorReflectable = requires(const Args<SIGNATURE>& args) {
    {
        T::reflect(args,
                   []([[maybe_unused]] std::string_view name,
                      // Note: This will be a TensorDescriptor<DT, RANK>, but the actual
                      // DT and RANK may differ depending on member.
                      [[maybe_unused]] const auto& desc,
                      [[maybe_unused]] void* T::*ptr) {})
    };
};

namespace detail {

/// The default alignment between tensors allocated separately
/// by `UniqueTensors`. This should be large enough to accomodate
/// any type. hipMalloc returns an alignment of 256 by default.
constexpr size_t TENSOR_ALIGNMENT = 256;

/// @brief Common type for automatically managing memory of sets of tensors.
///
/// This type implements the automatic memory management logic for `Inputs` and
/// `Outputs` that support reflection.
///
/// @tparam SIGNATURE The signature to specialize the structure for.
/// @tparam Tensors The `Inputs` or `Outputs` type corresponding to `SIGNATURE`.
template <auto SIGNATURE, typename Tensors>
    requires TensorReflectable<Tensors, SIGNATURE>
struct UniqueTensors
{
    /// @brief Allocate tensors.
    ///
    /// This function computes the total size of memory to allocate according to
    /// the tensors in `args`, and then allocates it as a continuous buffer.
    ///
    /// @param args The run-time arguments of the operation.
    explicit UniqueTensors(const Args<SIGNATURE>& args)
    {
        // First compute the total size of all tensors combined
        size_t total_size = 0;
        Tensors::reflect(args,
                         [&, this]([[maybe_unused]] std::string_view name,
                                   const auto& desc,
                                   [[maybe_unused]] void* Tensors::*ptr) {
                             total_size = align_fwd(total_size, TENSOR_ALIGNMENT);
                             total_size += desc.get_element_space_size_in_bytes();
                         });

        data_ = alloc_buffer(total_size);

        // Now assign the pointers based on the same offsets that
        // we computed in the first loop.
        size_t offset = 0;
        Tensors::reflect(args,
                         [&, this]([[maybe_unused]] std::string_view name,
                                   const auto& desc,
                                   [[maybe_unused]] void* Tensors::*ptr) {
                             offset        = align_fwd(offset, TENSOR_ALIGNMENT);
                             tensors_.*ptr = data_.get() + offset;
                             offset += desc.get_element_space_size_in_bytes();
                         });
    }

    /// @brief Return raw `Inputs` or `Outputs` type.
    ///
    /// @see ValidUniqueInputs
    /// @see ValidUniqueOutputs
    Tensors get() const { return tensors_; }

    private:
    /// Owning pointer of input memory
    DeviceBuffer data_;
    /// Struct with pointers to each tensor. Stored here so that we
    /// don't need to keep recomputing it.
    Tensors tensors_;
};

} // namespace detail

/// @brief Implementation of `UniqueInputs` for `Inputs` that support reflection.
///
/// @tparam SIGNATURE The signature to specialize for.
///
/// @see UniqueInputs
template <auto SIGNATURE>
    requires TensorReflectable<Inputs<SIGNATURE>, SIGNATURE>
struct UniqueInputs<SIGNATURE> : detail::UniqueTensors<SIGNATURE, Inputs<SIGNATURE>>
{
    using detail::UniqueTensors<SIGNATURE, Inputs<SIGNATURE>>::UniqueTensors;
};

/// @brief Implementation of `UniqueOutputs` for `Outputs` that support reflection.
///
/// @tparam SIGNATURE The signature to specialize for.
///
/// @see UniqueOutputs
template <auto SIGNATURE>
    requires TensorReflectable<Outputs<SIGNATURE>, SIGNATURE>
struct UniqueOutputs<SIGNATURE> : detail::UniqueTensors<SIGNATURE, Outputs<SIGNATURE>>
{
    using detail::UniqueTensors<SIGNATURE, Outputs<SIGNATURE>>::UniqueTensors;
};

/// @brief Implementation of `alloc_inputs` for `Inputs` that support reflection.
///
/// @tparam SIGNATURE The signature to specialize for.
///
/// @param args The run-time arguments of the operation.
///
/// @see alloc_inputs
template <auto SIGNATURE>
    requires TensorReflectable<Inputs<SIGNATURE>, SIGNATURE>
UniqueInputs<SIGNATURE> alloc_inputs(const Args<SIGNATURE>& args)
{
    static_assert(ValidUniqueInputs<SIGNATURE>, "sanity check");
    return UniqueInputs<SIGNATURE>(args);
}

/// @brief Implementation of `alloc_outputs` for `Outputs` that support reflection.
///
/// @tparam SIGNATURE The signature to specialize for.
///
/// @param args The run-time arguments of the operation.
///
/// @see alloc_outputs
template <auto SIGNATURE>
    requires TensorReflectable<Outputs<SIGNATURE>, SIGNATURE>
UniqueOutputs<SIGNATURE> alloc_outputs(const Args<SIGNATURE>& args)
{
    static_assert(ValidUniqueOutputs<SIGNATURE>, "sanity check");
    return UniqueOutputs<SIGNATURE>(args);
}

/// @brief Implementation of `validate` for `Outputs` that support reflection.
///
/// @tparam SIGNATURE The signature to specialize for.
///
/// @param args The run-time arguments of the operation.
/// @param actual The actual results, the results of the operation to-be-tested.
/// @param expected The expected results, the results of the reference implementation.
///
/// @see alloc_outputs
template <auto SIGNATURE>
    requires TensorReflectable<Outputs<SIGNATURE>, SIGNATURE>
ValidationReport
validate(const Args<SIGNATURE>& args, Outputs<SIGNATURE> actual, Outputs<SIGNATURE> expected)
{
    ValidationReport report;

    Outputs<SIGNATURE>::reflect(
        args, [&](std::string_view name, const auto& desc, void* Outputs<SIGNATURE>::*ptr) {
            report.check(name, desc, actual.*ptr, expected.*ptr);
        });

    return report;
}

} // namespace ck_tile::builder::test
