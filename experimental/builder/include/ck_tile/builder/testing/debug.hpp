// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/testing/tensor_descriptor.hpp"
#include <iostream>
#include <locale>
#include <string>

/// This file contains a few debugging utilities, mainly focused around
/// tensor data. The idea is that the functionality in this file is not
/// necessarily used in any testing directly, but is available for the
/// programmer to help with debugging problems. These utilities themselves
/// should be tested just the same, though, so that they don't undergo
/// bitrot while they are not actively being used.

namespace ck_tile::builder::test {

namespace detail {

/// @brief Custom number punctuation for CK-Builder debugging.
///
/// During debugging, the locale is usually left to the default C locale.
/// The C locale does not have any thousands separator, which makes
/// large numbers hard to read. This is a specialization of the default
/// C++ number punctuation (`std::numpuct`) which separates thousands
/// using `'`. This character is chosen because C++14 allows number literals
/// to have this character.
///
/// @note When using this locale, be sure to restore the old locale in the
/// event that the user actually wants to use a non-standard locale.
///
/// @see std::numpunct
struct numpunct : std::numpunct<char>
{
    char do_thousands_sep() const override { return '\''; }
    std::string do_grouping() const override { return "\3"; }
};

} // namespace detail

/// @brief Print information about a tensor descriptor.
///
/// This function dumps useful information from a tensor descriptor to a
/// stream, `std::cout` by default. This includes the number of elements
/// in the tensor, the size of the backing space, lengths, strides, etc.
///
/// @note All information is printed using a lightly modified locale to
/// get a unified printing experience. The original locale in `stream` is
/// temporarily replaced, but restored before the function returns.
///
/// @tparam DT The tensor element datatype
/// @tparam RANK The rank (number of spatial dimensions) of the tensor.
///
/// @param name A name for the tensor descriptor.
/// @param desc The tensor descriptor to print.
/// @param stream The stream to print to, `std::cout` by default.
template <DataType DT, size_t RANK>
void print_descriptor(const char* name,
                      const TensorDescriptor<DT, RANK>& desc,
                      std::ostream& stream = std::cout)
{
    // Print name along with some generic info
    const auto size   = desc.get_element_size();
    const auto space  = desc.get_element_space_size();
    const auto bytes  = desc.get_element_space_size_in_bytes();
    const auto packed = desc.is_packed();

    const auto orig_locale = stream.getloc();
    const auto orig_flags  = stream.flags();

    stream.imbue(std::locale(std::locale(), new detail::numpunct{}));

    stream << "Descriptor \"" << name << "\":\n"
           << "  data type: " << DT << '\n'
           << "  size:      " << size << " elements\n"
           << "  space:     " << space << " elements (" << bytes << " bytes)\n"
           << "  lengths:   " << desc.get_lengths() << '\n'
           << "  strides:   " << desc.get_strides() << '\n'
           << "  packed:    " << (packed ? "yes" : "no") << '\n';

    stream.imbue(orig_locale);
    stream.flags(orig_flags);
}

} // namespace ck_tile::builder::test
