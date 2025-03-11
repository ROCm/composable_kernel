.. meta::
  :description: Composable Kernel supported precision types and custom type support
  :keywords: composable kernel, precision, data types, ROCm

.. _precision-support:

********************************
Precision support
********************************

The Composable Kernel (CK) library provides support for a wide range of precision types for tensor
operations. This document outlines the supported data types and how custom types can be used within
the library.

Supported data types
====================

The Composable Kernel (CK) library supports the following scalar data types:

.. list-table::
    :header-rows: 1
    :widths: 25 15 60

    * - Type
      - Bit Width
      - Description

    * - ``double``
      - 64-bit
      - Standard IEEE 754 double precision floating point

    * - ``float``
      - 32-bit
      - Standard IEEE 754 single precision floating point

    * - ``int32_t``
      - 32-bit
      - Standard signed 32-bit integer

    * - ``int8_t``
      - 8-bit
      - Standard signed 8-bit integer

    * - ``uint8_t``
      - 8-bit
      - Standard unsigned 8-bit integer

    * - ``bool``
      - 1-bit
      - Boolean type

    * - ``ck::half_t``
      - 16-bit
      - IEEE 754 half precision floating point with 5 exponent bits, 10 mantissa bits, and 1 sign bit

    * - ``ck::bhalf_t``
      - 16-bit
      - Brain floating point with 8 exponent bits, 7 mantissa bits, and 1 sign bit

    * - ``ck::f8_t``
      - 8-bit
      - 8-bit floating point (E4M3 format) with 4 exponent bits, 3 mantissa bits, and 1 sign bit

    * - ``ck::bf8_t``
      - 8-bit
      - 8-bit brain floating point (E5M2 format) with 5 exponent bits, 2 mantissa bits, and 1 sign bit

    * - ``ck::f4_t``
      - 4-bit
      - 4-bit floating point format

    * - ``ck::f6_t``
      - 6-bit
      - 6-bit floating point format (e2m3 format)

    * - ``ck::bf6_t``
      - 6-bit
      - 6-bit brain floating point format (e3m2 format)

Vector types
============

In addition to scalar types, the library provides mechanisms to work with vector data for more
efficient data processing and computation. Rather than predefined vector types, the library offers
template utilities that allow users to create vector types with customizable widths.

The library provides the ``ck::vector_type`` template structure and related utilities to enable vector
operations. These utilities allow users to create vector types of any supported scalar type with custom
widths. For example:

* ``ck::vector_type<float, 4>`` for a 4-element float vector
* ``ck::vector_type<ck::half_t, 8>`` for an 8-element half-precision vector
* ``ck::vector_type<int8_t, 16>`` for a 16-element 8-bit integer vector

The template system is designed to prevent creating "vectors of vectors" accidentally, instead properly
flattening nested vector types into a single wider vector.

Vector operations can significantly improve computational throughput for data-parallel operations. For
vector operations to be valid, the underlying scalar type must be one of the supported native types
listed above, or a custom type that properly implements the required operations.

Custom type support
===================

Beyond the native types, the library's template-based architecture supports the use of custom
user-defined types. This allows users to implement specialized numerical formats tailored to their specific
applications.

To use custom types with the library, users need to define a C++ type that implements the necessary
operations for tensor computations. The library is designed to work with any type that provides the
appropriate operators and functions needed for the specific operations being performed.

Here is an example of a simple custom type that can be used:

**Complex Half-Precision Type**

.. code-block:: cpp

    struct complex_half_t
    {
        half_t real;
        half_t img;
    };

**Complex Half-Precision Type with Constructor**

.. code-block:: cpp

    struct complex_half_t
    {
        using type = half_t;
        type real;
        type img;

        complex_half_t() : real{type{}}, img{type{}} {}
        complex_half_t(type real_init, type img_init) : real{real_init}, img{img_init} {}
    };

When creating custom types for use with the library, consider implementing:

- Appropriate constructors and initialization methods
- Required arithmetic operators if the type will be used in computational operations
- Any conversion functions needed to interface with other parts of your application

Custom types can be particularly useful for specialized applications such as complex number arithmetic,
custom quantization schemes, or domain-specific number representations.
