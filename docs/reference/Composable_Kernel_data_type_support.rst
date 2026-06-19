.. meta::
  :description: Composable Kernel library precision support
  :keywords: composable kernel, scalar, data types, support, CK, ROCm, precision

.. _composablekernel-data-type-support:

***************************************************
Composable Kernel precision support
***************************************************

This topic lists the data type support for the Composable Kernel library on AMD
GPUs.

This page lists the data types supported by the library itself and does not
indicate hardware support. A type listed here is only usable if the GPU
architecture also supports it; otherwise it is unsupported. For data type support
across the other ROCm libraries and by GPU architecture, see the
:doc:`Data types and precision support page <rocm:reference/precision-support>`.

.. _composablekernel-input-output-type-support:

Supported data types overview
=============================

The following table summarizes the input and output data types supported by
Composable Kernel.

.. list-table::
    :header-rows: 1

    *
      - Icon
      - Definition
    *
      - ✅
      - Fully supported as both an input and output type.
    *
      - ⚠️
      - Partially supported as an input or output type.

Data types not listed in the table below are not supported.

.. datatemplate:yaml:: /data/reference/precision-support.yaml

    .. list-table::
        :header-rows: 1
        :widths: 70, 30

        *
            - Data type
            - Support
    {% for data_type in data.data_types %}
        *
            - {{ data_type.type }}
            - {{ data_type.support }}
    {% endfor %}

Related references
==================

For details about the underlying C++ types used by Composable Kernel, see the
following pages:

* :doc:`Supported scalar data types <Composable_Kernel_supported_scalar_types>` —
  the C++ scalar types, their bit widths, and bit-layout descriptions.
* :doc:`Custom types <Composable_Kernel_custom_types>` — the custom types
  Composable Kernel defines for specialized operations.
