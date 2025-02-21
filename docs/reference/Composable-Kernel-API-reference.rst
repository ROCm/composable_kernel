.. meta::
  :description: Composable Kernel documentation and API reference library
  :keywords: composable kernel, CK, ROCm, API, documentation

.. _api-reference:

********************************************************************
Composable Kernel API reference guide
********************************************************************

This document contains details of the APIs for the Composable Kernel library and introduces some of the key design principles that are used to write new classes that extend the functionality of the Composable Kernel library.

=================
DeviceMem
=================

.. doxygenstruct:: DeviceMem

=============================
Kernels For Flashattention
=============================

The Flashattention algorithm is defined in :cite:t:`dao2022flashattention`. This section lists
the classes that are used in the CK GPU implementation of Flashattention.

**Gridwise classes**

.. doxygenstruct:: ck::GridwiseBatchedGemmSoftmaxGemm_Xdl_CShuffle

**Blockwise classes**

.. doxygenstruct:: ck::ThreadGroupTensorSliceTransfer_v4r1

.. doxygenstruct:: ck::BlockwiseGemmXdlops_v2

.. doxygenstruct:: ck::BlockwiseSoftmax

**Threadwise classes**

.. doxygenstruct:: ck::ThreadwiseTensorSliceTransfer_StaticToStatic

.. bibliography::
