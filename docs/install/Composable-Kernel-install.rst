.. meta::
   :description: Installation instructions for Composable Kernel
   :keywords: ck, lib, composable, kernel, algorithm, install, sdk, rocm

.. _installation:

*************************
Install Composable Kernel
*************************

Before you begin, verify that your system is supported. For more information,
see :ref:`ROCm Core SDK components <rocm:release-components>`.

For advanced workflows, source builds, or custom configurations, see
:doc:`./Composable-Kernel-build`.

.. _install-rocm:

Install the ROCm Core SDK
=========================

Composable Kernel (CK) is included with the ROCm Core SDK on Linux and Windows.
For the most complete installation, we recommend that developers use the
``amdrocm-core-sdk`` meta package on Linux.

For instructions, see :doc:`Install AMD ROCm <rocm:install/rocm>`. Use the
selector panel on that page to view instructions appropriate for your system
environment.

.. _install-base:

Install the ROCm CK package on Linux
====================================

Alternatively, if you want to install Composable Kernel as part of the ROCm
without additional ROCm libraries and tools, install the ``amdrocm-ck``
package.

1. Complete the :doc:`ROCm installation prerequisites <rocm:install/rocm>` to
   install dependencies and configure GPU access permissions.

2. Install the ROCm CK package that matches your desired ROCm version. Package
   names use the following format:

   .. code-block:: shell-session

      amdrocm-ck<rocm_version>-<llvm_target>

   Where:

   * ``<rocm_version>`` is the ROCm Core SDK version to install. Omit this
     suffix to install the latest available version.

   * ``<llvm_target>`` (starting with ``gfx``) is used if you are installing
     for a single AMD GPU architecture. Omit this suffix to install for all
     architectures at the cost of disk space.

   For example, to install the latest Composable Kernel development package release for
   supported GPU architectures:

   .. tab-set::

      .. tab-item:: Debian-based distros

         .. code-block:: bash

            sudo apt install amdrocm-ck

      .. tab-item:: RHEL-based distros

         .. code-block:: bash

            sudo dnf install amdrocm-ck

      .. tab-item:: SLES

         .. code-block:: bash

            sudo zypper install amdrocm-ck

.. _install-nightly:

Install a nightly build
=======================

The `TheRock <https://github.com/ROCm/TheRock>`__ build system also publishes
nightly builds for the ROCm Core SDK and its components, including Composable
Kernel. See `Nightly release status
<https://github.com/ROCm/TheRock#nightly-release-status>`__ for details.

