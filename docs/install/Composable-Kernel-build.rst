.. meta::
   :description: Composable Kernel build and install
   :keywords: composable kernel, CK, ROCm, API, documentation, install

***********************************************
Build and install Composable Kernel from source
***********************************************

To build Composable Kernel (CK) as part of the ROCm Core SDK, see `TheRock build
instructions <https://github.com/ROCm/TheRock/blob/main/docs/development/README.md>`__. TheRock is the recommended way to build ROCm components from source.

Alternatively, you can build Composable Kernel standalone using the following instructions.

Prerequisites
=============

The following prerequisites are required to build and install Composable Kernel:

* `CMake <https://cmake.org/>`_ 3.21 or later
* `Python <https://www.python.org/>`_ 3.8 or later
* `Git <https://git-scm.com/>`_
* A C++ compiler from the ROCm install, typically ``/opt/rocm/llvm/bin/clang++`` or ``hipcc`` on Linux

Composable Kernel uses HIP to compile device code. Set ``CMAKE_PREFIX_PATH`` to the ROCm install prefix and ``CMAKE_CXX_COMPILER`` to the ROCm Clang or ``hipcc`` path when you configure the build.

Pre-built Docker images that bundle ROCm, CMake, and the LLVM toolchain are on `Docker Hub <https://hub.docker.com/r/rocm/composable_kernel/tags>`_. 

Build and install
=================

Before you begin, clone the `Composable Kernel project <https://github.com/ROCm/rocm-libraries/tree/develop/projects/composablekernel>`_.

Use sparse checkout when cloning the Composable Kernel project:

.. code-block:: bash

   git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries
   git sparse-checkout init --cone
   git sparse-checkout set projects/composablekernel

Then use ``git checkout`` to check out the branch you need.

The develop branch is intended for users who want to preview new features or contribute to the Composable Kernel codebase.

If you don't intend to contribute to the codebase and won't be previewing features, use a branch that matches the version of ROCm installed on your system.

Create the ``build`` directory under ``rocm-libraries/projects/composablekernel``:

.. code-block:: bash

   cd projects/composablekernel
   mkdir build

Change directory to the ``build`` directory and generate the makefile using the ``cmake`` command. Two build options are required:

* ``CMAKE_PREFIX_PATH``: The ROCm installation path. ROCm is installed in ``/opt/rocm`` by default.
* ``CMAKE_CXX_COMPILER``: The path to the Clang compiler. Clang is found at ``/opt/rocm/llvm/bin/clang++`` by default.

.. code-block:: bash

   cd build
   cmake ../. -D CMAKE_PREFIX_PATH="/opt/rocm" -D CMAKE_CXX_COMPILER="/opt/rocm/llvm/bin/clang++" [-D<OPTION1=VALUE1> [-D<OPTION2=VALUE2>] ...]

Other build options are:

* ``DISABLE_DL_KERNELS``: Set this to "ON" to not build deep learning (DL) and data parallel primitive (DPP) instances. DL and DPP instances are useful on architectures that don't support XDL or WMMA.
* ``CK_USE_FP8_ON_UNSUPPORTED_ARCH``: Set to ``ON`` to build FP8 data type instances on gfx90a without native FP8 support.
* ``GPU_TARGETS``: Target GPU architectures. The standard HIP variable. Composable Kernel forwards the list to HIP, so HIP's compatibility rules apply. List one architecture or a small set from the same family. Set this option to build the tests, examples, and tutorials. Enclose the list in quotation marks and separate entries with semicolons (``;``). For example, ``cmake -D GPU_TARGETS="gfx908;gfx90a"``.
* ``GPU_ARCHS``: Target GPU architectures. Composable Kernel-specific. Use this option to build the Composable Kernel library for architectures from different families. If you set ``GPU_ARCHS``, Composable Kernel clears ``GPU_TARGETS`` before configuring HIP. Composable Kernel then builds only the library and skips the tests, examples, and tutorials. Enclose the list in quotation marks and separate entries with semicolons (``;``). For example, ``cmake -D GPU_ARCHS="gfx908;gfx1100"``.
* ``CMAKE_BUILD_TYPE``: The build type. Can be ``None``, ``Release``, ``Debug``, ``RelWithDebInfo``, or ``MinSizeRel``. CMake uses ``Release`` by default.

.. note::

   When both ``GPU_TARGETS`` and ``GPU_ARCHS`` are set, Composable Kernel uses ``GPU_ARCHS`` and clears ``GPU_TARGETS``. When neither is set, Composable Kernel picks a default target list based on the detected HIP version. Composable Kernel drops the unsupported architectures ``gfx900``, ``gfx906``, and ``gfx90c``, along with any target the installed compiler can't build for.

Build Composable Kernel using the generated makefile. With a default configuration, the build produces the Composable Kernel libraries, the ``ckProfiler`` benchmarking tool, the example binaries, the tutorial binaries, and the test binaries. The output is saved to ``build/lib/`` and ``build/bin/``. The Composable Kernel headers stay in the source tree until ``make install`` copies them to the install location.

.. note::

   A default Composable Kernel build can take an hour or more on a workstation. Most of the time goes to compiling operator instances, and build time scales with the number of GPU architectures you select. Limit ``GPU_TARGETS`` to your hardware to cut build time.

.. code-block:: bash

   make -j20

The ``-j`` option runs build steps in parallel. For example, ``-j20`` runs up to twenty jobs at a time. Each parallel job can use several gigabytes of memory because Composable Kernel instance files expand large template instantiations, and link jobs use more memory than compile jobs. Pick a ``-j`` value that fits your available RAM, and lower it if compiles fail with out-of-memory errors.

.. note::

   Don't run ``-j`` without a number. Bare ``-j`` launches an unbounded number of jobs and can exhaust memory.

   With Ninja, set ``-D CK_PARALLEL_COMPILE_JOBS=N`` and ``-D CK_PARALLEL_LINK_JOBS=M`` at configure time to cap compile and link jobs separately. These options have no effect with Make.

Install the Composable Kernel library:

.. code-block:: bash

   make install

After running ``make install``, the Composable Kernel files will be saved to the following locations:

* Library files: ``/opt/rocm/lib/``
* Header files: ``/opt/rocm/include/ck/`` and ``/opt/rocm/include/ck_tile/``
* Examples, tests, and ckProfiler: ``/opt/rocm/bin/``

For information about ckProfiler, see `the ckProfiler readme file <https://github.com/ROCm/rocm-libraries/tree/develop/projects/composablekernel/profiler/README.md>`_.

For information about running the examples and tests, see :doc:`Composable Kernel examples and tests <../tutorial/Composable-Kernel-examples>`.
