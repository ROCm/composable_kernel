.. meta::
   :description: Composable Kernel build and install
   :keywords: composable kernel, CK, ROCm, API, documentation, install

***********************************************
Build and install Composable Kernel from source
***********************************************

To build Composable Kernel as part of the ROCm Core SDK, see `TheRock build
instructions
<https://github.com/ROCm/TheRock/blob/main/docs/development/README.md>`__.
TheRock is the recommended way to build ROCm components from source.

Alternatively, you can build Composable Kernel standalone using the following
instructions.

Prerequisites
=============

The following prerequisites are required to build and install Composable Kernel:

* cmake
* hip-rocclr
* iputils-ping
* jq
* libelf-dev
* libncurses5-dev
* libnuma-dev
* libpthread-stubs0-dev
* llvm-amdgpu
* mpich
* net-tools
* python3
* python3-dev
* python3-pip
* redis
* rocm-llvm-dev
* zlib1g-dev
* libzstd-dev
* openssh-server
* clang-format-18

Docker images that include all the required prerequisites for building Composable Kernel are available on `Docker Hub <https://hub.docker.com/r/rocm/composable_kernel/tags>`_.

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

* ``DISABLE_DL_KERNELS``: Set this to "ON" to not build deep learning (DL) and data parallel primitive (DPP) instances. 

  .. note::

     DL and DPP instances are useful on architectures that don't support XDL or WMMA.

* ``CK_USE_FP8_ON_UNSUPPORTED_ARCH``: Set to ``ON`` to build FP8 data type instances on gfx90a without native FP8 support.
* ``GPU_TARGETS``: Target architectures. Target architectures in this list must all be different versions of the same architectures. Enclose the list of targets in quotation marks. Separate multiple targets with semicolons (``;``). For example, ``cmake -D GPU_TARGETS="gfx908;gfx90a"``. This option is required to build tests and examples.
* ``GPU_ARCHS``: Target architectures. Target architectures in this list are not limited to different versions of the same architectures. Enclose the list of targets in quotation marks. Separate multiple targets with semicolons (``;``). For example, ``cmake -D GPU_TARGETS="gfx908;gfx1100"``.
* ``CMAKE_BUILD_TYPE``: The build type. Can be ``None``, ``Release``, ``Debug``, ``RelWithDebInfo``, or ``MinSizeRel``. CMake will use ``Release`` by default.

.. note::

   If neither ``GPU_TARGETS`` nor ``GPU_ARCHS`` is specified, Composable Kernel will be built for all targets supported by the compiler.

Build Composable Kernel using the generated makefile. This will build the library, the examples, and the tests, and save them to ``bin``.

.. code-block:: bash

   make -j20

The ``-j`` option speeds up the build by using multiple threads in parallel. For example, ``-j20`` uses twenty threads in parallel. On average, each thread will use 2GB of memory. Make sure that the number of threads you use doesn't exceed the available memory in your system.

Using ``-j`` alone will launch an unlimited number of threads and is not recommended.

Install the Composable Kernel library:

.. code-block:: bash

   make install

After running ``make install``, the Composable Kernel files will be saved to the following locations:

* Library files: ``/opt/rocm/lib/``
* Header files: ``/opt/rocm/include/ck/`` and ``/opt/rocm/include/ck_tile/``
* Examples, tests, and ckProfiler: ``/opt/rocm/bin/``

For information about ckProfiler, see `the ckProfiler readme file <https://github.com/ROCm/rocm-libraries/tree/develop/projects/composablekernel/profiler/README.md>`_.

For information about running the examples and tests, see :doc:`Composable Kernel examples and tests <../tutorial/Composable-Kernel-examples>`.
