===============
CK Hello world
===============

-------------------------------------
Motivation
-------------------------------------

This tutorial is aimed at engineers dealing with artificial intelligence and machine learning who would like to optimize their pipelines and squeeze every performance drop by adding Composable Kernel (CK) library to their projects. We would like to make the CK library approachable so the tutorial is not based on the latest release and doesn't have all the bleeding edge features, but it will be reproducible now and forever.

During this tutorial we will have an introduction to the CK library, we will build it and run some examples and tests, so to say we will run a "Hello world" example. In future tutorials we will go in depth and breadth and get familiar with other tools and ways to integrate CK into your project.

-------------------------------------
Description
-------------------------------------

Modern AI technology solves more and more problems in all imaginable fields, but crafting fast and efficient workflows is still challenging. CK is one of the tools to make AI heavy lifting as fast and efficient as possible. CK is a collection of optimized AI operator kernels and tools to create new ones. The library has components required for majority of modern neural networks architectures including matrix multiplication, convolution, contraction, reduction, attention modules, variety of activation functions, fused operators and many more.

So how do we (almost) reach the speed of light? CK acceleration abilities are based on:

* Layered structure.
* Tile-based computation model.
* Tensor coordinate transformation.
* Hardware acceleration use.
* Support of low precision data types including fp16, bf16, int8 and int4.

If you are excited and need more technical details and benchmarking results - read this awesome `blog post <https://community.amd.com/t5/instinct-accelerators/amd-composable-kernel-library-efficient-fused-kernels-for-ai/ba-p/553224>`_.

For more details visit our `github repo <https://github.com/ROCmSoftwarePlatform/composable_kernel>`_.

-------------------------------------
Hardware targets
-------------------------------------

CK library fully supports "gfx908" and "gfx90a" GPU architectures and only some operators are supported for "gfx1030". Let's check the hardware you have at hand and decide on the target GPU architecture

==========     =========
GPU Target     AMD GPU
==========     =========
gfx908 	       Radeon Instinct MI100
gfx90a 	       Radeon Instinct MI210, MI250, MI250X
gfx1030        Radeon PRO V620, W6800, W6800X, W6800X Duo, W6900X, RX 6800, RX 6800 XT, RX 6900 XT, RX 6900 XTX, RX 6950 XT
==========     =========

There are also `cloud options <https://aws.amazon.com/ec2/instance-types/g4/>`_ you can find if you don't have an AMD GPU at hand.

-------------------------------------
Build the library
-------------------------------------

First let's clone the library and rebase to the tested version::

    git clone https://github.com/ROCmSoftwarePlatform/composable_kernel.git
    cd composable_kernel/
    git checkout tutorial_hello_world

To make our lives easier we prepared `docker images <https://hub.docker.com/r/rocm/composable_kernel>`_ with all the necessary dependencies. Pick the right image and create a container. In this tutorial we use "rocm/composable_kernel:ck_ub20.04_rocm5.3_release" image, it is based on Ubuntu 20.04, ROCm v5.3, compiler release version.

If your current folder is ${HOME}, start the docker container with::

    docker run  \
    -it  \
    --privileged  \
    --group-add sudo  \
    -w /root/workspace  \
    -v ${HOME}:/root/workspace  \
    rocm/composable_kernel:ck_ub20.04_rocm5.3_release  \
    /bin/bash

If your current folder is different from ${HOME}, adjust the line `-v ${HOME}:/root/workspace` to fit your folder structure.

Inside the docker container current folder is "~/workspace", library path is "~/workspace/composable_kernel", navigate to the library::

    cd composable_kernel/

Create and go to the "build" directory::

    mkdir build && cd build

In the previous section we talked about target GPU architecture. Once you decide which one is right for you, run cmake using the right GPU_TARGETS flag::

    cmake  \
    -D CMAKE_PREFIX_PATH=/opt/rocm  \
    -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc  \
    -D CMAKE_CXX_FLAGS="-O3"  \
    -D CMAKE_BUILD_TYPE=Release  \
    -D BUILD_DEV=OFF  \
    -D GPU_TARGETS="gfx908;gfx90a;gfx1030" ..

If everything went well the cmake run will end up with::

    -- Configuring done
    -- Generating done
    -- Build files have been written to: "/root/workspace/composable_kernel/build"

Finally, we can build examples and tests::

    make -j examples tests

If everything is smooth, you'll see::

    Scanning dependencies of target tests
    [100%] Built target tests

---------------------------
Run examples and tests
---------------------------

Examples are listed as test cases as well, so we can run all examples and tests with::

    ctest

You can check the list of all tests by running::

    ctest -N

We can also run them separately, here is a separate example execution::

    ./bin/example_gemm_xdl_fp16 1 1 1

The arguments "1 1 1" mean that we want to run this example in the mode: verify results with CPU, initialize matrices with integers and benchmark the kernel execution. You can play around with these parameters and see how output and execution results change.

If everything goes well and you have a device based on gfx908 or gfx90a architecture you should see something like::

    a_m_k: dim 2, lengths {3840, 4096}, strides {4096, 1}
    b_k_n: dim 2, lengths {4096, 4096}, strides {1, 4096}
    c_m_n: dim 2, lengths {3840, 4096}, strides {4096, 1}
    launch_and_time_kernel: grid_dim {480, 1, 1}, block_dim {256, 1, 1}
    Warm up 1 time
    Start running 10 times...
    Perf: 1.10017 ms, 117.117 TFlops, 87.6854 GB/s, DeviceGemmXdl<256, 256, 128, 4, 8, 32, 32, 4, 2> NumPrefetch: 1, LoopScheduler: Default, PipelineVersion: v1

Meanwhile, running it on a gfx1030 device should result in::

    a_m_k: dim 2, lengths {3840, 4096}, strides {4096, 1}
    b_k_n: dim 2, lengths {4096, 4096}, strides {1, 4096}
    c_m_n: dim 2, lengths {3840, 4096}, strides {4096, 1}
    DeviceGemmXdl<256, 256, 128, 4, 8, 32, 32, 4, 2> NumPrefetch: 1, LoopScheduler: Default, PipelineVersion: v1 does not support this problem

But don't panic, some of the operators are supported on gfx1030 architecture, so you can run a separate example like::

    ./bin/example_gemm_dl_fp16 1 1 1

and it should result in something nice similar to::

    a_m_k: dim 2, lengths {3840, 4096}, strides {1, 4096}
    b_k_n: dim 2, lengths {4096, 4096}, strides {4096, 1}
    c_m_n: dim 2, lengths {3840, 4096}, strides {4096, 1}
    arg.a_grid_desc_k0_m0_m1_k1_{2048, 3840, 2}
    arg.b_grid_desc_k0_n0_n1_k1_{2048, 4096, 2}
    arg.c_grid_desc_m_n_{ 3840, 4096}
    launch_and_time_kernel: grid_dim {960, 1, 1}, block_dim {256, 1, 1}
    Warm up 1 time
    Start running 10 times...
    Perf: 3.65695 ms, 35.234 TFlops, 26.3797 GB/s, DeviceGemmDl<256, 128, 128, 16, 2, 4, 4, 1>

Or we can run a separate test::

    ctest -R test_gemm_fp16

If everything goes well you should see something like::

    Start 121: test_gemm_fp16
    1/1 Test #121: test_gemm_fp16 ...................   Passed   51.81 sec

    100% tests passed, 0 tests failed out of 1

-----------
Summary
-----------

In this tutorial we took the first look at the Composable Kernel library, built it on your system and ran some examples and tests. Stay tuned, in the next tutorial we will run kernels with different configs to find out the best one for your hardware and task.

P.S.: Don't forget to switch out the cloud instance if you have launched one, you can find better ways to spend your money for sure!
