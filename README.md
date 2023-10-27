# Composable Kernel

Composable Kernel (CK) library aims to provide a programming model for writing performance critical kernels for machine learning workloads across multiple architectures including GPUs, CPUs, etc, through general purpose kernel languages, like HIP C++.

CK utilizes two concepts to achieve performance portability and code maintainability:
* A tile-based programming model
* Algorithm complexity reduction for complex ML operators, using innovative technique we call "Tensor Coordinate Transformation".

![ALT](/docs/data/ck_component.png "CK Components")

## Code Structure

Current CK library are structured into 4 layers:
* "Templated Tile Operators" layer
* "Templated Kernel and Invoker" layer
* "Instantiated Kernel and Invoker" layer
* "Client API" layer

![ALT](/docs/data/ck_layer.png "CK Layers")

## Documentation

Run the steps below to build documentation locally.

```
cd docs
pip3 install -r sphinx/requirements.txt
python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

## Contributors

The list of developers and contributors is here: [Contributors](/CONTRIBUTORS.md)

## Citation

If you use CK, please use following citations:
* CK paper will be freely available on arXiv soon: [Realizing Tensor Operators Using Coordinate Transformations and Tile Based Programming](???)
* [CITATION.cff](/CITATION.cff)

## License

CK is released under the MIT license. [License File](/LICENSE)


# Build Composable Kernel

We recommend building Composable Kernel inside docker containers that include 
all necessary packages. Pre-built docker images are available from this public repo: 

https://hub.docker.com/r/rocm/composable_kernel/tags

In order to build a new docker image, you can use the Dockerfile provided with the source code as shown below:

```bash
DOCKER_BUILDKIT=1 docker build -t ck:latest -f Dockerfile .
```

The docker container can then be launched, e.g., using the following command:

```bash
docker run                                     \
-it                                            \
--privileged                                   \
--group-add sudo                               \
-w /root/workspace                             \
-v ${PATH_TO_LOCAL_WORKSPACE}:/root/workspace  \
ck:latest                                      \
/bin/bash
```

After launching the container you can clone Composable Kernel source code from the github repository and strat the build:

```bash
git clone https://github.com/ROCmSoftwarePlatform/composable_kernel.git && \
cd composable_kernel && \
mkdir build && \
cd build
```
You will then need to set the GPU_TARGETS macro to specify GPU target architecture(s) that you want 
to execute CK on, e.g., gfx908, or gfx908;gfx90a;gfx940.
You are can specify either single or multiple architectures (use semicolon to separate), e.g.:

```bash
cmake                                                                                             \
-D CMAKE_PREFIX_PATH=/opt/rocm                                                                    \
-D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                                                         \
-D CMAKE_BUILD_TYPE=Release                                                                       \
-D GPU_TARGETS="gfx908;gfx90a"                                                                    \
..
```
After that you can build the entire CK library with just 

```bash
make -j
```

## Install CK

```bash
make -j install
```

## Build examples and tests

```bash
 make -j examples tests
```

## Build and run all examples and tests

```bash
 make -j check
```

Instructions for running each individual examples are under [example](/example)


## Build ckProfiler

```bash
 make -j ckProfiler
```
Instructions for running ckProfiler are under [profiler](/profiler)

Please note the "-j" option for building with multiple threads in parallel. This speeds up the build significantly.
Depending on the number of CPU cores and the amount of RAM on your system, it may be advizable to limit the number of threads.
By default, "-j" will try to launch one thread per CPU core. This could potentially cause the build to run out of memory and crash,
for example if you have a 128-core CPU and 64Gb of RAM. In such cases, you can try to reduce the number of threads to 32 by using "-j32".

If GPU_TARGETS is not set on the cmake command line, CK will be built for all targets supported by the 
current compiler.

Additional cmake flags can be used to significantly speed-up the build:

INSTANCES_ONLY (by default is OFF) must be set to ON in order to build only the instances and library
while skipping all tests, examples, and profiler. This is useful for libraries that use CK as a dependency.

DTYPES (by default not set) can be set to any subset of "fp64;fp32;fp16;fp8;bf16;int8" to build instances 
of select data types only. Currently, building of int8 instances is taking a lot of time (the compiler fix is in the works).

DL_KERNELS (by default is OFF) must be set to ON in order to build the gemm_dl and batched_gemm_multi_d_dl 
instances. Those instances are only needed for the NAVI2x platforms.

## Using sccache for building

The default CK docker images come with pre-installed version of sccache which supports clang being used as hip-compiler
" -x hip". Using sccache can help reduce the time to re-build the code from hours to 1 - 2 minutes. In order to
invoke sccache, you need to run

```bash
 sccache --start-server
```
and add the following flags to the cmake command line:

```bash
 -DCMAKE_CXX_COMPILER_LAUNCHER=sccache -DCMAKE_C_COMPILER_LAUNCHER=sccache
```

## Using CK as pre-built kernel library

Instructions for using CK as a pre-built kernel library are under [client_example](/client_example)

## Contributing

When you contribute to Composable Kernel, make sure to run `clang-format` on all the changed files. We highly recommend using git hooks that are managed by the `pre-commit` framework. To install hooks, run:

```bash
sudo script/install_precommit.sh
```

This way, `pre-commit` will add the appropriate hooks to your local repository and automatically run `clang-format` (and possibly additional checks) before any commit is created.

If you need to uninstall hooks from the repository, you can do so by running the following command:

```bash
script/uninstall_precommit.sh
```

If for any reason, you need to temporarily disable precommit hooks, you can add the `--no-verify` option to the `git commit` command.

## Caveat
### Kernel Timing and Verification

CK's own kernel timer will warn up kernel once, and then run it multiple times
to get average kernel time. For some kernels that use atomic add, this will cause
output buffer to be accumulated multiple times, causing verification failure.
To work around it, do not use CK's own timer and do verification at the same time.
CK's own timer and verification in each example and ckProfiler can be enabled or
disabled from command line.
