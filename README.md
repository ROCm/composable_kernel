# Composable Kernel

## Methodology
Composable Kernel (CK) library aims to provide a programming model for writing performance critical kernels for Machine Learning workloads across multiple architectures including GPUs, CPUs, etc, through general purpose kernel progarmming languages, like HIP C++.

CK utilizes two concepts to achieve performance portabilatity and code maintainbility:
* A tile-based programming model
* Algorithm complexity reduction for complex ML operators, using innovative technique we call "Tensor Coordinate Transformation".

![ck_components drawio](https://user-images.githubusercontent.com/22615726/193490227-da9835fd-f942-4211-8131-f9d303f27c00.png)

## CK Structure
Current CK library are structure into 4 layers:
* "Templated Tile Operators"
* "Templated Kernel and Invoker" layer
* "Instantiated Kernel and Invoker" layer
* "Client API" layer

![ck_layers](https://user-images.githubusercontent.com/22615726/193490216-12d561d5-42ff-4a09-b65d-8e6ddfa2ac89.png)

## Contributor
### Developers
Chao Liu (https://github.com/asroy), Jing Zhang (https://github.com/zjing14), 2018-2022

Letao Qin (https://github.com/ltqin), Qianfeng Zhang (https://github.com/qianfengz), Liang Huang (https://github.com/carlushuang), Shaojie Wang (https://github.com/shaojiewang), 2019-2022

Anthony Chang (https://github.com/rosenrodt), Chunyu Lai (https://github.com/rocking5566), Illia Silin (https://github.com/illsilin), Adam Osewski (https://github.com/aosewski), Poyen Chen (https://github.com/poyenc), Rosty Geyyer (https://github.com/geyyer), 2022

Hanwen Chang, 2019-2021,

Tejash Shah, 2019-2020

Xiaoyan Zhou, 2020

Jianfeng Yan (https://github.com/j4yan), 2021-2022


### Product Manager
Jun Liu (jun.liu@amd.com)

### Contributors
Dan Yao (https://github.com/danyao12), Guangzhao Lu, Raman Jana (https://github.com/ramjana), Jehandad Khan (https://github.com/JehandadKhan)

### Acknowledgement
CK team works closely with Meta [AITemplate] (link to be added) team (led by Bing Xu, Ying Zhang). Most of the lucrative graph optimization opportunities in ML models were identified by AITemplate team, and we also co-designed many high performance fused kernels for AMD GPUs. Without this collaboration, CK would not reach its current potential.

## Citation
CK paper will be freely available on arXiv soon: 
```Realizing Tensor Operators Using Coordinate Transformations and Tile Based Programming```

## License
CK is released under the MIT license.


# Build CK

## Build docker image
```bash
DOCKER_BUILDKIT=1 docker build -t ck:latest -f Dockerfile .
```

## Launch docker
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

## Build CK
```bash
mkdir build && cd build

# Need to specify target ID, example below is for gfx908 and gfx90a
cmake                                                                                             \
-D CMAKE_PREFIX_PATH=/opt/rocm                                                                    \
-D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                                                         \
-D CMAKE_CXX_FLAGS="-O3"                                                                          \
-D CMAKE_BUILD_TYPE=Release                                                                       \
-D GPU_TARGETS=gfx908;gfx90a                                                                      \
..
```

### Build examples and tests
```bash
 make -j examples tests
 make test
```

Instructions for running each individual examples are under ```example/```


## Build ckProfiler
```bash
 make -j ckProfiler
```
Instructions for running ckProfiler are under ```profiler/```

## Install CK
```bash
make install
```

## Using CK as pre-built kernel library
Instructions for using CK as a pre-built kernel library are under ```client_example/```

## Caveat
### Kernel Timing and Verification
CK's own kernel timer will warn up kernel once, and then run it multiple times
to get average kernel time. For some kernels that use atomic add, this will cause
output buffer to be accumulated multiple times, causing verfication failure.
To work around it, do not use CK's own timer and do verification at the same time.
CK's own timer and verification in each example and ckProfiler can be enabled or
disabled from command line.
