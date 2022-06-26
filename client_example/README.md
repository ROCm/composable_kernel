##
Client application links to CK library, and therefore CK library needs to be installed before building client applications.

## Docker script
```bash
docker run                                     \
-it                                            \
--privileged                                   \
--group-add sudo                               \
-w /root/workspace                             \
-v ${PATH_TO_LOCAL_WORKSPACE}:/root/workspace  \
rocm/tensorflow:rocm5.1-tf2.6-dev              \
/bin/bash
```

## Build
```bash
mkdir -p client_example/build
cd client_example/build
```

```bash
cmake                                                                 \
-D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                             \
-D CMAKE_PREFIX_PATH=/opt/rocm                                        \
..
```

### Build client example
```bash
 make -j 
```
