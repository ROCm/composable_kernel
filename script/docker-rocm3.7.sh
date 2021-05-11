WORKSPACE=$1
echo "workspace: " $WORKSPACE

docker run                                                                   \
-it                                                                          \
--rm                                                                         \
--privileged                                                                 \
--group-add sudo                                                             \
-w /root/workspace                                                           \
-v $WORKSPACE:/root/workspace                                                \
asroy/tensorflow:rocm3.7-tf2.3-dev-omp                                       \
/bin/bash

#--network host                                                               \
