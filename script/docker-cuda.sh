WORKSPACE=$1
echo "workspace: " $WORKSPACE
sudo docker run  -it  -v $WORKSPACE:/root/workspace --group-add sudo --runtime=nvidia     asroy/cuda:10.1-cudnn7-devel-ubuntu18.04-latest /bin/bash
