DRIVER=$1
ARCH=$2
cuobjdump -xelf $ARCH ./driver/$DRIVER && nvdisasm --print-code -g $DRIVER.$ARCH.cubin > $DRIVER.$ARCH.asm
