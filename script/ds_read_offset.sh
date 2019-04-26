for((i=0;i<=4096;i=i+64))
do
    OFFSET=$i
    echo "if(offset == $OFFSET)"
    echo "{"
    echo "    asm volatile(\"\\n \\"
    echo "        ds_read_b128 %0, %1 offset:$OFFSET\n \\"
    echo "        \""
    echo "    : \"=v\"(r)"
    echo "    : \"v\"(__to_local(lds)));"
    echo "}"
done
