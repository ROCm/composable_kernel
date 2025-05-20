
EXE="$(find . -name tile_smoothquant -type f | head -n 1)"

$EXE -m=1 -n=1  -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=80  -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=128  -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=144  -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=168  -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=184  -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=256  -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=288  -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=344  -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=376  -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=448  -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=512  -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=924  -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=1024  -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=1078  -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=1996  -v=1 -prec=bf16 -repeat=1000
$EXE -m=700 -n=4080  -v=1 -prec=bf16 -repeat=1000

$EXE -m=700 -n=80  -v=1  -prec=fp16 -repeat=1000
$EXE -m=700 -n=128  -v=1 -prec=fp16 -repeat=1000
$EXE -m=700 -n=144  -v=1 -prec=fp16 -repeat=1000
$EXE -m=700 -n=168  -v=1 -prec=fp16 -repeat=1000
$EXE -m=700 -n=184  -v=1 -prec=fp16 -repeat=1000
$EXE -m=700 -n=256  -v=1 -prec=fp16 -repeat=1000
$EXE -m=700 -n=288  -v=1 -prec=fp16 -repeat=1000
$EXE -m=700 -n=344  -v=1 -prec=fp16 -repeat=1000
$EXE -m=700 -n=376  -v=1 -prec=fp16 -repeat=1000
$EXE -m=700 -n=448  -v=1 -prec=fp16 -repeat=1000
$EXE -m=700 -n=512  -v=1 -prec=fp16 -repeat=1000
$EXE -m=700 -n=924  -v=1 -prec=fp16 -repeat=1000
$EXE -m=700 -n=1024  -v=1 -prec=fp16 -repeat=1000
$EXE -m=700 -n=1078  -v=1 -prec=fp16 -repeat=1000
$EXE -m=700 -n=1996  -v=1 -prec=fp16 -repeat=1000
$EXE -m=700 -n=4080  -v=1 -prec=fp16 -repeat=1000