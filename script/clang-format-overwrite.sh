#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

git diff origin/develop...HEAD --name-only -- "*.{h,hpp,cpp,h.in,hpp.in,cpp.in,cl,cuh,cu,inc}" | \
  xargs --max-procs=16 --replace={} --verbose \
    'clang-format-12 -i -style=file {}'
