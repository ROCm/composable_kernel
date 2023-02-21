#!/bin/bash

set -eu

# Make this directory the PWD
cd "$(dirname "${BASH_SOURCE[0]}")"

# Build doxygen info
bash run_doxygen.sh

# Build sphinx docs
cd source
make clean
make -e SPHINXOPTS="-t html" html
make latexpdf
