#!/bin/bash

## The following will be used for CI

set -x

## for float
bin/test_reduce_with_index -D 64,4,280,82  -R 0,1,2,3  0 1
bin/test_reduce_with_index -D 64,4,280,82  -R 0,1,2  0 1
bin/test_reduce_with_index -D 64,4,280,82  -R 0,1,3  0 1
bin/test_reduce_with_index -D 64,4,280,82  -R 0,2,3  0 1
bin/test_reduce_with_index -D 64,4,280,82  -R 1,2,3  0 1
bin/test_reduce_with_index -D 64,4,280,82  -R 0  0 1
bin/test_reduce_with_index -D 64,4,280,82  -R 1  0 1
bin/test_reduce_with_index -D 64,4,280,82  -R 2  0 1
bin/test_reduce_with_index -D 64,4,280,82  -R 3  0 1

## for float16
bin/test_reduce_with_index -D 64,4,280,82  -R 0,1,2,3  1 1
bin/test_reduce_with_index -D 64,4,280,82  -R 0,1,2  1 1
bin/test_reduce_with_index -D 64,4,280,82  -R 0,1,3  1 1
bin/test_reduce_with_index -D 64,4,280,82  -R 0,2,3  1 1
bin/test_reduce_with_index -D 64,4,280,82  -R 1,2,3  1 1
bin/test_reduce_with_index -D 64,4,280,82  -R 0  1 1
bin/test_reduce_with_index -D 64,4,280,82  -R 1  1 1
bin/test_reduce_with_index -D 64,4,280,82  -R 2  1 1
bin/test_reduce_with_index -D 64,4,280,82  -R 3  1 1

## for int8_t
bin/test_reduce_with_index -D 64,4,280,82  -R 0,1,2,3  3 1
bin/test_reduce_with_index -D 64,4,280,82  -R 0,1,2  3 1
bin/test_reduce_with_index -D 64,4,280,82  -R 0,1,3  3 1
bin/test_reduce_with_index -D 64,4,280,82  -R 0,2,3  3 1
bin/test_reduce_with_index -D 64,4,280,82  -R 1,2,3  3 1
bin/test_reduce_with_index -D 64,4,280,82  -R 0  3 1
bin/test_reduce_with_index -D 64,4,280,82  -R 1  3 1
bin/test_reduce_with_index -D 64,4,280,82  -R 2  3 1
bin/test_reduce_with_index -D 64,4,280,82  -R 3  3 1

## for bfloat16
bin/test_reduce_with_index -D 64,4,280,82  -R 0,1,2,3  5 1
bin/test_reduce_with_index -D 64,4,280,82  -R 0,1,2  5 1
bin/test_reduce_with_index -D 64,4,280,82  -R 0,1,3  5 1
bin/test_reduce_with_index -D 64,4,280,82  -R 0,2,3  5 1
bin/test_reduce_with_index -D 64,4,280,82  -R 1,2,3  5 1
bin/test_reduce_with_index -D 64,4,280,82  -R 0  5 1
bin/test_reduce_with_index -D 64,4,280,82  -R 1  5 1
bin/test_reduce_with_index -D 64,4,280,82  -R 2  5 1
bin/test_reduce_with_index -D 64,4,280,82  -R 3  5 1

set +x

