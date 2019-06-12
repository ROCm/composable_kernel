#!/bin/bash

/root/workspace/rocprofiler_pkg/bin/rpl_run.sh --timestamp on -i /root/workspace/rocprofiler_pkg/input.xml -d ./trace ./driver/driver 0 10
