#!/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

current_year=$(date +%Y)
exit_code=0

# print names of files that are being checked
echo "Checking copyright year for the following files:"
for file in $@; do
    echo $file
done

for file in $@; do
    if grep -q "Copyright (c)" $file
    then
        if ! grep -q "Copyright (c).*$current_year" $file
        then
            echo "ERROR: File $file has a copyright notice without the current year ($current_year)."
            exit_code=1
        fi
    fi
done

exit $exit_code
