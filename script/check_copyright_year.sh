#!/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

exit_code=0

for file in $@; do
    # Check if file has any copyright notice
    if grep -q "Copyright.*Advanced Micro Devices" "$file"
    then
        # Check for the exact template
        if ! grep -q "Copyright © Advanced Micro Devices, Inc., or its affiliates." "$file"
        then
            echo "ERROR: File $file has incorrect copyright format. Expected: 'Copyright © Advanced Micro Devices, Inc., or its affiliates.'"
            exit_code=1
        fi
        
        if ! grep -q "SPDX-License-Identifier: MIT" "$file"
        then
            echo "ERROR: File $file missing SPDX-License-Identifier: MIT"
            exit_code=1
        fi
    else
        echo "ERROR: File $file missing copyright header"
        exit_code=1
    fi
done

exit $exit_code
