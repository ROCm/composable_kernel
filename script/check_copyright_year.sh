#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# This script checks if files have the correct copyright header template.
# It supports .hpp, .cpp, .inc, .py, .sh, and .cmake files.
#
# Usage: ./check_copyright_year.sh <file1> <file2> ...

exit_code=0

# Expected copyright header lines (without comment characters)
COPYRIGHT_LINE="Copyright (c) Advanced Micro Devices, Inc., or its affiliates."
SPDX_LINE="SPDX-License-Identifier: MIT"

check_file() {
    local file=$1
    local ext="${file##*.}"
    local comment_char

    # Determine comment character based on file extension
    case "$ext" in
        cpp|hpp|inc)
            comment_char="//"
            ;;
        py|sh|cmake)
            comment_char="#"
            ;;
        *)
            # If the filename is *.cmake or CMakeLists.txt, use #
            if [[ "$file" == *CMakeLists.txt ]] || [[ "$file" == *.cmake ]]; then
                comment_char="#"
            else
                # Skip files with unsupported extensions
                return 0
            fi
            ;;
    esac

    # Build expected header patterns
    expected_copyright="$comment_char $COPYRIGHT_LINE"
    expected_spdx="$comment_char $SPDX_LINE"

    # Check if file contains both required lines
    if ! grep -qF "$expected_copyright" "$file"; then
        echo "ERROR: File $file is missing the correct copyright header line."
        echo "  Expected: $expected_copyright"
        return 1
    fi

    if ! grep -qF "$expected_spdx" "$file"; then
        echo "ERROR: File $file is missing the correct SPDX license identifier line."
        echo "  Expected: $expected_spdx"
        return 1
    fi

    return 0
}

# Process each file provided as argument
for file in "$@"; do
    # Skip if file doesn't exist or is a directory
    if [[ ! -f "$file" ]]; then
        continue
    fi

    if ! check_file "$file"; then
        exit_code=1
    fi
done

exit $exit_code
