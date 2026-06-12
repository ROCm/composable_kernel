#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Rejects Windows CRLF line endings (a trailing carriage return) in the
# files passed as arguments. Used both by the local pre-commit hook and
# by the Jenkinsfile "CRLF Check" static-check stage.
#
# Usage: ./check_no_crlf.sh <file1> <file2> ...

exit_code=0

for file in "$@"; do
    [[ -f "$file" ]] || continue
    if LC_ALL=C grep -qP '\r$' "$file" 2>/dev/null; then
        echo "ERROR: $file contains CRLF (Windows) line endings:"
        LC_ALL=C grep -nP '\r$' "$file" | head -20 | sed 's/\r$/<CR>/'
        echo "  Fix: convert to LF, e.g. 'sed -i 's/\\r\$//' $file' or 'dos2unix $file'"
        exit_code=1
    fi
done

exit $exit_code
