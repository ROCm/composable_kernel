#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Rejects any byte outside printable ASCII (plus tab, LF, CR) in the
# files passed as arguments. Used both by the local pre-commit hook
# and by the Jenkinsfile "ASCII Only Check" static-check stage.
#
# Usage: ./check_ascii_only.sh <file1> <file2> ...

exit_code=0

for file in "$@"; do
    [[ -f "$file" ]] || continue
    if LC_ALL=C grep -qP '[^\x09\x0A\x0D\x20-\x7E]' "$file" 2>/dev/null; then
        echo "ERROR: $file contains non-ASCII bytes:"
        LC_ALL=C grep -nP '[^\x09\x0A\x0D\x20-\x7E]' "$file" | head -20
        echo "  Fix: replace with ASCII (em-dash -> --, smart quotes -> \", arrows -> ->, etc.)"
        exit_code=1
    fi
done

exit $exit_code
