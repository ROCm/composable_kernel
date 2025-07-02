#!/bin/bash

# Get list of staged files
STAGED_FILES=$(git diff --cached --name-only)

# Check if any staged file is under include/ck_tile/ or example/ck_tile/
if echo "$STAGED_FILES" | grep -qE '^(include/ck_tile/|example/ck_tile/)'; then
    echo "Detected changes in ck_tile-related files. Running remod.py..."

    # Run remod.py in both required locations
    (cd include/ck_tile/ && python3 remod.py)
    (cd example/ck_tile/ && python3 remod.py)

    echo "remod.py completed."
else
    echo "No changes in ck_tile-related files. Skipping remod.py."
fi
