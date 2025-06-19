#!/bin/bash

run_and_check() {
    "$@"
    status=$?
    if [ $status -ne 0 ]; then
        echo "Error with \"$@\": Exited with status $status"
        exit $status
    fi
    return $status
}

echo "I: Installing tools required for pre-commit checks..."
run_and_check sudo apt install -y clang-format-12

run_and_check python3 -m venv .venv
# Activate the virtual environment for the rest of the script
eval "$(.venv/bin/activate && export VIRTUAL_ENV=\"$VIRTUAL_ENV\" && export PATH=\"$PATH\")"
echo "I: Installing pre-commit itself..."
run_and_check .venv/bin/pip install pre-commit
run_and_check .venv/bin/pre-commit install

echo "I: Installation successful."
