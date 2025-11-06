# Static Analysis Quick Start

## Installation
```bash
# Use the project's installation script (recommended)
./script/install_precommit.sh

# Then activate the virtual environment
source .venv/bin/activate
```

**Alternative:** If you have pre-commit installed globally or in ~/.scripts:
```bash
pre-commit install
```

## Usage

### Run on staged files (recommended during development)
```bash
# Run both tools
pre-commit run --hook-stage manual

# Run individual tools
pre-commit run clang-tidy
pre-commit run cppcheck
```

### Run on specific files
```bash
# Test on a single file
pre-commit run clang-tidy --files include/ck_tile/core/tensor/tensor_view.hpp
```

### Run on all files (for CI/CD or comprehensive check)
```bash
pre-commit run --hook-stage manual --all-files
```

## Configuration Files
- `.clang-tidy` - Clang-tidy configuration with comprehensive checks
- `.cppcheck-suppressions.txt` - CppCheck suppressions
- `.pre-commit-config.yaml` - Pre-commit hook definitions

## Features
✅ **Optional** - Won't run automatically on every commit  
✅ **Fast** - Only checks modified files  
✅ **Parallel** - Uses 4 concurrent jobs  
✅ **Warnings-only** - Non-blocking, won't fail commits  
✅ **Comprehensive** - Matches CMake configuration

## Full Documentation
See [docs/STATIC_ANALYSIS.md](docs/STATIC_ANALYSIS.md) for detailed information.

## Comparison with CMake Targets
```bash
# Pre-commit (fast, modified files only)
pre-commit run --hook-stage manual

# CMake targets (slower, all project files)
cmake --build build --target tidy
cmake --build build --target cppcheck
