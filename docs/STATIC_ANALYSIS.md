# Static Analysis Tools Guide

This project includes optional static analysis tools (clang-tidy and CppCheck) integrated with pre-commit hooks for catching potential issues in C++ code.

## Overview

- **clang-tidy**: Performs comprehensive static analysis including code style, bug detection, and modernization suggestions
- **CppCheck**: Lightweight static analysis focused on finding bugs and undefined behavior

Both tools are configured to:
- ✅ Run only on modified files (fast feedback)
- ✅ Execute in parallel (4 jobs) for better performance
- ✅ Use warnings-only mode (non-blocking)
- ✅ Be optional (manual stage only)
- ✅ Respect project-specific suppressions

## Quick Start

### Installation

```bash
# Option 1: Use the project's installation script (recommended)
./script/install_precommit.sh

# Then activate the virtual environment
source .venv/bin/activate

# Option 2: If you have pre-commit installed globally or in ~/.scripts
pre-commit install
```

### Basic Usage

**Run on staged files:**
```bash
# Run clang-tidy only
pre-commit run clang-tidy

# Run cppcheck only
pre-commit run cppcheck

# Run both static analysis tools
pre-commit run --hook-stage manual
```

**Run on all files:**
```bash
# Run on entire codebase (takes longer)
pre-commit run clang-tidy --all-files
pre-commit run cppcheck --all-files
```

**Run on specific files:**
```bash
# Run on a specific file
pre-commit run clang-tidy --files path/to/file.cpp
```

## Configuration

### Clang-Tidy

Configuration file: `.clang-tidy`

- **Checks**: Comprehensive list with ~100+ rules enabled
- **Disabled checks**: Many noisy checks are disabled (see `.clang-tidy` for details)
- **Mode**: Warnings-only (WarningsAsErrors: '')
- **Parallel jobs**: 4 concurrent processes
- **Header filter**: Only checks `.hpp` files

### CppCheck

Configuration file: `.cppcheck-suppressions.txt`

- **Enabled checks**: warning, style, performance, portability
- **Suppressions**: Project-specific suppressions defined in `.cppcheck-suppressions.txt`
- **Parallel jobs**: 4 concurrent processes
- **Exit code**: 0 (non-blocking, warnings only)

## Integration with Development Workflow

### IDE Integration

#### VSCode
Install the clang-tidy extension and configure it to use the project's `.clang-tidy` file.

#### CLion/IntelliJ
CLion has built-in clang-tidy support. Enable it in:
Settings → Editor → Inspections → C/C++ → General → Clang-Tidy

### CI/CD Integration

To run static analysis in CI pipelines:

```bash
# In your CI script
pre-commit run --hook-stage manual --all-files
```

### Git Commit Integration

By default, these hooks do NOT run on every commit (they use `stages: [manual]`).

To run them on specific commits:
```bash
# Run manual-stage hooks before committing
pre-commit run --hook-stage manual
# Then commit as usual
git commit
```

## Advanced Usage

### Customizing Checks

**To add more clang-tidy checks:**
Edit `.clang-tidy` and add checks to the `Checks:` section.

**To suppress specific warnings:**
Add suppressions to `.cppcheck-suppressions.txt`.

### Performance Tuning

The hooks are configured with:
- **Parallel execution**: `-j=4` (adjust based on your CPU)
- **Quiet mode**: Reduces output noise
- **Smart file filtering**: Only processes C++ files

### Troubleshooting

**Hooks taking too long?**
- Reduce parallel jobs: Change `-j=4` to `-j=2` in `.pre-commit-config.yaml`
- Run on specific directories only
- Use `--files` flag to target specific files

**False positives?**
- Add suppressions to `.cppcheck-suppressions.txt` for CppCheck
- Add `// NOLINT` comments in code for clang-tidy
- Update `.clang-tidy` to disable specific checks

**Tools not found?**
Ensure clang-tidy and cppcheck are installed:
```bash
# On Ubuntu/Debian
sudo apt-get install clang-tidy cppcheck

# On Fedora/RHEL
sudo dnf install clang-tools-extra cppcheck

# On macOS
brew install llvm cppcheck
```

## Best Practices

1. **Run before pushing**: Make it a habit to run static analysis before pushing changes
   ```bash
   pre-commit run --hook-stage manual
   ```

2. **Start small**: Run on modified files first, then expand to full codebase
   
3. **Address warnings gradually**: Don't try to fix everything at once

4. **Document suppressions**: Add comments explaining why specific warnings are suppressed

5. **Update regularly**: Keep pre-commit hooks updated
   ```bash
   pre-commit autoupdate
   ```

## Comparison with CMake Targets

This project also has CMake targets for static analysis:
```bash
cmake --build build --target tidy      # Runs clang-tidy via CMake
cmake --build build --target cppcheck  # Runs cppcheck via CMake
```

**Pre-commit vs CMake targets:**

| Feature | Pre-commit Hooks | CMake Targets |
|---------|------------------|---------------|
| Modified files only | ✅ Yes | ❌ No (all files) |
| Parallel execution | ✅ Yes (4 jobs) | ✅ Yes |
| Warnings-only | ✅ Yes | ⚠️ Configurable |
| Integration | Git workflow | Build system |
| Speed | ⚡ Fast | 🐌 Slower |
| Use case | Development | CI/CD, full analysis |

## References

- [clang-tidy documentation](https://clang.llvm.org/extra/clang-tidy/)
- [CppCheck manual](http://cppcheck.sourceforge.net/manual.pdf)
- [pre-commit framework](https://pre-commit.com/)
