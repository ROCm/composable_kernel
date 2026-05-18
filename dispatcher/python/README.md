# CK Tile Dispatcher Python Utilities

This directory contains Python utilities used by the dispatcher examples.

## Contents

### Shared Utilities (used by both GEMM and Grouped Conv)

- `dispatcher_common.py` - Shared dispatcher infrastructure
  - Path helpers (`get_dispatcher_root`, `get_build_dir`, etc.)
  - `ValidationResultBase` - Structured validation feedback
  - `validate_wave_config`, `validate_warp_tile_config`, `validate_trait_combo`
  - `auto_correct_wave`, `auto_correct_trait` - Auto-correction helpers
  - `Colors` - Cross-platform ANSI color support
  - `print_phase`, `print_success`, `print_error`, `print_info` - Phased output
  - `cleanup_generated_kernels` - Cleanup helper

### GEMM Utilities

- `ctypes_utils.py` - Core ctypes utilities for GEMM Python examples
  - `KernelConfig` - Kernel configuration dataclass
  - `setup_gemm_dispatcher()` - Setup dispatcher with auto-correction
  - `cleanup_gemm()` - Cleanup dispatcher resources
  - `GemmRunner` - GPU execution helper
  - Auto-correction and validation utilities

### Grouped Convolution Utilities

- `grouped_conv_utils.py` - Utilities for grouped convolution
  - `GroupedConvValidationResult` - Validation result (extends `ValidationResultBase`)
  - `validate_grouped_conv_config` - Validate a grouped conv config
  - `auto_correct_grouped_conv_config` - Auto-correct invalid configs
  - `get_grouped_conv_default_config` - Get default config for a variant
  - `GroupedConvDataType` - Data type enum (FP16, BF16, FP32, FP8, BF8, INT8)
  - `format_grouped_conv_summary` - Human-readable config summary

## Usage

### GEMM Examples

The GEMM Python examples in `dispatcher/examples/gemm/python/` import:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from ctypes_utils import (
    KernelConfig,
    setup_gemm_dispatcher,
    cleanup_gemm,
    GemmRunner,
)
```

### Grouped Conv Usage

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from grouped_conv_utils import (
    validate_grouped_conv_config,
    auto_correct_grouped_conv_config,
    get_grouped_conv_default_config,
    GroupedConvDataType,
)

# Get a default config
config = get_grouped_conv_default_config(variant="forward", arch="gfx942")

# Validate
result = validate_grouped_conv_config(config)
print(f"Valid: {result.is_valid}")
```

## Requirements

- Python 3.8+
- NumPy
- HIP runtime (for GPU execution)
