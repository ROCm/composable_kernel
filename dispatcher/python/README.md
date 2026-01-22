# CK Tile Dispatcher Python Utilities

This directory contains Python utilities used by the dispatcher examples.

## Contents

- `ctypes_utils.py` - Core ctypes utilities for GEMM Python examples
  - `KernelConfig` - Kernel configuration dataclass
  - `setup_gemm_dispatcher()` - Setup dispatcher with auto-correction
  - `cleanup_gemm()` - Cleanup dispatcher resources
  - `GemmRunner` - GPU execution helper
  - Auto-correction and validation utilities

- `conv_utils.py` - Core utilities for Conv Python examples
  - `ConvSignature`, `ConvAlgorithm` - Convolution configuration
  - `ConvProblem` - Problem definition
  - `GpuConvRunner` - GPU execution helper
  - `EnhancedConvCodegenRunner` - Kernel codegen utilities

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

### Conv Examples

The Conv Python examples in `dispatcher/examples/conv/python/` import:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from conv_utils import (
    ConvSignature,
    ConvAlgorithm,
    ConvProblem,
    GpuConvRunner,
)
```

## Requirements

- Python 3.8+
- NumPy
- HIP runtime (for GPU execution)
