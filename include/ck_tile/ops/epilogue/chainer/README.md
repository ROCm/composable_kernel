# CK Tile Epilogue Chainer

## Overview

The Epilogue Chainer provides a modular epilogue processing framework for GEMM operations through scheduler-defined operation graphs.

## Architecture

### Core Design Principle
The chainer follows a **Scheduler-Graph-Node** architecture with shared context:
- **Scheduler**: Defines operation graphs and creates a shared context
- **Graph**: Composes multiple operations into sequential processing units
- **Node**: Wraps individual epilogue operations with their arguments

### EpilogueChainer
The `EpilogueChainer` struct serves as the modular epilogue processing facilitator. It delegates to schedulers for context creation and schedule generation, then processes the resulting operation graphs.

### EpilogueNode
Individual epilogue operations are wrapped in `EpilogueNode` structures that capture required arguments at construction time and automatically forward them during processing. Supports both parameterized and parameter-free operations.

### EpilogueGraph  
The `EpilogueGraph` composes multiple nodes into sequential processing units that iterate over multiple accesses if needed, running all operations in order for each iteration.

### Scheduler System
`CshuffleEpilogueSchedule` provides tagged schedule selection using schedule tags:

**Schedule Tags:**
- `DefaultScheduleTag` - Standard epilogue: Slice → Cast → PrepC → ApplyD → Store → Move
- `RowColQuantScheduleTag` - RowCol quantization: Slice → ScaleWindow → Cast → PrepC → ApplyD → Store → Move
- `TensorQuantScheduleTag` - Tensor quantization: Slice → ScaleScalar → Cast → PrepC → ApplyD → Store → Move

**Tag-Based Selection:**
```cpp
using Scheduler = CshuffleEpilogueSchedule<Problem, DefaultScheduleTag>;
```

## Files

- `epilogue_chainer.hpp` - Core chainer processing facilitator and graph composition utilities
- `cshuffle_epilogue_schedule.hpp` - Tagged scheduler providing pre-built operation graphs
- `common_epilogue_ops.hpp` - Reusable epilogue operations for graph composition
- `cshuffle_epilogue_chainer_ops.hpp` - CShuffle-specific problem configuration and base operations 