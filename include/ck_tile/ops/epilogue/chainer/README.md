# CK Tile Epilogue Chainer

## Overview

The Epilogue Chainer provides composability for building epilogue sequences for GEMM operations. It allows to chain multiple epilogue stages to be executed sequentially.

## Components

### EpilogueChainer
EpilogueChainer class, orchestrator that executes epilogue sequences. Takes an initialization epilogue and a sequence of operations to perform.

### EpilogueNode
Wrapper for individual epilogue operations with embedded arguments. Supports both parameterized and parameter-free epilogues.

### EpilogueLoop
Executes a sequence of epilogue nodes across multiple iterations (e.g., for tiled processing).

### Scheduler
Pre-built epilogue sequences for common patterns:
- `CshuffleEpilogueSchedule::make_base_schedule()` - Basic epilogue without scaling
- `CshuffleEpilogueSchedule::make_scale_schedule(m_scale, n_scale)` - Epilogue with row/column scaling

## Current Status

**Work in Progress** - The API is functional but evolving:
- Current implementation supports basic epilogue chaining
- Chaining demonstrated by breaking down the cshuffle epilogue into modular components and chaining them together. This will also be consequently modified and improved. 

## Usage Pattern

```cpp
// Create schedule with embedded arguments
auto schedule = CshuffleEpilogueSchedule<Problem>::make_scale_schedule(
    m_scale_window, n_scale_window);

// Execute epilogue chain
EpilogueChainer<InitEpilogue, decltype(schedule)>{}(
    output_window, acc_tile, d_tensors, smem, schedule);
```

- Concrete use case based example(s) to be added. 

## Files

- `epilogue_chainer.hpp` - Main chainer orchestrator
- `epilogue_graph.hpp` - Node and loop constructs
- `epilogue_schedule.hpp` - Pre-built schedules
- `cshuffle_chained_epilogues.hpp` - Individual epilogue implementations (To be modified and trimmed down) 