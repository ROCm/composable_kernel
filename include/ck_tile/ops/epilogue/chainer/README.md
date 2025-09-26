# CK Tile Epilogue Chainer

## Overview

The Epilogue Chainer provides composability for building epilogue sequences for GEMM operations. It provides a means to chain multiple epilogue stages so they can be run sequentially.

## Components

### EpilogueChainer
The `EpilogueChainer` class is an orchestrator that runs epilogue sequences. It takes an initialization epilogue and a sequence of operations.

### EpilogueNode
The `EpilogueNode`class is a wrapper for individual epilogue operations, it Supports both parameterized and parameter-free epilogues.

### EpilogueLoop
The `EpilogueLoop` class provides means to specify sequence of epilogue nodes that need to be looped together. It provides the `execute` method that runs the provided sequence.

### Scheduler
`epilogue_schedule.hpp` provides pre-built epilogue sequences for common patterns:
- `CshuffleEpilogueSchedule::make_base_schedule()` - Basic epilogue without scaling
- `CshuffleEpilogueSchedule::make_scale_schedule(m_scale, n_scale)` - Epilogue with row/column scaling

## Current Status

**Work in Progress** - The API is functional but evolving:
- Current implementation supports basic epilogue chaining
- Chaining is demonstrated by breaking down the cshuffle epilogue into modular components and chaining them together. This will also be consequently modified and improved. 

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
- `cshuffle_chained_epilogues.hpp` - Individual epilogue implementations 