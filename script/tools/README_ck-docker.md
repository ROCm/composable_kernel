# ck-docker

Build and test composable_kernel in Docker with ROCm support.

## Terminal Usage

Direct command-line usage:

```bash
# From composable_kernel directory
script/tools/ck-docker start
script/tools/ck-docker build test_amdgcn_mma
script/tools/ck-docker test test_amdgcn_mma --gtest_filter=*Fp16*
script/tools/ck-docker status
script/tools/ck-docker shell

# Or add to PATH
export PATH="$PATH:$PWD/script/tools"
ck-docker start
```

## LLM Assistant Integration

If using an LLM assistant, you can ask in natural language:
- "Start the docker container"
- "Build test_amdgcn_mma"
- "Run test_amdgcn_mma with filter *Fp16*"
- "Check container status"
- "Open a shell in the container"

## Commands

```
ck-docker start [name]                    Start Docker container
ck-docker build [target] [--reconfigure]  Build target (optionally reconfigure CMake)
ck-docker test <name> [options]           Run test
ck-docker shell [name]                    Interactive shell
ck-docker status [name]                   Check status
ck-docker stop [name]                     Stop container
```

## Configuration

- **Image**: rocm/composable_kernel:ck_ub24.04_rocm7.0.1
- **GPU**: Auto-detected via rocminfo (fallback: gfx950)
- **Compiler**: /opt/rocm/llvm/bin/clang++
- **Build**: Ninja + CMake (Release)
- **Mount**: Current directory → /workspace
- **Container Name**: Auto-generated as `ck_<username>_<branch>` to avoid clashes

## Environment

```bash
export CK_CONTAINER_NAME=my_build                                   # Override default container name
export CK_DOCKER_IMAGE=rocm/composable_kernel:ck_ub24.04_rocm7.0.1  # Override Docker image
export GPU_TARGET=gfx942                                             # Override GPU target detection
```

## Examples

```bash
# Start container
ck-docker start

# Build and run test
ck-docker build test_amdgcn_mma
ck-docker test test_amdgcn_mma

# Force clean CMake reconfiguration and build
ck-docker build --reconfigure test_amdgcn_mma

# Custom container
ck-docker start my_build
ck-docker build test_amdgcn_mma --name my_build
ck-docker test test_amdgcn_mma --name my_build

# Debug
ck-docker shell
ck-docker status
```
