# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host regressions for quant build targets, direct loading, and C++ guards."""

import ctypes
import importlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "codegen")]
OPS = ("rowcolquant", "tensorquant")
CXX = shutil.which("c++")


@pytest.mark.parametrize("op", OPS)
def test_custom_quant_tiles_require_explicit_arch(op, tmp_path):
    codegen = importlib.import_module(f"unified_grouped_gemm_{op}_codegen")
    config = codegen._default_config("gfx1250")
    output = tmp_path / "headers"
    with pytest.raises(ValueError, match="explicit.*gfx.arch"):
        codegen.generate_kernels(output, config=config, parallel=False)
    assert not output.exists()

    # The same custom tile is valid when its target is explicitly recorded.
    header = codegen.generate_kernels(output, config=config, gfx_arch="gfx1250:xnack-",
                                      parallel=False)[0].read_text()
    assert 'GfxArch = "gfx1250"' in header
    # Preserve legacy no-config generation without a target.
    legacy = codegen.generate_kernels(tmp_path / "legacy", parallel=False)[0].read_text()
    assert 'GfxArch = ""' in legacy
    assert "WarpTileM      = 32" in legacy


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("config_mode", ["--config", "--config-json"])
@pytest.mark.parametrize("list_names", [False, True])
def test_custom_quant_cli_requires_explicit_arch(op, config_mode, list_names, tmp_path):
    codegen = importlib.import_module(f"unified_grouped_gemm_{op}_codegen")
    config_json = json.dumps(codegen._default_config("gfx1250"))
    config_file = tmp_path / "config.json"
    config_file.write_text(config_json)
    output = tmp_path / "headers"
    args = [sys.executable, str(ROOT / f"codegen/unified_grouped_gemm_{op}_codegen.py"),
            "--output-dir", str(output), config_mode,
            str(config_file) if config_mode == "--config" else config_json]
    if list_names:
        args.append("--list-names")
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode != 0
    assert "explicit --gfx-arch" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("config_arch,build_arch", [("gfx1250", "gfx942"), ("gfx942", "gfx1250")])
@pytest.mark.parametrize("explicit", [False, True])
def test_config_build_mismatch_fails_before_codegen(op, config_arch, build_arch, explicit, monkeypatch, tmp_path):
    module = importlib.import_module(f"grouped_gemm_{op}_utils")
    monkeypatch.setattr(module, "_detect_gpu_arch", lambda: build_arch)
    monkeypatch.setattr(module, f"_generate_{op}_kernel", lambda *a: pytest.fail("codegen must not run"))
    setup = getattr(module, f"setup_multiple_{op}_dispatchers")
    with pytest.raises(ValueError, match="does not match build target"):
        setup([module.default_fp8_config(config_arch)], output_dir=tmp_path / "output",
              gfx_arch=build_arch if explicit else None)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("op", OPS)
def test_matching_suffix_config_generates_target_metadata(op, tmp_path):
    module = importlib.import_module(f"grouped_gemm_{op}_utils")
    config = module.default_fp8_config("gfx942:sramecc+:xnack-")
    config.gfx_arch = "gfx942:xnack-"
    header = getattr(module, f"_generate_{op}_kernel")(config, tmp_path)
    assert header is not None
    assert 'GfxArch = "gfx942"' in header.read_text()


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("arch", ["gfx90a", "gfx90a:xnack-", "gfx1200", "gfx1201"])
def test_unsupported_quant_targets_rejected_at_entry(op, arch, tmp_path):
    module = importlib.import_module(f"grouped_gemm_{op}_utils")
    with pytest.raises(ValueError, match="Unsupported GPU architecture"):
        module.default_fp8_config(arch)
    result = subprocess.run([sys.executable, str(ROOT / f"codegen/unified_grouped_gemm_{op}_codegen.py"),
                             "--gfx-arch", arch, "--output-dir", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Unsupported GPU architecture" in result.stderr
    assert not list(tmp_path.glob("*.hpp"))


@pytest.mark.parametrize("op,cls", [("rowcolquant", "RowColQuant"), ("tensorquant", "TensorQuant")])
@pytest.mark.parametrize("missing", ["dispatcher_get_tile_n", "dispatcher_get_pad_n"])
def test_direct_runner_reports_old_abi(op, cls, missing, monkeypatch, tmp_path):
    module = importlib.import_module(f"grouped_gemm_{op}_utils")
    lib = SimpleNamespace()
    lifecycle_calls = []
    for name in ("dispatcher_initialize", "dispatcher_run_gemm", "dispatcher_get_kernel_name",
                 "dispatcher_get_kernel_count", "dispatcher_get_tile_n", "dispatcher_get_pad_n",
                 "dispatcher_cleanup"):
        if name != missing:
            setattr(lib, name, lambda *args: 0)
    lib.dispatcher_initialize = lambda: lifecycle_calls.append("initialize")
    lib.dispatcher_cleanup = lambda: lifecycle_calls.append("cleanup")
    path = tmp_path / "old.so"
    path.touch()
    monkeypatch.setattr(module.ctypes, "CDLL", lambda *a: lib)
    with pytest.raises(RuntimeError, match="Incompatible quant bridge ABI.*Rebuild"):
        getattr(module, cls + "GpuGemmRunner")(path)
    assert lifecycle_calls == []


def compile_cpp(tmp_path, source, shared=False):
    if not CXX:
        pytest.skip("requires C++ compiler")
    cpp = tmp_path / "probe.cpp"
    cpp.write_text(source)
    output = tmp_path / ("probe.so" if shared else "probe.o")
    args = [CXX, "-std=c++17", str(cpp), "-o", str(output)]
    args += ["-shared", "-fPIC"] if shared else ["-c"]
    return subprocess.run(args, capture_output=True, text=True), output


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("generated,compiled,passes", [
    ("gfx1250", "gfx942", False), ("gfx942", "gfx1250", False),
    ("gfx1250", "gfx1250", True), ("gfx942", "gfx942", True), ("gfx950", "gfx950", True),
])
def test_production_cpp_rejects_header_target_mismatch(op, generated, compiled, passes, tmp_path):
    codegen = importlib.import_module(f"unified_grouped_gemm_{op}_codegen")
    header = codegen.generate_kernels(tmp_path / "headers", gfx_arch=generated, parallel=False)[0].read_text()
    # Compile the actual emitted constants and actual bridge guard block without
    # requiring HIP. Kernel implementation and GPU execution are tested separately.
    constants = header[header.index("    static constexpr const char* GfxArch"):header.index("    // Informational only:")]
    constants = constants.replace("ck_tile::index_t", "int")
    bridge = (ROOT / f"bindings/ctypes/grouped_gemm_{op}_ctypes_lib.cpp").read_text()
    guards = bridge[bridge.index("static constexpr bool ct_starts_with"):bridge.index('extern "C" {')]
    source = f'#define GFX_ARCH "{compiled}"\nstruct SelectedKernel {{\n{constants}\n}};\n{guards}'
    result, _ = compile_cpp(tmp_path, source)
    assert (result.returncode == 0) == passes, result.stderr
    if not passes:
        assert "Generated kernel architecture does not match" in result.stderr


@pytest.mark.parametrize("arch,pad_n,M,N,expected", [
    ("gfx1250", False, 128, 128, 0),
    ("gfx1250", False, 126, 128, -1),
    ("gfx1250", False, 128, 100, -1),
    ("gfx942", False, 126, 128, 0),
    ("gfx1250", True, 128, 100, 0),
])
def test_rowcolquant_actual_shape_guards(arch, pad_n, M, N, expected, tmp_path):
    bridge = (ROOT / "bindings/ctypes/grouped_gemm_rowcolquant_ctypes_lib.cpp").read_text()
    # Execute the production host-side prefix of dispatcher_run_gemm. Stop before
    # allocation/launch; valid inputs return a sentinel success in this harness.
    prefix = bridge[bridge.index("int dispatcher_run_gemm("):bridge.index("    // Only packed (contiguous) layouts")]
    source = f'''#include <atomic>
#include <cstdint>
#include <iostream>
#define GFX_ARCH "{arch}"
std::atomic<int> g_ref_count{{1}};
constexpr bool kCompiledForGfx12 = {str(arch == "gfx1250").lower()};
struct SelectedKernel {{ static constexpr bool kPadN = {str(pad_n).lower()}; static constexpr int TileN = 64; }};
extern "C" {{
{prefix}
return 0;
}}
}}
'''
    result, output = compile_cpp(tmp_path, source, shared=True)
    assert result.returncode == 0, result.stderr
    lib = ctypes.CDLL(str(output))
    run = lib.dispatcher_run_gemm
    run.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int64] * 10 + [ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    run.restype = ctypes.c_int
    elapsed = ctypes.c_float()
    assert run(*([1] * 5), M, N, 192, 192, 192, 1, 1, N, M, N, 1, ctypes.byref(elapsed)) == expected


@pytest.mark.parametrize("inherited_arches", [None, "gfx942;gfx1250", "gfx950;gfx942", "gfx1250;gfx950"])
@pytest.mark.parametrize("explicit", [True, False])
@pytest.mark.parametrize("arch", [
    "gfx942", "gfx942:sramecc+:xnack-",
    "gfx950", "gfx950:sramecc+:xnack-",
    "gfx1250", "gfx1250:xnack-",
])
def test_cmake_normalizes_explicit_and_inferred_arches(tmp_path, explicit, arch, inherited_arches):
    if not shutil.which("cmake"):
        pytest.skip("requires CMake")
    source = tmp_path / "source"
    source.mkdir()
    build = tmp_path / "build"
    headers = build / "generated_kernels"
    headers.mkdir(parents=True)
    for op in OPS:
        (headers / f"grouped_gemm_{op}_test.hpp").touch()
    overrides = '\n'.join(f'set(CK_TILE_{op.upper()}_GFX_ARCH "{arch}")' for op in OPS) if explicit else ''
    # Execute the real parent feature-definition block, including its defaults.
    parent_features = ""
    if inherited_arches:
        parent_cmake = (ROOT.parent / "CMakeLists.txt").read_text()
        start = parent_cmake.index('if (SUPPORTED_GPU_TARGETS MATCHES "gfx9|gfx11|gfx12"')
        end = parent_cmake.index('if ((SUPPORTED_GPU_TARGETS MATCHES "gfx942"', start)
        parent_features = f'set(SUPPORTED_GPU_TARGETS "{inherited_arches}")\n' + parent_cmake[start:end]
    configured_arches = f"{arch};{inherited_arches}" if inherited_arches else arch
    (source / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.16)
project(quant_cmake_probe LANGUAGES CXX)
add_library(hip::device INTERFACE IMPORTED)
{parent_features}
set(CMAKE_HIP_ARCHITECTURES "{configured_arches}")
{overrides}
add_subdirectory("{ROOT / 'bindings/ctypes'}" ctypes)
''')
    result = subprocess.run(["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    commands = json.loads((build / "compile_commands.json").read_text())
    from dispatcher_common import arch_feature_defines

    # Compare effective macros, preserving the generated command's -D/-U order.
    # Checking only flag presence misses inherited definitions and overrides.
    feature_flags = set().union(*(set(arch_feature_defines(a)) for a in ("gfx942", "gfx950", "gfx1250")))
    feature_flags.update({"-DUSE_NEW_UNIFIED_FRAMEWORK=0", "-DCK_USE_FNUZ_FP8",
                          "-DCK_USE_XDL", "-DCK_USE_GFX94", "-DCK_USE_GFX950",
                          "-DCK_USE_WMMA", "-DCK_USE_WMMA_FP8", "-DCK_GFX1030_SUPPORT"})
    feature_names = {flag[2:].split("=")[0] for flag in feature_flags}

    def effective_features(flags, compiler):
        result = subprocess.run([compiler, *flags, "-dM", "-E", "-x", "c++", "-"],
                                input="", capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        macros = {}
        for line in result.stdout.splitlines():
            _, name, *value = line.split(maxsplit=2)
            if name in feature_names:
                macros[name] = value[0] if value else ""
        return macros
    for op in OPS:
        command = next(c["command"] for c in commands if c["file"].endswith(f"grouped_gemm_{op}_ctypes_lib.cpp"))
        base = arch.split(":")[0]
        assert f'GFX_ARCH=\\"{base}\\"' in command
        assert f'GFX_ARCH=\\"{base}:' not in command
        assert f'-DCK_CMAKE_GPU_TARGET_IDS=0x{base[3:]}' in command
        expected = set(arch_feature_defines(base))
        if base == "gfx1250":
            expected.add("-DUSE_NEW_UNIFIED_FRAMEWORK=0")
        args = shlex.split(command)
        flags = [arg for arg in args[1:] if arg.startswith(("-D", "-U"))]
        actual = effective_features(flags, args[0])
        expected = effective_features(sorted(expected), args[0])
        assert actual == expected, f"{op} {arch}: CMake/JIT feature mismatch: {actual} != {expected}"
