# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import functools
import os
import shutil
import subprocess
import tempfile


@functools.lru_cache(None)
def library_path():
    return os.path.join(os.path.dirname(__file__), "library")


# Headers whose absence at compile time causes a silent failure this diagnostic
# exists to surface. Each is the entrypoint a generated kernel #includes for its
# backend family:
#   ck/ck.hpp       - classic CK (pulls ck/config.h transitively)
#   ck/config.h     - CMake-generated; supplied by $ROCM_HOME, not the wheel
#   ck_tile/core.hpp- CK-Tile; shipped in the wheel
_DIAGNOSTIC_HEADERS = ("ck/ck.hpp", "ck/config.h", "ck_tile/core.hpp")


def _ck_dir():
    """CK root the runtime uses, matching PyTorch's resolution order:
    TORCHINDUCTOR_CK_DIR env override, else this package's own directory."""
    return os.environ.get("TORCHINDUCTOR_CK_DIR") or os.path.dirname(__file__)


def _rocm_home():
    return (
        os.environ.get("ROCM_HOME")
        or os.environ.get("ROCM_PATH")
        or "/opt/rocm"
    )


def _include_roots(ck_dir, rocm_home):
    return [
        os.path.join(ck_dir, "include"),
        os.path.join(ck_dir, "library", "include"),
        os.path.join(rocm_home, "include"),
    ]


def include_roots():
    """The three -I roots the generated kernels are compiled against, in the
    same order as torch/_inductor/codegen/rocm/compile_command.py
    (_rocm_include_paths): CK first, ROCm last."""
    return _include_roots(_ck_dir(), _rocm_home())


def _try_compile(header, roots):
    """Optional: hipcc -fsyntax-only on `#include "<header>"` over the -I roots,
    to catch transitive-include breaks path-search cannot see. Returns True/False,
    or None when hipcc is unavailable (graceful degradation for CI/no-compiler)."""
    hipcc = shutil.which("hipcc")
    if not hipcc:
        return None
    src = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cpp", delete=False
        ) as f:
            f.write(f'#include "{header}"\nint main() {{ return 0; }}\n')
            src = f.name
        cmd = [hipcc, "-std=c++17", "-fsyntax-only"]
        for r in roots:
            cmd += ["-I", r]
        cmd.append(src)
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        return proc.returncode == 0
    except Exception:
        return None
    finally:
        if src and os.path.exists(src):
            os.unlink(src)


def check_headers(headers=_DIAGNOSTIC_HEADERS, try_compile=True):
    """Diagnose whether the CK headers the generated kernels #include actually
    resolve, using the same include roots the runtime compiles against.

    Backend-agnostic: reports every header; callers select which subset is
    relevant to the backend they are gating.

    Returns a dict:
        {
          "ck_dir": <resolved CK root>,
          "rocm_home": <resolved ROCm root>,
          "include_roots": [...],
          "headers": {
             "<header>": {
                "resolved": bool,        # found by path search under some root
                "found_in": <root|None>, # first root containing it
                "compiled": bool|None,   # hipcc syntax check, None if skipped
             }, ...
          },
          "ok": bool,   # every requested header path-resolved
        }
    """
    # Read each location once and derive everything from those values, so the
    # roots probed and the roots reported cannot disagree if the environment
    # changes mid-call.
    ck_dir = _ck_dir()
    rocm_home = _rocm_home()
    roots = _include_roots(ck_dir, rocm_home)
    results = {}
    for hdr in headers:
        found_in = None
        for root in roots:
            if os.path.exists(os.path.join(root, hdr)):
                found_in = root
                break
        results[hdr] = {
            "resolved": found_in is not None,
            "found_in": found_in,
            "compiled": _try_compile(hdr, roots) if try_compile else None,
        }
    return {
        "ck_dir": ck_dir,
        "rocm_home": rocm_home,
        "include_roots": roots,
        "headers": results,
        "ok": all(v["resolved"] for v in results.values()),
    }
