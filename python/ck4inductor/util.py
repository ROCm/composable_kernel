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


def canonical_instances(op_instances):
    """Return the enumerated instances in a deterministic, duplicate-free order.

    Two problems, one canonicalization.

    *Order.* The enumerators build their lists from `grep -R`, which walks
    directories in readdir order -- so the order differs per machine and changes
    on reinstall. Consumers that sample a subset under a fixed seed (PyTorch
    Inductor draws `ck_max_profiling_configs`) therefore got a different subset
    per machine from the same wheel. `name()` embeds every template parameter, so
    it is a total key.

    *Duplicates.* CK's headers repeat some instance lines, but in C++ those
    repeats are not duplicates -- they are one instance appearing in several
    disjoint sets. `..._merged_groups_instance.hpp` lists its only `_V3` entry in
    both the base alias and the gfx950 `_2x` alias, which are mutually exclusive
    arms of a `get_device_name()` branch; the WMMA header repeats its `// generic
    instance` into each partN list, and separate translation units consume those.
    Grepping a directory flattens those partitions into one pool, so the repeats
    become genuine duplicates *here* and nowhere else.
    """
    survivors = {}
    for op in sorted(op_instances, key=lambda op: op.name()):
        kept = survivors.setdefault(op.name(), op)
        # Dropping a same-name op is only legal because name() is a total key --
        # it is built from dict_items(), so equal names imply equal parameters.
        # Check that here rather than trusting it: after the fact both ops are
        # no longer available to compare, so this is the only point where the
        # loss is detectable.
        if kept is not op and kept != op:
            raise ValueError(
                "ck4inductor instances differ but share the name "
                f"{op.name()!r}; name() must uniquely encode every "
                "template parameter"
            )
    return list(survivors.values())
