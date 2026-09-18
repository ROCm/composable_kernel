#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for the arch gate used by the RowColQuant/TensorQuant GPU tests.

These run on CPU: they exercise the pure string predicate, not a device.

Why this file exists
--------------------
The GPU correctness tests decide whether to run or skip by comparing the detected
architecture against a tuple of supported targets. That comparison used to be an
exact ``in`` test, which is wrong: ``rocm_agent_enumerator`` (and
``hipDeviceProp_t::gcnArchName``) may report feature suffixes, so a perfectly
supported device can enumerate as ``"gfx942:sramecc+:xnack-"``. An exact compare
then evaluates False and the test SKIPS.

A skip is not a failure, so CI stays green while covering nothing -- the defect is
invisible by construction. That failure mode is why this predicate gets its own
regression test rather than being trusted to review.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_rowcolquant_gpu_correctness as rcq
import test_tensorquant_gpu_correctness as tq

# Both modules must gate identically; parametrizing over them keeps them in step.
MODULES = pytest.mark.parametrize("mod", [rcq, tq], ids=["rowcolquant", "tensorquant"])


@MODULES
@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1250"])
def test_bare_supported_arch_is_accepted(mod, arch):
    assert mod.arch_is_supported(arch) is True


@MODULES
@pytest.mark.parametrize(
    "arch",
    [
        "gfx942:sramecc+:xnack-",
        "gfx942:sramecc-:xnack-",
        "gfx950:sramecc+:xnack-",
        "gfx1250:xnack-",
    ],
)
def test_feature_suffixed_supported_arch_is_accepted(mod, arch):
    """The regression this file exists for: suffixes must not cause a skip."""
    assert mod.arch_is_supported(arch) is True, (
        f"{arch} is a supported device but the gate rejected it; "
        "the GPU test would silently skip and CI would be vacuously green"
    )


@MODULES
@pytest.mark.parametrize(
    "arch",
    [
        "gfx90a",
        "gfx90a:sramecc+:xnack-",  # unsupported and suffixed -> still unsupported
        "gfx1030",
        "gfx000",
        "",
    ],
)
def test_unsupported_arch_is_rejected(mod, arch):
    assert mod.arch_is_supported(arch) is False


@MODULES
def test_no_substring_false_positives(mod):
    """Prefix tolerance must not degrade into a substring match.

    ``gfx9421`` is a different (hypothetical) target and must not be accepted just
    because ``gfx942`` is a prefix of it. Splitting on ``:`` and comparing the whole
    base token gives us that for free; a naive ``startswith`` would not.
    """
    assert mod.arch_is_supported("gfx9421") is False
    assert mod.arch_is_supported("gfx12501") is False


@MODULES
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("gfx942:sramecc+:xnack-", "gfx942"),
        ("gfx1250", "gfx1250"),
        ("gfx950:xnack-", "gfx950"),
        ("", ""),
    ],
)
def test_normalize_strips_feature_suffix(mod, raw, expected):
    """Normalization also feeds ``--offload-arch``, so it must yield a bare target."""
    assert mod.normalize_gfx_arch(raw) == expected


@MODULES
def test_gate_matches_supported_tuple(mod):
    """Every entry of the declared tuple must pass its own gate, bare and suffixed."""
    for arch in mod._SUPPORTED_ARCHES:
        assert mod.arch_is_supported(arch) is True
        assert mod.arch_is_supported(f"{arch}:sramecc+:xnack-") is True




@MODULES
@pytest.mark.parametrize("detected", ["", "gfx90a"])
def test_standalone_reports_skip_for_unavailable_target(mod, detected, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [mod.__file__, "--output-dir", str(tmp_path)])
    monkeypatch.setattr(mod, "_GFX_ARCH", detected)
    assert mod.main() == 77


@MODULES
def test_standalone_suffix_override_reaches_all_cases(mod, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [mod.__file__, "--gfx", "gfx1250:xnack-",
                                     "--output-dir", str(tmp_path)])
    monkeypatch.setattr(mod, "_GFX_ARCH", "")
    monkeypatch.setattr(mod, "_has_hipcc", lambda: True)
    monkeypatch.setattr(mod, "_ml_dtypes", object())
    encoded = []
    def inputs(M, N, K, dtype, arch):
        encoded.append((M, N, K, dtype, arch))
        return ()
    monkeypatch.setattr(mod, "_make_inputs", inputs)
    monkeypatch.setattr(mod, "_run_one", lambda *a, **kw: ("PASS", "host test stub"))
    assert mod.main() == 0
    assert len(encoded) == 3
    assert all(case[-1] == "gfx1250" for case in encoded)
    assert (64, 256, 128, "fp8", "gfx1250") in encoded


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
