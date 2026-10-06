#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
CPU-only unit tests for grouped_gemm_rowcolquant_utils.py.

Tests kernel name generation, config serialization, and problem dimension helpers.
No GPU or hipcc required.

Run:
    python3 -m pytest dispatcher/tests/test_grouped_gemm_rowcolquant_utils.py -v
"""

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import grouped_gemm_rowcolquant_utils as UTILS
from grouped_gemm_rowcolquant_utils import (
    RowColQuantKernelConfig,
    RowColQuantGemmProblem,
    default_fp8_config,
    default_bf8_config,
)


# =============================================================================
# RowColQuantKernelConfig.name — byte-exact match with codegen KERNEL_NAME
# =============================================================================


class TestKernelName:

    def test_fp8_rcr_default_name(self):
        cfg = RowColQuantKernelConfig(
            dtype="fp8",
            layout="rcr",
            pipeline="compv3",
            epilogue="cshuffle",
            scheduler="intrawave",
            tile_m=128, tile_n=128, tile_k=64,
            warp_m=2, warp_n=2, warp_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
            pad_m=False, pad_n=False, pad_k=True,
            persistent=False,
        )
        assert cfg.name == (
            "grouped_gemm_rowcolquant_fp8_rcr_compv3_cshuffle_intrawave_"
            "False_False_True_False_"
            "128x128x64_2x2x1_32x32x16"
        )

    def test_bf8_rcr_default_name(self):
        cfg = RowColQuantKernelConfig(
            dtype="bf8",
            layout="rcr",
            pipeline="compv3",
            epilogue="cshuffle",
            scheduler="intrawave",
            tile_m=128, tile_n=128, tile_k=64,
            warp_m=2, warp_n=2, warp_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
            pad_m=False, pad_n=False, pad_k=True,
            persistent=False,
        )
        assert cfg.name == (
            "grouped_gemm_rowcolquant_bf8_rcr_compv3_cshuffle_intrawave_"
            "False_False_True_False_"
            "128x128x64_2x2x1_32x32x16"
        )

    def test_pad_flags_reflected_in_name(self):
        cfg = RowColQuantKernelConfig(
            dtype="fp8", layout="rcr", pipeline="compv3",
            epilogue="cshuffle", scheduler="intrawave",
            tile_m=128, tile_n=128, tile_k=64,
            warp_m=2, warp_n=2, warp_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
            pad_m=True, pad_n=True, pad_k=True, persistent=False,
        )
        assert "True_True_True_False" in cfg.name

    def test_persistent_reflected_in_name(self):
        cfg = RowColQuantKernelConfig(
            dtype="fp8", layout="rcr", pipeline="compv3",
            epilogue="cshuffle", scheduler="intrawave",
            tile_m=128, tile_n=128, tile_k=64,
            warp_m=2, warp_n=2, warp_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
            pad_m=False, pad_n=False, pad_k=False, persistent=True,
        )
        assert cfg.name.endswith("True_128x128x64_2x2x1_32x32x16")

    def test_name_no_spaces(self):
        cfg = default_fp8_config()
        assert " " not in cfg.name

    def test_name_only_valid_chars(self):
        import re
        cfg = default_fp8_config()
        assert re.match(r'^[a-zA-Z0-9_]+$', cfg.name), f"Invalid chars in name: {cfg.name}"

    def test_fp8_bf8_names_differ(self):
        fp8 = default_fp8_config()
        bf8 = default_bf8_config()
        assert fp8.name != bf8.name

    def test_different_tiles_produce_different_names(self):
        def make(tm, tn, tk):
            return RowColQuantKernelConfig(
                dtype="fp8", layout="rcr", pipeline="compv3",
                epilogue="cshuffle", scheduler="intrawave",
                tile_m=tm, tile_n=tn, tile_k=tk,
                warp_m=2, warp_n=2, warp_k=1,
                warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
            ).name
        names = [make(64, 64, 32), make(128, 64, 32), make(128, 128, 64)]
        assert len(names) == len(set(names))

    def test_default_fp8_config_name(self):
        cfg = default_fp8_config()
        assert "fp8" in cfg.name
        assert "grouped_gemm_rowcolquant" in cfg.name

    def test_default_bf8_config_name(self):
        cfg = default_bf8_config()
        assert "bf8" in cfg.name
        assert "grouped_gemm_rowcolquant" in cfg.name


# =============================================================================
# RowColQuantKernelConfig.to_codegen_config — round-trip shape
# =============================================================================


class TestCodegenConfig:

    def test_codegen_config_contains_correct_dtype(self):
        cfg = default_fp8_config()
        d = cfg.to_codegen_config()
        assert d["dtypes"] == ["fp8"]

    def test_codegen_config_tile_roundtrip(self):
        cfg = RowColQuantKernelConfig(
            dtype="fp8", layout="rcr", pipeline="compv3",
            epilogue="cshuffle", scheduler="intrawave",
            tile_m=64, tile_n=128, tile_k=32,
            warp_m=2, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
        )
        d = cfg.to_codegen_config()
        tc = d["tile_configs"][0]
        assert tc["tile_m"] == 64
        assert tc["tile_n"] == 128
        assert tc["tile_k"] == 32
        assert tc["warp_m"] == 2
        assert tc["warp_n"] == 4

    def test_codegen_config_single_layout(self):
        cfg = default_fp8_config()
        d = cfg.to_codegen_config()
        assert d["layouts"] == ["rcr"]

    def test_codegen_config_pad_flags(self):
        cfg = RowColQuantKernelConfig(
            dtype="fp8", layout="rcr", pipeline="compv3",
            epilogue="cshuffle", scheduler="intrawave",
            tile_m=128, tile_n=128, tile_k=64,
            warp_m=2, warp_n=2, warp_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
            pad_m=True, pad_n=False, pad_k=True, persistent=True,
        )
        d = cfg.to_codegen_config()
        assert d["pad_m"] is True
        assert d["pad_n"] is False
        assert d["pad_k"] is True
        assert d["persistent"] is True


# =============================================================================
# RowColQuantGemmProblem dimension helpers
# =============================================================================


class TestRowColQuantGemmProblem:

    def test_QK_A_equals_M(self):
        p = RowColQuantGemmProblem(M=16, N=64, K=256)
        assert p.QK_A == 16

    def test_QK_B_equals_N(self):
        p = RowColQuantGemmProblem(M=16, N=64, K=256)
        assert p.QK_B == 64

    def test_default_k_batch(self):
        p = RowColQuantGemmProblem(M=16, N=64, K=256)
        assert p.k_batch == 1

    def test_k_batch_set(self):
        p = RowColQuantGemmProblem(M=16, N=64, K=256, k_batch=4)
        assert p.k_batch == 4


# =============================================================================
# Name uniqueness across a small sweep
# =============================================================================


class TestNameUniqueness:

    def _make_configs(self):
        configs = []
        for dtype in ("fp8", "bf8"):
            for pad_k in (True, False):
                configs.append(RowColQuantKernelConfig(
                    dtype=dtype, layout="rcr", pipeline="compv3",
                    epilogue="cshuffle", scheduler="intrawave",
                    tile_m=128, tile_n=128, tile_k=64,
                    warp_m=2, warp_n=2, warp_k=1,
                    warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
                    pad_k=pad_k,
                ))
        return configs

    def test_all_names_unique(self):
        configs = self._make_configs()
        names = [c.name for c in configs]
        assert len(names) == len(set(names)), f"Duplicate names: {[n for n in names if names.count(n) > 1]}"


# =============================================================================
# Codegen header generation (CPU-only, no GPU)
# =============================================================================


class TestCodegenHeaderGeneration:

    def _generate_header(self, dtype="fp8"):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codegen"))
        from unified_grouped_gemm_rowcolquant_codegen import (
            RowColQuantKernelHeaderGenerator,
            RowColQuantKernelSpec,
            RowColQuantTileConfig,
        )
        tile = RowColQuantTileConfig(
            tile_m=128, tile_n=128, tile_k=64,
            warp_m=2, warp_n=2, warp_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
        )
        spec = RowColQuantKernelSpec(
            dtype=dtype, layout="rcr", pipeline="compv3",
            epilogue="cshuffle", scheduler="intrawave",
            pad_m=False, pad_n=False, pad_k=True, persistent=False,
            tile=tile,
        )
        gen = RowColQuantKernelHeaderGenerator()
        return gen.generate(spec)

    def test_header_contains_kernel_name(self):
        header = self._generate_header()
        assert "grouped_gemm_rowcolquant_fp8_rcr" in header

    def test_header_contains_rowcolquant_type(self):
        header = self._generate_header()
        assert "ck_tile::QuantType::RowColQuant" in header

    def test_header_contains_selected_kernel(self):
        header = self._generate_header()
        assert "using SelectedKernel" in header

    def test_header_contains_single_kernel_include_guard(self):
        header = self._generate_header()
        assert "CK_TILE_SINGLE_KERNEL_INCLUDE" in header

    def test_bf8_header_uses_bf8_datatype(self):
        header = self._generate_header(dtype="bf8")
        assert "ck_tile::bf8_t" in header

    def test_fp8_header_uses_fp8_datatype(self):
        header = self._generate_header(dtype="fp8")
        assert "ck_tile::fp8_t" in header

    def test_header_contains_aq_bq_layouts(self):
        header = self._generate_header()
        assert "using AQLayout" in header
        assert "using BQLayout" in header

    def test_header_launch_takes_vector_of_host_args(self):
        header = self._generate_header()
        assert "std::vector<ck_tile::QuantGroupedGemmHostArgs>" in header


# =============================================================================
# Default config alignment: utils default vs codegen default
# =============================================================================


class TestDefaultConfigAlignment:
    """Ensure default_fp8_config/default_bf8_config stay in sync with _default_config()."""

    # Both defaults are arch-dependent, so comparing them at a single architecture
    # cannot see a drift that only exists on another one -- which is exactly the
    # drift gfx1250 enablement introduces. Every arch the bridge supports is
    # checked, plus a suffixed spelling on each side.
    _ARCHES = ["gfx942", "gfx950", "gfx1250", "gfx1250:xnack-", "gfx942:sramecc+:xnack-"]

    def _codegen_default_names(self, gfx_arch=""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codegen"))
        from unified_grouped_gemm_rowcolquant_codegen import _default_config, _build_specs
        specs = _build_specs(_default_config(gfx_arch))
        return {s.name for s in specs}

    @pytest.mark.parametrize("gfx_arch", _ARCHES)
    def test_default_fp8_config_name_in_codegen_defaults(self, gfx_arch):
        cfg = default_fp8_config(gfx_arch)
        assert cfg.name in self._codegen_default_names(gfx_arch), (
            f"default_fp8_config('{gfx_arch}').name '{cfg.name}' is not produced by "
            f"_default_config('{gfx_arch}') in the codegen. The two defaults have "
            f"drifted — update one to match the other."
        )

    @pytest.mark.parametrize("gfx_arch", _ARCHES)
    def test_default_bf8_config_name_in_codegen_defaults(self, gfx_arch):
        cfg = default_bf8_config(gfx_arch)
        assert cfg.name in self._codegen_default_names(gfx_arch), (
            f"default_bf8_config('{gfx_arch}').name '{cfg.name}' is not produced by "
            f"_default_config('{gfx_arch}') in the codegen. The two defaults have "
            f"drifted — update one to match the other."
        )

    def test_gfx1250_default_actually_differs_from_the_gfx9_default(self):
        """Guard against the parametrization passing for the wrong reason.

        If _default_config() ever stops honouring gfx_arch, every case above
        still passes -- both sides would just fall back to the gfx9 tile
        together. Assert the two architectures really do produce different
        kernels, so that regression is visible.
        """
        assert self._codegen_default_names("gfx1250").isdisjoint(
            self._codegen_default_names("gfx942")
        )


# =============================================================================
# Arch-dependent hipcc defines
# =============================================================================


class TestArchDefines:
    """The OCP-FP8 define is deliberately a gfx12 *family* test.

    gfx1200/gfx1201/gfx1250 all use OCP FP8 encoding, so narrowing this to an
    exact gfx1250 match would be wrong. This is the opposite of the tile
    selector in codegen_common.rowcol_tensor_quant_default_tile(), which must be
    exact because gfx1200/gfx1201 have a different 8-bit warp fragment. Both
    behaviours are pinned so neither gets "fixed" into the other.
    """

    def _compile_argv(self, monkeypatch, tmp_path, gfx_arch):
        captured = []

        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(cmd, *a, **kw):
            captured.append(list(cmd))
            return _Result()

        monkeypatch.setattr(UTILS.subprocess, "run", fake_run)
        ok = UTILS._compile_rowcolquant_kernel(
            hpp_path=tmp_path / "k.hpp",
            so_path=tmp_path / "k.so",
            gfx_arch=gfx_arch,
        )
        assert ok, "compile helper should report success when hipcc succeeds"
        return captured[0]

    @pytest.mark.parametrize("arch", ["gfx1250", "gfx1250:xnack-"])
    def test_supported_gfx12_targets_get_ocp_fp8(self, monkeypatch, tmp_path, arch):
        argv = self._compile_argv(monkeypatch, tmp_path, arch)
        assert "-DUSE_NEW_UNIFIED_FRAMEWORK=0" in argv
        assert "-DCK_CMAKE_GPU_TARGET_IDS=0x1250" in argv
        assert "-DCK_USE_OCP_FP8" in argv
        assert "-DCK_TILE_USE_OCP_FP8" in argv

    def test_gfx950_gets_ocp_fp8_and_mx(self, monkeypatch, tmp_path):
        argv = self._compile_argv(monkeypatch, tmp_path, "gfx950")
        assert "-DCK_CMAKE_GPU_TARGET_IDS=0x950" in argv
        assert "-DCK_USE_OCP_FP8" in argv
        assert "-DCK_USE_NATIVE_MX_SUPPORT" in argv

    def test_gfx942_gets_neither(self, monkeypatch, tmp_path):
        argv = self._compile_argv(monkeypatch, tmp_path, "gfx942:sramecc+:xnack-")
        assert "-DCK_USE_OCP_FP8" not in argv
        assert "-DCK_USE_NATIVE_MX_SUPPORT" not in argv


# =============================================================================
# Compiler-flag arch normalization
# =============================================================================


class TestArchNormalizationInCompileFlags:
    """A suffixed target must never reach --offload-arch / -DGFX_ARCH.

    normalize_gfx_arch() used to be applied only to the *detected* arch. A caller
    passing gfx_arch="gfx1250:xnack-" therefore had the raw string forwarded into
    the compiler flags. Normalization now happens once at the entry boundary.
    """

    def _compile_argv(self, monkeypatch, tmp_path, gfx_arch):
        captured = []

        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(cmd, *a, **kw):
            captured.append(cmd)
            return _Result()

        monkeypatch.setattr(UTILS.subprocess, "run", fake_run)
        ok = UTILS._compile_rowcolquant_kernel(
            hpp_path=tmp_path / "k.hpp",
            so_path=tmp_path / "k.so",
            gfx_arch=gfx_arch,
        )
        assert ok
        return captured[0]

    def test_suffixed_arch_is_stripped_from_offload_arch(self, monkeypatch, tmp_path):
        argv = self._compile_argv(monkeypatch, tmp_path, "gfx1250:xnack-")
        assert "--offload-arch=gfx1250" in argv
        assert not any("xnack" in tok for tok in argv), argv

    def test_suffixed_arch_is_stripped_from_gfx_arch_define(self, monkeypatch, tmp_path):
        argv = self._compile_argv(monkeypatch, tmp_path, "gfx942:sramecc+:xnack-")
        assert '-DGFX_ARCH="gfx942"' in argv

    def test_default_config_stores_bare_arch(self):
        assert default_fp8_config(gfx_arch="gfx1250:xnack-").gfx_arch == "gfx1250"
        assert default_bf8_config(gfx_arch="gfx942:sramecc+").gfx_arch == "gfx942"


# =============================================================================
# .so cache key encodes the ABI
# =============================================================================


class TestSoCacheAbiKey:
    """A pre-ABI artifact in a persistent output_dir must not be reused.

    setup_multiple_rowcolquant_dispatchers() reuses a .so when the filename
    matches. The name used to be lib{kernel}_{arch}.so, which says nothing about
    the exported symbol set, so a .so predating the
    dispatcher_get_tile_n()/dispatcher_get_pad_n() exports was selected and then
    died at attribute lookup with "undefined symbol". The name now carries
    _SO_ABI, so the stale artifact simply cannot be selected.
    """

    def test_abi_is_versioned(self):
        assert isinstance(UTILS._SO_ABI, int) and UTILS._SO_ABI >= 2

    @pytest.mark.parametrize("old_suffix", ["", "_abi2"])
    def test_pre_abi_artifact_is_not_reused(self, monkeypatch, tmp_path, old_suffix):
        cfg = default_fp8_config(gfx_arch="gfx950")

        so_dir = tmp_path / "libs"
        so_dir.mkdir(parents=True)
        # Exactly the name the old code would have produced and reused.
        stale = so_dir / f"lib{cfg.name}_gfx950{old_suffix}.so"
        stale.write_bytes(b"stale pre-ABI artifact")

        compiled = []

        def fake_compile(hpp_path, so_path, gfx_arch, **kw):
            compiled.append(so_path)
            so_path.write_bytes(b"fresh")
            return True

        monkeypatch.setattr(UTILS, "_compile_rowcolquant_kernel", fake_compile)

        out = UTILS.setup_multiple_rowcolquant_dispatchers(
            configs=[cfg], output_dir=tmp_path, gfx_arch="gfx950", parallel=False,
        )

        assert compiled, "stale pre-ABI .so was reused instead of rebuilt"
        assert out[0] != stale
        assert f"_abi{UTILS._SO_ABI}.so" in out[0].name
        assert stale.read_bytes() == b"stale pre-ABI artifact", "stale file was clobbered"

    def test_second_call_hits_the_cache(self, monkeypatch, tmp_path):
        cfg = default_fp8_config(gfx_arch="gfx950")
        calls = []

        def fake_compile(hpp_path, so_path, gfx_arch, **kw):
            calls.append(so_path)
            so_path.write_bytes(b"fresh")
            return True

        monkeypatch.setattr(UTILS, "_compile_rowcolquant_kernel", fake_compile)

        first = UTILS.setup_multiple_rowcolquant_dispatchers(
            configs=[cfg], output_dir=tmp_path, gfx_arch="gfx950", parallel=False)
        second = UTILS.setup_multiple_rowcolquant_dispatchers(
            configs=[cfg], output_dir=tmp_path, gfx_arch="gfx950", parallel=False)

        assert len(calls) == 1, "second call should have hit the cache"
        assert first[0] == second[0]
