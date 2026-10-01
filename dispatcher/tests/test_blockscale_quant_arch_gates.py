#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
CPU-only unit tests for the gfx12 arch gates in the five non-grouped block-scale
quant bridges (aquant, abquant, bquant, rowcolquant, tensor_quant).

These bridges contain two DIFFERENT kinds of gfx12 predicate, and the whole point
of this suite is to pin the difference so a future "cleanup" cannot collapse one
into the other:

  * 8-bit ``warp_tile_k`` selectors -- must match gfx1250 EXACTLY.  gfx1250 is
    the only gfx12 part with a 16x16x128 8-bit WMMA fragment; gfx1200/gfx1201
    expose 16x16x16.  A K=128 tile on gfx1200/gfx1201 still compiles and then
    silently returns garbage, the same failure mode as K=128 on gfx942, so the
    negative cases below are correctness assertions, not style checks.

  * OCP fp8 encoding predicates -- must stay FAMILY-WIDE.  Every gfx12xx part
    uses OCP e4m3/e5m2, so narrowing these to gfx1250 would break fp8 on
    gfx1200/gfx1201.

Real devices report feature suffixes (``gfx1250:xnack-``), so every exact match
is also tested in suffixed form.  No GPU or hipcc required.

Run:
    python3 -m pytest dispatcher/tests/test_blockscale_quant_arch_gates.py -v
"""

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codegen"))

import codegen_common
import dispatcher_common
import gemm_abquant_utils
import gemm_aquant_utils
import gemm_bquant_utils
import gemm_rowcolquant_utils
import gemm_tensor_quant_utils


# gfx1250 with and without the feature suffix a real agent reports.
GFX1250_FORMS = ["gfx1250", "gfx1250:xnack-"]

# Other gfx12 parts: OCP fp8, but only a 16x16x16 8-bit WMMA fragment.
GFX12_NON_1250 = ["gfx1200", "gfx1201"]

# Legacy MFMA archs whose behaviour must be byte-identical to before this change.
LEGACY_ARCHS = ["gfx90a", "gfx942"]


# =============================================================================
# warp_tile_k selectors -- EXACT gfx1250, not the gfx12 family
#
# Each entry is (id, callable taking gfx_arch -> int).
# =============================================================================

# (id, callable, K expected on a NON-gfx1250 gfx12 part i.e. the legacy MFMA answer)
WARP_TILE_K_SELECTORS = [
    ("aquant_decode",
     lambda arch: gemm_aquant_utils._warp_tile_k_for(arch, preshuffle_aquant=False), 32),
    ("aquant_preshufflequant",
     lambda arch: gemm_aquant_utils._warp_tile_k_for(arch, preshuffle_aquant=True), 64),
    ("abquant_fp8",
     lambda arch: gemm_abquant_utils._warp_tile_k_for("fp8", arch), 32),
    ("abquant_bf8_flatmm",
     lambda arch: gemm_abquant_utils._warp_tile_k_for("bf8", arch, is_flat_mm=True), 64),
    ("bquant_decode",
     lambda arch: gemm_bquant_utils._warp_tile_k_for(arch, is_flatmm=False), 32),
    ("bquant_preshuffleb",
     lambda arch: gemm_bquant_utils._warp_tile_k_for(arch, is_flatmm=True), 64),
    ("rowcolquant_fp8",
     lambda arch: gemm_rowcolquant_utils._warp_tile_k_for("fp8", arch), 32),
    ("rowcolquant_bf8",
     lambda arch: gemm_rowcolquant_utils._warp_tile_k_for("bf8", arch), 32),
    ("tensor_quant_fp8",
     gemm_tensor_quant_utils.fp8_warp_tile_k_for_arch, 32),
]

WARP_TILE_K_IDS = [name for name, _, _ in WARP_TILE_K_SELECTORS]
WARP_TILE_K_FNS = [fn for _, fn, _ in WARP_TILE_K_SELECTORS]
WARP_TILE_K_LEGACY = [(fn, exp) for _, fn, exp in WARP_TILE_K_SELECTORS]


@pytest.mark.parametrize("selector", WARP_TILE_K_FNS, ids=WARP_TILE_K_IDS)
@pytest.mark.parametrize("arch", GFX1250_FORMS)
def test_gfx1250_selects_128(selector, arch):
    """gfx1250 has the 16x16x128 8-bit WMMA fragment -> K warp tile 128."""
    assert selector(arch) == 128


@pytest.mark.parametrize("selector,expected", WARP_TILE_K_LEGACY, ids=WARP_TILE_K_IDS)
@pytest.mark.parametrize("arch", GFX12_NON_1250)
def test_other_gfx12_parts_never_select_128(selector, expected, arch):
    """gfx1200/gfx1201 must NOT inherit gfx1250's K=128 tile.

    Their 8-bit WMMA fragment is 16x16x16; a K=128 kernel compiles for them and
    then silently returns wrong results, so a family-wide "gfx12" test here is a
    silent-correctness bug rather than a cosmetic one.

    Asserted as an EXACT value rather than "!= 128": a bare inequality passes for
    any wrong answer (16, 64, ...) and would not catch a selector that broke in a
    different direction.
    """
    assert selector(arch) == expected


@pytest.mark.parametrize("selector", WARP_TILE_K_FNS, ids=WARP_TILE_K_IDS)
def test_feature_suffix_does_not_change_result(selector):
    """``gfx1250:xnack-`` must normalize to the same answer as ``gfx1250``."""
    assert selector("gfx1250:xnack-") == selector("gfx1250")


# -----------------------------------------------------------------------------
# Legacy arch outputs: unchanged by gfx1250 enablement.
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("arch,expected", [
    ("gfx90a", 32),
    ("gfx942", 32),
    ("gfx950", 128),
])
def test_aquant_decode_legacy_unchanged(arch, expected):
    assert gemm_aquant_utils._warp_tile_k_for(arch, preshuffle_aquant=False) == expected


@pytest.mark.parametrize("arch,expected", [
    ("gfx90a", 64),
    ("gfx942", 64),
    ("gfx950", 128),   # gfx950 ignores IsFlatMM
])
def test_aquant_preshufflequant_legacy_unchanged(arch, expected):
    assert gemm_aquant_utils._warp_tile_k_for(arch, preshuffle_aquant=True) == expected


@pytest.mark.parametrize("variant,arch,is_flat_mm,expected", [
    ("fp8", "gfx942", False, 32),
    ("fp8", "gfx942", True, 64),
    ("fp8", "gfx950", False, 128),
    ("fp8", "gfx950", True, 128),    # IsFlatMM ignored on gfx950
    ("fp4", "gfx950", True, 32),     # non-8bit-float stays 32 on gfx950
    ("fp4", "gfx942", True, 64),     # ... and the legacy IsFlatMM rule on gfx942
])
def test_abquant_legacy_and_fp4_unchanged(variant, arch, is_flat_mm, expected):
    assert gemm_abquant_utils._warp_tile_k_for(variant, arch, is_flat_mm=is_flat_mm) == expected


@pytest.mark.parametrize("arch", GFX1250_FORMS)
@pytest.mark.parametrize("is_flat_mm", [False, True])
def test_abquant_fp4_uses_128_on_gfx1250(arch, is_flat_mm):
    """fp4 must use K=128 on gfx1250 -- 32 there is a silent-wrong-answer bug.

    gfx1250 differs from gfx950, where fp4 correctly stays at 32.  abquant
    dispatches on AComputeDataType and auto_compute_type collapses packed
    A==B==pk_fp4_t to fp8_t, so at K=32 the fp4 kernel selects the unguarded MFMA
    Dispatcher<fp8_t,fp8_t,float,16,16,32>, whose intrinsics are
    #if __gfx94__/__gfx95__ only and otherwise return CVecType{0.f}.

    GPU-measured on MI400: K=32 builds, launches, and yields a dead accumulator
    (~50% exact zeros, the rest uncorrelated noise, corr vs reference -0.004).
    Note it is NOT uniformly zero, so an ``all(C == 0)`` check does not catch it.
    K=64 and K=128 both verify at 4.74e-4 max relative error.
    """
    assert gemm_abquant_utils._warp_tile_k_for("fp4", arch, is_flat_mm=is_flat_mm) == 128


@pytest.mark.parametrize("arch", GFX1250_FORMS)
def test_abquant_default_config_set_keeps_fp4_on_gfx1250(arch):
    """fp4 stays in the gfx1250 default set, now carrying the 128 tile."""
    cfgs = gemm_abquant_utils.all_default_configs(arch)
    fp4 = [c for c in cfgs if c.variant_key == "fp4"]
    assert fp4, "fp4 configs must still be emitted for gfx1250"
    assert all(c.warp_tile_k == 128 for c in fp4)
    # ... while gfx950 keeps its own 32.
    assert all(c.warp_tile_k == 32
               for c in gemm_abquant_utils.all_default_configs("gfx950")
               if c.variant_key == "fp4")


@pytest.mark.parametrize("arch,is_flatmm,expected", [
    ("gfx90a", False, 32),
    ("gfx942", False, 32),
    ("gfx942", True, 64),
    ("gfx950", False, 128),
    ("gfx950", True, 128),
])
def test_bquant_legacy_unchanged(arch, is_flatmm, expected):
    assert gemm_bquant_utils._warp_tile_k_for(arch, is_flatmm=is_flatmm) == expected


@pytest.mark.parametrize("variant,arch,expected", [
    ("fp8", "gfx942", 32),
    ("bf8", "gfx942", 32),
    ("fp8", "gfx950", 128),
    ("bf8", "gfx950", 128),
])
def test_rowcolquant_legacy_unchanged(variant, arch, expected):
    assert gemm_rowcolquant_utils._warp_tile_k_for(variant, arch) == expected


@pytest.mark.parametrize("arch,expected", [
    ("gfx942", 32),
    ("gfx950", 128),
])
def test_tensor_quant_legacy_unchanged(arch, expected):
    assert gemm_tensor_quant_utils.fp8_warp_tile_k_for_arch(arch) == expected


@pytest.mark.parametrize("selector", WARP_TILE_K_FNS, ids=WARP_TILE_K_IDS)
@pytest.mark.parametrize("arch", [None, ""])
def test_selectors_never_silently_default(selector, arch):
    """A missing arch must raise, not fall through to the gfx942 legacy tile.

    These modules state a "never silently default" rule.  Note the call sites
    evaluate ``_is_gfx1250(gfx_arch)`` FIRST precisely so this guard is reachable:
    with ``"gfx950" in gfx_arch`` first, None raised an opaque TypeError from the
    ``in`` operator instead, and _is_gfx1250's documented None tolerance was dead
    code from the bridges' point of view.
    """
    with pytest.raises(ValueError):
        selector(arch)


# =============================================================================
# pk_int4 ("i4") variants are rejected on gfx1250
# =============================================================================
#
# GPU-measured on MI400: {fp8i4, bf8i4} x warp_tile_k {16,32,64,128} x N
# {128,256,512} -- all 24 combinations BUILT and LAUNCHED, and all 24 returned
# NaN.  C4/fp8 and C4/bf8 pass in the same harness with the same host codec, so
# the breakage is specific to the packed-int4 path, not to fp8/bf8 packing.
#
# Note the failure mode contradicts the prose in the grouped sibling, which says
# i4 "does NOT compile on gfx1250": it compiles cleanly and is wrong at runtime,
# which is the more dangerous direction.  These tests pin the loud rejection.

I4_CONSTRUCTORS = [
    ("aquant_fp8i4", gemm_aquant_utils.default_fp8i4_config),
    ("aquant_bf8i4", gemm_aquant_utils.default_bf8i4_config),
    ("aquant_fp8i4_psq", gemm_aquant_utils.default_fp8i4_preshufflequant_config),
    ("aquant_bf8i4_psq", gemm_aquant_utils.default_bf8i4_preshufflequant_config),
    ("bquant_fp8i4", gemm_bquant_utils.default_fp8i4_config),
    ("bquant_bf8i4", gemm_bquant_utils.default_bf8i4_config),
    ("bquant_fp8i4_psb", gemm_bquant_utils.default_fp8i4_preshuffleb_config),
    ("bquant_bf8i4_psb", gemm_bquant_utils.default_bf8i4_preshuffleb_config),
    ("bquant_fp8i4_psq", gemm_bquant_utils.default_fp8i4_preshufflequant_config),
    ("bquant_bf8i4_psq", gemm_bquant_utils.default_bf8i4_preshufflequant_config),
]
I4_IDS = [n for n, _ in I4_CONSTRUCTORS]
I4_FNS = [f for _, f in I4_CONSTRUCTORS]


@pytest.mark.parametrize("make_config", I4_FNS, ids=I4_IDS)
@pytest.mark.parametrize("arch", GFX1250_FORMS)
def test_i4_variants_rejected_on_gfx1250(make_config, arch):
    with pytest.raises(ValueError, match="not supported"):
        make_config(gfx_arch=arch)


@pytest.mark.parametrize("make_config", I4_FNS, ids=I4_IDS)
@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_i4_variants_still_work_on_gfx9(make_config, arch):
    """The rejection is gfx1250-only: gfx942/gfx950 i4 must be untouched."""
    cfg = make_config(gfx_arch=arch)
    assert cfg.gfx_arch == arch
    assert cfg.warp_tile_k in (32, 64, 128)


@pytest.mark.parametrize("arch", GFX1250_FORMS)
def test_fp8_bf8_still_allowed_on_gfx1250(arch):
    """Only i4 is gated -- the fp8/bf8 paths this PR enables must still build."""
    assert gemm_aquant_utils.default_fp8_config(gfx_arch=arch).warp_tile_k == 128
    assert gemm_bquant_utils.default_bf8_config(gfx_arch=arch).warp_tile_k == 128


# =============================================================================
# OCP fp8 predicates -- FAMILY-WIDE gfx12 is correct here
# =============================================================================

OCP_PREDICATES = [
    ("bquant", gemm_bquant_utils._uses_ocp_fp8),
    ("rowcolquant", gemm_rowcolquant_utils._uses_ocp_fp8),
]
OCP_IDS = [name for name, _ in OCP_PREDICATES]
OCP_FNS = [fn for _, fn in OCP_PREDICATES]


@pytest.mark.parametrize("predicate", OCP_FNS, ids=OCP_IDS)
@pytest.mark.parametrize("arch", GFX1250_FORMS + GFX12_NON_1250 + ["gfx950"])
def test_ocp_fp8_is_family_wide(predicate, arch):
    """Every gfx12xx part uses OCP fp8/bf8, so this predicate stays family-wide.

    Narrowing it to an exact gfx1250 match would silently switch gfx1200/gfx1201
    host-side encoding to FNUZ and produce mismatched/NaN comparisons.
    """
    assert predicate(arch) is True


@pytest.mark.parametrize("predicate", OCP_FNS, ids=OCP_IDS)
@pytest.mark.parametrize("arch", LEGACY_ARCHS)
def test_ocp_fp8_false_on_legacy_fnuz_archs(predicate, arch):
    assert predicate(arch) is False


@pytest.mark.parametrize("predicate", OCP_FNS, ids=OCP_IDS)
def test_ocp_fp8_defaults_to_ocp_when_arch_unknown(predicate):
    assert predicate(None) is True


# -----------------------------------------------------------------------------
# The compile-path -D flags, in all FIVE modules.
#
# _uses_ocp_fp8 above only exists in two of the bridges and governs HOST-side
# encoding.  The hipcc -D flags are a separate gate that every module has, and
# they had no coverage at all -- which is why abquant silently shipped a
# gfx950-only version of it.  ALL FIVE must be family-wide gfx12.
# -----------------------------------------------------------------------------

# The five bridges now route their compile-path -D flags through the SHARED
# dispatcher_common.arch_feature_defines(), as the five grouped bridges already
# did. That helper supplies the OCP pair AND the per-arch feature set (notably
# CK_TILE_USE_WMMA, which must be passed even when 0 -- an undefined identifier
# reads as 0, which is right on gfx942/gfx950 and wrong on every WMMA part).
# Pin both halves here: the family-wide OCP rule and the gfx1250 feature set.


@pytest.mark.parametrize("arch", GFX1250_FORMS + GFX12_NON_1250 + ["gfx950"])
def test_ocp_fp8_defines_emitted_family_wide(arch):
    """Every gfx12xx part -- not just gfx1250 -- must get the OCP fp8 defines.

    Narrowing this to an exact gfx1250 match would switch gfx1200/gfx1201 back to
    the FNUZ encodings. This is the exact inverse of the warp_tile_k rule, and the
    two must never be "tidied" into each other.
    """
    defines = dispatcher_common.arch_feature_defines(arch)
    assert "-DCK_USE_OCP_FP8" in defines
    assert "-DCK_TILE_USE_OCP_FP8" in defines


@pytest.mark.parametrize("arch", LEGACY_ARCHS + [None, ""])
def test_ocp_fp8_defines_absent_on_fnuz_archs(arch):
    defines = dispatcher_common.arch_feature_defines(arch)
    assert "-DCK_USE_OCP_FP8" not in defines
    assert "-DCK_TILE_USE_OCP_FP8" not in defines


@pytest.mark.parametrize("arch", GFX1250_FORMS)
def test_gfx1250_gets_the_wmma_feature_defines(arch):
    """gfx1250 needs the WMMA feature set, not just the OCP encoding pair.

    Without CK_TILE_USE_WMMA the generated K=128 configuration can compile down
    the non-WMMA feature path.
    """
    defines = dispatcher_common.arch_feature_defines(arch)
    for flag in ("-DCK_TILE_USE_WMMA=1", "-DCK_GFX12_SUPPORT",
                 "-DCK_USE_GFX1250", "-DCK_GFX1250_SUPPORT"):
        assert flag in defines, f"{flag} missing for {arch}: {defines}"


@pytest.mark.parametrize("arch", LEGACY_ARCHS + ["gfx950"])
def test_legacy_archs_do_not_get_wmma_defines(arch):
    assert "-DCK_TILE_USE_WMMA=1" not in dispatcher_common.arch_feature_defines(arch)


# =============================================================================
# Configs must not be compiled for a different arch than they were built for
# =============================================================================
#
# Every arch safeguard in these bridges runs when the CONFIG is constructed,
# keyed on that config's own gfx_arch. The compile entry points take their own
# gfx_arch, so without a cross-check a config built for one arch and compiled for
# another slips past all of them -- its literal tile is emitted verbatim.

SETUP_ENTRYPOINTS = [
    ("aquant", gemm_aquant_utils.setup_multiple_aquant_dispatchers,
     lambda: gemm_aquant_utils.default_fp8_config(gfx_arch="gfx950")),
    ("abquant", gemm_abquant_utils.setup_multiple_abquant_dispatchers,
     lambda: gemm_abquant_utils.default_fp4_config(gfx_arch="gfx950")),
    ("bquant", gemm_bquant_utils.setup_multiple_bquant_dispatchers,
     lambda: gemm_bquant_utils.default_fp8_config(gfx_arch="gfx950")),
]
SETUP_IDS = [n for n, _, _ in SETUP_ENTRYPOINTS]


@pytest.mark.parametrize("setup,make_cfg",
                         [(s, m) for _, s, m in SETUP_ENTRYPOINTS], ids=SETUP_IDS)
@pytest.mark.parametrize("arch", GFX1250_FORMS)
def test_setup_rejects_config_built_for_another_arch(setup, make_cfg, arch):
    """The motivating case: default_fp4_config("gfx950") records warp_tile_k=32.

    Compiling it for gfx1250 would emit a 16x16x32 tile there -- the GPU-confirmed
    dead-accumulator case -- because the fp4 rule already ran, for gfx950.
    """
    cfg = make_cfg()
    with pytest.raises(ValueError, match="different architecture"):
        setup([cfg], gfx_arch=arch)


def test_setup_accepts_matching_arch():
    """The guard must only fire on a genuine mismatch."""
    cfg = gemm_abquant_utils.default_fp4_config(gfx_arch="gfx950")
    # Same arch: must get past the guard (it will fail later for lack of hipcc,
    # which is fine -- we only assert the guard itself does not reject it).
    try:
        gemm_abquant_utils.setup_multiple_abquant_dispatchers([cfg], gfx_arch="gfx950")
    except ValueError as exc:
        assert "different architecture" not in str(exc)
    except Exception:
        pass


# =============================================================================
# Supported-arch lists accept gfx1250, including with a feature suffix
# =============================================================================

VALIDATORS = [
    ("aquant", gemm_aquant_utils._validate_arch),
    ("abquant", gemm_abquant_utils._validate_arch),
    ("tensor_quant", gemm_tensor_quant_utils._validate_arch),
]
VALIDATOR_IDS = [name for name, _ in VALIDATORS]
VALIDATOR_FNS = [fn for _, fn in VALIDATORS]


@pytest.mark.parametrize("validate", VALIDATOR_FNS, ids=VALIDATOR_IDS)
@pytest.mark.parametrize("arch", GFX1250_FORMS + ["gfx942", "gfx950"])
def test_validate_arch_accepts_supported(validate, arch):
    assert validate(arch) == arch


@pytest.mark.parametrize("validate", VALIDATOR_FNS, ids=VALIDATOR_IDS)
@pytest.mark.parametrize("arch", [
    "gfx1030", "gfx803", "sm_90",
    # Near-misses that a prefix test (arch.startswith) would WRONGLY accept.
    # These matter because an accepted "gfx12500" then fails _is_gfx1250, drops
    # to the legacy 32/64 tile, and proceeds -- a silent default, which is
    # exactly what these modules say they never do.
    "gfx12500", "gfx1250x", "gfx9500", "gfx942x",
    "", None,
])
def test_validate_arch_rejects_unsupported(validate, arch):
    with pytest.raises(ValueError):
        validate(arch)


@pytest.mark.parametrize("validate", VALIDATOR_FNS, ids=VALIDATOR_IDS)
@pytest.mark.parametrize("arch", ["gfx1250:xnack-", "gfx950:xnack-",
                                  "gfx942:sramecc+:xnack-"])
def test_validate_arch_returns_full_target_id(validate, arch):
    """Only the feature suffix is stripped for MATCHING; the caller gets it back.

    The returned string goes straight to --offload-arch and into the .so
    filename, so it must survive validation byte-for-byte.
    """
    assert validate(arch) == arch


@pytest.mark.parametrize("module", [
    gemm_aquant_utils,
    gemm_abquant_utils,
    gemm_tensor_quant_utils,
], ids=["aquant", "abquant", "tensor_quant"])
def test_gfx1250_listed_in_supported_archs(module):
    assert "gfx1250" in module._SUPPORTED_ARCHS


# =============================================================================
# The private normalizer itself
# =============================================================================

NORMALIZERS = [
    ("aquant", gemm_aquant_utils._is_gfx1250),
    ("abquant", gemm_abquant_utils._is_gfx1250),
    ("bquant", gemm_bquant_utils._is_gfx1250),
    ("rowcolquant", gemm_rowcolquant_utils._is_gfx1250),
    ("tensor_quant", gemm_tensor_quant_utils._is_gfx1250),
]
NORMALIZER_IDS = [name for name, _ in NORMALIZERS]
NORMALIZER_FNS = [fn for _, fn in NORMALIZERS]


@pytest.mark.parametrize("is_gfx1250", NORMALIZER_FNS, ids=NORMALIZER_IDS)
@pytest.mark.parametrize("arch,expected", [
    ("gfx1250", True),
    ("gfx1250:xnack-", True),
    ("gfx1250:xnack+", True),
    ("gfx1200", False),
    ("gfx1201", False),
    ("gfx12", False),
    ("gfx125", False),
    ("gfx12500", False),   # substring matching would wrongly accept this
    ("gfx942", False),
    ("gfx950", False),
    ("", False),
    (None, False),
])
def test_is_gfx1250_exact_match(is_gfx1250, arch, expected):
    assert is_gfx1250(arch) is expected


# =============================================================================
# The shared codegen helper must agree with the runtime bridges
# =============================================================================
#
# codegen_common.fp8_warp_tile_k_for_arch is the declared single source of truth
# for the arch -> WarpTileK rule, and its own docstring says a second, drifting
# copy is a silent-wrong-answer bug.  It is a genuinely separate implementation
# from the five bridges, so pin them together.  (They collapse onto the shared
# normalize_gfx_arch() once #11043 lands.)


@pytest.mark.parametrize("arch,expected", [
    ("gfx1250", 128),
    ("gfx1250:xnack-", 128),
    ("gfx950", 128),
    ("gfx942", 32),
    ("gfx1200", 32),   # NOT gfx1250: only a 16x16x16 8-bit fragment
    ("gfx1201", 32),
])
def test_codegen_common_matches_runtime_selector(arch, expected):
    assert codegen_common.fp8_warp_tile_k_for_arch(arch) == expected
    assert gemm_tensor_quant_utils.fp8_warp_tile_k_for_arch(arch) == expected


@pytest.mark.parametrize("arch", GFX1250_FORMS + ["gfx950", "gfx942", "gfx1200"])
def test_codegen_common_agrees_with_aquant_on_both_pipelines(arch):
    """AQuant is the one caller that uses the preshuffle_quant axis."""
    assert (codegen_common.fp8_warp_tile_k_for_arch(arch)
            == gemm_aquant_utils._warp_tile_k_for(arch, preshuffle_aquant=False))
    assert (codegen_common.fp8_warp_tile_k_for_arch(arch, preshuffle_quant=True)
            == gemm_aquant_utils._warp_tile_k_for(arch, preshuffle_aquant=True))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
