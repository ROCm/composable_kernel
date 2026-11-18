"""Tests for GemmUniversalVecFeatureEngine."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HEURISTICS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HEURISTICS))

from feature_engine import GemmUniversalFeatureEngine  # noqa: E402
from feature_engine_vec import (  # noqa: E402
    VEC_FEATURES,
    GemmUniversalVecFeatureEngine,
)

LAYOUTS = ("rcr", "rrr", "crr", "ccr")


def _problem(m=512, n=512, k=512, layout="rcr", dtype="bf16", upper=False):
    keys = ("M", "N", "K") if upper else ("m", "n", "k")
    return {
        "op_type": "gemm_universal",
        "dtype": dtype,
        "layout": layout,
        "arch": "gfx950",
        keys[0]: m,
        keys[1]: n,
        keys[2]: k,
        "split_k": 1,
    }


#: Geometry whose native widths differ by arch, layout and epilogue.
DISCRIMINATING = {
    "tile_m": 32,
    "tile_n": 32,
    "tile_k": 16,
    "warp_m": 2,
    "warp_n": 2,
    "warp_k": 1,
    "warp_tile_m": 32,
    "warp_tile_n": 32,
    "warp_tile_k": 16,
}


def _kernel(vec=(0, 0, 0), **over):
    return {
        "tile_m": 128,
        "tile_n": 128,
        "tile_k": 128,
        "warp_m": 2,
        "warp_n": 2,
        "warp_k": 1,
        "warp_tile_m": 32,
        "warp_tile_n": 32,
        "warp_tile_k": 16,
        "pipeline": "compv3",
        "epilogue": "default",
        "scheduler": "intrawave",
        "pad_m": False,
        "pad_n": False,
        "pad_k": False,
        "persistent": True,
        "vec_a": vec[0],
        "vec_b": vec[1],
        "vec_c": vec[2],
        **over,
    }


def _row(**kw):
    pk = ("m", "n", "k", "layout", "dtype")
    kernel_over = {k: v for k, v in kw.items() if k not in (*pk, "vec", "arch")}
    row = {
        **_problem(**{k: v for k, v in kw.items() if k in pk}),
        **_kernel(kw.get("vec", (0, 0, 0)), **kernel_over),
        "measured_tflops": 1.0,
        "latency_ms": 1.0,
        "is_valid": True,
    }
    if "arch" in kw:
        row["arch"] = kw["arch"]
    return row


def _frame(**kw):
    return pd.DataFrame([_row(**kw)])


def _legal(m, n, k, layout="rcr", dtype="bf16"):
    from codegen_common import gemm_problem_vector_sizes
    from feature_engine_vec import _out_dtype

    return gemm_problem_vector_sizes(m, n, k, layout, dtype, dtype, _out_dtype(dtype))


class TestSchema:
    def test_adds_exactly_six_features(self):
        base = len(GemmUniversalFeatureEngine().get_feature_names())
        ext = GemmUniversalVecFeatureEngine().get_feature_names()
        assert len(ext) == base + 6
        assert ext[-6:] == [
            "vec_a",
            "vec_b",
            "vec_c",
            "vec_frac_a",
            "vec_frac_b",
            "vec_frac_c",
        ]

    def test_base_engine_is_untouched(self):
        names = GemmUniversalFeatureEngine().get_feature_names()
        assert not any(f in names for f in VEC_FEATURES)

    def test_extract_width_matches_the_name_list(self):
        fe = GemmUniversalVecFeatureEngine()
        assert fe.extract(_problem(), _kernel()).shape == (len(fe.get_feature_names()),)

    def test_extract_batch_width_matches_the_name_list(self):
        fe = GemmUniversalVecFeatureEngine()
        assert fe.extract_batch(_frame()).shape == (1, len(fe.get_feature_names()))


class TestParity:
    @pytest.mark.parametrize("layout", LAYOUTS)
    @pytest.mark.parametrize(
        "vec", [(0, 0, 0), (8, 8, 8), (4, 4, 8), (1, 1, 1), (2, 8, 8)]
    )
    def test_single_matches_batch(self, layout, vec):
        fe = GemmUniversalVecFeatureEngine()
        p, kn = _problem(m=1792, n=300, k=20226, layout=layout), _kernel(vec)
        np.testing.assert_allclose(
            fe.extract(p, kn),
            fe.extract_batch(_frame(m=1792, n=300, k=20226, layout=layout, vec=vec))[0],
            rtol=0,
            atol=1e-12,
        )

    @pytest.mark.parametrize(
        "stored",
        [4, "4", 4.0, 8, "8"],
        ids=["int", "str", "float", "int8", "str8"],
    )
    def test_width_coercion_agrees(self, stored):
        fe = GemmUniversalVecFeatureEngine()
        kn = {**_kernel(), "vec_a": stored}
        df = _frame()
        df["vec_a"] = pd.Series([stored], dtype=object)
        np.testing.assert_allclose(
            fe.extract(_problem(), kn), fe.extract_batch(df)[0], rtol=0, atol=1e-12
        )

    @pytest.mark.parametrize(
        "stored", [float("nan"), "", None], ids=["nan", "empty-string", "none"]
    )
    def test_both_paths_reject_a_null_width(self, stored):
        fe = GemmUniversalVecFeatureEngine()
        with pytest.raises(ValueError, match="native"):
            fe.extract(_problem(), {**_kernel(), "vec_a": stored})
        df = _frame()
        df["vec_a"] = pd.Series([stored], dtype=object)
        with pytest.raises(ValueError, match="native"):
            fe.extract_batch(df)

    def test_parity_across_a_mixed_batch(self):
        fe = GemmUniversalVecFeatureEngine()
        specs = [
            dict(m=1, n=32, k=3072, layout="rrr", vec=(8, 1, 1)),
            dict(m=2048, n=512, k=192, layout="rcr", vec=(0, 0, 0)),
            dict(m=256, n=256, k=133120, layout="crr", vec=(2, 8, 8)),
            dict(m=1024, n=1024, k=1562, layout="ccr", vec=(8, 2, 8)),
        ]
        batch = fe.extract_batch(
            pd.concat([_frame(**s) for s in specs], ignore_index=True)
        )
        for i, s in enumerate(specs):
            p = _problem(**{k: v for k, v in s.items() if k != "vec"})
            np.testing.assert_allclose(
                fe.extract(p, _kernel(s["vec"])), batch[i], rtol=0, atol=1e-12
            )


class TestArgumentSplit:
    def test_widths_come_from_the_kernel_dict(self):
        fe = GemmUniversalVecFeatureEngine()
        v = fe.extract(_problem(), _kernel((4, 4, 8)))[-6:]
        assert list(v[:3]) == [4.0, 4.0, 8.0]

    def test_widths_in_the_problem_dict_are_ignored(self):
        fe = GemmUniversalVecFeatureEngine()
        p = {**_problem(), "vec_a": 1, "vec_b": 1, "vec_c": 1}
        v = fe.extract(p, _kernel((4, 4, 8)))[-6:]
        assert list(v[:3]) == [4.0, 4.0, 8.0], (
            "widths must be read from the kernel dict"
        )

    def test_extents_come_from_the_problem_dict(self):
        fe = GemmUniversalVecFeatureEngine()
        v = fe.extract(_problem(k=4), _kernel((4, 4, 8)))[-6:]
        assert v[3] == pytest.approx(1.0)

    def test_uppercase_extents_are_accepted(self):
        fe = GemmUniversalVecFeatureEngine()
        v = fe.extract(_problem(k=4, upper=True), _kernel((4, 4, 8)))[-6:]
        assert v[3] == pytest.approx(1.0), "uppercase M/N/K must resolve like lowercase"

    def test_missing_extent_raises(self):
        fe = GemmUniversalVecFeatureEngine()
        p = {k: v for k, v in _problem().items() if k != "k"}
        with pytest.raises(KeyError):
            fe.extract(p, _kernel())


class TestVectorFeatures:
    def test_native_resolves_to_the_kernels_own_width(self):
        fe = GemmUniversalVecFeatureEngine()
        v = fe.extract(_problem(), _kernel((0, 0, 0)))[-6:]
        assert list(v[:3]) == [0.0, 0.0, 0.0]
        np.testing.assert_allclose(v[3:5], [1.0, 1.0])
        assert v[5] < 1.0, "default-epilogue C is narrower than the legal width"

    def test_fraction_is_problem_relative(self):
        fe = GemmUniversalVecFeatureEngine()
        kn = _kernel((4, 4, 8))
        tight = fe.extract(_problem(k=4), kn)[-6:]  # rcr A/B contiguous in k
        loose = fe.extract(_problem(k=512), kn)[-6:]
        assert list(tight[:3]) == list(loose[:3])
        assert tight[3] == pytest.approx(1.0)  # 4 of 4 legal
        assert loose[3] == pytest.approx(0.5)  # 4 of 8 legal

    def test_fraction_exceeds_one_when_width_is_illegal(self):
        fe = GemmUniversalVecFeatureEngine()
        v = fe.extract(_problem(k=4), _kernel((8, 8, 8)))[-6:]
        assert v[3] == pytest.approx(2.0)
        np.testing.assert_allclose(
            fe.extract_batch(_frame(k=4, vec=(8, 8, 8)))[0][-6:], v
        )


class TestBatchValidation:
    @pytest.mark.parametrize("col", ["m", "n", "k", "layout", "dtype"])
    def test_nan_in_a_required_column_raises(self, col):
        fe = GemmUniversalVecFeatureEngine()
        df = _frame()
        df[col] = np.nan if col in ("m", "n", "k") else None
        with pytest.raises(ValueError, match="contains null"):
            fe.extract_batch(df)

    @pytest.mark.parametrize("col", ["m", "n", "k", "layout", "dtype"])
    def test_missing_required_column_raises(self, col):
        fe = GemmUniversalVecFeatureEngine()
        with pytest.raises(KeyError, match="required to compute vector-width"):
            fe.extract_batch(_frame().drop(columns=[col]))

    def test_unknown_layout_raises(self):
        fe = GemmUniversalVecFeatureEngine()
        df = _frame()
        df["layout"] = "zzz"
        with pytest.raises(ValueError, match="layout"):
            fe.extract_batch(df)

    def test_unknown_dtype_raises(self):
        fe = GemmUniversalVecFeatureEngine()
        df = _frame()
        df["dtype"] = "float128"
        with pytest.raises(KeyError):
            fe.extract_batch(df)

    def test_empty_frame_is_well_formed(self):
        fe = GemmUniversalVecFeatureEngine()
        assert fe.extract_batch(_frame().iloc[0:0]).shape == (
            0,
            len(fe.get_feature_names()),
        )


class TestWidthEncoding:
    def test_zero_resolves_to_the_kernels_native_width(self):
        fe = GemmUniversalVecFeatureEngine()
        v = fe.extract(_problem(), _kernel((0, 0, 0)))[-6:]
        assert list(v[:3]) == [0.0, 0.0, 0.0]
        np.testing.assert_allclose(v[3:5], [1.0, 1.0])

    def test_default_epilogue_native_c_is_one_element_not_the_legal_width(self):
        fe = GemmUniversalVecFeatureEngine()
        p = _problem(m=1792, n=1536, k=7168)  # gcd(1536, 8) == 8
        cs = fe.extract(p, {**_kernel((0, 0, 0)), "epilogue": "cshuffle"})[-1]
        de = fe.extract(p, {**_kernel((0, 0, 0)), "epilogue": "default"})[-1]
        assert cs == pytest.approx(1.0), "cshuffle reaches the legal width here"
        assert de == pytest.approx(0.125), "default epilogue stores 1 of 8"

    def test_the_epilogue_changes_only_the_c_fraction(self):
        fe = GemmUniversalVecFeatureEngine()
        p = _problem(m=1792, n=1536, k=7168)
        cs = fe.extract(p, {**_kernel((0, 0, 0)), "epilogue": "cshuffle"})[-6:]
        de = fe.extract(p, {**_kernel((0, 0, 0)), "epilogue": "default"})[-6:]
        np.testing.assert_allclose(cs[3:5], de[3:5])
        assert cs[5] != de[5]

    def test_a_fixed_width_ignores_the_epilogue(self):
        fe = GemmUniversalVecFeatureEngine()
        p = _problem(m=1792, n=1536, k=7168)
        cs = fe.extract(p, {**_kernel((4, 4, 4)), "epilogue": "cshuffle"})[-6:]
        de = fe.extract(p, {**_kernel((4, 4, 4)), "epilogue": "default"})[-6:]
        np.testing.assert_allclose(cs, de)

    def test_explicit_widths_are_not_collapsed(self):
        fe = GemmUniversalVecFeatureEngine()
        v = fe.extract(_problem(k=512), _kernel((1, 1, 1)))[-6:]
        assert list(v[:3]) == [1.0, 1.0, 1.0]
        assert v[3] == pytest.approx(0.125)

    def test_negative_width_raises(self):
        from feature_engine_vec import _width

        with pytest.raises(ValueError, match=">= 0"):
            _width(-4)

    def test_absent_key_raises_on_the_scalar_path(self):
        fe = GemmUniversalVecFeatureEngine()
        kn = {k: v for k, v in _kernel().items() if k != "vec_a"}
        with pytest.raises(KeyError, match="absent from both"):
            fe.extract(_problem(), kn)

    def test_c_width_uses_the_output_dtype_not_the_input(self):
        fe = GemmUniversalVecFeatureEngine()
        p = _problem(m=1792, n=1536, k=16, dtype="fp8")
        scalar = fe.extract(p, _kernel((16, 16, 8)))[-6:]
        batch = fe.extract_batch(
            _frame(m=1792, n=1536, k=16, dtype="fp8", vec=(16, 16, 8))
        )[0][-6:]
        np.testing.assert_allclose(scalar, batch)
        assert scalar[3] == pytest.approx(1.0), "vec_a=16 is the widest legal fp8 A"
        assert scalar[4] == pytest.approx(1.0), "vec_b=16 is the widest legal fp8 B"
        assert scalar[5] == pytest.approx(1.0), "vec_c=8 is the widest legal fp8 C"

    def test_b_operand_does_not_use_the_c_cap(self):
        fe = GemmUniversalVecFeatureEngine()
        v = fe.extract(_problem(m=1792, n=1536, k=16, dtype="fp8"), _kernel((8, 8, 8)))
        assert v[-3] == pytest.approx(0.5), "vec_a=8 is half the legal 16"
        assert v[-2] == pytest.approx(0.5), "vec_b=8 is half the legal 16"
        assert v[-1] == pytest.approx(1.0), "vec_c=8 is all of the legal 8"

    def test_bf16_c_width_is_unchanged(self):
        from feature_engine_vec import _out_dtype

        assert _out_dtype("bf16") == "bf16"


class TestRequiredVectorColumns:
    @pytest.mark.parametrize("col", ["vec_a", "vec_b", "vec_c"])
    def test_missing_vec_column_raises(self, col):
        fe = GemmUniversalVecFeatureEngine()
        with pytest.raises(KeyError, match="needs the fixed vector widths"):
            fe.extract_batch(_frame().drop(columns=[col]))


class TestLayoutValidation:
    def test_scalar_rejects_unknown_layout(self):
        fe = GemmUniversalVecFeatureEngine()
        with pytest.raises(ValueError, match="layout"):
            fe.extract(_problem(layout="zzz"), _kernel((4, 4, 8)))

    def test_batch_rejects_unknown_layout(self):
        fe = GemmUniversalVecFeatureEngine()
        df = _frame()
        df["layout"] = "zzz"
        with pytest.raises(ValueError, match="layout"):
            fe.extract_batch(df)

    def test_uppercase_layout_is_rejected_not_mis_resolved(self):
        fe = GemmUniversalVecFeatureEngine()
        with pytest.raises(ValueError, match="layout"):
            fe.extract(_problem(layout="RCR"), _kernel((4, 4, 8)))


class TestExtentValidation:
    def test_zero_extent_raises_scalar(self):
        fe = GemmUniversalVecFeatureEngine()
        with pytest.raises(ValueError, match=">= 1"):
            fe.extract(_problem(k=0), _kernel((4, 4, 8)))

    def test_zero_extent_raises_batch(self):
        fe = GemmUniversalVecFeatureEngine()
        with pytest.raises(ValueError, match=">= 1"):
            fe.extract_batch(_frame(k=0))


#: key -> (probe unlike the fixture, probe unlike the base engine's default,
#: dict the base engine reads the key from).
SCALAR_KEYS = {
    "layout": ("ccr", "ccr", "problem"),
    "dtype": ("fp32", "fp32", "problem"),
    "arch": ("gfx1250", None, None),
    "epilogue": ("cshuffle", "default", "kernel"),
}


class TestScalarKeysAreResolvedNotDefaulted:
    @pytest.mark.parametrize("key", sorted(SCALAR_KEYS))
    def test_absent_from_both_dicts_raises(self, key):
        fe = GemmUniversalVecFeatureEngine()
        problem = {k: v for k, v in _problem().items() if k != key}
        kernel = {k: v for k, v in _kernel().items() if k != key}
        with pytest.raises(KeyError, match="required to compute vector-width"):
            fe.extract(problem, kernel)

    @pytest.mark.parametrize("key", sorted(SCALAR_KEYS))
    def test_the_alternate_value_actually_moves_the_vector(self, key):
        fe = GemmUniversalVecFeatureEngine()
        value = SCALAR_KEYS[key][0]
        baseline = fe.extract(_problem(), _kernel(**DISCRIMINATING))
        moved = fe.extract(
            {**_problem(), key: value},
            {**_kernel(**DISCRIMINATING), key: value},
        )
        assert np.isfinite(baseline).all() and np.isfinite(moved).all(), (
            "a non-finite feature makes `not allclose` pass for the wrong "
            "reason -- NaN compares unequal to everything"
        )
        assert not np.allclose(baseline, moved), (
            f"{key}={value!r} produced the fixtures' vector, so the desync "
            "tests that consume this probe prove nothing"
        )


class TestTheBaseHalfReadsOneDictOnly:
    @pytest.mark.parametrize(
        "key", [k for k, (_, d, _) in SCALAR_KEYS.items() if d is not None]
    )
    def test_the_wrong_side_desynchronises_the_two_halves(self, key):
        fe = GemmUniversalVecFeatureEngine()
        _, probe, base_side = SCALAR_KEYS[key]

        def spelled_on(side):
            problem = {k: v for k, v in _problem().items() if k != key}
            kernel = {k: v for k, v in _kernel().items() if k != key}
            (problem if side == "problem" else kernel)[key] = probe
            return fe.extract(problem, kernel)

        other = "kernel" if base_side == "problem" else "problem"
        assert not np.allclose(spelled_on(base_side), spelled_on(other)), (
            f"{key}={probe!r} now gives the same vector from either dict. If "
            "the base engine was unified to search both, delete this test; "
            "otherwise the two halves are agreeing by accident."
        )

    def test_arch_is_genuinely_either_side(self):
        fe = GemmUniversalVecFeatureEngine()
        on_problem = fe.extract(
            {**_problem(), "arch": "gfx1250"}, _kernel(**DISCRIMINATING)
        )
        on_kernel = fe.extract(
            {k: v for k, v in _problem().items() if k != "arch"},
            {**_kernel(**DISCRIMINATING), "arch": "gfx1250"},
        )
        np.testing.assert_allclose(on_problem, on_kernel)


class TestLayoutSetTracksTheBaseEngine:
    def test_matches_the_base_engines_layout_map(self):
        from feature_engine import LAYOUT_MAP
        from feature_engine_vec import _LAYOUTS

        assert _LAYOUTS == frozenset(LAYOUT_MAP), (
            "_LAYOUTS, LAYOUT_MAP and parse_kernel_name must widen together"
        )

    def test_the_parser_accepts_exactly_those_layouts(self):
        from data_pipeline import parse_kernel_name
        from feature_engine_vec import _LAYOUTS

        tail = (
            "compv3_cshuffle_intrawave_False_False_False_True"
            "_128x128x128_2x2x1_32x32x16"
        )
        for layout in _LAYOUTS:
            assert parse_kernel_name(f"gemm_universal_bf16_{layout}_{tail}"), layout
        assert parse_kernel_name(f"gemm_universal_bf16_rcc_{tail}") == {}


class TestBaseBlockIsUnperturbed:
    def test_the_first_n_base_columns_equal_the_base_engines_output(self):
        df = _frame(m=1792, n=300, k=20226)
        base = GemmUniversalFeatureEngine().extract_batch(df)
        vec = GemmUniversalVecFeatureEngine().extract_batch(df)
        n = GemmUniversalVecFeatureEngine._N_BASE
        assert base.shape[1] == n
        np.testing.assert_allclose(vec[:, :n], base, rtol=0, atol=0)

    def test_the_scalar_path_agrees_too(self):
        p, kn = _problem(), _kernel((4, 4, 8))
        n = GemmUniversalVecFeatureEngine._N_BASE
        np.testing.assert_allclose(
            GemmUniversalVecFeatureEngine().extract(p, kn)[:n],
            GemmUniversalFeatureEngine().extract(p, kn),
            rtol=0,
            atol=0,
        )


class TestNativeWidthsAreKernelDerived:
    P = dict(m=1792, n=1536, k=7168, layout="rcr", dtype="bf16")

    def _native(self, **over):
        from codegen_common import gemm_native_vector_sizes

        g = {**DISCRIMINATING, **over}
        return gemm_native_vector_sizes(
            dtype_a=over.get("dtype", "bf16"),
            dtype_b=over.get("dtype", "bf16"),
            dtype_c=over.get("dtype_c", over.get("dtype", "bf16")),
            layout="rcr",
            tile=(g["tile_m"], g["tile_n"], g["tile_k"]),
            waves=(g["warp_m"], g["warp_n"], g["warp_k"]),
            warp_tile=(g["warp_tile_m"], g["warp_tile_n"], g["warp_tile_k"]),
            gfx_arch=over.get("arch", "gfx950"),
            epilogue=over.get("epilogue", "default"),
        )

    def test_the_geometry_actually_discriminates(self):
        from codegen_common import gemm_native_vector_sizes as nat

        g = DISCRIMINATING
        kw = dict(
            dtype_a="bf16",
            dtype_b="bf16",
            dtype_c="bf16",
            layout="rcr",
            gfx_arch="gfx950",
            epilogue="default",
        )
        right = nat(
            **kw,
            tile=(g["tile_m"], g["tile_n"], g["tile_k"]),
            waves=(2, 2, 1),
            warp_tile=(32, 32, 16),
        )
        swapped = nat(
            **kw,
            tile=(g["tile_m"], g["tile_n"], g["tile_k"]),
            waves=(32, 32, 16),
            warp_tile=(2, 2, 1),
        )
        assert right != swapped, "fixture no longer detects a waves/warp_tile swap"
        wave32 = nat(
            **{**kw, "gfx_arch": "gfx1250"},
            tile=(g["tile_m"], g["tile_n"], g["tile_k"]),
            waves=(2, 2, 1),
            warp_tile=(32, 32, 16),
        )
        assert right != wave32, "fixture no longer detects an arch change"

    def test_waves_and_warp_tile_are_not_transposed(self):
        fe = GemmUniversalVecFeatureEngine()
        legal = _legal(**self.P)
        got = fe.extract(_problem(**self.P), _kernel((0, 0, 0), **DISCRIMINATING))[-3:]
        exp = [nat / lim for nat, lim in zip(self._native(), legal)]
        np.testing.assert_allclose(got, exp)

    @pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp8", "int8"])
    def test_native_c_uses_the_output_dtype(self, dtype):
        from feature_engine_vec import _out_dtype

        fe = GemmUniversalVecFeatureEngine()
        P = {**self.P, "dtype": dtype}
        got = fe.extract(
            _problem(**P), _kernel((0, 0, 0), **DISCRIMINATING, epilogue="cshuffle")
        )[-3:]
        exp_nat = self._native(
            dtype=dtype, dtype_c=_out_dtype(dtype), epilogue="cshuffle"
        )
        exp = [nat / lim for nat, lim in zip(exp_nat, _legal(**P))]
        np.testing.assert_allclose(got, exp)

    @pytest.mark.parametrize("arch", ["gfx950", "gfx942", "gfx1250"])
    def test_the_rows_arch_selects_the_wavefront_size(self, arch):
        fe = GemmUniversalVecFeatureEngine()
        got = fe.extract(
            _problem(**self.P) | {"arch": arch},
            _kernel((0, 0, 0), **DISCRIMINATING),
        )[-3:]
        exp = [nat / lim for nat, lim in zip(self._native(arch=arch), _legal(**self.P))]
        np.testing.assert_allclose(got, exp)

    def test_a_wave32_arch_differs_from_a_wave64_one(self):
        fe = GemmUniversalVecFeatureEngine()
        k = _kernel((0, 0, 0), **DISCRIMINATING)
        a = fe.extract(_problem(**self.P) | {"arch": "gfx950"}, k)[-3:]
        b = fe.extract(_problem(**self.P) | {"arch": "gfx1250"}, k)[-3:]
        assert list(a) != list(b)


class TestHeterogeneousBatch:
    def _mixed(self):
        rows = [
            _row(m=1792, n=1536, k=7168, epilogue="default"),
            _row(m=1792, n=1536, k=7168, epilogue="cshuffle"),
            _row(m=1792, n=1536, k=7168, **DISCRIMINATING),
            _row(m=1792, n=1536, k=7168, **DISCRIMINATING, epilogue="cshuffle"),
            _row(m=1792, n=1536, k=7168, arch="gfx1250", **DISCRIMINATING),
            _row(m=512, n=300, k=20226, layout="ccr", vec=(4, 4, 8)),
        ]
        return pd.DataFrame(rows), rows

    def test_each_row_matches_its_own_scalar_extraction(self):
        fe = GemmUniversalVecFeatureEngine()
        df, rows = self._mixed()
        batch = fe.extract_batch(df)
        for i, row in enumerate(rows):
            np.testing.assert_allclose(
                batch[i],
                fe.extract(row, row),
                rtol=0,
                atol=1e-12,
                err_msg=f"row {i} disagrees with the scalar path",
            )

    def test_the_rows_are_not_all_the_same(self):
        fe = GemmUniversalVecFeatureEngine()
        df, _ = self._mixed()
        fracs = {tuple(r) for r in fe.extract_batch(df)[:, -3:]}
        assert len(fracs) >= 3, (
            f"frame is too homogeneous to detect a regroup bug: {fracs}"
        )


class TestRequiredColumns:
    EXPECTED = (
        "m",
        "n",
        "k",
        "layout",
        "dtype",
        "arch",
        "epilogue",
        "vec_a",
        "vec_b",
        "vec_c",
        "tile_m",
        "tile_n",
        "tile_k",
        "warp_m",
        "warp_n",
        "warp_k",
        "warp_tile_m",
        "warp_tile_n",
        "warp_tile_k",
    )

    def test_the_constant_lists_exactly_these(self):
        assert set(GemmUniversalVecFeatureEngine._REQUIRED) == set(self.EXPECTED)

    @pytest.mark.parametrize("col", EXPECTED)
    def test_dropping_any_required_column_raises_naming_it(self, col):
        fe = GemmUniversalVecFeatureEngine()
        with pytest.raises(KeyError, match=col):
            fe.extract_batch(_frame().drop(columns=[col]))


class TestInputValidation:
    REAL_ARCHES = (
        "gfx908",
        "gfx90a",
        "gfx942",
        "gfx950",
        "gfx1030",
        "gfx1100",
        "gfx1201",
        "gfx1250",
        "gfx950:xnack-",
    )

    @pytest.mark.parametrize("arch", REAL_ARCHES)
    def test_every_supported_arch_is_accepted(self, arch):
        fe = GemmUniversalVecFeatureEngine()
        assert (
            fe.extract(_problem() | {"arch": arch}, _kernel((0, 0, 0))).shape[0] == 78
        )

    @pytest.mark.parametrize(
        "arch",
        [
            "unknown",
            "nan",
            "",
            "GFX950",
            "gfx",
            "gfx8",
            "gfx9zzz",
            "gfx12345",
            "gfx1399",
            "gfx1500",
            "gfx9999",
            "gfx950a",
            "gfx90",
            "gfx99",
        ],
    )
    def test_an_unrecognised_arch_raises(self, arch):
        fe = GemmUniversalVecFeatureEngine()
        with pytest.raises(ValueError, match="architecture"):
            fe.extract(_problem() | {"arch": arch}, _kernel((0, 0, 0)))

    @pytest.mark.parametrize("epi", ["cshufle", "comp_tdm", "", "gemm"])
    def test_an_unmodelled_epilogue_raises(self, epi):
        fe = GemmUniversalVecFeatureEngine()
        with pytest.raises(ValueError, match="epilogue"):
            fe.extract(_problem(), _kernel((0, 0, 0), epilogue=epi))

    def test_the_batch_path_validates_too(self):
        fe = GemmUniversalVecFeatureEngine()
        with pytest.raises(ValueError, match="architecture"):
            fe.extract_batch(_frame(arch="unknown"))


class TestGeometryPrecedence:
    #: Knobs that change the native width at DISCRIMINATING.
    MOVING = {"warp_m": 999, "tile_k": 4}

    def test_the_chosen_knobs_actually_move_the_result(self):
        fe = GemmUniversalVecFeatureEngine()
        p = _problem(m=1792, n=1536, k=7168)
        base = fe.extract(p, _kernel((0, 0, 0), **DISCRIMINATING))[-3:]
        moved = fe.extract(p, _kernel((0, 0, 0), **{**DISCRIMINATING, **self.MOVING}))[
            -3:
        ]
        assert list(base) != list(moved)

    def test_the_kernels_geometry_beats_the_problems(self):
        fe = GemmUniversalVecFeatureEngine()
        p = {**_problem(m=1792, n=1536, k=7168), **self.MOVING}
        k = _kernel((0, 0, 0), **DISCRIMINATING)
        np.testing.assert_allclose(
            fe.extract(p, k), fe.extract(_problem(m=1792, n=1536, k=7168), k)
        )

    def test_a_missing_geometry_knob_raises_rather_than_defaulting(self):
        fe = GemmUniversalVecFeatureEngine()
        k = {kk: v for kk, v in _kernel((0, 0, 0)).items() if kk != "warp_tile_n"}
        with pytest.raises(KeyError, match="warp_tile_n"):
            fe.extract(_problem(), k)


class TestKeyPrecedence:
    P = dict(m=1792, n=1536, k=7168)

    #: Extents whose legal widths differ from the fixture's.
    POISON = {"m": 4, "n": 4, "k": 4}

    def test_the_poison_actually_moves_the_result(self):
        fe = GemmUniversalVecFeatureEngine()
        k = _kernel((0, 0, 0), **DISCRIMINATING)
        honest = fe.extract(_problem(**self.P), k)[-3:]
        moved = fe.extract(_problem(**self.POISON), k)[-3:]
        assert list(honest) != list(moved)

    def test_the_problems_extents_win(self):
        fe = GemmUniversalVecFeatureEngine()
        k = _kernel((0, 0, 0), **DISCRIMINATING)
        honest = fe.extract(_problem(**self.P), k)
        poisoned = fe.extract(_problem(**self.P), {**k, **self.POISON})
        np.testing.assert_allclose(honest, poisoned)

    def test_the_problems_layout_and_dtype_win(self):
        fe = GemmUniversalVecFeatureEngine()
        k = _kernel((0, 0, 0), **DISCRIMINATING)
        honest = fe.extract(_problem(**self.P), k)
        poisoned = fe.extract(
            _problem(**self.P), {**k, "layout": "crr", "dtype": "fp8"}
        )
        np.testing.assert_allclose(honest, poisoned)

    def test_that_poison_would_otherwise_move_the_result(self):
        fe = GemmUniversalVecFeatureEngine()
        k = _kernel((0, 0, 0), **DISCRIMINATING)
        a = fe.extract(_problem(**self.P), k)[-3:]
        b = fe.extract(_problem(**self.P, layout="crr", dtype="fp8"), k)[-3:]
        assert list(a) != list(b)

    def test_the_kernels_epilogue_wins(self):
        fe = GemmUniversalVecFeatureEngine()
        k = _kernel((0, 0, 0), **DISCRIMINATING, epilogue="cshuffle")
        honest = fe.extract(_problem(**self.P), k)
        poisoned = fe.extract({**_problem(**self.P), "epilogue": "default"}, k)
        np.testing.assert_allclose(honest, poisoned)

    def test_a_kernel_side_arch_is_reached_when_the_problem_omits_it(self):
        fe = GemmUniversalVecFeatureEngine()
        problem = {k: v for k, v in _problem(**self.P).items() if k != "arch"}
        kernel = _kernel((0, 0, 0), **DISCRIMINATING)
        wave32 = fe.extract(problem, {**kernel, "arch": "gfx1250"})
        wave64 = fe.extract(problem, {**kernel, "arch": "gfx950"})
        assert not np.allclose(wave32, wave64), (
            "a kernel-side arch did not reach the features, so the kernel half "
            "of the search is untested"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
