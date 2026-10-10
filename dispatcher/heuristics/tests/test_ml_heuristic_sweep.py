"""Tests for the candidate dicts ml_heuristic_sweep hands the predictor."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_heuristic_sweep import KERNEL_POOL, spec_to_feature_dict  # noqa: E402


@pytest.fixture
def feat():
    return spec_to_feature_dict(KERNEL_POOL[0], "bf16", "rcr", "gfx950")


class TestSpecToFeatureDict:
    def test_the_arch_is_threaded_from_the_argument(self):
        for arch in ("gfx950", "gfx942", "gfx1250"):
            assert (
                spec_to_feature_dict(KERNEL_POOL[0], "bf16", "rcr", arch)["arch"]
                == arch
            )

    def test_dtype_and_layout_are_threaded_too(self):
        f = spec_to_feature_dict(KERNEL_POOL[0], "fp8", "ccr", "gfx950")
        assert f["dtype"] == "fp8" and f["layout"] == "ccr"

    def test_the_pool_is_declared_native(self, feat):
        assert (feat["vec_a"], feat["vec_b"], feat["vec_c"]) == (0, 0, 0)

    def test_the_vec_engine_accepts_the_dict(self):
        from feature_engine_vec import GemmUniversalVecFeatureEngine

        problem = {
            "m": 1792,
            "n": 1536,
            "k": 7168,
            "dtype": "bf16",
            "layout": "rcr",
            "split_k": 1,
        }
        fe = GemmUniversalVecFeatureEngine()
        for spec in KERNEL_POOL[:5]:
            kc = spec_to_feature_dict(spec, "bf16", "rcr", "gfx950")
            assert fe.extract(problem, kc).shape[0] == len(fe.get_feature_names())

    def test_the_base_engine_accepts_it_too(self):
        from feature_engine import GemmUniversalFeatureEngine

        problem = {
            "m": 1792,
            "n": 1536,
            "k": 7168,
            "dtype": "bf16",
            "layout": "rcr",
            "split_k": 1,
        }
        fe = GemmUniversalFeatureEngine()
        kc = spec_to_feature_dict(KERNEL_POOL[0], "bf16", "rcr", "gfx950")
        assert fe.extract(problem, kc).shape[0] == len(fe.get_feature_names())

    def test_the_wave_warp_inversion_is_preserved(self, feat):
        spec = KERNEL_POOL[0]
        assert (feat["warp_m"], feat["warp_n"], feat["warp_k"]) == (
            spec.wave_m,
            spec.wave_n,
            spec.wave_k,
        )
        assert (feat["warp_tile_m"], feat["warp_tile_n"], feat["warp_tile_k"]) == (
            spec.warp_m,
            spec.warp_n,
            spec.warp_k,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
