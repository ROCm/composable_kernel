#!/usr/bin/env python3
"""Test that compressed models can be loaded and used."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predict import Predictor


def test_fp16_model_decompression():
    """Test that fp16 model is auto-decompressed and usable."""
    model_dir = Path(__file__).parent.parent / "models" / "gemm_universal_fp16_gfx950"

    # Ensure .lgbm.gz exists
    gz_file = model_dir / "model_tflops.lgbm.gz"

    assert gz_file.exists(), f"Compressed model not found: {gz_file}"

    # Load predictor - should auto-decompress
    predictor = Predictor(model_dir)

    # Test prediction
    problem = {"m": 128, "n": 1536, "k": 7168, "dtype": "fp16", "layout": "rcr"}
    kernel_config = {
        "tile_shape": {"m0": 128, "n0": 128, "k0": 16},
        "wave_shape": {"m1": 2, "n1": 2, "k1": 1},
        "warp_tile": {"m2": 32, "n2": 32, "k2": 8},
    }

    tflops = predictor.predict_tflops(problem, kernel_config)

    assert isinstance(tflops, float), f"Expected float, got {type(tflops)}"
    assert tflops > 0, f"Expected positive TFLOPS, got {tflops}"

    # Verify decompressed file was created
    lgbm_file = model_dir / "model_tflops.lgbm"
    assert lgbm_file.exists(), "Model should have been decompressed"

    print(f"✅ FP16 model decompression test passed")
    print(f"   Predicted TFLOPS: {tflops:.2f}")
    print(f"   Decompressed to: {lgbm_file}")
    return True


def test_fp8_model_decompression():
    """Test that fp8 model is auto-decompressed and usable."""
    model_dir = Path(__file__).parent.parent / "models" / "gemm_universal_fp8_gfx950"

    # Ensure .lgbm.gz exists
    gz_file = model_dir / "model_tflops.lgbm.gz"

    assert gz_file.exists(), f"Compressed model not found: {gz_file}"

    # Load predictor - should auto-decompress
    predictor = Predictor(model_dir)

    # Test prediction
    problem = {"m": 2048, "n": 2048, "k": 2048, "dtype": "fp8", "layout": "rcr"}
    kernel_config = {
        "tile_shape": {"m0": 256, "n0": 256, "k0": 64},
        "wave_shape": {"m1": 2, "n1": 2, "k1": 1},
        "warp_tile": {"m2": 32, "n2": 32, "k2": 16},
    }

    tflops = predictor.predict_tflops(problem, kernel_config)

    assert isinstance(tflops, float), f"Expected float, got {type(tflops)}"
    assert tflops > 0, f"Expected positive TFLOPS, got {tflops}"

    # Verify decompressed file was created
    lgbm_file = model_dir / "model_tflops.lgbm"
    assert lgbm_file.exists(), "Model should have been decompressed"

    print(f"✅ FP8 model decompression test passed")
    print(f"   Predicted TFLOPS: {tflops:.2f}")
    print(f"   Decompressed to: {lgbm_file}")
    return True


if __name__ == "__main__":
    print("Testing compressed model auto-decompression...")
    print()

    test_fp16_model_decompression()
    print()
    test_fp8_model_decompression()
    print()
    print("✅ All model compression tests passed!")
