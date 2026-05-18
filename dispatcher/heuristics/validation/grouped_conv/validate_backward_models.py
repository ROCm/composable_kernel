#!/usr/bin/env python3
"""
Validate backward pass ML models using actual training problem shapes.

Tests prediction quality on representative problems from the training set.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # heuristics

from predict import Predictor
from feature_engine_grouped_conv import GroupedConvFeatureEngine

# Representative test problems from training sets

BWD_DATA_TEST_PROBLEMS = [
    # Small problems (from bwd_data_training.py)
    {'N': 32, 'C': 1, 'K': 1, 'G': 1, 'Hi': 5, 'Wi': 5, 'Y': 3, 'X': 3, 'stride_h': 1, 'stride_w': 1, 'pad_h': 0, 'pad_w': 0},
    {'N': 64, 'C': 1, 'K': 1, 'G': 1, 'Hi': 5, 'Wi': 5, 'Y': 3, 'X': 3, 'stride_h': 1, 'stride_w': 1, 'pad_h': 0, 'pad_w': 0},
    {'N': 128, 'C': 256, 'K': 128, 'G': 1, 'Hi': 32, 'Wi': 32, 'Y': 3, 'X': 3, 'stride_h': 1, 'stride_w': 1, 'pad_h': 1, 'pad_w': 1},
    {'N': 2, 'C': 128, 'K': 256, 'G': 1, 'Hi': 32, 'Wi': 32, 'Y': 3, 'X': 3, 'stride_h': 1, 'stride_w': 1, 'pad_h': 1, 'pad_w': 1},
    {'N': 2, 'C': 256, 'K': 256, 'G': 1, 'Hi': 14, 'Wi': 14, 'Y': 1, 'X': 1, 'stride_h': 1, 'stride_w': 1, 'pad_h': 0, 'pad_w': 0},
]

BWD_WEIGHT_TEST_PROBLEMS = [
    # Small problems (from bwd_weight_synthetic.py)
    {'N': 1, 'C': 64, 'K': 64, 'G': 1, 'Hi': 7, 'Wi': 7, 'Y': 1, 'X': 1, 'stride_h': 1, 'stride_w': 1, 'pad_h': 0, 'pad_w': 0},
    {'N': 2, 'C': 64, 'K': 128, 'G': 1, 'Hi': 14, 'Wi': 14, 'Y': 1, 'X': 1, 'stride_h': 1, 'stride_w': 1, 'pad_h': 0, 'pad_w': 0},
    {'N': 8, 'C': 128, 'K': 128, 'G': 1, 'Hi': 28, 'Wi': 28, 'Y': 3, 'X': 3, 'stride_h': 1, 'stride_w': 1, 'pad_h': 1, 'pad_w': 1},
    # Medium problems
    {'N': 16, 'C': 128, 'K': 256, 'G': 1, 'Hi': 14, 'Wi': 14, 'Y': 3, 'X': 3, 'stride_h': 1, 'stride_w': 1, 'pad_h': 1, 'pad_w': 1},
    {'N': 32, 'C': 256, 'K': 512, 'G': 1, 'Hi': 28, 'Wi': 28, 'Y': 3, 'X': 3, 'stride_h': 1, 'stride_w': 1, 'pad_h': 1, 'pad_w': 1},
    # Large problems
    {'N': 64, 'C': 512, 'K': 1024, 'G': 1, 'Hi': 14, 'Wi': 14, 'Y': 3, 'X': 3, 'stride_h': 2, 'stride_w': 2, 'pad_h': 1, 'pad_w': 1},
    {'N': 128, 'C': 1024, 'K': 2048, 'G': 1, 'Hi': 28, 'Wi': 28, 'Y': 5, 'X': 5, 'stride_h': 1, 'stride_w': 1, 'pad_h': 2, 'pad_w': 2},
]

# Backward kernel configurations (compv3, mem)
BACKWARD_KERNELS = [
    {'block_size': 16, 'gemm_m_per_block': 64, 'gemm_n_per_block': 64, 'pipeline': 'compv3'},
    {'block_size': 16, 'gemm_m_per_block': 64, 'gemm_n_per_block': 64, 'pipeline': 'mem'},
    {'block_size': 32, 'gemm_m_per_block': 64, 'gemm_n_per_block': 64, 'pipeline': 'compv3'},
    {'block_size': 32, 'gemm_m_per_block': 64, 'gemm_n_per_block': 64, 'pipeline': 'mem'},
    {'block_size': 32, 'gemm_m_per_block': 128, 'gemm_n_per_block': 64, 'pipeline': 'compv3'},
    {'block_size': 32, 'gemm_m_per_block': 128, 'gemm_n_per_block': 64, 'pipeline': 'mem'},
    {'block_size': 64, 'gemm_m_per_block': 64, 'gemm_n_per_block': 64, 'pipeline': 'compv3'},
    {'block_size': 64, 'gemm_m_per_block': 64, 'gemm_n_per_block': 64, 'pipeline': 'mem'},
    {'block_size': 64, 'gemm_m_per_block': 128, 'gemm_n_per_block': 64, 'pipeline': 'compv3'},
    {'block_size': 64, 'gemm_m_per_block': 128, 'gemm_n_per_block': 64, 'pipeline': 'mem'},
    {'block_size': 128, 'gemm_m_per_block': 128, 'gemm_n_per_block': 64, 'pipeline': 'compv3'},
    {'block_size': 128, 'gemm_m_per_block': 128, 'gemm_n_per_block': 64, 'pipeline': 'mem'},
]


def format_problem(p):
    """Format problem for display."""
    Ho = (p['Hi'] + 2*p['pad_h'] - p['Y']) // p['stride_h'] + 1
    Wo = (p['Wi'] + 2*p['pad_w'] - p['X']) // p['stride_w'] + 1
    return f"N={p['N']:3d} C={p['C']:4d} K={p['K']:4d} {p['Hi']:2d}x{p['Wi']:2d}→{Ho:2d}x{Wo:2d} f{p['Y']}x{p['X']}"


def validate_variant(variant, test_problems, model_dir):
    """Validate a specific variant (bwd_data or bwd_weight)."""
    print("=" * 100)
    print(f"  VALIDATING {variant.upper()} MODEL")
    print("=" * 100)
    print(f"  Model: {model_dir}")
    print(f"  Problems: {len(test_problems)}")
    print()

    # Load model
    feature_engine = GroupedConvFeatureEngine()
    predictor = Predictor(model_dir, feature_engine=feature_engine)
    print("  ✓ Model loaded successfully")
    print()

    # Test each problem
    print(f"  {'Problem':<45} {'Best Kernel':<25} {'Pred TFLOPS':>12} {'Top-3 Kernels':<35}")
    print("  " + "-" * 117)

    all_predictions = []

    for problem in test_problems:
        # Add dtype
        problem_with_dtype = {**problem, 'dtype': 'bf16'}

        # Predict for all kernels
        predictions = []
        for kernel in BACKWARD_KERNELS:
            tflops = predictor.predict_tflops(problem_with_dtype, kernel)
            predictions.append({
                'tflops': tflops,
                'kernel': f"{kernel['block_size']}x{kernel['gemm_m_per_block']}x{kernel['gemm_n_per_block']}_{kernel['pipeline']}",
                'pipeline': kernel['pipeline']
            })

        # Sort by TFLOPS
        predictions.sort(key=lambda x: x['tflops'], reverse=True)
        all_predictions.append(predictions)

        # Format output
        prob_str = format_problem(problem)
        best = predictions[0]
        top3_str = f"{predictions[0]['kernel'][:18]}, {predictions[1]['kernel'][:18]}, {predictions[2]['kernel'][:18]}"

        print(f"  {prob_str:<45} {best['kernel']:<25} {best['tflops']:>12.2f} {top3_str:<35}")

    print()
    print("  " + "=" * 117)

    # Summary statistics
    print()
    print("  SUMMARY STATISTICS:")
    print(f"  {'Metric':<30} {'Value':>15}")
    print("  " + "-" * 47)

    # Average predicted TFLOPS
    avg_best_tflops = sum(p[0]['tflops'] for p in all_predictions) / len(all_predictions)
    print(f"  {'Avg Best Predicted TFLOPS':<30} {avg_best_tflops:>15.2f}")

    # Min/max predicted TFLOPS
    min_tflops = min(p[0]['tflops'] for p in all_predictions)
    max_tflops = max(p[0]['tflops'] for p in all_predictions)
    print(f"  {'Min Predicted TFLOPS':<30} {min_tflops:>15.2f}")
    print(f"  {'Max Predicted TFLOPS':<30} {max_tflops:>15.2f}")

    # Pipeline preference (how often each pipeline is selected)
    compv3_count = sum(1 for p in all_predictions if p[0]['pipeline'] == 'compv3')
    mem_count = sum(1 for p in all_predictions if p[0]['pipeline'] == 'mem')
    print(f"  {'Best pipeline: compv3':<30} {compv3_count:>15} ({100*compv3_count/len(all_predictions):.1f}%)")
    print(f"  {'Best pipeline: mem':<30} {mem_count:>15} ({100*mem_count/len(all_predictions):.1f}%)")

    # Top-3 accuracy approximation (how often best kernel is significantly better than 2nd/3rd)
    gaps = []
    for preds in all_predictions:
        gap = (preds[0]['tflops'] - preds[2]['tflops']) / preds[0]['tflops'] * 100
        gaps.append(gap)
    avg_gap = sum(gaps) / len(gaps)
    print(f"  {'Avg gap: best vs 3rd (%)':<30} {avg_gap:>15.1f}%")

    print()


def main():
    print()
    print("=" * 100)
    print("  BACKWARD PASS ML MODEL VALIDATION")
    print("  Testing predictions on training problem shapes")
    print("=" * 100)
    print()

    # Model directory is in heuristics/models/, not validation/grouped_conv/models/
    heuristics_dir = Path(__file__).parent.parent.parent  # Go up from validation/grouped_conv/ to heuristics/

    # Validate bwd_data
    bwd_data_model = heuristics_dir / "models" / "grouped_conv_bwd_data_bf16_gfx950"
    if bwd_data_model.exists():
        validate_variant("bwd_data", BWD_DATA_TEST_PROBLEMS, bwd_data_model)
    else:
        print(f"  ⚠ BWD_DATA model not found: {bwd_data_model}")

    print()

    # Validate bwd_weight
    bwd_weight_model = heuristics_dir / "models" / "grouped_conv_bwd_weight_bf16_gfx950"
    if bwd_weight_model.exists():
        validate_variant("bwd_weight", BWD_WEIGHT_TEST_PROBLEMS, bwd_weight_model)
    else:
        print(f"  ⚠ BWD_WEIGHT model not found: {bwd_weight_model}")

    print()
    print("=" * 100)
    print("  VALIDATION COMPLETE")
    print("=" * 100)
    print()


if __name__ == "__main__":
    main()
