#!/usr/bin/env python3

# Validation test set for BWD_DATA - 10 unseen shapes
# These are NOT in the training set and are sized to avoid GPU crashes
# Focus on realistic backward data gradient computation scenarios

import sys
from pathlib import Path

# Add dispatcher/python to path for grouped_conv_utils import
dispatcher_python = Path(__file__).resolve().parents[4] / "dispatcher" / "python"
sys.path.insert(0, str(dispatcher_python))

from grouped_conv_utils import GroupedConvProblem  # noqa: E402

VALIDATION_PROBLEMS_BWD_DATA = [
    # Small batch, moderate channels (typical validation/inference backprop)
    GroupedConvProblem(
        N=4,
        C=64,
        K=128,
        G=1,
        Hi=32,
        Wi=32,
        Y=3,
        X=3,
        stride_h=1,
        stride_w=1,
        dilation_h=1,
        dilation_w=1,
        pad_h=1,
        pad_w=1,
        direction="bwd_data",
    ),
    # 1x1 convolution (common in ResNet bottlenecks)
    GroupedConvProblem(
        N=8,
        C=256,
        K=64,
        G=1,
        Hi=14,
        Wi=14,
        Y=1,
        X=1,
        stride_h=1,
        stride_w=1,
        dilation_h=1,
        dilation_w=1,
        pad_h=0,
        pad_w=0,
        direction="bwd_data",
    ),
    # 3x3 stride 1 (common conv layer)
    GroupedConvProblem(
        N=16,
        C=128,
        K=128,
        G=1,
        Hi=28,
        Wi=28,
        Y=3,
        X=3,
        stride_h=1,
        stride_w=1,
        dilation_h=1,
        dilation_w=1,
        pad_h=1,
        pad_w=1,
        direction="bwd_data",
    ),
    # Small spatial, larger channels
    GroupedConvProblem(
        N=8,
        C=512,
        K=256,
        G=1,
        Hi=7,
        Wi=7,
        Y=3,
        X=3,
        stride_h=1,
        stride_w=1,
        dilation_h=1,
        dilation_w=1,
        pad_h=1,
        pad_w=1,
        direction="bwd_data",
    ),
    # Medium batch, medium channels
    GroupedConvProblem(
        N=32,
        C=64,
        K=64,
        G=1,
        Hi=56,
        Wi=56,
        Y=3,
        X=3,
        stride_h=1,
        stride_w=1,
        dilation_h=1,
        dilation_w=1,
        pad_h=1,
        pad_w=1,
        direction="bwd_data",
    ),
    # 1x1 downsampling
    GroupedConvProblem(
        N=16,
        C=512,
        K=256,
        G=1,
        Hi=14,
        Wi=14,
        Y=1,
        X=1,
        stride_h=1,
        stride_w=1,
        dilation_h=1,
        dilation_w=1,
        pad_h=0,
        pad_w=0,
        direction="bwd_data",
    ),
    # Larger spatial, smaller channels
    GroupedConvProblem(
        N=4,
        C=32,
        K=64,
        G=1,
        Hi=112,
        Wi=112,
        Y=3,
        X=3,
        stride_h=1,
        stride_w=1,
        dilation_h=1,
        dilation_w=1,
        pad_h=1,
        pad_w=1,
        direction="bwd_data",
    ),
    # Balanced problem
    GroupedConvProblem(
        N=8,
        C=128,
        K=256,
        G=1,
        Hi=32,
        Wi=32,
        Y=3,
        X=3,
        stride_h=1,
        stride_w=1,
        dilation_h=1,
        dilation_w=1,
        pad_h=1,
        pad_w=1,
        direction="bwd_data",
    ),
    # Small everything (quick test)
    GroupedConvProblem(
        N=2,
        C=64,
        K=64,
        G=1,
        Hi=28,
        Wi=28,
        Y=3,
        X=3,
        stride_h=1,
        stride_w=1,
        dilation_h=1,
        dilation_w=1,
        pad_h=1,
        pad_w=1,
        direction="bwd_data",
    ),
    # Moderate all dimensions
    GroupedConvProblem(
        N=16,
        C=256,
        K=128,
        G=1,
        Hi=14,
        Wi=14,
        Y=3,
        X=3,
        stride_h=1,
        stride_w=1,
        dilation_h=1,
        dilation_w=1,
        pad_h=1,
        pad_w=1,
        direction="bwd_data",
    ),
]

if __name__ == "__main__":
    print(
        f"Generated {len(VALIDATION_PROBLEMS_BWD_DATA)} validation problems for BWD_DATA"
    )
