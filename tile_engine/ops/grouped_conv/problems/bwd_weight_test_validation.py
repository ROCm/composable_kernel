#!/usr/bin/env python3
"""
Validation test set for BWD_WEIGHT - 10 unseen problems for testing ML model performance.

These problems are NEVER used in training and represent diverse real-world scenarios.
"""

import sys
from pathlib import Path

# Add dispatcher/python to path for grouped_conv_utils import
dispatcher_python = Path(__file__).resolve().parents[4] / "dispatcher" / "python"
sys.path.insert(0, str(dispatcher_python))

from grouped_conv_utils import GroupedConvProblem  # noqa: E402

VALIDATION_PROBLEMS_BWD_WEIGHT = [
    # 1. Small spatial + high channels (critical for validation)
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
        pad_h=1,
        pad_w=1,
        direction="bwd_weight",
    ),
    # 2. Small batch + small spatial
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
        pad_h=1,
        pad_w=1,
        direction="bwd_weight",
    ),
    # 3. Medium spatial + medium channels (common validation gap)
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
        pad_h=1,
        pad_w=1,
        direction="bwd_weight",
    ),
    # 4. Large batch + medium spatial
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
        pad_h=1,
        pad_w=1,
        direction="bwd_weight",
    ),
    # 5. Small spatial + 1x1 bottleneck
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
        pad_h=0,
        pad_w=0,
        direction="bwd_weight",
    ),
    # 6. Medium batch + high channels
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
        pad_h=0,
        pad_w=0,
        direction="bwd_weight",
    ),
    # 7. Large spatial + small channels (early layers)
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
        pad_h=1,
        pad_w=1,
        direction="bwd_weight",
    ),
    # 8. Medium spatial + asymmetric channels
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
        pad_h=1,
        pad_w=1,
        direction="bwd_weight",
    ),
    # 9. Medium batch + medium everything
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
        pad_h=1,
        pad_w=1,
        direction="bwd_weight",
    ),
    # 10. High channels + small spatial
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
        pad_h=1,
        pad_w=1,
        direction="bwd_weight",
    ),
]

if __name__ == "__main__":
    print(
        f"Generated {len(VALIDATION_PROBLEMS_BWD_WEIGHT)} validation problems for BWD_WEIGHT"
    )
