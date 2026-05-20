#!/usr/bin/env python3
"""
Extended synthetic training set for BWD_WEIGHT targeting validation gaps.

Based on validation analysis:
- Current model: 96.5% mean efficiency, 90.1% P10, 20% top-1 accuracy
- Needs better coverage for diverse problem sizes and channel combinations
- CRITICAL: Add dilation support (zero training data exists)
- Already has groups and stride-2 coverage

This set focuses on ~2000+ carefully selected problems covering weak areas + dilation.
"""

import sys
from pathlib import Path

# Add dispatcher/python to path for grouped_conv_utils import
dispatcher_python = Path(__file__).resolve().parents[4] / "dispatcher" / "python"
sys.path.insert(0, str(dispatcher_python))

from grouped_conv_utils import GroupedConvProblem  # noqa: E402

TRAINING_PROBLEMS_BWD_WEIGHT_SYNTHETIC = []

# 1. CRITICAL: Small spatial (7x7, 14x14) + Various channels
# This addresses validation cases like N=8 C=512 K=256 7x7 (96% efficiency)
for Hi in [7, 14]:
    for C in [64, 128, 256, 512, 1024]:
        for K in [64, 128, 256, 512, 1024]:
            # Skip if both are too large
            if C >= 1024 and K >= 1024:
                continue

            for N in [1, 2, 4, 8, 16, 32]:
                # 1x1 bottleneck
                TRAINING_PROBLEMS_BWD_WEIGHT_SYNTHETIC.append(
                    GroupedConvProblem(
                        N=N,
                        C=C,
                        K=K,
                        G=1,
                        Hi=Hi,
                        Wi=Hi,
                        Y=1,
                        X=1,
                        stride_h=1,
                        stride_w=1,
                        pad_h=0,
                        pad_w=0,
                        direction="bwd_weight",
                    )
                )

                # 3x3 standard conv
                TRAINING_PROBLEMS_BWD_WEIGHT_SYNTHETIC.append(
                    GroupedConvProblem(
                        N=N,
                        C=C,
                        K=K,
                        G=1,
                        Hi=Hi,
                        Wi=Hi,
                        Y=3,
                        X=3,
                        stride_h=1,
                        stride_w=1,
                        pad_h=1,
                        pad_w=1,
                        direction="bwd_weight",
                    )
                )

# 2. Medium spatial (28x28, 32x32, 56x56) + Various channels
# Addresses cases like N=2 C=64 K=64 28x28 (90.1% efficiency)
for Hi in [28, 32, 56]:
    for C in [32, 64, 128, 256, 512]:
        for K in [64, 128, 256, 512]:
            for N in [1, 2, 4, 8, 16, 32]:
                # 1x1 projection
                TRAINING_PROBLEMS_BWD_WEIGHT_SYNTHETIC.append(
                    GroupedConvProblem(
                        N=N,
                        C=C,
                        K=K,
                        G=1,
                        Hi=Hi,
                        Wi=Hi,
                        Y=1,
                        X=1,
                        stride_h=1,
                        stride_w=1,
                        pad_h=0,
                        pad_w=0,
                        direction="bwd_weight",
                    )
                )

                # 3x3 conv
                TRAINING_PROBLEMS_BWD_WEIGHT_SYNTHETIC.append(
                    GroupedConvProblem(
                        N=N,
                        C=C,
                        K=K,
                        G=1,
                        Hi=Hi,
                        Wi=Hi,
                        Y=3,
                        X=3,
                        stride_h=1,
                        stride_w=1,
                        pad_h=1,
                        pad_w=1,
                        direction="bwd_weight",
                    )
                )

# 3. Large spatial (112x112) + Small/Medium channels (early conv layers)
for Hi in [112]:
    for C in [16, 32, 64, 128, 256]:
        for K in [32, 64, 128, 256]:
            for N in [1, 2, 4, 8]:
                # 3x3 conv
                TRAINING_PROBLEMS_BWD_WEIGHT_SYNTHETIC.append(
                    GroupedConvProblem(
                        N=N,
                        C=C,
                        K=K,
                        G=1,
                        Hi=Hi,
                        Wi=Hi,
                        Y=3,
                        X=3,
                        stride_h=1,
                        stride_w=1,
                        pad_h=1,
                        pad_w=1,
                        direction="bwd_weight",
                    )
                )

                # 7x7 stride 2 (ResNet first layer style)
                if C <= 128:
                    TRAINING_PROBLEMS_BWD_WEIGHT_SYNTHETIC.append(
                        GroupedConvProblem(
                            N=N,
                            C=C,
                            K=K,
                            G=1,
                            Hi=Hi,
                            Wi=Hi,
                            Y=7,
                            X=7,
                            stride_h=2,
                            stride_w=2,
                            pad_h=3,
                            pad_w=3,
                            direction="bwd_weight",
                        )
                    )

# 4. Asymmetric C/K combinations (common in architecture transitions)
for Hi in [14, 28, 56]:
    for C, K in [(64, 256), (128, 512), (256, 64), (256, 128), (512, 256), (256, 1024)]:
        for N in [4, 8, 16, 32]:
            # 1x1 for channel change
            TRAINING_PROBLEMS_BWD_WEIGHT_SYNTHETIC.append(
                GroupedConvProblem(
                    N=N,
                    C=C,
                    K=K,
                    G=1,
                    Hi=Hi,
                    Wi=Hi,
                    Y=1,
                    X=1,
                    stride_h=1,
                    stride_w=1,
                    pad_h=0,
                    pad_w=0,
                    direction="bwd_weight",
                )
            )

            # 3x3 conv
            TRAINING_PROBLEMS_BWD_WEIGHT_SYNTHETIC.append(
                GroupedConvProblem(
                    N=N,
                    C=C,
                    K=K,
                    G=1,
                    Hi=Hi,
                    Wi=Hi,
                    Y=3,
                    X=3,
                    stride_h=1,
                    stride_w=1,
                    pad_h=1,
                    pad_w=1,
                    direction="bwd_weight",
                )
            )

# 5. Very small batch (inference/validation scenarios)
for N in [1, 2]:
    for Hi in [7, 14, 28, 56]:
        for C, K in [(64, 128), (128, 256), (256, 512), (512, 1024)]:
            # 1x1 conv
            TRAINING_PROBLEMS_BWD_WEIGHT_SYNTHETIC.append(
                GroupedConvProblem(
                    N=N,
                    C=C,
                    K=K,
                    G=1,
                    Hi=Hi,
                    Wi=Hi,
                    Y=1,
                    X=1,
                    stride_h=1,
                    stride_w=1,
                    pad_h=0,
                    pad_w=0,
                    direction="bwd_weight",
                )
            )

# 6. Large batch (distributed training)
for N in [64, 128]:
    for Hi in [7, 14, 28]:
        for C, K in [(64, 64), (128, 128), (256, 256), (512, 512)]:
            # 3x3 conv
            TRAINING_PROBLEMS_BWD_WEIGHT_SYNTHETIC.append(
                GroupedConvProblem(
                    N=N,
                    C=C,
                    K=K,
                    G=1,
                    Hi=Hi,
                    Wi=Hi,
                    Y=3,
                    X=3,
                    stride_h=1,
                    stride_w=1,
                    pad_h=1,
                    pad_w=1,
                    direction="bwd_weight",
                )
            )

            # 1x1 conv
            TRAINING_PROBLEMS_BWD_WEIGHT_SYNTHETIC.append(
                GroupedConvProblem(
                    N=N,
                    C=C,
                    K=K,
                    G=1,
                    Hi=Hi,
                    Wi=Hi,
                    Y=1,
                    X=1,
                    stride_h=1,
                    stride_w=1,
                    pad_h=0,
                    pad_w=0,
                    direction="bwd_weight",
                )
            )

# 7. Grouped convolutions (G > 1) - Group convs
for G in [2, 4, 8]:
    for Hi in [14, 28, 56]:
        # Ensure C and K are divisible by G
        for base_c in [64, 128, 256]:
            C = base_c * G  # Total channels
            K = base_c * G  # Total output channels
            for N in [1, 4, 8, 16]:
                # 3x3 grouped conv
                TRAINING_PROBLEMS_BWD_WEIGHT_SYNTHETIC.append(
                    GroupedConvProblem(
                        N=N,
                        C=C,
                        K=K,
                        G=G,
                        Hi=Hi,
                        Wi=Hi,
                        Y=3,
                        X=3,
                        stride_h=1,
                        stride_w=1,
                        pad_h=1,
                        pad_w=1,
                        direction="bwd_weight",
                    )
                )

                # 1x1 grouped conv
                TRAINING_PROBLEMS_BWD_WEIGHT_SYNTHETIC.append(
                    GroupedConvProblem(
                        N=N,
                        C=C,
                        K=K,
                        G=G,
                        Hi=Hi,
                        Wi=Hi,
                        Y=1,
                        X=1,
                        stride_h=1,
                        stride_w=1,
                        pad_h=0,
                        pad_w=0,
                        direction="bwd_weight",
                    )
                )

# 8. Depthwise convolution (G = C = K) - MobileNet style
for Hi in [14, 28, 56, 112]:
    for C in [64, 128, 256, 512]:
        for N in [1, 4, 8]:
            TRAINING_PROBLEMS_BWD_WEIGHT_SYNTHETIC.append(
                GroupedConvProblem(
                    N=N,
                    C=C,
                    K=C,
                    G=C,  # Depthwise: each channel is its own group
                    Hi=Hi,
                    Wi=Hi,
                    Y=3,
                    X=3,
                    stride_h=1,
                    stride_w=1,
                    pad_h=1,
                    pad_w=1,
                    direction="bwd_weight",
                )
            )

# 9. Stride-2 convolutions (common for downsampling)
for Hi in [14, 28, 56]:
    for C in [64, 128, 256]:
        for K in [128, 256, 512]:
            for N in [4, 8, 16]:
                TRAINING_PROBLEMS_BWD_WEIGHT_SYNTHETIC.append(
                    GroupedConvProblem(
                        N=N,
                        C=C,
                        K=K,
                        G=1,
                        Hi=Hi,
                        Wi=Hi,
                        Y=3,
                        X=3,
                        stride_h=2,
                        stride_w=2,
                        pad_h=1,
                        pad_w=1,
                        direction="bwd_weight",
                    )
                )

# 10. DILATED CONVOLUTIONS - Critical for semantic segmentation backward weight
# Common dilations: 2, 4, 6 with 3x3 filters (DeepLab, PSPNet)
for dilation in [2, 4, 6]:
    for Hi in [14, 28, 56]:
        for C, K in [(64, 128), (128, 256), (256, 512), (128, 128), (256, 256)]:
            for N in [1, 4, 8, 16]:
                # 3x3 dilated conv backward weight
                pad = dilation * (3 - 1) // 2
                TRAINING_PROBLEMS_BWD_WEIGHT_SYNTHETIC.append(
                    GroupedConvProblem(
                        N=N,
                        C=C,
                        K=K,
                        G=1,
                        Hi=Hi,
                        Wi=Hi,
                        Y=3,
                        X=3,
                        stride_h=1,
                        stride_w=1,
                        pad_h=pad,
                        pad_w=pad,
                        dilation_h=dilation,
                        dilation_w=dilation,
                        direction="bwd_weight",
                    )
                )

# 11. Additional dilated convolutions with different spatial sizes
for dilation in [2, 4]:
    for Hi in [7, 32, 112]:
        for C, K in [(64, 64), (128, 128), (256, 256)]:
            for N in [2, 8]:
                pad = dilation * (3 - 1) // 2
                TRAINING_PROBLEMS_BWD_WEIGHT_SYNTHETIC.append(
                    GroupedConvProblem(
                        N=N,
                        C=C,
                        K=K,
                        G=1,
                        Hi=Hi,
                        Wi=Hi,
                        Y=3,
                        X=3,
                        stride_h=1,
                        stride_w=1,
                        pad_h=pad,
                        pad_w=pad,
                        dilation_h=dilation,
                        dilation_w=dilation,
                        direction="bwd_weight",
                    )
                )

if __name__ == "__main__":
    num_dilated = sum(
        1 for p in TRAINING_PROBLEMS_BWD_WEIGHT_SYNTHETIC if p.dilation_h > 1 or p.dilation_w > 1
    )
    num_stride2_3x3 = sum(
        1
        for p in TRAINING_PROBLEMS_BWD_WEIGHT_SYNTHETIC
        if p.Y == 3 and p.X == 3 and p.stride_h == 2 and p.stride_w == 2
    )

    print(
        f"Generated {len(TRAINING_PROBLEMS_BWD_WEIGHT_SYNTHETIC)} extended synthetic training problems for BWD_WEIGHT"
    )
    print(f"  Dilated problems: {num_dilated}")
    print(f"  Stride-2 3x3 problems: {num_stride2_3x3}")
    print()
    print("Coverage:")
    print("  Batch sizes: 1-128")
    print("  Channels: 16-1024")
    print("  Groups: 1, 2, 4, 8, depthwise")
    print("  Spatial: 7x7 to 112x112")
    print("  Filters: 1x1, 3x3, 7x7")
    print("  Strides: 1, 2")
    print("  Dilations: 1 (standard), 2, 4, 6 (atrous)")
    print()
    print("NEW in this version:")
    print("  ✓ Dilated convolutions (dilation=2,4,6)")
