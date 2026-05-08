#!/usr/bin/env python3
"""
Extended synthetic training set for BWD_DATA targeting validation gaps.

Based on validation analysis:
- Low efficiency on small spatial + high channels (7x7, 14x14 with C/K >= 256)
- Low efficiency on moderate spatial + moderate channels (28x28, 32x32)
- Good efficiency on large spatial + small channels (already covered)
- CRITICAL: Add stride-2 with 3x3 filter (missing common downsampling pattern)
- CRITICAL: Add dilation support (zero training data exists)
- CRITICAL: Add 3D convolution support (infrastructure ready, zero data)

This set focuses on ~1500+ carefully selected problems covering weak areas + dilation + 3D.
"""

import sys
from pathlib import Path

# Add dispatcher/python to path for grouped_conv_utils import
dispatcher_python = Path(__file__).resolve().parents[4] / "dispatcher" / "python"
sys.path.insert(0, str(dispatcher_python))

from grouped_conv_utils import GroupedConvProblem  # noqa: E402

TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC = []

# 1. CRITICAL: Small spatial (7x7, 14x14) + High channels (256-2048)
# This addresses validation failures like N=8 C=512 K=256 7x7 (38% efficiency)
for Hi in [7, 14]:
    for C in [256, 512, 1024]:
        for K in [64, 128, 256, 512, 1024]:
            # Skip if both are too large
            if C >= 1024 and K >= 1024:
                continue

            for N in [1, 4, 8, 16, 32]:
                # 1x1 bottleneck
                TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC.append(
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
                        direction="bwd_data",
                    )
                )

                # 3x3 standard conv
                TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC.append(
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
                        direction="bwd_data",
                    )
                )

# 2. Medium spatial (28x28, 32x32, 56x56) + Medium channels (64-512)
# Addresses validation gaps like N=4 C=64 K=128 32x32 (56% efficiency)
for Hi in [28, 32, 56]:
    for C in [64, 128, 256, 512]:
        for K in [64, 128, 256, 512]:
            for N in [2, 4, 8, 16, 32]:
                # 1x1 projection
                TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC.append(
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
                        direction="bwd_data",
                    )
                )

                # 3x3 conv
                TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC.append(
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
                        direction="bwd_data",
                    )
                )

# 3. Large spatial (112x112) + Small/Medium channels (32-256)
# Early conv layers in networks
for Hi in [112]:
    for C in [32, 64, 128, 256]:
        for K in [64, 128, 256]:
            for N in [1, 2, 4, 8]:
                # 3x3 conv
                TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC.append(
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
                        direction="bwd_data",
                    )
                )

                # 7x7 stride 2 (ResNet first layer style)
                if C <= 128:
                    TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC.append(
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
                            direction="bwd_data",
                        )
                    )

# 4. Asymmetric C/K combinations (common in architecture transitions)
for Hi in [14, 28, 56]:
    for C, K in [(64, 256), (128, 512), (256, 64), (256, 128), (512, 256)]:
        for N in [4, 8, 16]:
            # 1x1 for channel change
            TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC.append(
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
                    direction="bwd_data",
                )
            )

            # 3x3 conv
            TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC.append(
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
                    direction="bwd_data",
                )
            )

# 5. Very small batch (inference/validation scenarios)
for N in [1, 2]:
    for Hi in [7, 14, 28, 56]:
        for C, K in [(64, 128), (128, 256), (256, 512), (512, 1024)]:
            # 1x1 conv
            TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC.append(
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
                    direction="bwd_data",
                )
            )

# 6. Large batch (distributed training)
for N in [64, 128]:
    for Hi in [14, 28]:
        for C, K in [(64, 64), (128, 128), (256, 256)]:
            # 3x3 conv
            TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC.append(
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
                    direction="bwd_data",
                )
            )

# 7. Grouped convolutions (G > 1) - Depthwise-separable and group convs
for G in [2, 4, 8]:
    for Hi in [14, 28, 56]:
        # Ensure C and K are divisible by G
        for base_c in [64, 128, 256]:
            C = base_c * G  # Total channels
            K = base_c * G  # Total output channels
            for N in [1, 4, 8, 16]:
                # 3x3 grouped conv
                TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC.append(
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
                        direction="bwd_data",
                    )
                )

                # 1x1 grouped conv
                TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC.append(
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
                        direction="bwd_data",
                    )
                )

# 8. Depthwise convolution (G = C = K) - MobileNet style
for Hi in [14, 28, 56, 112]:
    for C in [64, 128, 256, 512]:
        for N in [1, 4, 8]:
            TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC.append(
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
                    direction="bwd_data",
                )
            )

# 9. CRITICAL: Stride-2 with 3x3 filter (most common downsampling in ResNet backward)
# This combination is currently MISSING from training data
for Hi in [28, 56, 112]:
    for C, K in [(64, 128), (128, 256), (256, 512), (128, 128), (256, 256)]:
        for N in [1, 4, 8, 16]:
            # 3x3 stride 2 backward data
            TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC.append(
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
                    direction="bwd_data",
                )
            )

# 10. DILATED CONVOLUTIONS - Critical for semantic segmentation backward pass
# Common dilations: 2, 4, 6 with 3x3 filters (DeepLab, PSPNet)
for dilation in [2, 4, 6]:
    for Hi in [14, 28, 56]:
        for C, K in [(64, 128), (128, 256), (256, 512), (128, 128), (256, 256)]:
            for N in [1, 4, 8, 16]:
                # 3x3 dilated conv backward data
                pad = dilation * (3 - 1) // 2
                TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC.append(
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
                        direction="bwd_data",
                    )
                )

# 11. 3D CONVOLUTIONS - For video and medical imaging backward pass
# Common 3D patterns: small depth (8-32) with moderate spatial (28-56)
for Di in [8, 16, 32]:
    for Hi in [28, 56]:
        for C, K in [(64, 128), (128, 256), (128, 128)]:
            for N in [1, 2, 4, 8]:
                # 3x3x3 3D conv backward data
                TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC.append(
                    GroupedConvProblem(
                        N=N,
                        C=C,
                        K=K,
                        G=1,
                        Di=Di,
                        Hi=Hi,
                        Wi=Hi,
                        Z=3,
                        Y=3,
                        X=3,
                        stride_d=1,
                        stride_h=1,
                        stride_w=1,
                        pad_d=1,
                        pad_h=1,
                        pad_w=1,
                        direction="bwd_data",
                    )
                )

                # 1x1x1 3D pointwise backward data
                TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC.append(
                    GroupedConvProblem(
                        N=N,
                        C=C,
                        K=K,
                        G=1,
                        Di=Di,
                        Hi=Hi,
                        Wi=Hi,
                        Z=1,
                        Y=1,
                        X=1,
                        stride_d=1,
                        stride_h=1,
                        stride_w=1,
                        pad_d=0,
                        pad_h=0,
                        pad_w=0,
                        direction="bwd_data",
                    )
                )

# 12. 3D temporal convolutions with stride (video downsampling backward)
for Di in [16, 32]:
    for Hi in [28, 56]:
        for C, K in [(64, 128), (128, 256)]:
            for N in [1, 2, 4]:
                # 3x3x3 with stride 2 in temporal dimension
                TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC.append(
                    GroupedConvProblem(
                        N=N,
                        C=C,
                        K=K,
                        G=1,
                        Di=Di,
                        Hi=Hi,
                        Wi=Hi,
                        Z=3,
                        Y=3,
                        X=3,
                        stride_d=2,
                        stride_h=1,
                        stride_w=1,
                        pad_d=1,
                        pad_h=1,
                        pad_w=1,
                        direction="bwd_data",
                    )
                )

if __name__ == "__main__":
    # Count 2D vs 3D problems
    num_2d = sum(1 for p in TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC if not p.is_3d)
    num_3d = sum(1 for p in TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC if p.is_3d)
    num_dilated = sum(
        1 for p in TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC if p.dilation_h > 1 or p.dilation_w > 1
    )
    num_stride2_3x3 = sum(
        1
        for p in TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC
        if p.Y == 3 and p.X == 3 and p.stride_h == 2 and p.stride_w == 2 and not p.is_3d
    )

    print(
        f"Generated {len(TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC)} extended synthetic training problems for BWD_DATA"
    )
    print(f"  2D problems: {num_2d}")
    print(f"  3D problems: {num_3d}")
    print(f"  Dilated problems: {num_dilated}")
    print(f"  Stride-2 3x3 problems: {num_stride2_3x3}")
    print()
    print("Coverage:")
    print("  Batch sizes: 1-128")
    print("  Channels: 32-2048")
    print("  Groups: 1, 2, 4, 8, depthwise")
    print("  Spatial 2D: 7x7 to 112x112")
    print("  Spatial 3D: depth 8-32, HW 28-56")
    print("  Filters: 1x1, 3x3, 7x7 (2D), 1x1x1, 3x3x3 (3D)")
    print("  Strides: 1, 2")
    print("  Dilations: 1 (standard), 2, 4, 6 (atrous)")
    print()
    print("NEW in this version:")
    print("  ✓ Stride-2 with 3x3 filter (critical missing pattern)")
    print("  ✓ Dilated convolutions (dilation=2,4,6)")
    print("  ✓ 3D convolution support")
