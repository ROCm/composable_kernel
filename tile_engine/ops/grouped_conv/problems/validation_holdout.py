"""
Validation holdout set for heuristic testing.
300 problems (250 2D + 50 3D) randomly sampled for validation.
"""

from grouped_conv_utils import GroupedConvProblem

VALIDATION_PROBLEMS = [
    GroupedConvProblem(
        N=4, C=256, K=256, G=4,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=32, C=1024, K=256, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=128, K=64, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=64, K=128, G=1,
        Di=1, Hi=112, Wi=112,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=2, stride_w=2,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=256, K=64, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=2, C=64, K=256, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=1024, K=128, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=64, K=1024, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=256, K=512, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=2, stride_w=2,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=32, C=1024, K=64, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=64, K=64, G=8,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=64, K=128, G=1,
        Di=1, Hi=112, Wi=112,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=2, stride_w=2,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=256, K=512, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=32, C=512, K=256, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=64, K=64, G=2,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=128, K=256, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=128, K=128, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=128, K=128, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=512, K=1024, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=1024, K=64, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=1024, K=512, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=256, K=64, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=32, K=32, G=2,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=128, K=64, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=128, C=128, K=128, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=512, K=128, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=64, K=64, G=64,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=128, K=128, G=4,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=512, K=512, G=8,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=32, C=128, K=128, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=1024, K=128, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=32, C=256, K=512, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=64, K=64, G=2,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=32, C=256, K=64, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=256, K=256, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=4, pad_w=4,
        dilation_d=1, dilation_h=4, dilation_w=4
    ),
    GroupedConvProblem(
        N=4, C=64, K=128, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=256, K=256, G=1,
        Di=1, Hi=14, Wi=14,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=2, pad_w=2,
        dilation_d=1, dilation_h=2, dilation_w=2
    ),
    GroupedConvProblem(
        N=1, C=128, K=128, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=4, pad_w=4,
        dilation_d=1, dilation_h=4, dilation_w=4
    ),
    GroupedConvProblem(
        N=16, C=128, K=512, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=256, K=256, G=8,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=128, K=256, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=128, K=128, G=1,
        Di=1, Hi=32, Wi=32,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=32, C=128, K=256, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=1024, K=64, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=128, K=256, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=2, pad_w=2,
        dilation_d=1, dilation_h=2, dilation_w=2
    ),
    GroupedConvProblem(
        N=32, C=128, K=64, G=1,
        Di=1, Hi=32, Wi=32,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=32, C=512, K=512, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=32, C=512, K=128, G=1,
        Di=1, Hi=32, Wi=32,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=32, C=512, K=1024, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=256, K=128, G=1,
        Di=1, Hi=32, Wi=32,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=128, K=256, G=1,
        Di=1, Hi=32, Wi=32,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=64, K=64, G=2,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=2, C=128, K=256, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=128, K=128, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=4, pad_w=4,
        dilation_d=1, dilation_h=4, dilation_w=4
    ),
    GroupedConvProblem(
        N=32, C=128, K=512, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=64, K=128, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=2, stride_w=2,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=2, C=512, K=256, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=256, K=128, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=64, K=128, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=512, K=128, G=1,
        Di=1, Hi=32, Wi=32,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=512, K=128, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=2, C=64, K=128, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=2, C=512, K=64, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=64, K=128, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=6, pad_w=6,
        dilation_d=1, dilation_h=6, dilation_w=6
    ),
    GroupedConvProblem(
        N=8, C=32, K=32, G=4,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=128, K=128, G=4,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=2, C=256, K=256, G=1,
        Di=1, Hi=32, Wi=32,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=128, K=128, G=2,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=512, K=64, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=64, K=64, G=64,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=256, K=1024, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=128, K=1024, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=128, K=128, G=8,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=1024, K=256, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=128, K=256, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=2, pad_w=2,
        dilation_d=1, dilation_h=2, dilation_w=2
    ),
    GroupedConvProblem(
        N=4, C=256, K=512, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=512, K=128, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=1024, K=64, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=32, C=512, K=64, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=128, K=128, G=4,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=256, K=512, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=6, pad_w=6,
        dilation_d=1, dilation_h=6, dilation_w=6
    ),
    GroupedConvProblem(
        N=16, C=256, K=256, G=1,
        Di=1, Hi=32, Wi=32,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=128, K=1024, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=32, C=256, K=512, G=1,
        Di=1, Hi=32, Wi=32,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=256, K=64, G=1,
        Di=1, Hi=32, Wi=32,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=32, C=512, K=512, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=512, K=512, G=8,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=128, K=256, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=512, K=512, G=8,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=256, K=256, G=8,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=64, K=64, G=4,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=256, K=1024, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=128, K=128, G=2,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=128, K=128, G=4,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=256, K=128, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=256, K=64, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=64, K=128, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=2, stride_w=2,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=128, K=1024, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=512, K=256, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=512, K=128, G=1,
        Di=1, Hi=32, Wi=32,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=128, K=512, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=256, K=256, G=4,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=512, K=512, G=8,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=2, C=256, K=512, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=32, C=1024, K=64, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=512, K=512, G=8,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=64, K=256, G=1,
        Di=1, Hi=32, Wi=32,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=128, K=128, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=2, pad_w=2,
        dilation_d=1, dilation_h=2, dilation_w=2
    ),
    GroupedConvProblem(
        N=16, C=256, K=512, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=2, pad_w=2,
        dilation_d=1, dilation_h=2, dilation_w=2
    ),
    GroupedConvProblem(
        N=8, C=256, K=128, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=32, C=1024, K=256, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=128, K=256, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=6, pad_w=6,
        dilation_d=1, dilation_h=6, dilation_w=6
    ),
    GroupedConvProblem(
        N=8, C=256, K=512, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=4, pad_w=4,
        dilation_d=1, dilation_h=4, dilation_w=4
    ),
    GroupedConvProblem(
        N=16, C=64, K=64, G=1,
        Di=1, Hi=32, Wi=32,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=2, C=64, K=256, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=32, C=128, K=256, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=256, K=512, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=512, K=1024, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=16, K=16, G=2,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=128, K=128, G=128,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=64, K=64, G=1,
        Di=1, Hi=32, Wi=32,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=512, K=1024, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=256, K=256, G=4,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=64, K=256, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=2, C=64, K=128, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=32, K=32, G=4,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=32, C=512, K=128, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=128, K=64, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=128, K=256, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=64, K=256, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=128, K=256, G=1,
        Di=1, Hi=112, Wi=112,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=2, stride_w=2,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=256, K=128, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=32, C=128, K=128, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=128, K=256, G=1,
        Di=1, Hi=112, Wi=112,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=2, stride_w=2,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=32, K=32, G=2,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=128, K=512, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=32, K=32, G=2,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=64, K=128, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=2, C=256, K=256, G=1,
        Di=1, Hi=32, Wi=32,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=64, K=1024, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=256, K=128, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=256, K=64, G=1,
        Di=1, Hi=112, Wi=112,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=512, K=512, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=256, K=512, G=1,
        Di=1, Hi=14, Wi=14,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=2, pad_w=2,
        dilation_d=1, dilation_h=2, dilation_w=2
    ),
    GroupedConvProblem(
        N=16, C=128, K=1024, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=1024, K=256, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=128, K=256, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=32, K=32, G=4,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=128, K=512, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=128, K=256, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=4, pad_w=4,
        dilation_d=1, dilation_h=4, dilation_w=4
    ),
    GroupedConvProblem(
        N=16, C=128, K=128, G=8,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=16, K=16, G=2,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=256, K=512, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=32, C=512, K=256, G=1,
        Di=1, Hi=32, Wi=32,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=2, C=512, K=256, G=1,
        Di=1, Hi=32, Wi=32,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=16, K=16, G=2,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=32, C=512, K=512, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=64, K=256, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=64, K=512, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=2, C=512, K=1024, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=64, K=256, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=512, K=512, G=512,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=1024, K=512, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=256, K=256, G=8,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=256, K=128, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=256, K=512, G=1,
        Di=1, Hi=112, Wi=112,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=2, stride_w=2,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=256, K=256, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=2, pad_w=2,
        dilation_d=1, dilation_h=2, dilation_w=2
    ),
    GroupedConvProblem(
        N=16, C=256, K=256, G=4,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=2, C=512, K=64, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=2, C=256, K=128, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=128, K=512, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=32, C=512, K=64, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=64, K=64, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=64, K=64, G=64,
        Di=1, Hi=112, Wi=112,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=256, K=256, G=256,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=32, C=128, K=128, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=64, K=256, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=64, K=128, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=2, pad_w=2,
        dilation_d=1, dilation_h=2, dilation_w=2
    ),
    GroupedConvProblem(
        N=8, C=256, K=256, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=256, K=128, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=1024, K=128, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=256, K=256, G=4,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=2, C=512, K=128, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=256, K=256, G=4,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=512, K=64, G=1,
        Di=1, Hi=32, Wi=32,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=64, K=128, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=2, pad_w=2,
        dilation_d=1, dilation_h=2, dilation_w=2
    ),
    GroupedConvProblem(
        N=1, C=128, K=1024, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=512, K=256, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=256, K=256, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=2, pad_w=2,
        dilation_d=1, dilation_h=2, dilation_w=2
    ),
    GroupedConvProblem(
        N=4, C=128, K=512, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=64, K=64, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=32, K=32, G=2,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=64, K=256, G=1,
        Di=1, Hi=32, Wi=32,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=128, K=256, G=1,
        Di=1, Hi=112, Wi=112,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=32, C=64, K=512, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=32, C=1024, K=256, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=256, K=256, G=1,
        Di=1, Hi=32, Wi=32,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=128, K=256, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=6, pad_w=6,
        dilation_d=1, dilation_h=6, dilation_w=6
    ),
    GroupedConvProblem(
        N=32, C=512, K=256, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=2, C=64, K=128, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=32, C=256, K=128, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=64, K=128, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=512, K=256, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=128, K=256, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=2, stride_w=2,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=2, C=128, K=128, G=1,
        Di=1, Hi=112, Wi=112,
        Z=1, Y=7, X=7,
        stride_d=1, stride_h=2, stride_w=2,
        pad_d=0, pad_h=3, pad_w=3,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=128, K=128, G=1,
        Di=1, Hi=14, Wi=14,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=6, pad_w=6,
        dilation_d=1, dilation_h=6, dilation_w=6
    ),
    GroupedConvProblem(
        N=1, C=256, K=1024, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=256, K=256, G=1,
        Di=1, Hi=14, Wi=14,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=2, pad_w=2,
        dilation_d=1, dilation_h=2, dilation_w=2
    ),
    GroupedConvProblem(
        N=16, C=128, K=512, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=256, K=512, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=2, pad_w=2,
        dilation_d=1, dilation_h=2, dilation_w=2
    ),
    GroupedConvProblem(
        N=16, C=64, K=128, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=64, K=256, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=64, K=64, G=8,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=64, K=128, G=1,
        Di=1, Hi=112, Wi=112,
        Z=1, Y=7, X=7,
        stride_d=1, stride_h=2, stride_w=2,
        pad_d=0, pad_h=3, pad_w=3,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=512, K=1024, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=32, C=128, K=1024, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=64, K=256, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=128, K=128, G=2,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=128, K=256, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=512, K=1024, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=256, K=512, G=1,
        Di=1, Hi=14, Wi=14,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=2, pad_w=2,
        dilation_d=1, dilation_h=2, dilation_w=2
    ),
    GroupedConvProblem(
        N=4, C=64, K=64, G=8,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=512, K=256, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=256, K=256, G=256,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=32, C=512, K=512, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=512, K=512, G=8,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=512, K=64, G=1,
        Di=1, Hi=32, Wi=32,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=256, K=128, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=128, K=512, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=512, K=64, G=1,
        Di=1, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=64, K=64, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=64, K=256, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=1024, K=256, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=256, K=512, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=128, K=128, G=1,
        Di=1, Hi=32, Wi=32,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=64, K=1024, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=64, K=64, G=2,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=2, C=64, K=128, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=1024, K=512, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=64, K=64, G=2,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=64, K=128, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=256, K=1024, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=256, K=256, G=256,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=64, K=64, G=4,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=256, K=256, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=32, C=1024, K=256, G=1,
        Di=1, Hi=16, Wi=16,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=32, C=256, K=512, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=128, K=128, G=1,
        Di=1, Hi=8, Wi=8,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=16, C=64, K=512, G=1,
        Di=1, Hi=32, Wi=32,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=64, K=256, G=1,
        Di=1, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=128, K=256, G=1,
        Di=16, Hi=28, Wi=28,
        Z=3, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=1, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=128, K=128, G=1,
        Di=16, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=64, K=128, G=1,
        Di=32, Hi=56, Wi=56,
        Z=3, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=1, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=64, K=128, G=1,
        Di=16, Hi=28, Wi=28,
        Z=3, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=1, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=2, C=128, K=128, G=1,
        Di=32, Hi=56, Wi=56,
        Z=3, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=1, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=128, K=256, G=1,
        Di=8, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=64, K=128, G=1,
        Di=16, Hi=28, Wi=28,
        Z=3, Y=3, X=3,
        stride_d=2, stride_h=1, stride_w=1,
        pad_d=1, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=128, K=256, G=1,
        Di=16, Hi=28, Wi=28,
        Z=3, Y=3, X=3,
        stride_d=2, stride_h=1, stride_w=1,
        pad_d=1, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=64, K=128, G=1,
        Di=32, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=128, K=256, G=1,
        Di=16, Hi=28, Wi=28,
        Z=3, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=1, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=128, K=256, G=1,
        Di=32, Hi=28, Wi=28,
        Z=3, Y=3, X=3,
        stride_d=2, stride_h=1, stride_w=1,
        pad_d=1, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=128, K=256, G=1,
        Di=32, Hi=28, Wi=28,
        Z=3, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=1, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=64, K=128, G=1,
        Di=16, Hi=28, Wi=28,
        Z=3, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=1, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=64, K=128, G=1,
        Di=8, Hi=56, Wi=56,
        Z=3, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=1, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=2, C=128, K=256, G=1,
        Di=16, Hi=56, Wi=56,
        Z=3, Y=3, X=3,
        stride_d=2, stride_h=1, stride_w=1,
        pad_d=1, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=128, K=256, G=1,
        Di=32, Hi=28, Wi=28,
        Z=3, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=1, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=2, C=128, K=128, G=1,
        Di=16, Hi=56, Wi=56,
        Z=3, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=1, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=128, K=256, G=1,
        Di=32, Hi=28, Wi=28,
        Z=3, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=1, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=128, K=256, G=1,
        Di=32, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=128, K=128, G=1,
        Di=32, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=128, K=256, G=1,
        Di=8, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=64, K=128, G=1,
        Di=8, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=128, K=256, G=1,
        Di=8, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=64, K=128, G=1,
        Di=32, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=128, K=256, G=1,
        Di=16, Hi=56, Wi=56,
        Z=3, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=1, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=2, C=64, K=128, G=1,
        Di=8, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=128, K=256, G=1,
        Di=16, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=64, K=128, G=1,
        Di=16, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=128, K=256, G=1,
        Di=16, Hi=56, Wi=56,
        Z=3, Y=3, X=3,
        stride_d=2, stride_h=1, stride_w=1,
        pad_d=1, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=128, K=128, G=1,
        Di=32, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=2, C=128, K=128, G=1,
        Di=32, Hi=28, Wi=28,
        Z=3, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=1, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=2, C=128, K=256, G=1,
        Di=8, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=64, K=128, G=1,
        Di=16, Hi=56, Wi=56,
        Z=3, Y=3, X=3,
        stride_d=2, stride_h=1, stride_w=1,
        pad_d=1, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=128, K=128, G=1,
        Di=8, Hi=56, Wi=56,
        Z=3, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=1, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=2, C=128, K=256, G=1,
        Di=32, Hi=56, Wi=56,
        Z=3, Y=3, X=3,
        stride_d=2, stride_h=1, stride_w=1,
        pad_d=1, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=128, K=128, G=1,
        Di=32, Hi=28, Wi=28,
        Z=3, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=1, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=128, K=128, G=1,
        Di=32, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=2, C=128, K=128, G=1,
        Di=8, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=128, K=128, G=1,
        Di=32, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=128, K=256, G=1,
        Di=32, Hi=28, Wi=28,
        Z=3, Y=3, X=3,
        stride_d=2, stride_h=1, stride_w=1,
        pad_d=1, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=128, K=256, G=1,
        Di=8, Hi=28, Wi=28,
        Z=3, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=1, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=2, C=128, K=256, G=1,
        Di=16, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=128, K=128, G=1,
        Di=16, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=2, C=128, K=256, G=1,
        Di=32, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=64, K=128, G=1,
        Di=8, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=2, C=128, K=256, G=1,
        Di=8, Hi=28, Wi=28,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=4, C=128, K=256, G=1,
        Di=32, Hi=56, Wi=56,
        Z=3, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=1, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=1, C=64, K=128, G=1,
        Di=32, Hi=28, Wi=28,
        Z=3, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=1, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=128, K=256, G=1,
        Di=32, Hi=56, Wi=56,
        Z=1, Y=1, X=1,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=0, pad_h=0, pad_w=0,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
    GroupedConvProblem(
        N=8, C=64, K=128, G=1,
        Di=8, Hi=56, Wi=56,
        Z=3, Y=3, X=3,
        stride_d=1, stride_h=1, stride_w=1,
        pad_d=1, pad_h=1, pad_w=1,
        dilation_d=1, dilation_h=1, dilation_w=1
    ),
]
