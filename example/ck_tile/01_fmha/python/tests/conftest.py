# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import torch

import ck_tile_fmha
from ck_tile_fmha import _C


def pytest_report_header(config):
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    return [
        f"ck_tile_fmha: {ck_tile_fmha.__file__}",
        f"native extension: {_C.__file__}",
        f"PyTorch {torch.__version__}, ROCm {torch.version.hip}, GPU {device.gcnArchName}",
    ]
