# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# generate kernel instances to speed up compilation

import copy
import os.path as path


def update_file(file_path, content):
    """Update the file at file_path with the given content if it differs from the existing content.

    It avoids unnecessary touching of the file which triggers rebuilds
    """

    existing_content = ""
    if path.exists(file_path):
        with open(file_path, "r") as file:
            existing_content = file.read()
    if existing_content == content:
        return
    with open(file_path, "w") as file:
        file.write(content)

def indent(code: str, indent: str='    ') -> str:
    return ''.join(indent + s if s.strip() != '' else '' for s in code.splitlines(keepends=True))

def if_(i: int) -> str:
    return 'if' if i == 0 else 'else if'

def group_kernels_by_filename(all_kernels):
    """Combine kernels that have the same value of the filename property, unique F_archs of the
    kernels in each group are combined into one list.

    Relative order of kernels is preserved.
    """

    kernels_by_file = {}
    for kernel in all_kernels:
        kernels_by_file.setdefault(kernel.filename, []).append(kernel)
    grouped_kernels = []
    for kernels in kernels_by_file.values():
        kernel = copy.deepcopy(kernels[0])
        kernel.F_archs.clear()
        kernel.F_archs.extend(sorted(list(set(arch for k in kernels for arch in k.F_archs))))
        grouped_kernels.append(kernel)
    return grouped_kernels
