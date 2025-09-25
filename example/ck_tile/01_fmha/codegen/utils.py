# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# generate kernel instances to speed up compilation

import copy
import dataclasses
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

def check_duplicates_and_paddings(traits, trait):
    """Check
     * if the traits list does not contain a trait with the same parameters;
     * if paddings are consitent: the previous kernel can be incorrectly called before the new one,
       for example, f, _t_, f, t cannot be before f, _f_, f, t.
    """

    fields = [f.name for f in dataclasses.fields(trait)]
    pad_fields = [f for f in fields if 'pad' in f]
    non_pad_fields = [f for f in fields if 'pad' not in f]
    for prev_trait in traits:
        if any(getattr(trait, f) != getattr(prev_trait, f) for f in non_pad_fields):
            continue
        if all(getattr(trait, f) == getattr(prev_trait, f) for f in pad_fields):
            raise Exception(f'Duplicate found {trait}')
        # Check if the previous kernel can be incorrectly used before the current one
        # for example, f, _t_, f, t cannot be before f, _f_, f, t
        is_prev_more_restrictive = False
        is_curr_more_restrictive = False
        for f in pad_fields:
            if getattr(prev_trait, f) == 't' and getattr(trait, f) == 'f':
                is_prev_more_restrictive = True
            elif getattr(prev_trait, f) == 'f' and getattr(trait, f) == 't':
                is_curr_more_restrictive = True
        if is_prev_more_restrictive and not is_curr_more_restrictive:
            raise Exception(f'Kernel will never be used because paddings are not ordered correctly: {prev_trait} supersedes {trait}')
