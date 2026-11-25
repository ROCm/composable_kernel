# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# generate kernel instances to speed up compilation
import dataclasses
import os.path as path
import textwrap


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


def indent(code: str, indent: str = "    ") -> str:
    return textwrap.indent(code, indent)


def if_(i: int) -> str:
    return "if" if i == 0 else "else if"


def non_padding_key(trait):
    """A hashable key of the trait's non-padding fields: equal for traits whose non-padding fields
    compare equal (repr, as some field types are unhashable dataclasses)."""
    return tuple(
        repr(getattr(trait, f.name))
        for f in dataclasses.fields(trait)
        if "pad" not in f.name
    )


def check_duplicates_and_paddings(traits, trait, by_key=None):
    """Check
    * if the traits list does not contain a trait with the same parameters;
    * if paddings are consitent: the previous kernel can be incorrectly called before the new one,
      for example, f, _t_, f, t cannot be before f, _f_, f, t.

    by_key: optional dict from non_padding_key() to the earlier traits with that key, kept by the
    caller. Only traits with equal non-padding fields can fail either check, so with it the check
    compares against those alone (linear rather than quadratic in the number of traits).
    """

    fields = [f.name for f in dataclasses.fields(trait)]
    pad_fields = [f for f in fields if "pad" in f]
    non_pad_fields = [f for f in fields if "pad" not in f]
    if by_key is not None:
        traits = by_key.get(non_padding_key(trait), ())
    for prev_trait in traits:
        if any(getattr(trait, f) != getattr(prev_trait, f) for f in non_pad_fields):
            continue
        if all(getattr(trait, f) == getattr(prev_trait, f) for f in pad_fields):
            raise Exception(f"Duplicate found {trait}")
        # Check if the previous kernel can be incorrectly used before the current one
        # for example, f, _t_, f, t cannot be before f, _f_, f, t
        is_prev_more_restrictive = False
        is_curr_more_restrictive = False
        for f in pad_fields:
            prev_pad = getattr(prev_trait, f)
            pad = getattr(trait, f)
            if isinstance(prev_pad, str):
                prev_pad = 1000000 if prev_pad == "f" else 1
                pad = 1000000 if pad == "f" else 1
            elif isinstance(prev_pad, int):
                prev_pad = 1000000 if prev_pad == 0 else prev_pad
                pad = 1000000 if pad == 0 else pad
            else:
                assert False
            if prev_pad < pad:
                is_prev_more_restrictive = True
            elif prev_pad > pad:
                is_curr_more_restrictive = True
        if is_prev_more_restrictive and not is_curr_more_restrictive:
            raise Exception(
                f"Kernel will never be used because paddings are not ordered correctly:\n{prev_trait} supersedes\n{trait}"
            )
