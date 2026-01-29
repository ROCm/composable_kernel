# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Template detail string parser for C++ template instantiations.

This module provides functions to parse the arg_detail strings from
Clang's -ftime-trace output into structured components.
"""

import re
from typing import Dict


def parse_template_detail(detail_str: str) -> Dict[str, any]:
    """
    Parse a template detail string into structured components.

    Args:
        detail_str: The arg_detail string from -ftime-trace

    Returns:
        Dictionary with parsed fields:
        - namespace: Top-level namespace (e.g., 'std', 'ck')
        - template_name: Template name without parameters
        - full_qualified_name: Full namespace::template_name
        - param_count: Number of template parameters
        - is_ck_type: Boolean indicating if this is a CK library type
        - is_nested: Boolean indicating if contains nested templates

    Example:
        >>> parse_template_detail('std::basic_string<char>')
        {
            'namespace': 'std',
            'template_name': 'basic_string',
            'full_qualified_name': 'std::basic_string',
            'param_count': 1,
            'is_ck_type': False,
            'is_nested': False
        }
    """
    # Handle empty or invalid strings
    if not detail_str or not isinstance(detail_str, str):
        return _empty_result()

    # Remove surrounding quotes if present
    detail_str = detail_str.strip('"')

    # Extract components
    namespace = extract_namespace(detail_str)
    template_name = extract_template_name(detail_str)
    full_qualified_name = extract_full_qualified_name(detail_str)
    param_count = count_template_params(detail_str)
    is_ck = is_ck_template(detail_str)
    is_nested = is_nested_template(detail_str)

    return {
        "namespace": namespace,
        "template_name": template_name,
        "full_qualified_name": full_qualified_name,
        "param_count": param_count,
        "is_ck_type": is_ck,
        "is_nested": is_nested,
    }


def extract_namespace(detail_str: str) -> str:
    """
    Extract the top-level namespace from a template detail string.

    Args:
        detail_str: The template detail string

    Returns:
        The top-level namespace, or empty string if none found

    Example:
        >>> extract_namespace('std::basic_string<char>')
        'std'
        >>> extract_namespace('ck::tensor_operation::device::DeviceConv2d<...>')
        'ck'
    """
    if not detail_str:
        return ""

    # Remove quotes
    detail_str = detail_str.strip('"')

    # Find first :: separator
    match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)::", detail_str)
    if match:
        return match.group(1)

    # No namespace found - check if it's a simple type
    match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)", detail_str)
    if match:
        return match.group(1)

    return ""


def extract_template_name(detail_str: str) -> str:
    """
    Extract the template name without namespace or parameters.

    Args:
        detail_str: The template detail string

    Returns:
        The template name without namespace or parameters

    Example:
        >>> extract_template_name('std::basic_string<char>')
        'basic_string'
        >>> extract_template_name('ck::GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3<...>')
        'GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3'
    """
    if not detail_str:
        return ""

    # Remove quotes
    detail_str = detail_str.strip('"')

    # Find the last component before < or end of string
    # This handles nested namespaces like ck::tensor_operation::device::DeviceConv2d
    match = re.search(r"::([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:<|$)", detail_str)
    if match:
        return match.group(1)

    # No :: found, try to get name before <
    match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:<|$)", detail_str)
    if match:
        return match.group(1)

    return ""


def extract_full_qualified_name(detail_str: str) -> str:
    """
    Extract the full qualified name (namespace::...::template_name).

    Args:
        detail_str: The template detail string

    Returns:
        The full qualified name without template parameters

    Example:
        >>> extract_full_qualified_name('std::basic_string<char>')
        'std::basic_string'
        >>> extract_full_qualified_name('ck::tensor_operation::device::DeviceConv2d<...>')
        'ck::tensor_operation::device::DeviceConv2d'
    """
    if not detail_str:
        return ""

    # Remove quotes
    detail_str = detail_str.strip('"')

    # Match everything up to the first < or end of string
    match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\s*(?:<|$)", detail_str)
    if match:
        return match.group(1)

    return ""


def count_template_params(detail_str: str) -> int:
    """
    Count the number of top-level template parameters.

    This counts commas at the top level of template brackets,
    not commas inside nested templates.

    Args:
        detail_str: The template detail string

    Returns:
        Number of template parameters, or 0 if not a template

    Example:
        >>> count_template_params('std::basic_string<char>')
        1
        >>> count_template_params('std::tuple<int, float, double>')
        3
    """
    if not detail_str or "<" not in detail_str:
        return 0

    # Remove quotes
    detail_str = detail_str.strip('"')

    # Find the template parameter section
    start = detail_str.find("<")
    if start == -1:
        return 0

    # Track bracket depth to only count top-level commas
    depth = 0
    param_count = 1  # Start with 1 (if there's a <, there's at least one param)
    in_template = False

    for i in range(start, len(detail_str)):
        char = detail_str[i]

        if char == "<":
            depth += 1
            in_template = True
        elif char == ">":
            depth -= 1
            if depth == 0:
                # We've closed the outermost template
                break
        elif char == "," and depth == 1:
            # Top-level comma
            param_count += 1

    return param_count if in_template else 0


def is_ck_template(detail_str: str) -> bool:
    """
    Check if this is a CK library template.

    Args:
        detail_str: The template detail string

    Returns:
        True if this is a CK library type, False otherwise

    Example:
        >>> is_ck_template('ck::tensor_operation::device::DeviceConv2d<...>')
        True
        >>> is_ck_template('std::basic_string<char>')
        False
    """
    if not detail_str:
        return False

    # Remove quotes
    detail_str = detail_str.strip('"')

    # Check if it starts with ck:: or contains ::ck::
    return detail_str.startswith("ck::") or "::ck::" in detail_str


def is_nested_template(detail_str: str) -> bool:
    """
    Check if this template contains nested template instantiations.

    Args:
        detail_str: The template detail string

    Returns:
        True if contains nested templates, False otherwise

    Example:
        >>> is_nested_template('std::vector<int>')
        False
        >>> is_nested_template('std::vector<std::string>')
        True
    """
    if not detail_str or "<" not in detail_str:
        return False

    # Remove quotes
    detail_str = detail_str.strip('"')

    # Find the template parameter section
    start = detail_str.find("<")
    if start == -1:
        return False

    # Look for nested < after the first one
    depth = 0
    for i in range(start, len(detail_str)):
        char = detail_str[i]

        if char == "<":
            depth += 1
            if depth > 1:
                # Found a nested template
                return True
        elif char == ">":
            depth -= 1
            if depth == 0:
                break

    return False


def _empty_result() -> Dict[str, any]:
    """Return an empty result dictionary with default values."""
    return {
        "namespace": "",
        "template_name": "",
        "full_qualified_name": "",
        "param_count": 0,
        "is_ck_type": False,
        "is_nested": False,
    }
