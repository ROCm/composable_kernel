# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Template instantiation analysis for Clang -ftime-trace data.

This module provides specialized functions for analyzing C++ template
instantiation costs from Clang's -ftime-trace output.
"""

import pandas as pd
from .template_parser import parse_template_detail


def get_template_instantiation_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to template instantiation events and parse arg_detail into structured columns.

    Returns events for:
    - InstantiateFunction: Function template instantiations
    - InstantiateClass: Class template instantiations

    The returned DataFrame includes parsed columns from arg_detail:
    - namespace: Top-level namespace (e.g., 'std', 'ck')
    - template_name: Template name without parameters
    - full_qualified_name: Full namespace::template_name
    - param_count: Number of template parameters
    - is_ck_type: Boolean indicating if this is a CK library type
    - is_nested: Boolean indicating if contains nested templates

    Args:
        df: DataFrame from parse_file()

    Returns:
        Filtered DataFrame containing template instantiation events with parsed columns

    Example:
        >>> df = parse_file('trace.json')
        >>> templates = get_template_instantiation_events(df)
        >>> templates.sort_values('dur', ascending=False).head(10)
        >>> # Filter to CK types only
        >>> ck_templates = templates[templates['is_ck_type']]
        >>> # Group by template name
        >>> templates.groupby('template_name')['dur'].sum()
    """
    # Filter to template instantiation events
    filtered_df = (
        df[
            df["name"].isin(
                [
                    "InstantiateClass",
                    "InstantiateFunction",
                ]
            )
        ]
        .drop(
            columns=[
                "arg_avg ms",
                "arg_count",
                "arg_name",
                "cat",
                "id",
                "ph",
                "pid",
                "tid",
            ]
        )
        .reset_index(drop=True)
    )

    # Parse arg_detail into structured columns
    parsed_data = filtered_df["arg_detail"].apply(parse_template_detail)

    # Convert list of dicts to DataFrame and join with original
    parsed_df = pd.DataFrame(parsed_data.tolist())

    # Combine with original data
    result_df = pd.concat([filtered_df, parsed_df], axis=1)

    return result_df
