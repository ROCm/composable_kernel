# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Functional pipeline for parallel processing of trace files.

This module provides a fluent API for building data processing pipelines with
support for parallel execution, progress tracking, and multiple output branches.

Example:
    >>> from trace_analysis import Pipeline, find_trace_files
    >>> from trace_analysis.parse_file import parse_file
    >>>
    >>> files = find_trace_files(Path("build"))
    >>> dfs = Pipeline(files).map(parse_file, workers=8).collect()
"""

from typing import Any, Callable, List, Optional, Tuple, Union
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm


class Pipeline:
    """
    Functional pipeline for processing data with parallel execution support.

    Provides a fluent API for chaining operations like map, filter, and reduce.
    Supports parallel processing with multiprocessing and progress tracking with tqdm.

    Features:
    - Fluent API with method chaining
    - Parallel processing with configurable worker count
    - Progress bars in Jupyter notebooks (tqdm)
    - Fail-fast error handling
    - In-memory processing for speed
    - Tee operation for branching into multiple outputs

    Attributes:
        _items: Current list of items in the pipeline
        _is_reduced: Flag indicating if pipeline has been reduced to single value

    Example:
        Basic parallel processing:
        >>> files = find_trace_files(Path("build"))
        >>> dfs = Pipeline(files).map(parse_file, workers=8).collect()

        Multi-stage pipeline:
        >>> results = (
        ...     Pipeline(files)
        ...     .map(parse_file, workers=8)
        ...     .filter(lambda df: len(df) > 1000)
        ...     .collect()
        ... )

        Multiple outputs with tee:
        >>> pipeline = Pipeline(files).map(parse_file, workers=8)
        >>> all_events, metadata, stats = pipeline.tee(
        ...     lambda dfs: pd.concat(dfs, ignore_index=True),
        ...     lambda dfs: [get_metadata(df) for df in dfs],
        ...     lambda dfs: {"count": len(dfs)}
        ... )
    """

    def __init__(self, items: List[Any]):
        """
        Initialize a new pipeline with a list of items.

        Args:
            items: Initial list of items to process
        """
        self._items = items
        self._is_reduced = False

    def map(
        self,
        func: Callable[[Any], Any],
        workers: Optional[int] = None,
        desc: Optional[str] = None,
    ) -> "Pipeline":
        """
        Apply a function to each item in the pipeline.

        Args:
            func: Function to apply to each item. Should accept a single argument
                and return a transformed value.
            workers: Number of parallel workers to use:
                - None: Sequential processing (single-threaded)
                - -1: Use all available CPUs
                - N > 0: Use N worker processes
            desc: Description for the progress bar. If None, uses a default description.

        Returns:
            Self for method chaining

        Raises:
            ValueError: If pipeline has already been reduced
            Exception: Any exception raised by func is re-raised with context

        Example:
            >>> # Sequential processing
            >>> Pipeline(files).map(parse_file).collect()
            >>>
            >>> # Parallel processing with all CPUs
            >>> Pipeline(files).map(parse_file, workers=-1).collect()
            >>>
            >>> # Parallel with custom worker count and description
            >>> Pipeline(files).map(parse_file, workers=8, desc="Parsing").collect()
        """
        if self._is_reduced:
            raise ValueError("Cannot map after reduce operation")

        if not self._items:
            return self

        # Determine worker count
        if workers == -1:
            workers = cpu_count()

        # Set default description
        if desc is None:
            desc = "Processing items"

        # Sequential processing
        if workers is None or workers == 1:
            results = []
            for item in tqdm(self._items, desc=desc):
                try:
                    results.append(func(item))
                except Exception as e:
                    raise type(e)(f"Error processing item {item}: {e}") from e
            self._items = results
            return self

        # Parallel processing
        try:
            with Pool(processes=workers) as pool:
                # Use imap_unordered for better performance (results as they complete)
                # Wrap with tqdm for progress tracking
                results = list(
                    tqdm(
                        pool.imap_unordered(func, self._items),
                        total=len(self._items),
                        desc=desc,
                    )
                )
            self._items = results
            return self
        except Exception as e:
            # Re-raise with context
            raise type(e)(f"Error in parallel map operation: {e}") from e

    def filter(self, predicate: Callable[[Any], bool]) -> "Pipeline":
        """
        Filter items based on a predicate function.

        Args:
            predicate: Function that returns True for items to keep, False to discard.
                Should accept a single argument and return a boolean.

        Returns:
            Self for method chaining

        Raises:
            ValueError: If pipeline has already been reduced

        Example:
            >>> # Keep only large DataFrames
            >>> Pipeline(dfs).filter(lambda df: len(df) > 1000).collect()
            >>>
            >>> # Keep only successful builds
            >>> Pipeline(dfs).filter(
            ...     lambda df: 'ExecuteCompiler' in df['name'].values
            ... ).collect()
        """
        if self._is_reduced:
            raise ValueError("Cannot filter after reduce operation")

        self._items = [item for item in self._items if predicate(item)]
        return self

    def reduce(self, func: Callable[[List[Any]], Any]) -> "Pipeline":
        """
        Reduce all items to a single value using an aggregation function.

        After reduction, the pipeline contains a single value and no further
        map or filter operations are allowed.

        Args:
            func: Aggregation function that accepts a list of all items and
                returns a single aggregated value.

        Returns:
            Self for method chaining

        Raises:
            ValueError: If pipeline has already been reduced

        Example:
            >>> # Concatenate all DataFrames
            >>> Pipeline(dfs).reduce(
            ...     lambda dfs: pd.concat(dfs, ignore_index=True)
            ... ).collect()
            >>>
            >>> # Sum all values
            >>> Pipeline(numbers).reduce(sum).collect()
            >>>
            >>> # Custom aggregation
            >>> Pipeline(dfs).reduce(
            ...     lambda dfs: {
            ...         "total_files": len(dfs),
            ...         "total_events": sum(len(df) for df in dfs)
            ...     }
            ... ).collect()
        """
        if self._is_reduced:
            raise ValueError("Cannot reduce twice")

        try:
            self._items = [func(self._items)]
            self._is_reduced = True
            return self
        except Exception as e:
            raise type(e)(f"Error in reduce operation: {e}") from e

    def tee(self, *funcs: Callable[[List[Any]], Any]) -> Tuple[Any, ...]:
        """
        Branch the pipeline into multiple outputs.

        Each function receives the full list of current items and produces
        an independent output. This is useful for generating multiple
        aggregations or analyses from the same data.

        This operation automatically collects the pipeline results.

        Args:
            *funcs: Variable number of functions, each accepting the full list
                of items and returning a result. Each function is applied
                independently to the same input data.

        Returns:
            Tuple of results, one per function, in the same order as the functions

        Raises:
            ValueError: If no functions are provided
            Exception: Any exception raised by a function is re-raised with context

        Example:
            >>> pipeline = Pipeline(files).map(parse_file, workers=8)
            >>>
            >>> # Create three different outputs from the same data
            >>> all_events, metadata_df, stats = pipeline.tee(
            ...     # Output 1: Concatenated DataFrame
            ...     lambda dfs: pd.concat(dfs, ignore_index=True),
            ...     # Output 2: Metadata summary
            ...     lambda dfs: pd.DataFrame([get_metadata(df).__dict__ for df in dfs]),
            ...     # Output 3: Statistics dictionary
            ...     lambda dfs: {
            ...         "total_files": len(dfs),
            ...         "total_events": sum(len(df) for df in dfs)
            ...     }
            ... )
        """
        if not funcs:
            raise ValueError("At least one function must be provided to tee")

        results = []
        for i, func in enumerate(funcs):
            try:
                results.append(func(self._items))
            except Exception as e:
                raise type(e)(f"Error in tee function {i}: {e}") from e

        return tuple(results)

    def collect(self) -> Union[List[Any], Any]:
        """
        Execute the pipeline and return the results.

        Returns:
            If the pipeline has been reduced, returns the single reduced value.
            Otherwise, returns the list of items.

        Example:
            >>> # Returns list of DataFrames
            >>> dfs = Pipeline(files).map(parse_file, workers=8).collect()
            >>>
            >>> # Returns single concatenated DataFrame
            >>> df = Pipeline(files).map(parse_file, workers=8).reduce(pd.concat).collect()
        """
        if self._is_reduced:
            return self._items[0]
        return self._items
