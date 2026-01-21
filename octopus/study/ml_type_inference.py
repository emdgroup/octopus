"""ML type inference utilities."""

import numpy as np
import pandas as pd

from octopus.logger import get_logger

from .types import MLType

logger = get_logger()


def infer_ml_type(
    data: pd.DataFrame,
    target_columns: list[str],
    silent: bool = False,
) -> tuple[MLType, dict[str, str]]:
    """Automatically infer the ML type based on target columns and data characteristics.

    Args:
        data: DataFrame containing the dataset
        target_columns: List of target column names
        silent: If True, suppresses logging output

    Returns:
        Tuple of (inferred_ml_type, target_assignments)
        - For single target: target_assignments is empty dict
        - For timetoevent: target_assignments maps "duration" and "event" to actual column names

    Raises:
        ValueError: If target columns configuration is invalid or columns don't exist

    Rules:
        Single target:
        - 2 unique values → classification
        - 3+ unique values + categorical/object/string → multiclass
        - numeric → regression

        Dual target (timetoevent):
        - One column: only 2 unique values (0,1) or (0.0, 1.0) or bool
        - Other column: more than 2 values of type int/float
        - Returns target_assignments: {"duration": col_name, "event": col_name}

        3+ targets:
        - Not supported, raises error
    """
    # Validate target columns exist in data
    missing_cols = [col for col in target_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Target columns not found in data: {missing_cols}")

    # Case 1: Single target column
    if len(target_columns) == 1:
        target_col = target_columns[0]
        y = data[target_col].dropna()

        if len(y) == 0:
            raise ValueError(f"Target column '{target_col}' has no valid (non-NaN) values")

        unique_count = y.nunique()

        # Classification: exactly 2 unique values
        if unique_count == 2:
            ml_type = MLType.CLASSIFICATION
            reason = "2 unique values"

        # Multiclass: 3+ unique values and categorical/object/string type
        elif y.dtype.name in ["object", "category", "string"]:
            ml_type = MLType.MULTICLASS
            reason = f"categorical type (dtype={y.dtype.name})"

        # Regression: numeric type
        elif np.issubdtype(y.dtype, np.number):  # type: ignore[arg-type]
            ml_type = MLType.REGRESSION
            reason = "numeric type"

        else:
            raise ValueError(f"Unsupported target dtype: {y.dtype}")

        if not silent:
            logger.info(f"Inferred ml_type: '{ml_type.value}' ({reason})")
            logger.info(f"  Target column: '{target_col}'")
            logger.info(f"  Unique values: {unique_count}")

        # Set default target assignment for single-target scenarios
        target_assignments = {"default": target_col}
        return ml_type, target_assignments

    # Case 2: Two target columns - timetoevent
    elif len(target_columns) == 2:
        ml_type, target_assignments = _infer_timetoevent(data, target_columns, silent)
        return ml_type, target_assignments

    # Case 3: 3+ target columns - not supported
    else:
        raise ValueError(
            f"Unsupported number of target columns: {len(target_columns)}. "
            f"Expected 1 (classification/multiclass/regression) or 2 (timetoevent)."
        )


def _infer_timetoevent(
    data: pd.DataFrame,
    target_columns: list[str],
    silent: bool = False,
) -> tuple[MLType, dict[str, str]]:
    """Infer timetoevent ML type and extract target assignments.

    Requirements:
    - One column: only 2 unique values (0,1) or (0.0, 1.0) or bool
    - Other column: more than 2 values of type int/float

    Args:
        data: DataFrame containing the dataset
        target_columns: List of exactly 2 target column names
        silent: If True, suppresses logging

    Returns:
        Tuple of (MLType.TIMETOEVENT, target_assignments)
        where target_assignments = {"duration": duration_col, "event": event_col}

    Raises:
        ValueError: If columns don't meet timetoevent requirements
    """
    col1, col2 = target_columns

    # Get non-null values for each column
    data1 = data[col1].dropna()
    data2 = data[col2].dropna()

    if len(data1) == 0 or len(data2) == 0:
        raise ValueError("Target columns must have valid (non-NaN) values")

    unique1 = data1.nunique()
    unique2 = data2.nunique()

    # Check which column is the event indicator (binary)
    event_col = None
    duration_col = None

    # Check col1 as potential event column
    if unique1 == 2:
        unique_vals1 = set(data1.unique())
        # Check if values are binary: {0, 1} or {0.0, 1.0} or boolean
        # and if col2 is numeric with >2 values
        if unique_vals1 <= {0, 1} and np.issubdtype(data2.dtype, np.number) and unique2 > 2:  # type: ignore[arg-type]
            event_col = col1
            duration_col = col2

    # Check col2 as potential event column if not found yet
    if event_col is None and unique2 == 2:
        unique_vals2 = set(data2.unique())
        # Check if values are binary and if col1 is numeric with >2 values
        if unique_vals2 <= {0, 1} and np.issubdtype(data1.dtype, np.number) and unique1 > 2:  # type: ignore[arg-type]
            event_col = col2
            duration_col = col1

    # Validate we found valid event and duration columns
    if event_col is None or duration_col is None:
        raise ValueError(
            f"Could not identify timetoevent structure in target columns {target_columns}.\n"
            f"Requirements:\n"
            f"  - One column must have exactly 2 unique values: (0,1), (0.0,1.0), or boolean\n"
            f"  - Other column must be numeric with >2 values\n"
            f"Column '{col1}': {unique1} unique values, dtype={data1.dtype}\n"
            f"Column '{col2}': {unique2} unique values, dtype={data2.dtype}"
        )

    target_assignments = {"duration": duration_col, "event": event_col}

    if not silent:
        logger.info("Inferred ml_type: 'timetoevent'")
        logger.info(f"  Duration column: '{duration_col}'")
        logger.info(f"  Event column: '{event_col}'")
        event_rate = data[event_col].mean()
        logger.info(f"  Event rate: {event_rate:.2%}")

    return MLType.TIMETOEVENT, target_assignments
