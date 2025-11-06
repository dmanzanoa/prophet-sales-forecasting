"""Data loading and preprocessing utilities.

This module contains helper functions to read time series data from CSV and
apply common transformations.  Keeping data handling separate from model
logic makes the code easier to maintain and reuse in other projects.
"""

from typing import Tuple
import pandas as pd

try:
    # SciPy provides winsorization, which can be used to limit the impact of
    # extreme outliers.  It is optional; if unavailable we fall back to no
    # winsorization.
    from scipy import stats  # type: ignore
except ImportError:  # pragma: no cover
    stats = None  # type: ignore


def load_data(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    The returned DataFrame will have its `ds` column parsed as a datetime.  No
    additional cleaning is performed; downstream code should drop missing values
    as appropriate.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing at least the columns `ds`, `y` and `region`.
    """
    df = pd.read_csv(file_path, parse_dates=["ds"])
    return df


def winsorize_per_region(
    df: pd.DataFrame,
    column: str = "y",
    limits: Tuple[float, float] = (0.0, 0.05),
) -> pd.DataFrame:
    """Apply winsorization to a numerical column within each region.

    Winsorization caps extreme values at a specified lower and upper percentile
    boundary.  This can reduce the impact of outliers on model training.  The
    function groups the DataFrame by the `region` column and applies
    SciPy's ``stats.mstats.winsorize`` on the given column.  If SciPy is not
    installed, the function returns the original DataFrame unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing a `region` column.
    column : str, default "y"
        The numeric column to winsorize.
    limits : tuple of float, default ``(0.0, 0.05)``
        The lower and upper tail proportions to cut off.  For example,
        ``(0.0, 0.05)`` caps the top 5% of values without touching the lower end.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the specified column winsorized per region.
    """
    if stats is None:
        # SciPy not installed; return a copy to avoid surprising callers
        return df.copy()

    def _winsorize_series(s: pd.Series) -> pd.Series:
        # SciPy's winsorize returns a numpy masked array; convert back to Series
        arr = stats.mstats.winsorize(s, limits=limits)  # type: ignore[arg-type]
        return pd.Series(arr, index=s.index)

    df = df.copy()
    df[column] = df.groupby("region")[column].transform(_winsorize_series)
    return df
