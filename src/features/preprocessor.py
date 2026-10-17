"""Feature preparation for the ML engines and the executive UI.

* :meth:`ScoutingPreprocessor.scale_features` - Z-score or IQR scaling of the
  9-feature scouting vector for clustering, PCA and similarity search.
* :meth:`ScoutingPreprocessor.add_positional_percentiles` - eCDF percentiles
  computed *within* each ``Position_Group`` cohort.
* :meth:`ScoutingPreprocessor.add_composite_scores` - 0-100 sub-scores per
  feature category, built from those positional percentiles.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, Union

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.config import (
    COMPOSITE_SCORES,
    CORE_FEATURES,
    FEATURE_CATEGORIES,
    PERCENTILE_SUFFIX,
)
from src.exceptions import InsufficientDataError

__all__ = ["InsufficientDataError", "ScalerType", "ScoutingPreprocessor", "ecdf_percentile"]

ScalerType = Literal["standard", "robust"]
FittedScaler = Union[StandardScaler, RobustScaler]

_SCALERS: dict[str, type[FittedScaler]] = {
    "standard": StandardScaler,
    "robust": RobustScaler,
}


def ecdf_percentile(values: pd.Series | np.ndarray) -> np.ndarray:
    """Mid-rank empirical CDF percentile in ``(0, 100)``.

    ``P(x) = 100 * (#{v < x} + 0.5 * #{v == x}) / n``, the same definition as
    ``scipy.stats.percentileofscore(kind="mean")``. Ties share a percentile, so
    the many defenders on 0 goals sit mid-table rather than being ranked
    arbitrarily. NaNs stay NaN and are excluded from ``n``.
    """
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan)
    mask = ~np.isnan(arr)
    n = int(mask.sum())
    if n:
        out[mask] = 100.0 * (rankdata(arr[mask], method="average") - 0.5) / n
    return out


class ScoutingPreprocessor:
    """Scales features and builds positional percentiles and composite scores.

    Args:
        features: Feature columns forming the scouting vector.
        group_col: Cohort column used to isolate percentile ranking.
        categories: Category name -> member features (for composites).
        composite_names: Category name -> output column name.
    """

    def __init__(
        self,
        features: Sequence[str] = CORE_FEATURES,
        group_col: str = "Position_Group",
        categories: Mapping[str, Sequence[str]] = FEATURE_CATEGORIES,
        composite_names: Mapping[str, str] = COMPOSITE_SCORES,
    ) -> None:
        self.features: tuple[str, ...] = tuple(features)
        self.group_col = group_col
        self.categories = {k: tuple(v) for k, v in categories.items()}
        self.composite_names = dict(composite_names)

        unknown = {f for members in self.categories.values() for f in members} - set(self.features)
        if unknown:
            raise ValueError(f"Category features not in feature vector: {sorted(unknown)}")
        if set(self.composite_names) != set(self.categories):
            raise ValueError("composite_names must map exactly the configured categories")

    @property
    def percentile_columns(self) -> list[str]:
        return [f"{f}{PERCENTILE_SUFFIX}" for f in self.features]

    @property
    def composite_columns(self) -> list[str]:
        return list(self.composite_names.values())

    # ------------------------------------------------------------------ #
    # Scaling
    # ------------------------------------------------------------------ #
    def scale_features(
        self, df: pd.DataFrame, scaler_type: ScalerType = "standard"
    ) -> tuple[np.ndarray, FittedScaler]:
        """Fit a scaler on the feature vector and return ``(X_scaled, scaler)``.

        ``"robust"`` centres on the median and scales by the IQR, so a few
        Haaland-tier outliers cannot stretch the axis for everyone else.
        Zero-variance features are left at 0 rather than dividing by zero
        (both sklearn scalers guard this internally).

        Raises:
            ValueError: unknown ``scaler_type`` or missing feature columns.
            InsufficientDataError: empty input.
        """
        if scaler_type not in _SCALERS:
            raise ValueError(f"scaler_type must be one of {sorted(_SCALERS)}, got {scaler_type!r}")
        X = self._feature_matrix(df)
        if X.shape[0] == 0:
            raise InsufficientDataError("Cannot scale an empty cohort; relax the filters.")
        scaler = _SCALERS[scaler_type]()
        X_scaled = scaler.fit_transform(X)
        return np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0), scaler

    # ------------------------------------------------------------------ #
    # Percentiles & composites
    # ------------------------------------------------------------------ #
    def add_positional_percentiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with ``<feature>_pct`` columns ranked within position group."""
        self._require_columns(df, (*self.features, self.group_col))
        out = df.copy()
        groups = out.groupby(self.group_col, sort=False, dropna=False)
        for feature, pct_col in zip(self.features, self.percentile_columns):
            out[pct_col] = groups[feature].transform(ecdf_percentile).astype(float)
        return out

    def add_composite_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with 0-100 category scores (mean of member percentiles).

        Percentile columns are computed first if they are not already present.
        """
        out = df if set(self.percentile_columns) <= set(df.columns) else self.add_positional_percentiles(df)
        out = out.copy()
        for category, members in self.categories.items():
            pct_cols = [f"{f}{PERCENTILE_SUFFIX}" for f in members]
            out[self.composite_names[category]] = out[pct_cols].mean(axis=1).clip(0.0, 100.0)
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: positional percentiles followed by composite scores."""
        return self.add_composite_scores(self.add_positional_percentiles(df))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        self._require_columns(df, self.features)
        X = df.loc[:, list(self.features)].to_numpy(dtype=float)
        # Impute any residual NaNs with the column median so one gap in an
        # uploaded CSV does not take down the whole model run.
        if np.isnan(X).any():
            medians = np.nanmedian(np.where(np.isnan(X).all(axis=0), 0.0, X), axis=0)
            X = np.where(np.isnan(X), medians, X)
        return X

    @staticmethod
    def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
