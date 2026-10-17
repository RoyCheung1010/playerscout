"""2-D tactical projection of the scaled scouting vector.

PCA compresses the 9 scaled features onto the two orthogonal axes that carry
the most variance. Loadings tell analysts what each axis *means* in football
terms (e.g. PC1 = "attacking output vs. defensive work").
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError

from src.config import RANDOM_STATE
from src.exceptions import InsufficientDataError

PC_NAMES: tuple[str, str] = ("PC1", "PC2")


class TacticalPCA:
    """Thin, validated wrapper around ``PCA(n_components=2)``."""

    n_components: int = 2

    def __init__(self, random_state: int = RANDOM_STATE) -> None:
        self.random_state = random_state
        self._pca: PCA | None = None

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray) -> TacticalPCA:
        """Fit on a scaled ``(n_samples, n_features)`` matrix.

        Raises:
            InsufficientDataError: fewer than 2 samples or 2 features.
        """
        X = self._validate(X)
        if min(X.shape) < self.n_components:
            raise InsufficientDataError(
                f"PCA needs at least {self.n_components} players and features; got shape {X.shape}."
            )
        with warnings.catch_warnings():
            # Zero total variance makes sklearn divide 0/0; the ratio
            # properties below report that as 0% retained instead.
            warnings.simplefilter("ignore", RuntimeWarning)
            self._pca = PCA(n_components=self.n_components, random_state=self.random_state).fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project onto ``(PC1, PC2)``; returns shape ``(n_samples, 2)``."""
        pca = self._fitted()
        X = self._validate(X)
        if X.shape[1] != pca.n_features_in_:
            raise ValueError(f"Expected {pca.n_features_in_} features, got {X.shape[1]}")
        return pca.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------ #
    # Interpretation
    # ------------------------------------------------------------------ #
    @property
    def is_fitted(self) -> bool:
        return self._pca is not None

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Share of total variance captured by PC1 and PC2."""
        return np.nan_to_num(self._fitted().explained_variance_ratio_, nan=0.0)

    @property
    def cumulative_variance(self) -> float:
        """Total variance retained by the 2-D map, in ``[0, 1]``."""
        return float(np.clip(self.explained_variance_ratio.sum(), 0.0, 1.0))

    def get_loadings_df(self, feature_names: Sequence[str]) -> pd.DataFrame:
        """Eigenvector weights: rows ``PC1``/``PC2``, one column per feature.

        A large positive weight means players high on that feature sit to the
        right (PC1) or top (PC2) of the map.
        """
        pca = self._fitted()
        names = list(feature_names)
        if len(names) != pca.n_features_in_:
            raise ValueError(f"Expected {pca.n_features_in_} feature names, got {len(names)}")
        return pd.DataFrame(pca.components_, index=list(PC_NAMES), columns=names)

    def to_frame(self, coords: np.ndarray, index: pd.Index | None = None) -> pd.DataFrame:
        """Wrap projected coordinates as a ``PC1``/``PC2`` DataFrame."""
        return pd.DataFrame(coords, columns=list(PC_NAMES), index=index)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _fitted(self) -> PCA:
        if self._pca is None:
            raise NotFittedError("TacticalPCA is not fitted; call fit() first.")
        return self._pca

    @staticmethod
    def _validate(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Expected a 2-D matrix, got {X.ndim}-D")
        if not np.isfinite(X).all():
            raise ValueError("Input contains NaN or infinite values")
        return X
