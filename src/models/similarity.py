"""Weighted multi-metric replacement search with Moneyball recruitment filters.

Two complementary notions of "similar":

* **Cosine** compares the *shape* of a profile (what a player does relative to
  their own volume), so a quieter player with the same style still matches.
* **Euclidean** compares exact *magnitude*, so it rewards like-for-like output.

Category weights are applied as ``X * sqrt(w)``, which makes squared Euclidean
distance (and cosine's dot product) weight each feature by exactly ``w``.

Scores are computed against the full comparison pool *before* recruitment
filters are applied, so tightening the budget never changes a candidate's score,
only whether they are shown.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from src.config import CORE_FEATURES, CURRENT_SEASON_END_YEAR, FEATURE_CATEGORIES
from src.exceptions import TargetPlayerNotFoundError

Metric = Literal["cosine", "euclidean"]

RESULT_COLUMNS: tuple[str, ...] = (
    "Rank",
    "Player",
    "Squad",
    "League",
    "Position_Group",
    "Age",
    "Market_Value_EUR_M",
    "Contract_Expiry_Year",
    "Match_Score_Pct",
    "Value_Diff_EUR_M",
    "Assigned_Role",
)

# Short names accepted for the four category weights (UI-friendly).
CATEGORY_ALIASES: dict[str, str] = {
    "Creation": "Creation & Passing",
    "Passing": "Creation & Passing",
    "Defensive": "Defending",
    "Defence": "Defending",
    "Defense": "Defending",
    "Finishing": "Scoring",
}

# Euclidean scores are normalised by this percentile of pool pairwise distances.
EUCLIDEAN_REFERENCE_PERCENTILE: float = 95.0


class ScoutingSimilarityEngine:
    """Ranks replacement candidates for a target player.

    Args:
        features: Column order of ``scaled_features``.
        categories: Category -> member features (drives the weight sliders).
        reference_year: Season end year that contract "years left" count from.
    """

    def __init__(
        self,
        features: Sequence[str] = CORE_FEATURES,
        categories: Mapping[str, Sequence[str]] = FEATURE_CATEGORIES,
        reference_year: int = CURRENT_SEASON_END_YEAR,
    ) -> None:
        self.features = tuple(features)
        self.categories = {k: tuple(v) for k, v in categories.items()}
        self.reference_year = reference_year
        uncovered = set(self.features) - {f for members in self.categories.values() for f in members}
        if uncovered:
            raise ValueError(f"Features without a weight category: {sorted(uncovered)}")

    # ------------------------------------------------------------------ #
    # Weights
    # ------------------------------------------------------------------ #
    def feature_weight_vector(self, feature_weights: Mapping[str, float] | None = None) -> np.ndarray:
        """Expand category weights to one weight per feature (default 1.0).

        Raises:
            ValueError: unknown category, negative weight, or all weights zero.
        """
        resolved: dict[str, float] = {category: 1.0 for category in self.categories}
        for key, value in (feature_weights or {}).items():
            category = CATEGORY_ALIASES.get(key, key)
            if category not in resolved:
                raise ValueError(f"Unknown weight category {key!r}; expected one of {sorted(resolved)}")
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Weight for {key!r} must be a non-negative number, got {value}")
            resolved[category] = float(value)

        per_feature = {f: resolved[c] for c, members in self.categories.items() for f in members}
        weights = np.array([per_feature[f] for f in self.features], dtype=float)
        if not weights.any():
            raise ValueError("At least one category weight must be greater than zero")
        return weights

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def search(
        self,
        target_player_name: str,
        df: pd.DataFrame,
        scaled_features: np.ndarray,
        metric: Metric = "cosine",
        feature_weights: Mapping[str, float] | None = None,
        same_position_only: bool = True,
        max_age: int | None = None,
        max_market_value: float | None = None,
        contract_expiry_years_max: int | None = None,
        excluded_leagues: Sequence[str] | None = None,
        top_n: int = 10,
        target_squad: str | None = None,
    ) -> pd.DataFrame:
        """Return the ``top_n`` best replacements, ranked by ``Match_Score_Pct``.

        ``df`` and ``scaled_features`` must be row-aligned. ``target_squad``
        disambiguates players who share a name. An empty frame (with the
        standard columns) is returned when no candidate passes the filters.

        Raises:
            TargetPlayerNotFoundError: target not in ``df``.
            ValueError: bad metric, misaligned inputs or ambiguous target.
        """
        if metric not in ("cosine", "euclidean"):
            raise ValueError(f"metric must be 'cosine' or 'euclidean', got {metric!r}")
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        X = np.asarray(scaled_features, dtype=float)
        if X.ndim != 2 or X.shape != (len(df), len(self.features)):
            raise ValueError(
                f"scaled_features shape {X.shape} must be ({len(df)}, {len(self.features)})"
            )

        frame = df.reset_index(drop=True)
        target_pos = self._locate_target(frame, target_player_name, target_squad)

        pool = np.ones(len(frame), dtype=bool)
        if same_position_only and "Position_Group" in frame.columns:
            pool &= (frame["Position_Group"] == frame.at[target_pos, "Position_Group"]).to_numpy()

        X_weighted = X * np.sqrt(self.feature_weight_vector(feature_weights))
        scores = np.full(len(frame), np.nan)
        scores[pool] = self._match_scores(X_weighted[pool], X_weighted[target_pos], metric)

        keep = pool & self._recruitment_mask(
            frame, max_age, max_market_value, contract_expiry_years_max, excluded_leagues
        )
        keep[target_pos] = False

        if not keep.any():
            return pd.DataFrame(columns=list(RESULT_COLUMNS))

        target_value = float(frame.at[target_pos, "Market_Value_EUR_M"])
        candidates = frame.loc[keep].assign(Match_Score_Pct=scores[keep])
        candidates = candidates.sort_values(
            ["Match_Score_Pct", "Market_Value_EUR_M"], ascending=[False, True], kind="mergesort"
        ).head(top_n)
        return self._format(candidates, target_value)

    def match_scores_for(
        self,
        target_player_name: str,
        df: pd.DataFrame,
        scaled_features: np.ndarray,
        metric: Metric = "cosine",
        feature_weights: Mapping[str, float] | None = None,
        target_squad: str | None = None,
    ) -> pd.Series:
        """Match score of every row against the target (no filters), indexed like ``df``."""
        frame = df.reset_index(drop=True)
        target_pos = self._locate_target(frame, target_player_name, target_squad)
        X_weighted = np.asarray(scaled_features, dtype=float) * np.sqrt(
            self.feature_weight_vector(feature_weights)
        )
        scores = self._match_scores(X_weighted, X_weighted[target_pos], metric)
        return pd.Series(scores, index=df.index, name="Match_Score_Pct")

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    @staticmethod
    def _match_scores(pool: np.ndarray, target: np.ndarray, metric: Metric) -> np.ndarray:
        """Map similarity to a 0-100% score for every row of ``pool``."""
        target_row = target.reshape(1, -1)
        if metric == "cosine":
            # (cos + 1) / 2 keeps a full ordering: opposite = 0%, orthogonal = 50%.
            # Zero vectors get cos = 0 from sklearn, i.e. a neutral 50%.
            sim = cosine_similarity(pool, target_row).ravel()
            return np.clip((sim + 1.0) / 2.0 * 100.0, 0.0, 100.0)

        dist = euclidean_distances(pool, target_row).ravel()
        reference = _reference_distance(pool)
        if reference <= 0:
            # Every player in the pool is identical: all perfect matches.
            return np.full(len(pool), 100.0)
        return np.clip(100.0 * (1.0 - dist / reference), 0.0, 100.0)

    # ------------------------------------------------------------------ #
    # Filtering & formatting
    # ------------------------------------------------------------------ #
    def _recruitment_mask(
        self,
        frame: pd.DataFrame,
        max_age: int | None,
        max_market_value: float | None,
        contract_expiry_years_max: int | None,
        excluded_leagues: Sequence[str] | None,
    ) -> np.ndarray:
        mask = pd.Series(True, index=frame.index)
        if max_age is not None:
            mask &= frame["Age"] <= max_age
        if max_market_value is not None:
            mask &= frame["Market_Value_EUR_M"] <= max_market_value
        if contract_expiry_years_max is not None:
            mask &= (frame["Contract_Expiry_Year"] - self.reference_year) <= contract_expiry_years_max
        if excluded_leagues:
            mask &= ~frame["League"].isin(list(excluded_leagues))
        return mask.fillna(False).to_numpy(dtype=bool)

    @staticmethod
    def _locate_target(frame: pd.DataFrame, name: str, squad: str | None) -> int:
        names = frame["Player"].astype(str)
        hits = names == name
        if not hits.any():
            hits = names.str.strip().str.casefold() == str(name).strip().casefold()
        if squad is not None:
            hits &= frame["Squad"].astype(str) == squad
        positions = np.flatnonzero(hits.to_numpy())
        if positions.size == 0:
            where = f" at {squad}" if squad else ""
            raise TargetPlayerNotFoundError(f"Player {name!r}{where} is not in the current cohort.")
        if positions.size > 1:
            clubs = sorted(frame.loc[positions, "Squad"].astype(str))
            raise ValueError(f"{name!r} is ambiguous ({clubs}); pass target_squad to choose one.")
        return int(positions[0])

    @staticmethod
    def _format(candidates: pd.DataFrame, target_value: float) -> pd.DataFrame:
        out = candidates.copy()
        out["Rank"] = np.arange(1, len(out) + 1)
        out["Value_Diff_EUR_M"] = (target_value - out["Market_Value_EUR_M"]).round(2)
        out["Match_Score_Pct"] = out["Match_Score_Pct"].round(2)
        if "Assigned_Role" not in out.columns:
            out["Assigned_Role"] = "Unassigned"
        if "Position_Group" not in out.columns:
            out["Position_Group"] = pd.NA
        return out.loc[:, list(RESULT_COLUMNS)].reset_index(drop=True)


def _reference_distance(pool: np.ndarray) -> float:
    """Robust "far apart" distance for the pool: 95th pct of pairwise distances.

    Using a high percentile instead of the maximum stops one outlier pair from
    compressing everyone else's scores towards 100%.
    """
    if len(pool) < 2:
        return 0.0
    pairwise = euclidean_distances(pool)
    upper = pairwise[np.triu_indices(len(pool), k=1)]
    return float(np.percentile(upper, EUCLIDEAN_REFERENCE_PERCENTILE))
