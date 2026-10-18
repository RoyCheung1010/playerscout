"""Streamlit-free orchestration layer between the dashboard and the engines.

``app.py`` only handles widgets, caching and layout; every computation it
displays comes from a function here, so the full dashboard logic is testable
without a browser.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import (
    COMPOSITE_SCORES,
    CORE_FEATURES,
    CURRENT_SEASON_END_YEAR,
    DATA_ORIGIN_COLUMN,
    FEATURE_LABELS,
    PERCENTILE_SUFFIX,
    season_end_year,
)
from src.exceptions import InsufficientDataError
from src.features.preprocessor import ScalerType, ScoutingPreprocessor
from src.models.clustering import DEFAULT_K_RANGE, TacticalClusterer
from src.models.pca import TacticalPCA
from src.models.similarity import Metric, ScoutingSimilarityEngine

GROUND_TRUTH_COLUMN = "Generator_Archetype"

# Display names for the composite scores, in sidebar/category order.
COMPOSITE_DISPLAY: dict[str, str] = {
    COMPOSITE_SCORES["Scoring"]: "Finishing",
    COMPOSITE_SCORES["Creation & Passing"]: "Creation & Passing",
    COMPOSITE_SCORES["Dribbling"]: "Dribbling & Carrying",
    COMPOSITE_SCORES["Defending"]: "Defending",
}

# Shortlist columns renamed for the executive table.
SHORTLIST_DISPLAY: dict[str, str] = {
    "Rank": "Rank",
    "Player": "Player",
    "Squad": "Squad",
    "League": "League",
    "Age": "Age",
    "Market_Value_EUR_M": "Market Value (€M)",
    "Value_Diff_EUR_M": "Value Diff (€M)",
    "Contract_Expiry_Year": "Contract Expiry",
    "Match_Score_Pct": "Match Score (%)",
    "Assigned_Role": "Tactical Role",
}


@dataclass
class CohortModel:
    """Everything fitted on one positional cohort."""

    cohort: pd.DataFrame  # players + Cluster, Assigned_Role, PC1, PC2
    X: np.ndarray  # scaled feature matrix, row-aligned with ``cohort``
    pca: TacticalPCA
    centroids: np.ndarray
    role_names: dict[int, str]
    scaler_type: ScalerType
    features: tuple[str, ...] = field(default=CORE_FEATURES)

    @property
    def labels(self) -> np.ndarray:
        return self.cohort["Cluster"].to_numpy()

    @property
    def pca_frame(self) -> pd.DataFrame:
        return self.cohort[["Player", "Squad", "PC1", "PC2"]]


@dataclass(frozen=True)
class RecruitmentFilters:
    """Moneyball constraints applied to the replacement search only."""

    max_age: int | None = None
    max_market_value: float | None = None
    contract_years_max: int | None = None

    def describe(self) -> str:
        parts = []
        if self.max_age is not None:
            parts.append(f"Age ≤ {self.max_age}")
        if self.max_market_value is not None:
            parts.append(f"Value ≤ €{self.max_market_value:,.1f}M")
        if self.contract_years_max is not None:
            parts.append(f"Contract ≤ {self.contract_years_max} yr left")
        return " · ".join(parts) or "No constraints"


# --------------------------------------------------------------------------- #
# Dataset context
# --------------------------------------------------------------------------- #
def dataset_reference_year(df: pd.DataFrame) -> int:
    """Season end year the data describes (contract years count from it).

    Uses the most common ``Season`` value (e.g. ``"2023-2024"`` -> 2024) and
    falls back to :data:`CURRENT_SEASON_END_YEAR` when absent or unparseable.
    """
    if "Season" not in df.columns or df["Season"].dropna().empty:
        return CURRENT_SEASON_END_YEAR
    try:
        return season_end_year(str(df["Season"].dropna().mode().iloc[0]))
    except ValueError:
        return CURRENT_SEASON_END_YEAR


def contract_reference_year(df: pd.DataFrame) -> int:
    """Year that contract "years remaining" count from.

    Real Transfermarkt contracts are the ones listed at the dataset's
    snapshot, so the ingestion records that year in
    ``Contract_Reference_Year``; otherwise the season end is used.
    """
    column = "Contract_Reference_Year"
    if column in df.columns and not df[column].dropna().empty:
        return int(df[column].dropna().mode().iloc[0])
    return dataset_reference_year(df)


def source_share(df: pd.DataFrame, column: str, source: str) -> float:
    """Fraction of rows whose provenance ``column`` equals ``source``."""
    if column not in df.columns or df.empty:
        return 0.0
    return float((df[column] == source).mean())


def dataset_origin(df: pd.DataFrame) -> str | None:
    """Provenance label (``"fbref"``, ``"synthetic-benchmark"``) if recorded."""
    if DATA_ORIGIN_COLUMN not in df.columns or df[DATA_ORIGIN_COLUMN].dropna().empty:
        return None
    return str(df[DATA_ORIGIN_COLUMN].dropna().mode().iloc[0])


def proxy_share(df: pd.DataFrame, column: str) -> float:
    """Fraction of rows whose ``column`` (e.g. ``Market_Value_Source``) is ``"proxy"``."""
    if column not in df.columns or df.empty:
        return 0.0
    return float((df[column] == "proxy").mean())


# --------------------------------------------------------------------------- #
# Cohort & model fitting
# --------------------------------------------------------------------------- #
def filter_cohort(df: pd.DataFrame, position_group: str, leagues: Sequence[str] | None) -> pd.DataFrame:
    """Players in one position group (and, optionally, a subset of leagues)."""
    mask = df["Position_Group"] == position_group
    if leagues is not None:
        mask &= df["League"].isin(list(leagues))
    return df.loc[mask].reset_index(drop=True)


def min_players_for(k: int) -> int:
    """Smallest cohort that supports ``k`` clusters and a 2-D PCA."""
    return max(k + 1, 3)


def build_cohort_model(cohort: pd.DataFrame, n_clusters: int, scaler_type: ScalerType) -> CohortModel:
    """Scale, project and cluster a cohort, then label every player's role.

    Raises:
        InsufficientDataError: the cohort is too small for ``n_clusters``.
    """
    if len(cohort) < min_players_for(n_clusters):
        raise InsufficientDataError(
            f"Only {len(cohort)} players match the cohort filters; at least "
            f"{min_players_for(n_clusters)} are needed for k = {n_clusters}."
        )
    X, _ = ScoutingPreprocessor().scale_features(cohort, scaler_type)
    pca = TacticalPCA().fit(X)
    clusterer = TacticalClusterer(n_clusters=n_clusters)
    labels = clusterer.fit_predict(X)
    roles = clusterer.role_labels(list(CORE_FEATURES))

    enriched = cohort.reset_index(drop=True).copy()
    enriched["Cluster"] = labels
    enriched["Assigned_Role"] = pd.Series(labels).map(roles).to_numpy()
    enriched[["PC1", "PC2"]] = pca.transform(X)
    return CohortModel(enriched, X, pca, clusterer.centroids_, roles, scaler_type)


def compute_diagnostics(X: np.ndarray, k_range: range = DEFAULT_K_RANGE) -> dict[str, list[float]]:
    """Elbow/silhouette curves; thin wrapper so the UI can cache it separately."""
    return TacticalClusterer.compute_diagnostics(X, k_range)


def best_k(diagnostics: Mapping[str, Sequence[float]]) -> int | None:
    """k with the highest silhouette score, or ``None`` if none is finite."""
    scores = np.asarray(diagnostics.get("silhouette_scores", []), dtype=float)
    if scores.size == 0 or not np.isfinite(scores).any():
        return None
    return int(np.asarray(diagnostics["k"])[int(np.nanargmax(scores))])


# --------------------------------------------------------------------------- #
# Player selection
# --------------------------------------------------------------------------- #
def player_options(cohort: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Selectbox label -> (Player, Squad), most valuable first.

    Names shared by several players get their squad appended, e.g.
    ``"Lucas Martin (Lille)"``; namesakes at the same club also get their age,
    then a counter, so every label is unique. Unique names stay bare.
    """
    ordered = cohort.sort_values(["Market_Value_EUR_M", "Player"], ascending=[False, True])
    name_dup = ordered["Player"].duplicated(keep=False)
    club_dup = ordered.duplicated(["Player", "Squad"], keep=False)
    options: dict[str, tuple[str, str]] = {}
    for (_, row), by_name, by_club in zip(ordered.iterrows(), name_dup, club_dup):
        if by_club:
            label = f"{row['Player']} ({row['Squad']}, age {int(row['Age'])})"
        elif by_name:
            label = f"{row['Player']} ({row['Squad']})"
        else:
            label = str(row["Player"])
        base, n = label, 2
        while label in options:
            label, n = f"{base} #{n}", n + 1
        options[label] = (str(row["Player"]), str(row["Squad"]))
    return options


def get_player_row(cohort: pd.DataFrame, player: str, squad: str) -> pd.Series:
    hits = cohort[(cohort["Player"] == player) & (cohort["Squad"] == squad)]
    if hits.empty:
        raise KeyError(f"{player} ({squad}) is not in the cohort")
    return hits.iloc[0]


def overall_percentile(row: pd.Series) -> float:
    """Mean within-position percentile across the 9 core features."""
    values = [row.get(f"{f}{PERCENTILE_SUFFIX}") for f in CORE_FEATURES]
    finite = [float(v) for v in values if v is not None and pd.notna(v)]
    return float(np.mean(finite)) if finite else 0.0


def score_badge(score: float) -> tuple[str, str]:
    """Qualitative tier and badge colour for a 0-100 score."""
    if score >= 85:
        return "Elite", "green"
    if score >= 65:
        return "Strong", "blue"
    if score >= 40:
        return "Average", "gray"
    if score >= 20:
        return "Below Average", "orange"
    return "Weak", "red"


# --------------------------------------------------------------------------- #
# Similarity
# --------------------------------------------------------------------------- #
def run_shortlist(
    model: CohortModel,
    player: str,
    squad: str,
    metric: Metric,
    weights: Mapping[str, float],
    filters: RecruitmentFilters,
    top_n: int = 10,
    reference_year: int = CURRENT_SEASON_END_YEAR,
) -> pd.DataFrame:
    """Top-``top_n`` replacements for the target inside the cohort."""
    return ScoutingSimilarityEngine(reference_year=reference_year).search(
        player,
        model.cohort,
        model.X,
        metric=metric,
        feature_weights=weights,
        same_position_only=True,
        max_age=filters.max_age,
        max_market_value=filters.max_market_value,
        contract_expiry_years_max=filters.contract_years_max,
        top_n=top_n,
        target_squad=squad,
    )


def match_scores(
    model: CohortModel, player: str, squad: str, metric: Metric, weights: Mapping[str, float]
) -> pd.Series:
    """Unfiltered match score of every cohort player against the target."""
    return ScoutingSimilarityEngine().match_scores_for(
        player, model.cohort, model.X, metric=metric, feature_weights=weights, target_squad=squad
    )


def format_shortlist(shortlist: pd.DataFrame) -> pd.DataFrame:
    """Rename and order the similarity output for the executive table."""
    return shortlist.loc[:, list(SHORTLIST_DISPLAY)].rename(columns=SHORTLIST_DISPLAY)


# --------------------------------------------------------------------------- #
# Clusters
# --------------------------------------------------------------------------- #
def top_traits(centroid: np.ndarray, n: int = 2) -> list[str]:
    """Human-readable names of the ``n`` most positive centroid features."""
    order = np.argsort(np.asarray(centroid))[::-1][:n]
    return [FEATURE_LABELS.get(CORE_FEATURES[i], CORE_FEATURES[i]) for i in order]


def cluster_summary(model: CohortModel) -> pd.DataFrame:
    """One row per cluster: role, size, share of cohort, signature traits."""
    sizes = model.cohort["Cluster"].value_counts()
    total = max(len(model.cohort), 1)
    rows = []
    for cid, role in sorted(model.role_names.items()):
        size = int(sizes.get(cid, 0))
        rows.append({
            "Cluster": cid,
            "Tactical Role": role,
            "Players": size,
            "Share (%)": round(100.0 * size / total, 1),
            "Top Traits": " + ".join(top_traits(model.centroids[cid])),
            "Median Value (€M)": round(
                float(model.cohort.loc[model.cohort["Cluster"] == cid, "Market_Value_EUR_M"].median()), 1
            ) if size else 0.0,
        })
    return pd.DataFrame(rows)


def ground_truth_report(model: CohortModel) -> pd.DataFrame | None:
    """Cluster x hidden-archetype cross-tab (NMI in ``attrs``), if available."""
    if GROUND_TRUTH_COLUMN not in model.cohort.columns:
        return None
    return TacticalClusterer.evaluate_against_ground_truth(
        model.labels, model.cohort[GROUND_TRUTH_COLUMN]
    )


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def build_export(model: CohortModel, scores: pd.Series | None = None) -> pd.DataFrame:
    """Full cohort with clusters, roles, PCA coordinates, composites and match %."""
    out = model.cohort.copy()
    if scores is not None:
        out["Match_Score_Pct"] = scores.reindex(out.index).round(2).to_numpy()
    lead = ["Player", "Squad", "League", "Position", "Position_Group", "Age", "Market_Value_EUR_M",
            "Contract_Expiry_Year", "Cluster", "Assigned_Role", "PC1", "PC2",
            *COMPOSITE_SCORES.values()]
    if "Match_Score_Pct" in out.columns:
        lead.append("Match_Score_Pct")
    lead = [c for c in lead if c in out.columns]
    rest = [c for c in out.columns if c not in lead]
    out = out[lead + rest]
    if "Match_Score_Pct" in out.columns:
        out = out.sort_values("Match_Score_Pct", ascending=False, kind="mergesort")
    return out.reset_index(drop=True)


def export_filename(position_group: str, now: datetime | None = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"scoutintel_{position_group}_shortlist_{stamp}.csv"
