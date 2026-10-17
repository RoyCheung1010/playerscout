"""K-Means tactical role clustering, k-selection diagnostics and auto-labelling.

Clusters are fitted in scaled feature space, so a centroid coordinate reads as
"how far above/below the cohort norm this group sits" on each feature. The two
most positive coordinates define the cluster's football identity.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

from src.config import FEATURE_LABELS, RANDOM_STATE
from src.exceptions import InsufficientDataError

DEFAULT_K_RANGE: range = range(2, 11)  # k in [2, 10]

# Unordered feature pair -> archetype name.
ROLE_LOOKUP: dict[frozenset[str], str] = {
    frozenset(pair): name
    for pair, name in {
        ("Goals_p90", "Shots_p90"): "Direct Goal Threat / Poacher",
        ("Progressive_Passes_p90", "Pass_Completion_Pct"): "Deep-Lying Playmaker / Recycler",
        ("pAdj_Tackles_Won_p90", "pAdj_Interceptions_p90"): "Ball-Winning Destroyer",
        ("Successful_Take_Ons_p90", "Key_Passes_p90"): "Dynamic 1v1 Creator",
        ("Key_Passes_p90", "Assists_p90"): "Final-Third Creator",
        ("Key_Passes_p90", "Progressive_Passes_p90"): "High-Volume Creator & Ball Progressor",
        ("Successful_Take_Ons_p90", "Shots_p90"): "Direct Dribbler & Box Threat",
        ("Successful_Take_Ons_p90", "Goals_p90"): "Inverted Goalscoring Winger",
        ("Successful_Take_Ons_p90", "Assists_p90"): "Wide Dribbler & Provider",
        ("Successful_Take_Ons_p90", "Progressive_Passes_p90"): "Progressive Ball Carrier",
        ("Goals_p90", "Assists_p90"): "Complete Forward / Goal Contributor",
        ("Shots_p90", "Key_Passes_p90"): "Advanced Playmaker & Shooter",
        ("Goals_p90", "Key_Passes_p90"): "Second Striker / Shadow Forward",
        ("Shots_p90", "Assists_p90"): "Attacking Midfield Goal Threat",
        ("Assists_p90", "Progressive_Passes_p90"): "Advanced Wide Provider",
        ("Key_Passes_p90", "Pass_Completion_Pct"): "Possession Creator",
        ("Pass_Completion_Pct", "pAdj_Interceptions_p90"): "Defensive Anchor & Recycler",
        ("Pass_Completion_Pct", "pAdj_Tackles_Won_p90"): "Press-Resistant Ball Winner",
        ("Progressive_Passes_p90", "pAdj_Interceptions_p90"): "Ball-Playing Interceptor",
        ("Progressive_Passes_p90", "pAdj_Tackles_Won_p90"): "Box-to-Box Ball Progressor",
        ("Successful_Take_Ons_p90", "pAdj_Tackles_Won_p90"): "Dynamic Box-to-Box Carrier",
        ("Successful_Take_Ons_p90", "Pass_Completion_Pct"): "Press-Resistant Dribbler",
        ("Goals_p90", "pAdj_Tackles_Won_p90"): "Pressing Forward",
        ("Shots_p90", "pAdj_Tackles_Won_p90"): "Pressing Forward",
    }.items()
}

# Below this scaled deviation a "dominant" feature is too weak to name a role.
MIN_DOMINANT_DEVIATION: float = 0.25


class TacticalClusterer:
    """K-Means wrapper exposing labels, centroids, diagnostics and role names."""

    def __init__(self, n_clusters: int = 5, random_state: int = RANDOM_STATE) -> None:
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2")
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._model: KMeans | None = None

    # ------------------------------------------------------------------ #
    # Clustering
    # ------------------------------------------------------------------ #
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit K-Means (``n_init=10``) and return integer labels in ``[0, k-1]``.

        Raises:
            InsufficientDataError: fewer players than clusters.
        """
        X = _validate(X)
        if X.shape[0] < self.n_clusters:
            raise InsufficientDataError(
                f"Cannot form {self.n_clusters} clusters from {X.shape[0]} players; "
                "lower k or relax the filters."
            )
        self._model = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
        return self._model.fit_predict(X)

    @property
    def centroids_(self) -> np.ndarray:
        """Cluster centres in scaled space, shape ``(n_clusters, n_features)``."""
        return self._fitted().cluster_centers_.copy()

    @property
    def labels_(self) -> np.ndarray:
        return self._fitted().labels_.copy()

    @property
    def inertia_(self) -> float:
        return float(self._fitted().inertia_)

    def role_labels(self, feature_names: Sequence[str]) -> dict[int, str]:
        """Role names for this fitted model's centroids."""
        return self.generate_role_labels(self.centroids_, feature_names)

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_diagnostics(
        X: np.ndarray,
        k_range: range = DEFAULT_K_RANGE,
        random_state: int = RANDOM_STATE,
    ) -> dict[str, list[float]]:
        """Elbow (inertia) and silhouette curves across ``k_range``.

        Silhouette is only defined for ``2 <= k <= n_samples - 1``, so values
        of k outside that window are skipped. The returned ``"k"`` list says
        which k each score belongs to.

        Raises:
            InsufficientDataError: no k in ``k_range`` is valid for this cohort.
        """
        X = _validate(X)
        n_samples = X.shape[0]
        valid_k = [k for k in k_range if 2 <= k <= n_samples - 1]
        if not valid_k:
            raise InsufficientDataError(
                f"No valid k in {list(k_range)} for {n_samples} players "
                "(silhouette needs 2 <= k < n_samples)."
            )

        inertias: list[float] = []
        silhouettes: list[float] = []
        for k in valid_k:
            model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            with warnings.catch_warnings():
                # Duplicate points can yield fewer clusters than k; handled below.
                warnings.simplefilter("ignore", ConvergenceWarning)
                labels = model.fit_predict(X)
            inertias.append(float(model.inertia_))
            # Duplicate points can collapse clusters; silhouette then degenerates.
            n_found = len(np.unique(labels))
            silhouettes.append(
                float(silhouette_score(X, labels, random_state=random_state))
                if 2 <= n_found <= n_samples - 1
                else float("nan")
            )
        return {"k": [float(k) for k in valid_k], "inertias": inertias, "silhouette_scores": silhouettes}

    # ------------------------------------------------------------------ #
    # Semantic role labelling
    # ------------------------------------------------------------------ #
    @staticmethod
    def generate_role_labels(centroids: np.ndarray, feature_names: Sequence[str]) -> dict[int, str]:
        """Name each cluster from its two most positive centroid features.

        Known feature pairs map through :data:`ROLE_LOOKUP`; anything else (or
        a pair whose deviations are too weak to be distinctive) becomes
        ``"Balanced <Feature 1> & <Feature 2> Specialist"``. Labels are made
        unique so they can serve as UI keys.
        """
        centroids = np.asarray(centroids, dtype=float)
        names = list(feature_names)
        if centroids.ndim != 2 or centroids.shape[1] != len(names):
            raise ValueError(
                f"centroids shape {centroids.shape} does not match {len(names)} feature names"
            )
        if len(names) < 2:
            raise ValueError("At least two features are needed to name roles")

        labels: dict[int, str] = {}
        for cluster_id, centroid in enumerate(centroids):
            order = np.argsort(centroid)[::-1]
            first, second = names[order[0]], names[order[1]]
            strong = centroid[order[1]] >= MIN_DOMINANT_DEVIATION
            role = ROLE_LOOKUP.get(frozenset((first, second))) if strong else None
            labels[cluster_id] = role or (
                f"Balanced {_label(first)} & {_label(second)} Specialist"
            )
        return _deduplicate(labels, centroids, names)

    # ------------------------------------------------------------------ #
    # Validation hook
    # ------------------------------------------------------------------ #
    @staticmethod
    def evaluate_against_ground_truth(
        cluster_labels: np.ndarray, generator_archetypes: pd.Series
    ) -> pd.DataFrame:
        """Cross-tab clusters vs. the generator's hidden archetypes.

        The NMI score (0 = unrelated, 1 = identical partitions) is attached as
        ``result.attrs["nmi"]``.
        """
        cluster_labels = np.asarray(cluster_labels)
        truth = pd.Series(generator_archetypes).reset_index(drop=True)
        if len(cluster_labels) != len(truth):
            raise ValueError(
                f"Length mismatch: {len(cluster_labels)} labels vs {len(truth)} archetypes"
            )
        table = pd.crosstab(
            pd.Series(cluster_labels, name="Cluster"),
            truth.rename("Generator_Archetype"),
        )
        table.attrs["nmi"] = (
            float(normalized_mutual_info_score(truth, cluster_labels)) if len(truth) else 0.0
        )
        return table

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _fitted(self) -> KMeans:
        if self._model is None:
            raise NotFittedError("TacticalClusterer is not fitted; call fit_predict() first.")
        return self._model


def _label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


def _deduplicate(labels: dict[int, str], centroids: np.ndarray, names: list[str]) -> dict[int, str]:
    """Disambiguate repeated role names with each cluster's third-strongest trait."""
    counts = pd.Series(labels).value_counts()
    result = dict(labels)
    for cluster_id, role in labels.items():
        if counts[role] > 1:
            third = names[int(np.argsort(centroids[cluster_id])[::-1][2])] if len(names) > 2 else None
            result[cluster_id] = f"{role} (+ {_label(third)})" if third else role
    # Fall back to numbering if third traits also collide.
    seen: dict[str, int] = {}
    for cluster_id in sorted(result):
        role = result[cluster_id]
        seen[role] = seen.get(role, 0) + 1
        if seen[role] > 1:
            result[cluster_id] = f"{role} #{seen[role]}"
    return result


def _validate(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix, got {X.ndim}-D")
    if not np.isfinite(X).all():
        raise ValueError("Input contains NaN or infinite values")
    return X
