"""Tests for K-Means role clustering, diagnostics and semantic labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.config import CORE_FEATURES
from src.exceptions import InsufficientDataError
from src.features.preprocessor import ScoutingPreprocessor
from src.models.clustering import ROLE_LOOKUP, TacticalClusterer

FEATURES = list(CORE_FEATURES)


def centroid(**deviations: float) -> np.ndarray:
    """A centroid at the cohort mean except for the given feature deviations."""
    row = np.zeros(len(FEATURES))
    for name, value in deviations.items():
        row[FEATURES.index(name)] = value
    return row


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [2, 5, 8])
def test_labels_in_valid_range(scaled_matrix: np.ndarray, k: int) -> None:
    clusterer = TacticalClusterer(n_clusters=k)
    labels = clusterer.fit_predict(scaled_matrix)
    assert labels.shape == (scaled_matrix.shape[0],)
    assert set(np.unique(labels)) == set(range(k))
    assert clusterer.centroids_.shape == (k, len(FEATURES))


def test_fit_is_deterministic(scaled_matrix: np.ndarray) -> None:
    a = TacticalClusterer(5).fit_predict(scaled_matrix)
    b = TacticalClusterer(5).fit_predict(scaled_matrix)
    np.testing.assert_array_equal(a, b)


def test_centroids_are_cluster_means(scaled_matrix: np.ndarray) -> None:
    clusterer = TacticalClusterer(4)
    labels = clusterer.fit_predict(scaled_matrix)
    for k in range(4):
        np.testing.assert_allclose(
            clusterer.centroids_[k], scaled_matrix[labels == k].mean(axis=0), atol=1e-6
        )


def test_too_few_players_raises() -> None:
    with pytest.raises(InsufficientDataError):
        TacticalClusterer(5).fit_predict(np.random.default_rng(0).normal(size=(3, 9)))


def test_invalid_k_rejected() -> None:
    with pytest.raises(ValueError):
        TacticalClusterer(1)


def test_unfitted_access_raises() -> None:
    with pytest.raises(NotFittedError):
        _ = TacticalClusterer().centroids_


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def test_diagnostics_lengths_and_ranges(scaled_matrix: np.ndarray) -> None:
    k_range = range(2, 11)
    diag = TacticalClusterer.compute_diagnostics(scaled_matrix, k_range)
    assert diag["k"] == [float(k) for k in k_range]
    assert len(diag["inertias"]) == len(diag["silhouette_scores"]) == len(k_range)
    assert all(-1.0 <= s <= 1.0 for s in diag["silhouette_scores"])
    assert all(i > 0 for i in diag["inertias"])


def test_inertia_decreases_with_k(scaled_matrix: np.ndarray) -> None:
    inertias = TacticalClusterer.compute_diagnostics(scaled_matrix, range(2, 8))["inertias"]
    assert all(a > b for a, b in zip(inertias, inertias[1:]))


def test_diagnostics_skip_k_at_or_above_sample_count() -> None:
    X = np.random.default_rng(1).normal(size=(5, 9))
    diag = TacticalClusterer.compute_diagnostics(X, range(2, 10))
    assert diag["k"] == [2.0, 3.0, 4.0]  # silhouette needs k <= n - 1


def test_diagnostics_raise_when_no_k_is_valid() -> None:
    with pytest.raises(InsufficientDataError):
        TacticalClusterer.compute_diagnostics(np.zeros((2, 9)), range(2, 10))


def test_diagnostics_survive_duplicate_points() -> None:
    X = np.vstack([np.zeros((6, 9)), np.ones((6, 9))])
    diag = TacticalClusterer.compute_diagnostics(X, range(2, 5))
    assert diag["silhouette_scores"][0] == pytest.approx(1.0)
    assert len(diag["silhouette_scores"]) == 3


# --------------------------------------------------------------------------- #
# Role labelling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("deviations", "expected"),
    [
        ({"Goals_p90": 1.5, "Shots_p90": 1.2}, "Direct Goal Threat / Poacher"),
        ({"Progressive_Passes_p90": 1.1, "Pass_Completion_Pct": 0.9}, "Deep-Lying Playmaker / Recycler"),
        ({"pAdj_Tackles_Won_p90": 1.0, "pAdj_Interceptions_p90": 1.3}, "Ball-Winning Destroyer"),
        ({"Successful_Take_Ons_p90": 2.0, "Key_Passes_p90": 0.8}, "Dynamic 1v1 Creator"),
    ],
)
def test_lookup_roles(deviations: dict[str, float], expected: str) -> None:
    labels = TacticalClusterer.generate_role_labels(np.array([centroid(**deviations)]), FEATURES)
    assert labels == {0: expected}


def test_unmapped_pair_falls_back_to_balanced_label() -> None:
    labels = TacticalClusterer.generate_role_labels(
        np.array([centroid(Goals_p90=1.0, pAdj_Interceptions_p90=0.9)]), FEATURES
    )
    assert labels[0] == "Balanced Clinical Finishing & Reading of Play Specialist"


def test_weak_deviation_falls_back_to_balanced_label() -> None:
    labels = TacticalClusterer.generate_role_labels(
        np.array([centroid(Goals_p90=0.2, Shots_p90=0.1)]), FEATURES
    )
    assert labels[0].startswith("Balanced ")


def test_duplicate_roles_are_disambiguated() -> None:
    centroids = np.array([
        centroid(Goals_p90=1.5, Shots_p90=1.2, Assists_p90=0.5),
        centroid(Goals_p90=1.4, Shots_p90=1.3, pAdj_Tackles_Won_p90=0.6),
        centroid(Goals_p90=1.3, Shots_p90=1.1, pAdj_Tackles_Won_p90=0.4),
    ])
    labels = TacticalClusterer.generate_role_labels(centroids, FEATURES)
    assert len(set(labels.values())) == 3
    assert all(label.startswith("Direct Goal Threat / Poacher") for label in labels.values())


def test_labels_distinct_and_non_empty_on_real_clusters(scaled_matrix: np.ndarray) -> None:
    for k in range(2, 11):
        clusterer = TacticalClusterer(k)
        clusterer.fit_predict(scaled_matrix)
        labels = clusterer.role_labels(FEATURES)
        assert set(labels) == set(range(k))
        assert all(isinstance(v, str) and v.strip() for v in labels.values())
        assert len(set(labels.values())) == k


def test_lookup_keys_use_known_features() -> None:
    assert all(len(pair) == 2 and pair <= set(FEATURES) for pair in ROLE_LOOKUP)


def test_label_shape_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        TacticalClusterer.generate_role_labels(np.zeros((2, 4)), FEATURES)


# --------------------------------------------------------------------------- #
# Ground-truth validation
# --------------------------------------------------------------------------- #
def test_nmi_against_ground_truth_full_cohort(
    processed_players: pd.DataFrame, scaled_matrix: np.ndarray
) -> None:
    labels = TacticalClusterer(8).fit_predict(scaled_matrix)
    table = TacticalClusterer.evaluate_against_ground_truth(
        labels, processed_players["Generator_Archetype"]
    )
    assert table.attrs["nmi"] > 0.40
    assert table.to_numpy().sum() == len(processed_players)
    assert set(table.columns) == set(processed_players["Generator_Archetype"])


@pytest.mark.parametrize(("group", "k"), [("DF", 3), ("MF", 4), ("FW", 4)])
def test_nmi_within_position_groups(processed_players: pd.DataFrame, group: str, k: int) -> None:
    cohort = processed_players[processed_players["Position_Group"] == group]
    X, _ = ScoutingPreprocessor().scale_features(cohort, "robust")
    labels = TacticalClusterer(k).fit_predict(X)
    nmi = TacticalClusterer.evaluate_against_ground_truth(labels, cohort["Generator_Archetype"]).attrs["nmi"]
    assert nmi > 0.40


def test_random_labels_score_low_nmi(processed_players: pd.DataFrame) -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 8, size=len(processed_players))
    nmi = TacticalClusterer.evaluate_against_ground_truth(
        labels, processed_players["Generator_Archetype"]
    ).attrs["nmi"]
    assert nmi < 0.05


def test_ground_truth_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        TacticalClusterer.evaluate_against_ground_truth(np.array([0, 1]), pd.Series(["a"]))
