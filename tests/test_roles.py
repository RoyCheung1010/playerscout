"""Position-aware role naming from percentile profiles."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import pipeline
from src.config import CORE_FEATURES, PERCENTILE_SUFFIX
from src.models import roles
from src.models.clustering import TacticalClusterer

FEATURES = list(CORE_FEATURES)


def profile(**percentiles: float) -> np.ndarray:
    """Percentile profile in [-1, 1]; unspecified features sit at the median (50)."""
    values = np.full(len(FEATURES), 50.0)
    for name, pct in percentiles.items():
        values[FEATURES.index(name)] = pct
    return (values - 50.0) / 50.0


STRIKER = profile(Goals_p90=90, Shots_p90=85, Key_Passes_p90=30, Successful_Take_Ons_p90=30, Progressive_Passes_p90=25)
WINGER = profile(Successful_Take_Ons_p90=92, Key_Passes_p90=80, Assists_p90=70, Progressive_Passes_p90=70)
PRESSER = profile(pAdj_Tackles_Won_p90=88, pAdj_Interceptions_p90=85, Goals_p90=25)
QUIET = profile(**{f: 30 for f in FEATURES if f != "Pass_Completion_Pct"})

BALL_PLAYING_CB = profile(Pass_Completion_Pct=80, Progressive_Passes_p90=85, Successful_Take_Ons_p90=30,
                          Key_Passes_p90=30, Assists_p90=35)
POSITIONAL_CB = profile(Pass_Completion_Pct=62, Key_Passes_p90=25, Successful_Take_Ons_p90=28, Assists_p90=30,
                        Shots_p90=30, Progressive_Passes_p90=35, pAdj_Tackles_Won_p90=33)
ATTACKING_FB = profile(Key_Passes_p90=90, Assists_p90=85, Successful_Take_Ons_p90=84, Progressive_Passes_p90=65)
SET_PIECE_CB = profile(Goals_p90=85, Shots_p90=75, Key_Passes_p90=35, Successful_Take_Ons_p90=35)


def test_forward_roles_named_by_profile() -> None:
    labels = roles.positional_role_labels(np.vstack([STRIKER, WINGER, PRESSER, QUIET]), FEATURES, "FW")
    assert labels == {
        0: "Penalty-Box Striker",
        1: "Dribbling Winger",
        2: "Pressing Forward",
        3: "Low-Involvement Forward",
    }


def test_defender_roles_never_call_centre_backs_finishers() -> None:
    labels = roles.positional_role_labels(
        np.vstack([BALL_PLAYING_CB, POSITIONAL_CB, ATTACKING_FB, SET_PIECE_CB]), FEATURES, "DF")
    assert labels == {
        0: "Ball-Playing Centre-Back",
        1: "Positional Centre-Back",
        2: "Attacking Full-Back",
        3: "Set-Piece Threat Centre-Back",
    }
    assert not any("Finishing" in name for name in labels.values())


def test_unknown_position_returns_none() -> None:
    assert roles.positional_role_labels(np.vstack([STRIKER, WINGER]), FEATURES, "GK") is None


def test_labels_unique_and_lookalikes_qualified() -> None:
    strong = ATTACKING_FB
    weaker = profile(Key_Passes_p90=72, Assists_p90=68, Successful_Take_Ons_p90=70, Progressive_Passes_p90=55)
    labels = roles.positional_role_labels(np.vstack([strong, weaker]), FEATURES, "DF")
    assert len(set(labels.values())) == 2
    assert labels[0].startswith("Attacking Full-Back (+ ")
    assert labels[1] == "Attacking Full-Back (Lower Volume)"  # behind its namesake everywhere


def test_more_clusters_than_roles_still_unique() -> None:
    rng = np.random.default_rng(0)
    profiles = np.clip(rng.normal(0, 0.4, size=(10, len(FEATURES))), -1, 1)
    labels = roles.positional_role_labels(profiles, FEATURES, "MF")
    assert len(labels) == 10 and len(set(labels.values())) == 10


def test_flat_profile_gets_hybrid_name() -> None:
    flat = np.zeros((1, len(FEATURES)))
    flat[0, FEATURES.index("Pass_Completion_Pct")] = 0.05
    flat[0, FEATURES.index("pAdj_Interceptions_p90")] = -0.05  # fits no template
    label = roles.positional_role_labels(flat, FEATURES, "FW")[0]
    assert label.startswith("Hybrid ") and label.endswith(" Profile")


def test_every_template_uses_known_features() -> None:
    for group, catalogue in roles.ROLE_TEMPLATES.items():
        assert len(catalogue) >= 5, group
        for weights in catalogue.values():
            assert set(weights) <= set(FEATURES)
            assert all(-1.0 <= w <= 1.0 for w in weights.values())


def test_percentile_profiles_from_cohort() -> None:
    cohort = pd.DataFrame({f"{f}{PERCENTILE_SUFFIX}": [100.0, 0.0, 50.0] for f in FEATURES})
    prof = roles.percentile_profiles(cohort, np.array([0, 0, 1]), FEATURES)
    assert prof.shape == (2, len(FEATURES))
    np.testing.assert_allclose(prof, 0.0)  # (100 + 0) / 2 = 50 -> 0 ; 50 -> 0
    assert roles.percentile_profiles(cohort.iloc[:, :3], np.array([0, 0, 1]), FEATURES) is None


@pytest.mark.parametrize("group", ["FW", "MF", "DF"])
def test_pipeline_uses_positional_roles(processed_players: pd.DataFrame, group: str) -> None:
    model = pipeline.build_cohort_model(pipeline.filter_cohort(processed_players, group, None), 5, "robust")
    catalogue = set(roles.ROLE_TEMPLATES[group])
    for name in model.role_names.values():
        base = name.split(" (")[0]
        assert base in catalogue or name.startswith("Hybrid "), name
    assert len(set(model.role_names.values())) == 5


def test_mixed_position_cohort_falls_back_to_generic_lookup(processed_players: pd.DataFrame) -> None:
    mixed = processed_players.reset_index(drop=True)
    X = np.random.default_rng(1).normal(size=(len(mixed), len(FEATURES)))
    clusterer = TacticalClusterer(n_clusters=3)
    labels = clusterer.fit_predict(X)
    names = pipeline.role_names_for(mixed, labels, clusterer)
    assert names == clusterer.role_labels(FEATURES)


def test_cluster_traits_read_percentiles(processed_players: pd.DataFrame) -> None:
    model = pipeline.build_cohort_model(pipeline.filter_cohort(processed_players, "DF", None), 4, "robust")
    summary = pipeline.cluster_summary(model)
    prof = roles.percentile_profiles(model.cohort, model.labels, FEATURES)
    expected = " + ".join(pipeline.top_traits(prof[0]))
    assert summary.loc[summary.Cluster == 0, "Top Traits"].item() == expected


def test_aerial_dominance_names_target_forwards_and_stoppers() -> None:
    target = profile(Aerial_Win_Pct=92, Shots_p90=70, Non_Penalty_xG_p90=68, Successful_Take_Ons_p90=20,
                     Progressive_Carries_p90=20, Pass_Completion_Pct=35)
    labels = roles.positional_role_labels(np.vstack([target, STRIKER, WINGER]), FEATURES, "FW")
    assert labels[0] == "Target Forward"
    stopper = profile(Aerial_Win_Pct=90, pAdj_Interceptions_p90=85, Key_Passes_p90=20,
                      Successful_Take_Ons_p90=20, Assists_p90=25, Progressive_Carries_p90=25)
    labels = roles.positional_role_labels(np.vstack([stopper, BALL_PLAYING_CB, ATTACKING_FB]), FEATURES, "DF")
    assert labels[0] == "Stopper Centre-Back"


def test_carriers_and_progressive_playmakers_separated() -> None:
    carrier = profile(Progressive_Carries_p90=97, Successful_Take_Ons_p90=95, Key_Passes_p90=75,
                      Shots_p90=75, Pass_Completion_Pct=35, Aerial_Win_Pct=20)
    playmaker = profile(Progressive_Passes_p90=92, Key_Passes_p90=85, Assists_p90=78, Pass_Completion_Pct=62)
    winner = profile(pAdj_Tackles_Won_p90=88, pAdj_Interceptions_p90=85, Key_Passes_p90=35, Goals_p90=35)
    labels = roles.positional_role_labels(np.vstack([carrier, playmaker, winner]), FEATURES, "MF")
    assert labels == {0: "Ball-Carrying Midfielder", 1: "Progressive Playmaker", 2: "Ball-Winning Midfielder"}
