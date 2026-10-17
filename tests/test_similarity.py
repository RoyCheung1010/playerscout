"""Tests for the weighted replacement-search engine and recruitment filters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import CORE_FEATURES
from src.exceptions import TargetPlayerNotFoundError
from src.models.similarity import RESULT_COLUMNS, ScoutingSimilarityEngine

FEATURES = list(CORE_FEATURES)


@pytest.fixture(scope="module")
def engine() -> ScoutingSimilarityEngine:
    return ScoutingSimilarityEngine()


@pytest.fixture(scope="module")
def target(processed_players: pd.DataFrame) -> str:
    """An expensive, high-minutes forward - a classic replacement brief."""
    forwards = processed_players[processed_players["Position_Group"] == "FW"]
    return str(forwards.sort_values("Market_Value_EUR_M").iloc[-5]["Player"])


def toy_frame(n: int = 6) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "Player": [f"P{i}" for i in range(n)],
        "Squad": [f"Club {i}" for i in range(n)],
        "League": ["Premier League"] * (n - 1) + ["Eredivisie"],
        "Position_Group": ["FW"] * n,
        "Age": [30, 21, 22, 25, 28, 19][:n],
        "Market_Value_EUR_M": [80.0, 5.0, 20.0, 45.0, 10.0, 2.0][:n],
        "Contract_Expiry_Year": [2030, 2027, 2028, 2029, 2027, 2031][:n],
    })
    X = rng.normal(size=(n, len(FEATURES)))
    return df, X


# --------------------------------------------------------------------------- #
# Output contract
# --------------------------------------------------------------------------- #
def test_result_columns_and_ranking(
    engine: ScoutingSimilarityEngine, processed_players: pd.DataFrame, scaled_matrix: np.ndarray, target: str
) -> None:
    out = engine.search(target, processed_players, scaled_matrix)
    assert list(out.columns) == list(RESULT_COLUMNS)
    assert len(out) == 10
    assert out["Rank"].tolist() == list(range(1, 11))
    assert out["Match_Score_Pct"].is_monotonic_decreasing
    assert out["Match_Score_Pct"].between(0, 100).all()


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_target_never_in_own_list(
    engine: ScoutingSimilarityEngine,
    processed_players: pd.DataFrame,
    scaled_matrix: np.ndarray,
    target: str,
    metric: str,
) -> None:
    out = engine.search(target, processed_players, scaled_matrix, metric=metric, top_n=500)  # type: ignore[arg-type]
    assert target not in set(out["Player"])


def test_value_diff_is_target_minus_candidate(
    engine: ScoutingSimilarityEngine, processed_players: pd.DataFrame, scaled_matrix: np.ndarray, target: str
) -> None:
    out = engine.search(target, processed_players, scaled_matrix)
    target_value = processed_players.loc[processed_players["Player"] == target, "Market_Value_EUR_M"].iloc[0]
    np.testing.assert_allclose(out["Value_Diff_EUR_M"], (target_value - out["Market_Value_EUR_M"]).round(2))


def test_same_position_only(
    engine: ScoutingSimilarityEngine, processed_players: pd.DataFrame, scaled_matrix: np.ndarray, target: str
) -> None:
    out = engine.search(target, processed_players, scaled_matrix, top_n=1000)
    assert set(out["Position_Group"]) == {"FW"}
    mixed = engine.search(target, processed_players, scaled_matrix, same_position_only=False, top_n=1000)
    assert len(mixed) == len(processed_players) - 1


def test_assigned_role_passthrough(engine: ScoutingSimilarityEngine) -> None:
    df, X = toy_frame()
    assert (engine.search("P0", df, X)["Assigned_Role"] == "Unassigned").all()
    df["Assigned_Role"] = [f"Role {i}" for i in range(len(df))]
    out = engine.search("P0", df, X)
    assert set(out["Assigned_Role"]) <= set(df["Assigned_Role"])


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_identical_vector_scores_100(engine: ScoutingSimilarityEngine, metric: str) -> None:
    df, X = toy_frame()
    X[3] = X[0]  # P3 is a statistical clone of P0
    out = engine.search("P0", df, X, metric=metric)  # type: ignore[arg-type]
    assert out.iloc[0]["Player"] == "P3"
    assert out.iloc[0]["Match_Score_Pct"] == pytest.approx(100.0)


def test_cosine_ignores_volume_euclidean_does_not(engine: ScoutingSimilarityEngine) -> None:
    df, X = toy_frame()
    X[1] = X[0] * 0.4  # same style, 40% of the output
    cos = engine.search("P0", df, X, metric="cosine").set_index("Player")["Match_Score_Pct"]
    euc = engine.search("P0", df, X, metric="euclidean").set_index("Player")["Match_Score_Pct"]
    assert cos["P1"] == pytest.approx(100.0)
    assert euc["P1"] < 100.0


def test_opposite_profile_scores_zero_cosine(engine: ScoutingSimilarityEngine) -> None:
    df, X = toy_frame()
    X[2] = -X[0]
    scores = engine.search("P0", df, X, metric="cosine", top_n=10).set_index("Player")["Match_Score_Pct"]
    assert scores["P2"] == pytest.approx(0.0)


def test_scores_unchanged_by_recruitment_filters(
    engine: ScoutingSimilarityEngine, processed_players: pd.DataFrame, scaled_matrix: np.ndarray, target: str
) -> None:
    for metric in ("cosine", "euclidean"):
        full = engine.search(target, processed_players, scaled_matrix, metric=metric, top_n=1000)  # type: ignore[arg-type]
        cheap = engine.search(target, processed_players, scaled_matrix, metric=metric, max_market_value=10.0)  # type: ignore[arg-type]
        merged = cheap.merge(full, on="Player", suffixes=("_cheap", "_full"))
        np.testing.assert_allclose(merged["Match_Score_Pct_cheap"], merged["Match_Score_Pct_full"])


def test_zero_vectors_are_handled(engine: ScoutingSimilarityEngine) -> None:
    df, X = toy_frame()
    X[:] = 0.0
    for metric in ("cosine", "euclidean"):
        out = engine.search("P0", df, X, metric=metric)  # type: ignore[arg-type]
        assert np.isfinite(out["Match_Score_Pct"]).all()


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #
def test_default_weights_are_ones(engine: ScoutingSimilarityEngine) -> None:
    np.testing.assert_array_equal(engine.feature_weight_vector(), np.ones(len(FEATURES)))


def test_category_weights_expand_to_features(engine: ScoutingSimilarityEngine) -> None:
    w = engine.feature_weight_vector({"Scoring": 3.0, "Creation": 0.5, "Defending": 0.0})
    expected = {
        "Goals_p90": 3.0, "Shots_p90": 3.0,
        "Assists_p90": 0.5, "Key_Passes_p90": 0.5, "Pass_Completion_Pct": 0.5,
        "Progressive_Passes_p90": 0.5,
        "Successful_Take_Ons_p90": 1.0,
        "pAdj_Tackles_Won_p90": 0.0, "pAdj_Interceptions_p90": 0.0,
    }
    np.testing.assert_array_equal(w, [expected[f] for f in FEATURES])


def test_sqrt_weighting_makes_squared_distance_weighted_by_w(engine: ScoutingSimilarityEngine) -> None:
    u, v = np.ones(len(FEATURES)), np.zeros(len(FEATURES))
    w = engine.feature_weight_vector({"Scoring": 4.0})
    d2 = np.sum(((u - v) * np.sqrt(w)) ** 2)
    assert d2 == pytest.approx(np.sum(w * (u - v) ** 2))


def test_weights_change_the_ranking(engine: ScoutingSimilarityEngine) -> None:
    df, X = toy_frame()
    goals, tackles = FEATURES.index("Goals_p90"), FEATURES.index("pAdj_Tackles_Won_p90")
    X[:] = 0.0
    X[0, [goals, tackles]] = [2.0, 2.0]
    X[1, goals] = 2.0    # matches the target's scoring only
    X[2, tackles] = 2.0  # matches the target's defending only
    scoring_first = engine.search("P0", df, X, feature_weights={"Scoring": 5.0, "Defending": 0.2})
    defending_first = engine.search("P0", df, X, feature_weights={"Scoring": 0.2, "Defending": 5.0})
    assert scoring_first.iloc[0]["Player"] == "P1"
    assert defending_first.iloc[0]["Player"] == "P2"


@pytest.mark.parametrize(
    "weights",
    [{"Vibes": 1.0}, {"Scoring": -1.0}, {"Scoring": float("nan")},
     {"Scoring": 0, "Creation & Passing": 0, "Dribbling": 0, "Defending": 0}],
)
def test_invalid_weights_rejected(engine: ScoutingSimilarityEngine, weights: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        engine.feature_weight_vector(weights)


# --------------------------------------------------------------------------- #
# Recruitment filters
# --------------------------------------------------------------------------- #
def test_max_age_and_value_strictly_prune(
    engine: ScoutingSimilarityEngine, processed_players: pd.DataFrame, scaled_matrix: np.ndarray, target: str
) -> None:
    out = engine.search(target, processed_players, scaled_matrix, max_age=23, max_market_value=15.0, top_n=1000)
    assert len(out) > 0
    assert (out["Age"] <= 23).all()
    assert (out["Market_Value_EUR_M"] <= 15.0).all()
    # Nothing in-bounds was lost: every eligible forward is present.
    pool = processed_players[
        (processed_players["Position_Group"] == "FW")
        & (processed_players["Age"] <= 23)
        & (processed_players["Market_Value_EUR_M"] <= 15.0)
        & (processed_players["Player"] != target)
    ]
    assert set(out["Player"]) == set(pool["Player"])


def test_contract_filter_counts_years_from_reference(engine: ScoutingSimilarityEngine) -> None:
    df, X = toy_frame()
    out = engine.search("P0", df, X, contract_expiry_years_max=1)  # expires by 2027
    assert set(out["Player"]) == {"P1", "P4"}
    shifted = ScoutingSimilarityEngine(reference_year=2027).search("P0", df, X, contract_expiry_years_max=1)
    assert set(shifted["Player"]) == {"P1", "P2", "P4"}


def test_excluded_leagues(engine: ScoutingSimilarityEngine) -> None:
    df, X = toy_frame()
    out = engine.search("P0", df, X, excluded_leagues=["Eredivisie"])
    assert "Eredivisie" not in set(out["League"])


def test_top_n_limits_rows(engine: ScoutingSimilarityEngine) -> None:
    df, X = toy_frame()
    assert len(engine.search("P0", df, X, top_n=2)) == 2
    with pytest.raises(ValueError):
        engine.search("P0", df, X, top_n=0)


def test_no_candidates_returns_empty_frame(engine: ScoutingSimilarityEngine) -> None:
    df, X = toy_frame()
    out = engine.search("P0", df, X, max_age=16)
    assert out.empty
    assert list(out.columns) == list(RESULT_COLUMNS)


def test_single_player_cohort_returns_empty(engine: ScoutingSimilarityEngine) -> None:
    df, X = toy_frame(1)
    assert engine.search("P0", df, X).empty


# --------------------------------------------------------------------------- #
# Target lookup & input validation
# --------------------------------------------------------------------------- #
def test_missing_target_raises(engine: ScoutingSimilarityEngine) -> None:
    df, X = toy_frame()
    with pytest.raises(TargetPlayerNotFoundError):
        engine.search("Nobody", df, X)


def test_target_lookup_is_case_insensitive(engine: ScoutingSimilarityEngine) -> None:
    df, X = toy_frame()
    assert "P0" not in set(engine.search(" p0 ", df, X)["Player"])


def test_ambiguous_target_needs_squad(engine: ScoutingSimilarityEngine) -> None:
    df, X = toy_frame()
    df.loc[1, "Player"] = "P0"
    with pytest.raises(ValueError, match="ambiguous"):
        engine.search("P0", df, X)
    out = engine.search("P0", df, X, target_squad="Club 1")
    assert "Club 1" not in set(out["Squad"])
    assert "Club 0" in set(out["Squad"])


def test_non_default_index_is_supported(engine: ScoutingSimilarityEngine) -> None:
    df, X = toy_frame()
    df.index = [100, 7, 55, 3, 9, 42]
    assert len(engine.search("P0", df, X)) == 5


def test_misaligned_inputs_rejected(engine: ScoutingSimilarityEngine) -> None:
    df, X = toy_frame()
    with pytest.raises(ValueError):
        engine.search("P0", df, X[:-1])
    with pytest.raises(ValueError):
        engine.search("P0", df, X, metric="manhattan")  # type: ignore[arg-type]


def test_match_scores_for_covers_every_row(engine: ScoutingSimilarityEngine) -> None:
    df, X = toy_frame()
    scores = engine.match_scores_for("P0", df, X)
    assert scores.index.equals(df.index)
    assert scores.iloc[0] == pytest.approx(100.0)
