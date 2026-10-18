"""Tests for feature scaling, positional percentiles and composite scores."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import percentileofscore
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.config import COMPOSITE_SCORES, CORE_FEATURES
from src.data.loader import DataLoader
from src.data.synthetic_generator import GeneratorConfig
from src.features.preprocessor import (
    InsufficientDataError,
    ScoutingPreprocessor,
    ecdf_percentile,
)


@pytest.fixture(scope="module")
def players() -> pd.DataFrame:
    return DataLoader().load()


@pytest.fixture(scope="module")
def prep() -> ScoutingPreprocessor:
    return ScoutingPreprocessor()


@pytest.fixture(scope="module")
def scored(players: pd.DataFrame, prep: ScoutingPreprocessor) -> pd.DataFrame:
    return prep.transform(players)


def toy_cohort() -> pd.DataFrame:
    """Three defenders and three forwards with controlled tackle numbers."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.uniform(0.1, 1.0, size=(6, len(CORE_FEATURES))), columns=list(CORE_FEATURES))
    df["Position_Group"] = ["DF", "DF", "DF", "FW", "FW", "FW"]
    # The defender's 2.0 is low against forwards' 5.0+, but best among defenders.
    df["pAdj_Tackles_Won_p90"] = [2.0, 1.0, 0.5, 5.0, 6.0, 7.0]
    return df


# --------------------------------------------------------------------------- #
# Scaling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("scaler_type", "cls"), [("standard", StandardScaler), ("robust", RobustScaler)])
def test_scaler_shapes_and_types(
    players: pd.DataFrame, prep: ScoutingPreprocessor, scaler_type: str, cls: type
) -> None:
    X, scaler = prep.scale_features(players, scaler_type)  # type: ignore[arg-type]
    assert X.shape == (len(players), len(CORE_FEATURES))
    assert isinstance(scaler, cls)
    assert np.isfinite(X).all()


def test_standard_scaler_is_zero_mean_unit_variance(players: pd.DataFrame, prep: ScoutingPreprocessor) -> None:
    X, _ = prep.scale_features(players, "standard")
    np.testing.assert_allclose(X.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(X.std(axis=0), 1.0, atol=1e-9)


def test_robust_scaler_centres_on_median(players: pd.DataFrame, prep: ScoutingPreprocessor) -> None:
    X, _ = prep.scale_features(players, "robust")
    np.testing.assert_allclose(np.median(X, axis=0), 0.0, atol=1e-9)


def test_robust_scaler_resists_outliers(prep: ScoutingPreprocessor) -> None:
    df = toy_cohort()
    df["Goals_p90"] = [0.1, 0.2, 0.3, 0.4, 0.5, 50.0]  # one absurd outlier
    goals = list(CORE_FEATURES).index("Goals_p90")
    X_std, _ = prep.scale_features(df, "standard")
    X_rob, _ = prep.scale_features(df, "robust")
    # Standard scaling squashes the normal players together; robust keeps them apart.
    assert np.ptp(X_rob[:5, goals]) > np.ptp(X_std[:5, goals])


def test_zero_variance_feature_does_not_produce_nan(prep: ScoutingPreprocessor) -> None:
    df = toy_cohort()
    df["Pass_Completion_Pct"] = 80.0
    for scaler_type in ("standard", "robust"):
        X, _ = prep.scale_features(df, scaler_type)  # type: ignore[arg-type]
        assert np.isfinite(X).all()


def test_nan_values_are_imputed(prep: ScoutingPreprocessor) -> None:
    df = toy_cohort()
    df.loc[0, "Goals_p90"] = np.nan
    X, _ = prep.scale_features(df)
    assert np.isfinite(X).all()


def test_empty_cohort_raises_clear_error(prep: ScoutingPreprocessor, players: pd.DataFrame) -> None:
    with pytest.raises(InsufficientDataError):
        prep.scale_features(players.iloc[0:0])


def test_unknown_scaler_rejected(prep: ScoutingPreprocessor, players: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="scaler_type"):
        prep.scale_features(players, "minmax")  # type: ignore[arg-type]


def test_missing_feature_column_rejected(prep: ScoutingPreprocessor, players: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Goals_p90"):
        prep.scale_features(players.drop(columns=["Goals_p90"]))


# --------------------------------------------------------------------------- #
# Percentiles
# --------------------------------------------------------------------------- #
def test_ecdf_matches_scipy_mean_kind() -> None:
    values = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 5.0])
    expected = [percentileofscore(values, v, kind="mean") for v in values]
    np.testing.assert_allclose(ecdf_percentile(values), expected)


def test_ecdf_edge_cases() -> None:
    assert ecdf_percentile(np.array([])).size == 0
    assert ecdf_percentile(np.array([3.0])).tolist() == [50.0]
    np.testing.assert_allclose(ecdf_percentile(np.array([1.0, 1.0, 1.0])), 50.0)
    out = ecdf_percentile(np.array([1.0, np.nan, 2.0]))
    assert np.isnan(out[1]) and out[0] < out[2]


def test_percentiles_bounded(scored: pd.DataFrame, prep: ScoutingPreprocessor) -> None:
    pct = scored[prep.percentile_columns]
    assert not pct.isna().any().any()
    assert ((pct >= 0.0) & (pct <= 100.0)).all().all()


def test_percentiles_are_isolated_by_position_group(prep: ScoutingPreprocessor) -> None:
    out = prep.add_positional_percentiles(toy_cohort())
    col = "pAdj_Tackles_Won_p90_pct"
    best_defender = out.loc[0, col]
    # Best tackler among defenders despite being below every forward in raw terms.
    assert best_defender == out.loc[out["Position_Group"] == "DF", col].max()
    assert best_defender > 80.0
    # Pooled ranking would have put this defender in the bottom half.
    pooled = ecdf_percentile(toy_cohort()["pAdj_Tackles_Won_p90"])
    assert pooled[0] < 50.0


def test_each_group_spans_its_own_percentile_range(scored: pd.DataFrame) -> None:
    for _, cohort in scored.groupby("Position_Group"):
        col = cohort["Goals_p90_pct"]
        assert col.min() < 20.0 and col.max() > 80.0


def test_defender_with_high_tackles_ranks_high_in_df(scored: pd.DataFrame) -> None:
    defenders = scored[scored["Position_Group"] == "DF"]
    top_tackler = defenders.loc[defenders["pAdj_Tackles_Won_p90"].idxmax()]
    assert top_tackler["pAdj_Tackles_Won_p90_pct"] > 99.0


def test_percentiles_do_not_mutate_input(prep: ScoutingPreprocessor) -> None:
    df = toy_cohort()
    before = df.copy()
    prep.add_positional_percentiles(df)
    pd.testing.assert_frame_equal(df, before)


# --------------------------------------------------------------------------- #
# Composite scores
# --------------------------------------------------------------------------- #
def test_composite_columns_exist_and_are_bounded(scored: pd.DataFrame) -> None:
    cols = list(COMPOSITE_SCORES.values())
    assert cols == ["Score_Finishing", "Score_Creation", "Score_Dribbling", "Score_Defending"]
    scores = scored[cols]
    assert not scores.isna().any().any()
    assert ((scores >= 0.0) & (scores <= 100.0)).all().all()


def test_composite_is_mean_of_member_percentiles(scored: pd.DataFrame) -> None:
    expected = scored[["Goals_p90_pct", "Shots_p90_pct", "Non_Penalty_xG_p90_pct"]].mean(axis=1)
    pd.testing.assert_series_equal(scored["Score_Finishing"], expected, check_names=False)
    dribbling = scored[["Successful_Take_Ons_p90_pct", "Progressive_Carries_p90_pct"]].mean(axis=1)
    pd.testing.assert_series_equal(scored["Score_Dribbling"], dribbling, check_names=False)


def test_composites_compute_percentiles_if_missing(players: pd.DataFrame, prep: ScoutingPreprocessor) -> None:
    out = prep.add_composite_scores(players)
    assert set(prep.percentile_columns) <= set(out.columns)


def test_composites_reflect_football_roles(scored: pd.DataFrame) -> None:
    by_group = scored.groupby("Position_Group")
    # Within-group percentiles centre every group near 50; extremes still differ.
    assert by_group["Score_Finishing"].mean().between(40, 60).all()
    raw = DataLoader().load(generator_config=GeneratorConfig(include_ground_truth=True))
    raw = ScoutingPreprocessor().transform(raw)

    def best_archetype(group: str, score: str) -> str:
        cohort = raw[raw["Position_Group"] == group]
        return str(cohort.groupby("Generator_Archetype")[score].mean().idxmax())

    # Scores are positional, so the leader is judged within each cohort.
    assert best_archetype("FW", "Score_Dribbling") == "Touchline Dribbler"
    assert best_archetype("DF", "Score_Dribbling") == "Attacking Full-Back"
    assert best_archetype("MF", "Score_Defending") == "Defensive Destroyer"
    assert best_archetype("FW", "Score_Finishing") == "Penalty-Box Poacher"


def test_invalid_category_config_rejected() -> None:
    with pytest.raises(ValueError):
        ScoutingPreprocessor(categories={"Scoring": ("xG_p90",)}, composite_names={"Scoring": "S"})
    with pytest.raises(ValueError):
        ScoutingPreprocessor(composite_names={"Scoring": "S"})
