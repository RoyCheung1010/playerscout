"""Contract and realism tests for the synthetic FBref-style generator."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import (
    ALL_LEAGUES,
    DEFAULT_MIN_MINUTES,
    LEAGUE_MATCHES,
    POSITION_GROUPS,
    RAW_COUNT_COLUMNS,
    REQUIRED_RAW_COLUMNS,
    TOP_5_LEAGUES,
)
from src.data.synthetic_generator import (
    GeneratorConfig,
    SyntheticPlayerGenerator,
    generate_synthetic_players,
)


@pytest.fixture(scope="module")
def players() -> pd.DataFrame:
    return SyntheticPlayerGenerator(GeneratorConfig(include_ground_truth=True)).generate()


@pytest.fixture(scope="module")
def qualified(players: pd.DataFrame) -> pd.DataFrame:
    return players[players["Minutes_Played"] >= DEFAULT_MIN_MINUTES]


def per90(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] / df["Minutes_Played"] * 90


# --------------------------------------------------------------------------- #
# Schema & volume
# --------------------------------------------------------------------------- #
def test_has_all_required_columns(players: pd.DataFrame) -> None:
    missing = set(REQUIRED_RAW_COLUMNS) - set(players.columns)
    assert not missing


def test_volume_meets_spec(players: pd.DataFrame, qualified: pd.DataFrame) -> None:
    assert len(players) >= 500
    # Enough players survive the default minutes filter to cluster per position.
    assert len(qualified) >= 500
    for group in POSITION_GROUPS:
        assert (qualified["Position"].str.split(",").str[0] == group).sum() >= 100


def test_minutes_filter_has_something_to_remove(players: pd.DataFrame) -> None:
    assert (players["Minutes_Played"] < DEFAULT_MIN_MINUTES).sum() > 0


def test_covers_all_leagues(players: pd.DataFrame) -> None:
    assert set(players["League"]) == set(ALL_LEAGUES)


def test_no_nulls_and_unique_names(players: pd.DataFrame) -> None:
    assert not players.isna().any().any()
    assert players["Player"].is_unique


def test_primary_positions_are_valid(players: pd.DataFrame) -> None:
    primary = players["Position"].str.split(",").str[0]
    assert set(primary) <= set(POSITION_GROUPS)


# --------------------------------------------------------------------------- #
# Internal consistency
# --------------------------------------------------------------------------- #
def test_counts_are_non_negative_integers(players: pd.DataFrame) -> None:
    for col in RAW_COUNT_COLUMNS:
        assert pd.api.types.is_integer_dtype(players[col]), col
        assert (players[col] >= 0).all(), col


def test_subset_relationships(players: pd.DataFrame) -> None:
    assert (players["Goals"] <= players["Shots"]).all()
    assert (players["Assists"] <= players["Key_Passes"]).all()
    assert (players["Passes_Completed"] <= players["Passes_Attempted"]).all()
    assert (players["Successful_Take_Ons"] <= players["Take_Ons_Attempted"]).all()
    assert (players["Starts"] <= players["Matches_Played"]).all()


def test_pass_completion_matches_counts(players: pd.DataFrame) -> None:
    has_passes = players["Passes_Attempted"] > 0
    expected = (
        100 * players.loc[has_passes, "Passes_Completed"] / players.loc[has_passes, "Passes_Attempted"]
    ).round(1)
    pd.testing.assert_series_equal(
        players.loc[has_passes, "Pass_Completion_Pct"], expected, check_names=False
    )
    assert players["Pass_Completion_Pct"].between(0, 100).all()


def test_minutes_within_season_bounds(players: pd.DataFrame) -> None:
    cap = players["League"].map(LEAGUE_MATCHES) * 90
    assert (players["Minutes_Played"] <= cap).all()
    assert (players["Matches_Played"] <= players["League"].map(LEAGUE_MATCHES)).all()


def test_metadata_ranges(players: pd.DataFrame) -> None:
    assert players["Age"].between(17, 37).all()
    assert players["Team_Possession_Pct"].between(30, 70).all()
    assert (players["Market_Value_EUR_M"] > 0).all()
    assert players["Contract_Expiry_Year"].between(2027, 2031).all()


def test_possession_is_constant_within_squad(players: pd.DataFrame) -> None:
    assert (players.groupby("Squad")["Team_Possession_Pct"].nunique() == 1).all()


# --------------------------------------------------------------------------- #
# Football realism (the signals downstream models must recover)
# --------------------------------------------------------------------------- #
def test_positional_signatures(qualified: pd.DataFrame) -> None:
    primary = qualified["Position"].str.split(",").str[0]
    goals = per90(qualified, "Goals").groupby(primary).mean()
    interceptions = per90(qualified, "Interceptions").groupby(primary).mean()
    assert goals["FW"] > goals["MF"] > goals["DF"]
    assert interceptions["DF"] > interceptions["MF"] > interceptions["FW"]


def test_archetype_signatures(qualified: pd.DataFrame) -> None:
    by_arch = qualified.assign(
        take_ons=per90(qualified, "Successful_Take_Ons"),
        prog=per90(qualified, "Progressive_Passes"),
    ).groupby("Generator_Archetype")
    take_ons = by_arch["take_ons"].mean()
    prog = by_arch["prog"].mean()
    assert take_ons.idxmax() == "Touchline Dribbler"
    assert prog.idxmax() == "Deep-Lying Playmaker"


def test_rates_are_plausible(qualified: pd.DataFrame) -> None:
    goals = per90(qualified, "Goals")
    assert goals.max() < 2.0  # nobody scores two a game over a season
    assert 0.05 < goals.mean() < 0.35


def test_possession_bias_in_raw_defending(qualified: pd.DataFrame) -> None:
    """High-possession sides make fewer raw tackles - the bias pAdj corrects."""
    defenders = qualified[qualified["Position"].str.startswith("DF")]
    tackles = per90(defenders, "Tackles_Won")
    high = tackles[defenders["Team_Possession_Pct"] >= 55].mean()
    low = tackles[defenders["Team_Possession_Pct"] <= 45].mean()
    assert high < low


def test_top5_players_are_worth_more(players: pd.DataFrame) -> None:
    is_top5 = players["League"].isin(TOP_5_LEAGUES)
    assert (
        players.loc[is_top5, "Market_Value_EUR_M"].median()
        > players.loc[~is_top5, "Market_Value_EUR_M"].median()
    )


def test_contains_statistical_outliers(qualified: pd.DataFrame) -> None:
    """Elite forwards should exist so RobustScaler has something to resist."""
    goals = per90(qualified, "Goals")
    assert goals.max() > goals.quantile(0.75) + 3 * (goals.quantile(0.75) - goals.quantile(0.25))


# --------------------------------------------------------------------------- #
# Configuration & reproducibility
# --------------------------------------------------------------------------- #
def test_deterministic_for_same_seed() -> None:
    a = generate_synthetic_players(GeneratorConfig(seed=7))
    b = generate_synthetic_players(GeneratorConfig(seed=7))
    pd.testing.assert_frame_equal(a, b)


def test_repeated_generate_calls_are_identical() -> None:
    gen = SyntheticPlayerGenerator(GeneratorConfig(seed=3, clubs_per_league=2))
    pd.testing.assert_frame_equal(gen.generate(), gen.generate())


def test_different_seeds_differ() -> None:
    a = generate_synthetic_players(GeneratorConfig(seed=1))
    b = generate_synthetic_players(GeneratorConfig(seed=2))
    assert not a["Goals"].equals(b["Goals"])


def test_ground_truth_hidden_by_default() -> None:
    df = generate_synthetic_players(GeneratorConfig(clubs_per_league=1))
    assert "Generator_Archetype" not in df.columns


def test_league_subset() -> None:
    df = generate_synthetic_players(GeneratorConfig(leagues=("Eredivisie",), clubs_per_league=3))
    assert set(df["League"]) == {"Eredivisie"}
    assert df["Squad"].nunique() == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"leagues": ("MLS",)},
        {"clubs_per_league": 0},
        {"clubs_per_league": 11},
        {"regular_share": 1.5},
        {"elite_outlier_share": -0.1},
    ],
)
def test_invalid_config_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        GeneratorConfig(**kwargs)
