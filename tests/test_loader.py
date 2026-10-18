"""Tests for ingestion, filtering, per-90, pAdj and league scaling."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from src.config import CORE_FEATURES, REQUIRED_RAW_COLUMNS
from src.data.loader import (
    DataLoader,
    DataValidationError,
    LoaderConfig,
    parse_position_group,
    per90,
    possession_multiplier,
)
from src.data.synthetic_generator import GeneratorConfig


def make_row(**overrides: object) -> dict[str, object]:
    """A complete raw row at 50% possession in a Top 5 league."""
    row: dict[str, object] = {
        "Player": "Test Player",
        "Squad": "Test FC",
        "League": "Premier League",
        "Position": "MF",
        "Age": 25,
        "Minutes_Played": 1800,
        "Market_Value_EUR_M": 10.0,
        "Contract_Expiry_Year": 2028,
        "Team_Possession_Pct": 50.0,
        "Goals": 10,
        "Shots": 40,
        "Assists": 6,
        "Key_Passes": 30,
        "Pass_Completion_Pct": 85.0,
        "Progressive_Passes": 100,
        "Successful_Take_Ons": 20,
        "Tackles_Won": 30,
        "Interceptions": 20,
        "Non_Penalty_xG": 8.5,
        "Progressive_Carries": 40,
        "Aerial_Win_Pct": 55.0,
    }
    row.update(overrides)
    return row


def frame(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


@pytest.fixture
def loader() -> DataLoader:
    return DataLoader(LoaderConfig(min_minutes=900))


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def test_loads_synthetic_db_by_default(loader: DataLoader) -> None:
    df = loader.load(generator_config=GeneratorConfig(clubs_per_league=3))
    assert len(df) > 0
    assert set(CORE_FEATURES) <= set(df.columns)
    assert not df[list(CORE_FEATURES)].isna().any().any()


def test_loads_csv_path(tmp_path, loader: DataLoader) -> None:
    path = tmp_path / "players.csv"
    frame(make_row(), make_row(Player="B")).to_csv(path, index=False)
    assert len(loader.load(str(path))) == 2
    assert len(loader.load(path)) == 2  # os.PathLike


def test_loads_bytes_buffer_like_streamlit_upload(loader: DataLoader) -> None:
    csv_bytes = frame(make_row(Player="Jürgen Ödegård")).to_csv(index=False).encode("utf-8")
    df = loader.load(io.BytesIO(csv_bytes))
    assert df.loc[0, "Player"] == "Jürgen Ödegård"


def test_loads_latin1_buffer(loader: DataLoader) -> None:
    csv_bytes = frame(make_row(Player="José")).to_csv(index=False).encode("latin-1")
    assert loader.load(io.BytesIO(csv_bytes)).loc[0, "Player"] == "José"


def test_loads_text_buffer(loader: DataLoader) -> None:
    df = loader.load(io.StringIO(frame(make_row()).to_csv(index=False)))
    assert len(df) == 1


def test_missing_required_columns_raise(loader: DataLoader) -> None:
    with pytest.raises(DataValidationError, match="Interceptions"):
        loader.load(frame(make_row()).drop(columns=["Interceptions"]))


def test_empty_file_raises(loader: DataLoader) -> None:
    with pytest.raises(DataValidationError, match="empty"):
        loader.load(io.BytesIO(b""))


def test_unsupported_source_raises(loader: DataLoader) -> None:
    with pytest.raises(DataValidationError):
        loader.load(42)  # type: ignore[arg-type]


def test_column_whitespace_is_tolerated(loader: DataLoader) -> None:
    df = frame(make_row()).rename(columns={"Goals": " Goals "})
    assert len(loader.load(df)) == 1


def test_invalid_rows_dropped_and_reported(loader: DataLoader) -> None:
    df = frame(
        make_row(Player="ok"),
        make_row(Player="bad minutes", Minutes_Played="n/a"),
        make_row(Player="bad possession", Team_Possession_Pct=140),
        make_row(Player="negative", Goals=-1),
        make_row(Player=""),
    )
    out = loader.load(df)
    assert out["Player"].tolist() == ["ok"]
    assert loader.last_report.n_invalid == 4
    assert loader.last_report.n_output == 1


def test_input_frame_is_not_mutated(loader: DataLoader) -> None:
    df = frame(make_row())
    before = df.copy()
    loader.load(df)
    pd.testing.assert_frame_equal(df, before)


# --------------------------------------------------------------------------- #
# Minutes & position filtering
# --------------------------------------------------------------------------- #
def test_min_minutes_filter_is_inclusive() -> None:
    df = frame(
        make_row(Player="below", Minutes_Played=899),
        make_row(Player="exact", Minutes_Played=900),
        make_row(Player="above", Minutes_Played=2500),
    )
    loader = DataLoader(LoaderConfig(min_minutes=900))
    out = loader.load(df)
    assert sorted(out["Player"]) == ["above", "exact"]
    assert loader.last_report.n_below_min_minutes == 1


def test_min_minutes_is_configurable() -> None:
    df = frame(make_row(Minutes_Played=500))
    assert DataLoader(LoaderConfig(min_minutes=900)).load(df).empty
    assert len(DataLoader(LoaderConfig(min_minutes=0)).load(df)) == 1


def test_everything_filtered_returns_empty_frame_with_schema(loader: DataLoader) -> None:
    out = loader.load(frame(make_row(Minutes_Played=10)))
    assert out.empty
    assert set(CORE_FEATURES) <= set(out.columns)


def test_negative_min_minutes_rejected() -> None:
    with pytest.raises(ValueError):
        LoaderConfig(min_minutes=-1)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("FW,MF", "FW"),
        ("MF,FW", "MF"),
        ("DF/MF", "DF"),
        ("df", "DF"),
        (" MF ", "MF"),
        ("FW-MF", "FW"),
        ("MF DF", "MF"),
    ],
)
def test_parse_position_group(raw: str, expected: str) -> None:
    assert parse_position_group(pd.Series([raw])).iloc[0] == expected


@pytest.mark.parametrize("raw", ["GK", "", None, "XX,FW"])
def test_unparseable_positions_become_nan(raw: object) -> None:
    assert pd.isna(parse_position_group(pd.Series([raw], dtype=object)).iloc[0])


def test_goalkeepers_dropped_and_reported(loader: DataLoader) -> None:
    out = loader.load(frame(make_row(Position="GK"), make_row(Player="B", Position="DF,MF")))
    assert out["Position_Group"].tolist() == ["DF"]
    assert loader.last_report.n_unknown_position == 1


# --------------------------------------------------------------------------- #
# Per-90
# --------------------------------------------------------------------------- #
def test_per90_math(loader: DataLoader) -> None:
    out = loader.load(frame(make_row(Minutes_Played=1800, Goals=10, Shots=40)))
    # 1800 mins = 20 nineties
    assert out.loc[0, "Goals_p90"] == pytest.approx(0.5)
    assert out.loc[0, "Shots_p90"] == pytest.approx(2.0)


def test_per90_zero_minutes_guard() -> None:
    result = per90(pd.Series([5.0, 3.0]), pd.Series([0.0, 90.0]))
    assert result.tolist() == [0.0, 3.0]
    out = DataLoader(LoaderConfig(min_minutes=0)).load(frame(make_row(Minutes_Played=0)))
    assert np.isfinite(out[list(CORE_FEATURES)].to_numpy()).all()


# --------------------------------------------------------------------------- #
# Possession adjustment
# --------------------------------------------------------------------------- #
def test_possession_multiplier_direction() -> None:
    assert possession_multiplier(50.0) == pytest.approx(1.0)
    assert possession_multiplier(60.0) > 1.0
    assert possession_multiplier(40.0) < 1.0
    # Symmetric around 50 and bounded within (0, 2).
    assert possession_multiplier(60.0) + possession_multiplier(40.0) == pytest.approx(2.0)
    extremes = possession_multiplier(np.array([0.0, 100.0]))
    assert (extremes > 0).all() and (extremes < 2).all()


def test_padj_steepness_controls_strength() -> None:
    gentle = possession_multiplier(60.0, steepness=0.05)
    strong = possession_multiplier(60.0, steepness=0.1)
    assert 1.0 < gentle < strong
    assert gentle == pytest.approx(1.2449, abs=1e-4)
    assert possession_multiplier(70.0, steepness=0.0) == pytest.approx(1.0)


def test_padj_steepness_is_configurable() -> None:
    df = frame(make_row(Team_Possession_Pct=60.0))
    default = DataLoader().load(df)
    strong = DataLoader(LoaderConfig(padj_steepness=0.1)).load(df)
    off = DataLoader(LoaderConfig(padj_steepness=0.0)).load(df)
    assert LoaderConfig().padj_steepness == 0.05
    assert strong.loc[0, "pAdj_Tackles_Won_p90"] > default.loc[0, "pAdj_Tackles_Won_p90"]
    assert off.loc[0, "pAdj_Tackles_Won_p90"] == pytest.approx(off.loc[0, "Tackles_Won_p90"])


def test_negative_padj_steepness_rejected() -> None:
    with pytest.raises(ValueError, match="padj_steepness"):
        LoaderConfig(padj_steepness=-0.1)


def test_padj_applied_to_defensive_stats(loader: DataLoader) -> None:
    out = loader.load(
        frame(
            make_row(Player="high", Team_Possession_Pct=60.0),
            make_row(Player="low", Team_Possession_Pct=40.0),
        )
    ).set_index("Player")
    expected_mult = 2.0 / (1.0 + np.exp(-0.05 * (60.0 - 50.0)))  # default steepness
    raw_tackles_p90 = 30 / 1800 * 90
    assert out.loc["high", "pAdj_Tackles_Won_p90"] == pytest.approx(raw_tackles_p90 * expected_mult)
    assert out.loc["high", "pAdj_Tackles_Won_p90"] > out.loc["high", "Tackles_Won_p90"]
    assert out.loc["low", "pAdj_Interceptions_p90"] < out.loc["low", "Interceptions_p90"]


def test_padj_does_not_touch_attacking_stats(loader: DataLoader) -> None:
    out = loader.load(
        frame(make_row(Player="a", Team_Possession_Pct=65.0), make_row(Player="b", Team_Possession_Pct=35.0))
    )
    assert out["Goals_p90"].nunique() == 1


# --------------------------------------------------------------------------- #
# League scaling
# --------------------------------------------------------------------------- #
def _two_leagues(apply: bool) -> pd.DataFrame:
    df = frame(
        make_row(Player="top5", League="Premier League"),
        make_row(Player="secondary", League="Eredivisie"),
    )
    return DataLoader(LoaderConfig(apply_league_scaling=apply)).load(df).set_index("Player")


def test_league_scaling_discounts_secondary_volume_stats() -> None:
    out = _two_leagues(apply=True)
    volume = [f for f in CORE_FEATURES if f.endswith("_p90")]
    ratio = out.loc["secondary", volume] / out.loc["top5", volume]
    assert np.allclose(ratio.to_numpy(dtype=float), 0.85)
    assert out.loc["secondary", "League_Coefficient"] == pytest.approx(0.85)


def test_league_scaling_leaves_pass_completion_untouched() -> None:
    out = _two_leagues(apply=True)
    assert out.loc["secondary", "Pass_Completion_Pct"] == pytest.approx(85.0)
    assert out.loc["top5", "Pass_Completion_Pct"] == pytest.approx(85.0)
    assert out.loc["secondary", "Aerial_Win_Pct"] == pytest.approx(55.0)  # rates are never scaled


def test_league_scaling_can_be_disabled() -> None:
    out = _two_leagues(apply=False)
    assert out.loc["secondary", "Goals_p90"] == pytest.approx(out.loc["top5", "Goals_p90"])
    assert (out["League_Coefficient"] == 1.0).all()


def test_unknown_league_uses_default_and_is_reported() -> None:
    loader = DataLoader(LoaderConfig(default_league_coefficient=0.7))
    out = loader.load(frame(make_row(League="MLS")))
    assert out.loc[0, "League_Coefficient"] == pytest.approx(0.7)
    assert loader.last_report.unknown_leagues == ["MLS"]


def test_full_pipeline_on_synthetic_schema_round_trip(tmp_path) -> None:
    """A synthetic export written to CSV loads identically to the in-memory frame."""
    loader = DataLoader()
    raw = loader.read(generator_config=GeneratorConfig(clubs_per_league=2))
    path = tmp_path / "export.csv"
    raw.to_csv(path, index=False)
    from_memory = loader.process(raw)
    from_csv = loader.load(path)
    assert set(REQUIRED_RAW_COLUMNS) <= set(raw.columns)
    np.testing.assert_allclose(
        from_memory[list(CORE_FEATURES)].to_numpy(), from_csv[list(CORE_FEATURES)].to_numpy()
    )


def test_new_volume_stats_converted_to_per90(loader: DataLoader) -> None:
    out = loader.process(frame(make_row()))
    assert out.loc[0, "Non_Penalty_xG_p90"] == pytest.approx(8.5 / 1800 * 90)
    assert out.loc[0, "Progressive_Carries_p90"] == pytest.approx(40 / 1800 * 90)
    assert out.loc[0, "Aerial_Win_Pct"] == pytest.approx(55.0)


def test_aerial_rate_out_of_range_rows_dropped(loader: DataLoader) -> None:
    out = loader.process(frame(make_row(), make_row(Player="Bad", Aerial_Win_Pct=140.0)))
    assert list(out["Player"]) == ["Test Player"]
