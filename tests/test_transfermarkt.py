"""Transfermarkt enrichment: name matching, season-end values, listed contracts.

Fixtures mirror ``davidcariboo/player-scores``: ``players.csv`` (date of
birth, listed contract, ``last_season``) and ``player_valuations.csv`` (dated
values tagged with the domestic league at the time).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import pipeline
from src.data import transfermarkt as tm
from src.data.real_data_enricher import CONTRACT_SOURCE_COLUMN, SOURCE_PROXY, VALUE_SOURCE_COLUMN

SEASON = "2023-2024"

TM_PLAYERS = pd.DataFrame(
    [
        # player_id, name, date_of_birth, contract_expiration_date, last_season
        (1, "Harry Kane", "1993-07-28", "2027-06-30", 2025),
        (2, "Min-jae Kim", "1996-11-15", "2028-06-30", 2025),
        (3, "Gabriel", "1997-12-19", "2029-06-30", 2025),
        (4, "Alejandro Grimaldo", "1995-09-20", "2027-06-30", 2025),
        (5, "Rodri", "1996-06-22", "2027-06-30", 2025),
        (6, "Rodri Sánchez", "2000-05-16", "2028-06-30", 2025),
        (7, "Retired Veteran", "1985-01-01", "2024-06-30", 2023),  # no longer tracked
        (8, "Twin Smith", "1999-01-01", "2026-06-30", 2025),
        (9, "Twin Smith", "1999-05-05", "2027-06-30", 2025),  # same name & year, other league
        (10, "Gabriel Strefezza", "1997-04-18", "2027-06-30", 2025),  # different birth year
    ],
    columns=["player_id", "name", "date_of_birth", "contract_expiration_date", "last_season"],
)

TM_VALUATIONS = pd.DataFrame(
    [
        # player_id, date, value, league
        (1, "2023-05-01", 90_000_000, "GB1"),
        (1, "2023-12-15", 110_000_000, "L1"),
        (1, "2024-06-01", 100_000_000, "L1"),  # season-end value
        (1, "2025-06-01", 75_000_000, "L1"),  # after the season: ignored
        (2, "2024-03-01", 50_000_000, "L1"),
        (3, "2024-05-01", 65_000_000, "GB1"),
        (4, "2024-05-01", 40_000_000, "L1"),
        (5, "2024-05-01", 120_000_000, "GB1"),
        (6, "2024-05-01", 8_000_000, "ES1"),
        (7, "2024-02-01", 1_000_000, "ES1"),
        (8, "2024-02-01", 5_000_000, "GB1"),
        (9, "2024-02-01", 6_000_000, "IT1"),
        (10, "2024-02-01", 9_000_000, "IT1"),
        (1, "2026-06-12", 70_000_000, "L1"),  # snapshot date
    ],
    columns=["player_id", "date", "market_value_in_eur", "player_club_domestic_competition_id"],
)

OURS = pd.DataFrame(
    [
        # Player, Born, League, Squad, proxy value, proxy contract
        ("Harry Kane", 1993, "Bundesliga", "Bayern Munich", 44.9, 2028),
        ("Kim Min-jae", 1996, "Bundesliga", "Bayern Munich", 20.0, 2027),
        ("Gabriel Magalhães", 1997, "Premier League", "Arsenal", 30.0, 2026),
        ("Álex Grimaldo", 1995, "Bundesliga", "Leverkusen", 25.0, 2029),
        ("Rodri", 1996, "Premier League", "Manchester City", 90.0, 2026),
        ("Rodri", 2000, "La Liga", "Betis", 5.3, 2026),
        ("Retired Veteran", 1985, "La Liga", "Getafe", 0.5, 2027),
        ("Twin Smith", 1999, "Premier League", "Fulham", 3.0, 2028),
        ("Nobody Known", 2001, "Ligue 1", "Lens", 2.0, 2029),
    ],
    columns=["Player", "Born", "League", "Squad", "Market_Value_EUR_M", "Contract_Expiry_Year"],
)


@pytest.fixture()
def enriched() -> pd.DataFrame:
    frame = OURS.copy()
    frame[VALUE_SOURCE_COLUMN] = SOURCE_PROXY
    frame[CONTRACT_SOURCE_COLUMN] = SOURCE_PROXY
    return frame


@pytest.fixture()
def attached(enriched: pd.DataFrame) -> tuple[pd.DataFrame, tm.MatchReport]:
    return tm.attach_transfermarkt(enriched, TM_PLAYERS, TM_VALUATIONS, SEASON)


def row(df: pd.DataFrame, player: str, squad: str | None = None) -> pd.Series:
    mask = df["Player"] == player
    if squad is not None:
        mask &= df["Squad"] == squad
    return df[mask].iloc[0]


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        ("Kim Min-jae", "Min-jae Kim"),
        ("Ohis Felix Uduokhai", "Felix Uduokhai"),
        ("Gio Reyna", "Giovanni Reyna"),
        ("Álex Grimaldo", "Alejandro Grimaldo"),
        ("Gabriel Magalhães", "Gabriel"),
        ("Nacho", "Nacho Fernández"),
        ("Przemysław Frankowski", "Przemyslaw Frankowski"),
    ],
)
def test_names_compatible_accepts_known_variants(ours: str, theirs: str) -> None:
    assert tm.names_compatible(tm.name_key(ours), tm.name_key(theirs))


@pytest.mark.parametrize(
    ("ours", "theirs"),
    [("Harry Kane", "Harry Maguire"), ("Leo", "Leonardo Bonucci"), ("Jordan Henderson", "Jordan Pickford")],
)
def test_names_compatible_rejects_different_players(ours: str, theirs: str) -> None:
    assert not tm.names_compatible(tm.name_key(ours), tm.name_key(theirs))


def test_name_key_strips_accents_and_punctuation() -> None:
    assert tm.name_key("Pierre-Emile Højbjerg") == "pierre emile hjbjerg"
    assert tm.name_key("N'Golo  Kanté") == "n golo kante"


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def test_exact_and_fuzzy_matches(attached: tuple[pd.DataFrame, tm.MatchReport]) -> None:
    out, report = attached
    assert row(out, "Harry Kane")[tm.TM_MATCH_COLUMN] == "exact"
    for player in ("Kim Min-jae", "Gabriel Magalhães", "Álex Grimaldo"):
        assert row(out, player)[tm.TM_MATCH_COLUMN] == "fuzzy", player
    assert row(out, "Nobody Known")[tm.TM_MATCH_COLUMN] == "unmatched"
    assert report.n_exact + report.n_fuzzy == report.n_matched


def test_namesakes_matched_by_birth_year(attached: tuple[pd.DataFrame, tm.MatchReport]) -> None:
    out, _ = attached
    assert row(out, "Rodri", "Manchester City")[tm.TM_ID_COLUMN] == 5
    assert row(out, "Rodri", "Betis")[tm.TM_ID_COLUMN] == 6


def test_same_name_and_year_resolved_by_league(attached: tuple[pd.DataFrame, tm.MatchReport]) -> None:
    out, _ = attached
    assert row(out, "Twin Smith")[tm.TM_ID_COLUMN] == 8  # the Premier League one


def test_fuzzy_never_crosses_birth_years(attached: tuple[pd.DataFrame, tm.MatchReport]) -> None:
    out, _ = attached
    assert 10 not in set(out[tm.TM_ID_COLUMN].dropna())


def test_one_transfermarkt_player_never_claimed_twice() -> None:
    ours = pd.DataFrame({"Player": ["Rodri", "Rodri Hernández"], "Born": [1996, 1996],
                         "League": ["Premier League", "Premier League"]})
    facts = pd.DataFrame({"player_id": [5], "key": ["rodri"], "born": [1996], "league": ["Premier League"],
                          "value_eur_m": [120.0], "contract_year": [2027], "last_season": [2025]})
    matched = tm.match_players(ours, facts)
    assert matched["player_id"].isna().all()


# --------------------------------------------------------------------------- #
# Values and contracts
# --------------------------------------------------------------------------- #
def test_value_is_latest_within_the_season(attached: tuple[pd.DataFrame, tm.MatchReport]) -> None:
    out, _ = attached
    kane = row(out, "Harry Kane")
    assert kane["Market_Value_EUR_M"] == 100.0  # 1 June 2024, not the 2025 or 2026 values
    assert kane[VALUE_SOURCE_COLUMN] == "transfermarkt"


def test_unmatched_players_keep_proxies(attached: tuple[pd.DataFrame, tm.MatchReport]) -> None:
    out, _ = attached
    nobody = row(out, "Nobody Known")
    assert (nobody["Market_Value_EUR_M"], nobody["Contract_Expiry_Year"]) == (2.0, 2029)
    assert nobody[VALUE_SOURCE_COLUMN] == SOURCE_PROXY and nobody[CONTRACT_SOURCE_COLUMN] == SOURCE_PROXY


def test_listed_contract_used_only_for_tracked_players(attached: tuple[pd.DataFrame, tm.MatchReport]) -> None:
    out, report = attached
    kane = row(out, "Harry Kane")
    assert kane["Contract_Expiry_Year"] == 2027 and kane[CONTRACT_SOURCE_COLUMN] == "transfermarkt"
    veteran = row(out, "Retired Veteran")
    assert veteran[VALUE_SOURCE_COLUMN] == "transfermarkt"  # value from the season is still real
    assert veteran[CONTRACT_SOURCE_COLUMN] == SOURCE_PROXY  # stale listed contract ignored
    assert veteran["Contract_Expiry_Year"] == 2027
    assert report.snapshot_date == pd.Timestamp("2026-06-12")


def test_contract_reference_year_is_the_snapshot(attached: tuple[pd.DataFrame, tm.MatchReport]) -> None:
    out, _ = attached
    assert (out[tm.CONTRACT_REFERENCE_COLUMN] == 2026).all()
    assert pipeline.contract_reference_year(out) == 2026
    assert pipeline.contract_reference_year(pd.DataFrame({"Season": ["2023-2024"]})) == 2024


def test_no_matches_keeps_season_reference(enriched: pd.DataFrame) -> None:
    empty_players = TM_PLAYERS.iloc[0:0]
    out, report = tm.attach_transfermarkt(enriched, empty_players, TM_VALUATIONS, SEASON)
    assert report.n_matched == 0
    assert (out[tm.CONTRACT_REFERENCE_COLUMN] == 2024).all()
    assert (out[VALUE_SOURCE_COLUMN] == SOURCE_PROXY).all()


def test_source_share_helper(attached: tuple[pd.DataFrame, tm.MatchReport]) -> None:
    out, _ = attached
    share = pipeline.source_share(out, VALUE_SOURCE_COLUMN, "transfermarkt")
    assert share == pytest.approx(8 / 9)
    assert pipeline.source_share(out, "missing", "transfermarkt") == 0.0


def test_season_window() -> None:
    window = tm.SeasonWindow.for_season(SEASON)
    assert (window.start, window.end) == (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-06-30"))


def test_read_transfermarkt_requires_both_files(tmp_path) -> None:
    TM_PLAYERS.to_csv(tmp_path / "players.csv", index=False)
    with pytest.raises(FileNotFoundError, match="player_valuations.csv"):
        tm.read_transfermarkt(tmp_path)
    TM_VALUATIONS.to_csv(tmp_path / "player_valuations.csv", index=False)
    players, valuations = tm.read_transfermarkt(tmp_path)
    assert len(players) == len(TM_PLAYERS) and len(valuations) == len(TM_VALUATIONS)
