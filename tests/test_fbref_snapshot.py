"""FBref 2023-24 snapshot adapter: schema mapping, possession, transfers, script.

Fixtures mirror the Kaggle export ``cleaned_2023-24.csv`` exactly (column
names, the mislabelled ``Avg Mins per Match`` total-minutes column, take-ons as
attempts + success rate, one row per player per club) and FBref's cached
league page, whose squad table may sit inside an HTML comment.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from src import pipeline
from src.config import CORE_FEATURES, DATA_ORIGIN_COLUMN, ORIGIN_FBREF, REQUIRED_RAW_COLUMNS
from src.data import fbref_snapshot as snap
from src.data.fbref_transform import FBrefSchemaError
from src.data.loader import DataLoader
from src.data.real_data_enricher import enrich_real_players
from src.features.preprocessor import ScoutingPreprocessor

ROOT = Path(__file__).resolve().parents[1]
SEASON = "2023-2024"

# (player, pos, squad, comp, age, born, MP, Min, Gls, Ast, Sh, Cmp, Att, KP, PrgP, TO_Att, TO_Succ%, TklW, Int)
ROWS = [
    ("Harry Kane", "FW", "Bayern Munich", "Bundesliga", 30, 1993, 32, 2839, 36, 8, 141, 700, 980, 31, 102, 50, 48.0, 9, 6),
    ("Eric Dier", "DF", "Tottenham", "Premier League", 29, 1994, 4, 120, 0, 0, 1, 90, 100, 0, 5, 0, 0.0, 2, 1),
    ("Eric Dier", "DF", "Bayern Munich", "Bundesliga", 29, 1994, 15, 1247, 1, 0, 6, 900, 1000, 3, 60, 2, 50.0, 10, 9),
    ("Rodri", "MF", "Manchester City", "Premier League", 27, 1996, 34, 2931, 8, 9, 62, 2700, 2935, 49, 376, 60, 70.0, 43, 26),
    ("Rodri", "FW", "Betis", "La Liga", 23, 2000, 25, 1051, 0, 0, 18, 300, 381, 15, 65, 50, 48.0, 10, 8),
    ("Bruno Guimarães", "MF", "Newcastle Utd", "Premier League", 26, 1997, 37, 3200, 7, 8, 50, 2000, 2400, 60, 250, 70, 55.0, 60, 30),
    ("Gianluigi Donnarumma", "GK", "Paris S-G", "Ligue 1", 24, 1999, 30, 2700, 0, 0, 0, 900, 1000, 0, 2, 0, 0.0, 0, 1),
    ("Vitinha", "MF", "Paris S-G", "Ligue 1", 23, 2000, 30, 2400, 6, 4, 50, 2100, 2300, 40, 220, 40, 60.0, 30, 15),
    ("Rafael Leão", "FW", "Milan", "Serie A", 24, 1999, 34, 2650, 9, 9, 80, 900, 1150, 50, 70, 180, 50.0, 6, 4),
    ("Lamine Yamal", "FW", "Barcelona", "La Liga", 16, 2007, 37, 2400, 5, 6, 70, 900, 1100, 60, 100, 150, 50.0, 10, 8),
    ("Kobbie Mainoo", "MF", "Manchester Utd", "Premier League", 18, 2005, 24, 1800, 3, 1, 20, 900, 1050, 15, 80, 30, 60.0, 30, 20),
    ("Jonathan Tah", "DF", "Leverkusen", "Bundesliga", 27, 1996, 31, 2700, 4, 1, 20, 2400, 2600, 5, 200, 4, 75.0, 25, 30),
    ("William Saliba", "DF", "Arsenal", "Premier League", 22, 2001, 38, 3420, 2, 1, 15, 2600, 2800, 4, 180, 10, 70.0, 30, 35),
]
COLUMNS = [
    "player", "pos", "squad", "comp", "age", "born", "Matches Played", "Avg Mins per Match", "Goals", "Assists",
    "Total Shots", "Passes Completed", "Passes Attempted", "Key passes", "Progressive Passes",
    "Take ons attempted", "% Successful take-ons", "Tackles Won", "Interceptions",
]

POSSESSION_PAGES = {
    "ENG-Premier League": [("Tottenham", 61.0), ("Manchester City", 65.2), ("Newcastle", 51.9),
                           ("Manchester Utd", 50.6), ("Arsenal", 59.4)],
    "GER-Bundesliga": [("Bayern Munich", 61.3), ("Leverkusen", 61.8)],
    "ESP-La Liga": [("Real Betis", 54.0), ("Barcelona", 64.3)],
    "ITA-Serie A": [("Milan", 53.3)],
    "FRA-Ligue 1": [("Paris SG", 65.2)],
}


def make_snapshot() -> pd.DataFrame:
    df = pd.DataFrame(ROWS, columns=COLUMNS)
    df.insert(0, "rk", range(1, len(df) + 1))
    df.insert(2, "nation", "Somewhere")
    df["Expected Goals"] = 0.0  # extra export columns must be ignored
    df["season"] = SEASON
    return df


def _squad_table(rows: list[tuple[str, float]]) -> str:
    body = "".join(
        f'<tr><th data-stat="team"><a href="#">{squad}</a></th><td data-stat="possession">{poss}</td></tr>'
        for squad, poss in rows
    )
    return (f'<table id="stats_squads_standard_for"><thead><tr><th>Squad</th><th>Poss</th></tr></thead>'
            f"<tbody>{body}</tbody></table>")


def write_cache(cache: Path, commented_leagues: tuple[str, ...] = ("GER-Bundesliga",)) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    for league, rows in POSSESSION_PAGES.items():
        table = _squad_table(rows)
        if league in commented_leagues:  # FBref hides some tables inside comments
            table = f"<div id='all_stats_squads'><!--\n{table}\n--></div>"
        (cache / f"players_{league}_2324_standard.html").write_text(
            f"<html><body>{table}</body></html>", encoding="utf-8")
    return cache


@pytest.fixture()
def snapshot() -> pd.DataFrame:
    return make_snapshot()


@pytest.fixture()
def possession(tmp_path: Path) -> pd.DataFrame:
    return snap.parse_cached_possession(write_cache(tmp_path / "FBref"), "2324")


@pytest.fixture()
def built(snapshot: pd.DataFrame, possession: pd.DataFrame) -> pd.DataFrame:
    raw, _ = snap.build_snapshot_dataset(snapshot, possession, SEASON)
    return raw


def _load_script():
    spec = importlib.util.spec_from_file_location("ingest_fbref_snapshot", ROOT / "scripts" / "ingest_fbref_snapshot.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["ingest_fbref_snapshot"] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Schema mapping
# --------------------------------------------------------------------------- #
def test_snapshot_maps_onto_raw_schema(snapshot: pd.DataFrame) -> None:
    raw = snap.snapshot_to_raw(snapshot, SEASON)
    kane = raw[raw.Player == "Harry Kane"].iloc[0]
    assert kane["Minutes_Played"] == 2839  # "Avg Mins per Match" really is season minutes
    assert (kane["Goals"], kane["Shots"], kane["Key_Passes"], kane["Progressive_Passes"]) == (36, 141, 31, 102)
    assert "Expected Goals" not in raw.columns and "rk" not in raw.columns


def test_successful_take_ons_rebuilt_from_rate(snapshot: pd.DataFrame) -> None:
    raw = snap.snapshot_to_raw(snapshot)
    leao = raw[raw.Player == "Rafael Leão"].iloc[0]
    assert leao["Successful_Take_Ons"] == 90  # 180 attempts x 50 %
    assert (raw["Successful_Take_Ons"] <= raw["Take_Ons_Attempted"]).all()


def test_missing_snapshot_column_raises(snapshot: pd.DataFrame) -> None:
    with pytest.raises(FBrefSchemaError, match="Key passes"):
        snap.snapshot_to_raw(snapshot.drop(columns=["Key passes"]))


def test_wrong_season_raises(snapshot: pd.DataFrame) -> None:
    with pytest.raises(FBrefSchemaError, match="2022-2023"):
        snap.snapshot_to_raw(snapshot, "2022-2023")


# --------------------------------------------------------------------------- #
# Possession from the cached FBref pages
# --------------------------------------------------------------------------- #
def test_possession_parsed_including_commented_tables(possession: pd.DataFrame) -> None:
    assert len(possession) == sum(len(v) for v in POSSESSION_PAGES.values())
    bayern = possession[(possession.League == "Bundesliga") & (possession.Squad == "Bayern Munich")]
    assert bayern["Team_Possession_Pct"].item() == 61.3  # Bundesliga page is comment-wrapped


def test_squad_aliases_match_player_table_names(possession: pd.DataFrame) -> None:
    squads = set(possession.Squad)
    assert {"Newcastle Utd", "Paris S-G", "Betis"} <= squads
    assert not {"Newcastle", "Paris SG", "Real Betis"} & squads


def test_missing_cache_yields_empty_possession(tmp_path: Path) -> None:
    assert snap.parse_cached_possession(tmp_path / "nothing", "2324").empty


def test_every_player_gets_real_possession(built: pd.DataFrame) -> None:
    vitinha = built[built.Player == "Vitinha"].iloc[0]
    assert vitinha["Team_Possession_Pct"] == 65.2
    assert built["Team_Possession_Pct"].between(35, 70).all()


# --------------------------------------------------------------------------- #
# Transfers and namesakes
# --------------------------------------------------------------------------- #
def test_transfer_merged_into_season_totals(built: pd.DataFrame) -> None:
    dier = built[built.Player == "Eric Dier"]
    assert len(dier) == 1
    row = dier.iloc[0]
    assert row["Squad"] == "Bayern Munich" and row["League"] == "Bundesliga"  # most-minutes club
    assert row["Minutes_Played"] == 1367 and row["Interceptions"] == 10 and row["Clubs_In_Season"] == 2
    expected = (61.0 * 120 + 61.3 * 1247) / 1367
    assert row["Team_Possession_Pct"] == pytest.approx(expected, abs=0.05)


def test_namesakes_kept_apart(built: pd.DataFrame) -> None:
    assert sorted(built.loc[built.Player == "Rodri", "Squad"]) == ["Betis", "Manchester City"]


def test_pass_completion_recomputed_from_totals(built: pd.DataFrame) -> None:
    rodri = built[(built.Player == "Rodri") & (built.Squad == "Manchester City")].iloc[0]
    assert rodri["Pass_Completion_Pct"] == pytest.approx(92.0, abs=0.05)


# --------------------------------------------------------------------------- #
# Enrichment, loader and model
# --------------------------------------------------------------------------- #
def test_enriched_snapshot_has_full_schema_without_nulls(built: pd.DataFrame) -> None:
    enriched = enrich_real_players(built, fetch_values=None)
    assert set(REQUIRED_RAW_COLUMNS) <= set(enriched.columns)
    assert not enriched[list(REQUIRED_RAW_COLUMNS)].isna().any().any()
    assert (enriched[DATA_ORIGIN_COLUMN] == ORIGIN_FBREF).all()


def test_snapshot_runs_through_loader_and_model(tmp_path: Path, built: pd.DataFrame) -> None:
    enriched = enrich_real_players(built, fetch_values=None)
    enriched.to_parquet(tmp_path / "real_players_2023_2024.parquet", index=False)
    df = ScoutingPreprocessor().transform(DataLoader().load_real(SEASON, data_dir=tmp_path))
    assert "GK" not in set(df["Position_Group"])
    assert pipeline.dataset_reference_year(df) == 2024
    cohort = pipeline.filter_cohort(df, "MF", None)
    model = pipeline.build_cohort_model(cohort, 2, "robust")
    assert model.X.shape == (len(cohort), len(CORE_FEATURES))
    assert set(model.cohort["Cluster"]) == {0, 1}


def test_season_codes() -> None:
    assert snap.season_code("2023-2024") == "2324"
    assert snap.season_code("1999-2000") == "9900"
    with pytest.raises(ValueError):
        snap.season_code("2324")


# --------------------------------------------------------------------------- #
# Ingestion script
# --------------------------------------------------------------------------- #
def test_script_writes_real_dataset(tmp_path: Path, snapshot: pd.DataFrame) -> None:
    module = _load_script()
    csv = tmp_path / "cleaned_2023-24.csv"
    snapshot.to_csv(csv, index=False)
    cache = write_cache(tmp_path / "FBref")
    out = tmp_path / "processed"
    code = module.main(["--csv", str(csv), "--fbref-cache", str(cache), "--out-dir", str(out)])
    assert code == 0
    written = pd.read_parquet(out / "real_players_2023_2024.parquet")
    assert len(written) == len(snapshot) - 1  # Dier's two clubs merged
    assert (out / "real_players_2023_2024.csv").exists()
    assert not list(out.glob("benchmark_*"))  # never a synthetic fallback


def test_script_without_possession_cache_still_succeeds(tmp_path: Path, snapshot: pd.DataFrame) -> None:
    module = _load_script()
    csv = tmp_path / "s.csv"
    snapshot.to_csv(csv, index=False)
    code = module.main(["--csv", str(csv), "--fbref-cache", str(tmp_path / "none"), "--out-dir", str(tmp_path)])
    assert code == 0
    written = pd.read_parquet(tmp_path / "real_players_2023_2024.parquet")
    assert (written["Team_Possession_Pct"] == 50.0).all()  # neutral pAdj, logged as a warning


def test_script_fails_cleanly_and_writes_nothing(tmp_path: Path) -> None:
    module = _load_script()
    assert module.main(["--csv", str(tmp_path / "missing.csv"), "--out-dir", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out").exists()


def test_snapshot_filename_for_season() -> None:
    assert _load_script().snapshot_filename("2023-2024") == "cleaned_2023-24.csv"


def test_script_attaches_transfermarkt(tmp_path: Path, snapshot: pd.DataFrame) -> None:
    module = _load_script()
    csv = tmp_path / "s.csv"
    snapshot.to_csv(csv, index=False)
    tm_dir = tmp_path / "tm"
    tm_dir.mkdir()
    pd.DataFrame({
        "player_id": [1], "name": ["Harry Kane"], "date_of_birth": ["1993-07-28"],
        "contract_expiration_date": ["2027-06-30"], "last_season": [2025],
    }).to_csv(tm_dir / "players.csv", index=False)
    pd.DataFrame({
        "player_id": [1, 1], "date": ["2024-06-01", "2026-06-12"], "market_value_in_eur": [100_000_000, 70_000_000],
        "player_club_domestic_competition_id": ["L1", "L1"],
    }).to_csv(tm_dir / "player_valuations.csv", index=False)
    code = module.main(["--csv", str(csv), "--fbref-cache", str(write_cache(tmp_path / "FBref")),
                        "--transfermarkt", str(tm_dir), "--out-dir", str(tmp_path / "out")])
    assert code == 0
    written = pd.read_parquet(tmp_path / "out" / "real_players_2023_2024.parquet")
    kane = written[written.Player == "Harry Kane"].iloc[0]
    assert (kane["Market_Value_EUR_M"], kane["Contract_Expiry_Year"]) == (100.0, 2027)
    assert kane["Market_Value_Source"] == "transfermarkt"
    assert (written["Contract_Reference_Year"] == 2026).all()
    others = written[written.Player != "Harry Kane"]
    assert (others["Market_Value_Source"] == "proxy").all()
