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
    df["Exp NPG"] = (df["Total Shots"] * 0.11).round(1)
    df["Progressive Carries"] = df["Take ons attempted"] // 2
    # Aerial win rate; blank (no duels) for the keeper, as FBref leaves it.
    df["% Aerial Duels won"] = [None if p == "GK" else 40.0 + 3 * (i % 7) for i, p in enumerate(df["pos"])]
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
    code = module.main(["--csv", str(csv), "--season", SEASON, "--fbref-cache", str(cache), "--out-dir", str(out)])
    assert code == 0
    written = pd.read_parquet(out / "real_players_2023_2024.parquet")
    assert len(written) == len(snapshot) - 1  # Dier's two clubs merged
    assert (out / "real_players_2023_2024.csv").exists()
    assert not list(out.glob("benchmark_*"))  # never a synthetic fallback


def test_script_without_possession_cache_still_succeeds(tmp_path: Path, snapshot: pd.DataFrame) -> None:
    module = _load_script()
    csv = tmp_path / "s.csv"
    snapshot.to_csv(csv, index=False)
    code = module.main(["--csv", str(csv), "--season", SEASON, "--fbref-cache", str(tmp_path / "none"),
                        "--out-dir", str(tmp_path)])
    assert code == 0
    written = pd.read_parquet(tmp_path / "real_players_2023_2024.parquet")
    assert (written["Possession_Source"] == "estimated").all()  # pass-share estimate, not a flat 50%
    assert written["Team_Possession_Pct"].between(30, 75).all()


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
    code = module.main(["--csv", str(csv), "--season", SEASON, "--fbref-cache", str(write_cache(tmp_path / "FBref")),
                        "--transfermarkt", str(tm_dir), "--out-dir", str(tmp_path / "out")])
    assert code == 0
    written = pd.read_parquet(tmp_path / "out" / "real_players_2023_2024.parquet")
    kane = written[written.Player == "Harry Kane"].iloc[0]
    assert (kane["Market_Value_EUR_M"], kane["Contract_Expiry_Year"]) == (100.0, 2027)
    assert kane["Market_Value_Source"] == "transfermarkt"
    assert (written["Contract_Reference_Year"] == 2026).all()
    others = written[written.Player != "Harry Kane"]
    assert (others["Market_Value_Source"] == "proxy").all()



# --------------------------------------------------------------------------- #
# New stats (v1.4)
# --------------------------------------------------------------------------- #
def test_new_stats_mapped_and_summed_across_transfers(built: pd.DataFrame) -> None:
    dier = built[built.Player == "Eric Dier"].iloc[0]
    assert dier["Non_Penalty_xG"] == pytest.approx(round(1 * 0.11, 1) + round(6 * 0.11, 1))
    assert dier["Progressive_Carries"] == 0 + 1  # (0 // 2) + (2 // 2)
    assert built["Aerial_Win_Pct"].between(0, 100).all() and built["Aerial_Win_Pct"].notna().all()


def test_missing_aerial_rate_filled_from_position_median(snapshot: pd.DataFrame) -> None:
    raw = snap.snapshot_to_raw(snapshot)
    keeper = raw[raw.Position == "GK"].iloc[0]
    assert 0 <= keeper["Aerial_Win_Pct"] <= 100  # no other GK: overall median


# --------------------------------------------------------------------------- #
# 2024-25 export layout (FBref column codes)
# --------------------------------------------------------------------------- #
def make_big5() -> pd.DataFrame:
    """Rows shaped like hubertsidorowicz/football-players-stats-2024-2025."""
    return pd.DataFrame({
        "Rk": [1, 2, 3], "Player": ["Mohamed Salah", "Max Aarons", "Max Aarons"],
        "Nation": ["eg EGY", "eng ENG", "eng ENG"], "Pos": ["FW", "DF", "DF,MF"],
        "Squad": ["Liverpool", "Bournemouth", "Valencia"],
        "Comp": ["eng Premier League", "eng Premier League", "es La Liga"],
        "Age": [32.0, 24.0, 24.0], "Born": [1992.0, 2000.0, 2000.0], "MP": [38, 3, 4], "Starts": [38, 1, 1],
        "Min": [3371, 86, 120], "90s": [37.5, 1.0, 1.3], "Gls": [29, 0, 0], "Ast": [18, 0, 0],
        "Sh": [130, 1, 0], "Cmp": [892, 40, 60], "Att": [1263, 50, 70], "Cmp%": [70.6, 80.0, 85.7],
        "KP": [88, 1, 0], "PrgP": [140, 3, 4], "Att_stats_possession": [137, 2, 1], "Succ": [58, 1, 0],
        "TklW": [11, 2, 3], "Int": [9, 1, 2], "npxG": [18.2, 0.1, 0.0], "PrgC": [154, 2, 3],
        "Won%": [39.1, None, 50.0], "Att_stats_defense": [30, 5, 6],  # same code, other table: ignored
    })


def test_big5_format_detected_and_mapped() -> None:
    frame = make_big5()
    assert snap.detect_format(frame).name == "fbref-big5"
    raw = snap.snapshot_to_raw(frame, "2024-2025")
    salah = raw[raw.Player == "Mohamed Salah"].iloc[0]
    assert (salah["League"], salah["Nation"], salah["Season"]) == ("Premier League", "EGY", "2024-2025")
    assert (salah["Passes_Attempted"], salah["Take_Ons_Attempted"], salah["Successful_Take_Ons"]) == (1263, 137, 58)
    assert (salah["Non_Penalty_xG"], salah["Progressive_Carries"], salah["Aerial_Win_Pct"]) == (18.2, 154, 39.1)


def test_big5_transfer_across_leagues_merged() -> None:
    raw, report = snap.build_snapshot_dataset(make_big5(), None, "2024-2025")
    aarons = raw[raw.Player == "Max Aarons"].iloc[0]
    assert report.n_transferred == 1
    assert aarons["Squad"] == "Valencia" and aarons["Minutes_Played"] == 206  # most-minutes club, summed


def test_unknown_layout_raises() -> None:
    with pytest.raises(FBrefSchemaError, match="unrecognised"):
        snap.detect_format(pd.DataFrame({"Player": ["x"], "Min": [90]}))


# --------------------------------------------------------------------------- #
# Possession estimate
# --------------------------------------------------------------------------- #
def test_possession_estimate_orders_clubs_by_pass_share() -> None:
    raw = pd.DataFrame({
        "League": ["L"] * 4, "Squad": ["Dominant", "Average", "Average2", "Deep"],
        "Passes_Attempted": [24000, 16000, 16000, 10000],
    })
    est = snap.estimate_possession(raw).set_index("Squad")["Team_Possession_Pct"]
    assert est["Dominant"] > est["Average"] > est["Deep"]
    assert est["Average"] == pytest.approx(est["Average2"])
    assert est.between(30, 75).all()


def test_possession_calibration_stretches_around_fifty() -> None:
    raw = pd.DataFrame({"League": ["L", "L"], "Squad": ["A", "B"], "Passes_Attempted": [600, 400]})
    flat = snap.estimate_possession(raw, slope=1.0).set_index("Squad")["Team_Possession_Pct"]
    cal = snap.estimate_possession(raw).set_index("Squad")["Team_Possession_Pct"]
    assert flat["A"] == pytest.approx(60.0)
    assert cal["A"] == pytest.approx(50 + snap.POSSESSION_CALIBRATION_SLOPE * 10, abs=0.05)


def test_cached_possession_preferred_and_labelled(snapshot: pd.DataFrame, possession: pd.DataFrame) -> None:
    partial = possession[possession.Squad != "Milan"]
    raw, _ = snap.build_snapshot_dataset(snapshot, partial, SEASON)
    by_club = raw.drop_duplicates("Squad").set_index("Squad")["Possession_Source"]
    assert by_club["Bayern Munich"] == "fbref" and by_club["Milan"] == "estimated"
    assert raw.loc[raw.Squad == "Bayern Munich", "Team_Possession_Pct"].iloc[0] == 61.3
