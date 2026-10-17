"""Real-data pipeline: FBref parsing, transfer dedup, enrichment, loading, fallback.

All tests run offline against fixtures shaped exactly like ``soccerdata.FBref``
output: a ``(league, season, team, player)`` index and two-level column
headers, with FBref quirks (string numbers with thousands separators,
``"25-123"`` ages, a mid-season transfer, namesakes, a goalkeeper).
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import pipeline
from src.config import (
    CORE_FEATURES,
    DATA_ORIGIN_COLUMN,
    ORIGIN_BENCHMARK,
    ORIGIN_FBREF,
    REQUIRED_RAW_COLUMNS,
    real_data_filename,
    season_end_year,
)
from src.data import fbref_transform as ft
from src.data import real_data_enricher as enr
from src.data.loader import DataLoader, RealDataNotFoundError, find_real_dataset

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- #
# FBref-shaped fixtures
# --------------------------------------------------------------------------- #
# (league, team, player, nation, pos, age, born, MP, Starts, Min, 90s, Gls, Ast,
#  Sh, Cmp, Att, KP, PrgP, TO_Att, TO_Succ, TklW, Int)
PLAYERS = [
    ("ENG-Premier League", "Arsenal", "Alpha Striker", "ENG", "FW", "25-100", 1998,
     34, 32, "2,800", 31.1, 21, 6, 98, 420, 600, 40, 55, 60, 30, 12, 6),
    ("ENG-Premier League", "Arsenal", "Beta Mid", "ESP", "MF", "27-010", 1996,
     36, 35, "3,100", 34.4, 4, 9, 40, 2300, 2600, 70, 260, 35, 22, 50, 40),
    ("ENG-Premier League", "Chelsea", "Gamma Def", "FRA", "DF,MF", "22-200", 2001,
     30, 30, "2,650", 29.4, 2, 3, 18, 1900, 2150, 20, 180, 20, 12, 45, 52),
    # Mid-season transfer inside the league: more minutes at Chelsea.
    ("ENG-Premier League", "Chelsea", "Delta Transfer", "BRA", "FW,MF", "24-050", 1999,
     17, 14, "1,250", 13.9, 7, 2, 40, 300, 420, 18, 25, 40, 18, 5, 2),
    ("ENG-Premier League", "Brighton", "Delta Transfer", "BRA", "FW,MF", "24-050", 1999,
     12, 9, "820", 9.1, 3, 1, 22, 210, 300, 10, 16, 30, 12, 3, 2),
    # Namesakes: same name, different birth years -> two different players.
    ("ENG-Premier League", "Brighton", "Same Name", "ENG", "MF", "30-001", 1993,
     25, 20, "1,900", 21.1, 1, 2, 15, 900, 1050, 15, 80, 10, 5, 30, 20),
    ("GER-Bundesliga", "Bayern Munich", "Same Name", "GER", "DF", "21-300", 2002,
     28, 26, "2,300", 25.6, 1, 1, 8, 1500, 1650, 6, 110, 6, 3, 35, 38),
    ("GER-Bundesliga", "Bayern Munich", "Keeper One", "GER", "GK", "29-000", 1994,
     34, 34, "3,060", 34.0, 0, 0, 0, 800, 1000, 0, 2, 0, 0, 0, 1),
]

TEAMS = [
    ("ENG-Premier League", "Arsenal", 58.4),
    ("ENG-Premier League", "Chelsea", 61.0),
    ("ENG-Premier League", "Brighton", 63.5),
    ("GER-Bundesliga", "Bayern Munich", 64.2),
]

SEASON = "2324"


def _index(rows: list[tuple]) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        [(r[0], SEASON, r[1], r[2]) for r in rows], names=["league", "season", "team", "player"]
    )


def fbref_tables(rows: list[tuple] = PLAYERS) -> dict[str, pd.DataFrame]:
    """Five soccerdata-style player stat tables with two-level headers."""
    idx = _index(rows)

    def frame(columns: list[tuple[str, str]], values: list[list]) -> pd.DataFrame:
        return pd.DataFrame(values, index=idx, columns=pd.MultiIndex.from_tuples(columns))

    standard = frame(
        [("nation", ""), ("pos", ""), ("age", ""), ("born", ""), ("Playing Time", "MP"),
         ("Playing Time", "Starts"), ("Playing Time", "Min"), ("Playing Time", "90s"),
         ("Performance", "Gls"), ("Performance", "Ast"), ("Expected", "xG")],
        [[r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], 0.0] for r in rows],
    )
    shooting = frame([("Standard", "Gls"), ("Standard", "Sh"), ("Standard", "SoT")],
                     [[r[11], r[13], r[13] // 3] for r in rows])
    passing = frame(
        [("Total", "Cmp"), ("Total", "Att"), ("Total", "Cmp%"), ("Ast", ""), ("KP", ""), ("PrgP", "")],
        [[r[14], r[15], 0.0, r[12], r[16], r[17]] for r in rows],
    )
    possession = frame([("Take-Ons", "Att"), ("Take-Ons", "Succ"), ("Take-Ons", "Succ%")],
                       [[r[18], r[19], 0.0] for r in rows])
    defense = frame([("Tackles", "Tkl"), ("Tackles", "TklW"), ("Int", "")],
                    [[r[20] + 5, r[20], r[21]] for r in rows])
    return {"standard": standard, "shooting": shooting, "passing": passing,
            "possession": possession, "defense": defense}


def fbref_team_table(teams: list[tuple] = TEAMS) -> pd.DataFrame:
    idx = pd.MultiIndex.from_tuples([(t[0], SEASON, t[1]) for t in teams], names=["league", "season", "team"])
    return pd.DataFrame(
        [[38, t[2], 70] for t in teams], index=idx,
        columns=pd.MultiIndex.from_tuples([("Playing Time", "MP"), ("Poss", ""), ("Performance", "Gls")]),
    )


@pytest.fixture(scope="module")
def raw_players() -> pd.DataFrame:
    raw, _ = ft.build_raw_player_table(fbref_tables(), ft.team_possession(fbref_team_table()))
    return raw


@pytest.fixture(scope="module")
def enriched(raw_players: pd.DataFrame) -> pd.DataFrame:
    return enr.enrich_real_players(raw_players, fetch_values=None)


def load_script():
    spec = importlib.util.spec_from_file_location("ingest_fbref", ROOT / "scripts" / "ingest_fbref.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ingest_fbref"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# --------------------------------------------------------------------------- #
# Column flattening -> raw schema
# --------------------------------------------------------------------------- #
def test_flatten_multiindex_columns() -> None:
    flat = ft.flatten_fbref_columns(fbref_tables()["standard"])
    for name in ("league", "season", "team", "player", "nation", "pos",
                 "Playing Time_Min", "Performance_Gls", "Expected_xG"):
        assert name in flat.columns
    assert not any(col.endswith("_") for col in flat.columns)


def test_flatten_drops_unnamed_levels_and_passes_flat_frames() -> None:
    df = pd.DataFrame([[1, 2]], columns=pd.MultiIndex.from_tuples([("KP", "Unnamed: 5_level_1"), ("Int", "")]))
    assert list(ft.flatten_fbref_columns(df).columns) == ["KP", "Int"]
    plain = pd.DataFrame({" a ": [1]})
    assert list(ft.flatten_fbref_columns(plain).columns) == ["a"]


def test_flattened_tables_map_onto_raw_schema(raw_players: pd.DataFrame) -> None:
    assert set(REQUIRED_RAW_COLUMNS) - {"Market_Value_EUR_M", "Contract_Expiry_Year"} <= set(raw_players.columns)
    alpha = raw_players.set_index("Player").loc["Alpha Striker"]
    assert alpha["Minutes_Played"] == 2800  # "2,800" parsed
    assert alpha["Goals"] == 21 and alpha["Shots"] == 98
    assert alpha["Key_Passes"] == 40 and alpha["Progressive_Passes"] == 55
    assert alpha["Successful_Take_Ons"] == 30 and alpha["Tackles_Won"] == 12
    assert alpha["Age"] == 25  # "25-100" -> 25
    assert alpha["League"] == "Premier League"
    assert alpha["Season"] == "2023-2024"


def test_header_matching_tolerates_case_and_spacing() -> None:
    tables = fbref_tables()
    shooting = tables["shooting"].copy()
    shooting.columns = pd.MultiIndex.from_tuples([("STANDARD", " Gls"), ("Standard ", "SH"), ("Standard", "SoT")])
    tables["shooting"] = shooting
    raw, _ = ft.build_raw_player_table(tables)
    assert raw.set_index("Player").loc["Alpha Striker", "Shots"] == 98


def test_missing_fbref_column_raises_schema_error() -> None:
    tables = fbref_tables()
    tables["defense"] = tables["defense"].drop(columns=[("Int", "")])
    with pytest.raises(ft.FBrefSchemaError, match="int"):
        ft.build_raw_player_table(tables)


def test_missing_table_raises_schema_error() -> None:
    tables = fbref_tables()
    del tables["possession"]
    with pytest.raises(ft.FBrefSchemaError, match="possession"):
        ft.build_raw_player_table(tables)


def test_pass_completion_recomputed_from_totals(raw_players: pd.DataFrame) -> None:
    beta = raw_players.set_index("Player").loc["Beta Mid"]
    assert beta["Pass_Completion_Pct"] == pytest.approx(round(100 * 2300 / 2600, 1))


def test_player_absent_from_secondary_table_gets_zero() -> None:
    tables = fbref_tables()
    tables["defense"] = tables["defense"].drop(index=("ENG-Premier League", SEASON, "Arsenal", "Beta Mid"))
    raw, _ = ft.build_raw_player_table(tables)
    beta = raw.set_index("Player").loc["Beta Mid"]
    assert beta["Tackles_Won"] == 0 and beta["Interceptions"] == 0


def test_number_and_age_parsers() -> None:
    assert ft.to_number(pd.Series(["1,234", " 56 ", "", None])).tolist()[:2] == [1234.0, 56.0]
    assert ft.to_number(pd.Series(["", None])).isna().all()
    assert ft.parse_age(pd.Series(["23-145", "31", None])).tolist()[:2] == [23, 31]


# --------------------------------------------------------------------------- #
# Transfer deduplication
# --------------------------------------------------------------------------- #
def test_no_duplicate_players(raw_players: pd.DataFrame) -> None:
    assert not raw_players.duplicated(["Player", "Born"]).any()
    assert len(raw_players) == len(PLAYERS) - 1  # one transfer collapsed


def test_transfer_stats_are_season_totals(raw_players: pd.DataFrame) -> None:
    delta = raw_players.set_index("Player").loc["Delta Transfer"]
    assert delta["Minutes_Played"] == 1250 + 820
    assert delta["Goals"] == 7 + 3
    assert delta["Shots"] == 40 + 22
    assert delta["Pass_Completion_Pct"] == pytest.approx(round(100 * 510 / 720, 1))
    assert delta["Clubs_In_Season"] == 2


def test_transfer_assigned_to_primary_club(raw_players: pd.DataFrame) -> None:
    assert raw_players.set_index("Player").loc["Delta Transfer", "Squad"] == "Chelsea"


def test_transfer_possession_is_minutes_weighted(raw_players: pd.DataFrame) -> None:
    expected = (61.0 * 1250 + 63.5 * 820) / (1250 + 820)
    assert raw_players.set_index("Player").loc["Delta Transfer", "Team_Possession_Pct"] == pytest.approx(expected, abs=0.05)


def test_namesakes_are_not_merged(raw_players: pd.DataFrame) -> None:
    same = raw_players[raw_players["Player"] == "Same Name"]
    assert len(same) == 2
    assert set(same["Squad"]) == {"Brighton", "Bayern Munich"}


def test_dedup_report() -> None:
    _, report = ft.build_raw_player_table(fbref_tables())
    assert (report.n_rows_in, report.n_players_out, report.n_transferred) == (len(PLAYERS), len(PLAYERS) - 1, 1)


def test_season_labels() -> None:
    assert ft._season_label("2324") == "2023-2024"
    assert ft._season_label("9900") == "1999-2000"
    assert ft._season_label("2023-2024") == "2023-2024"
    assert season_end_year("2023-2024") == 2024 and season_end_year("2324") == 2024


# --------------------------------------------------------------------------- #
# Enrichment
# --------------------------------------------------------------------------- #
def test_team_possession_mapping_and_fallbacks() -> None:
    possession = ft.team_possession(fbref_team_table())
    assert possession.set_index("Squad").loc["Arsenal", "Team_Possession_Pct"] == pytest.approx(58.4)
    players = pd.DataFrame({"League": ["Premier League", "Premier League", "Serie A"],
                            "Squad": ["Arsenal", "Unknown FC", "Inter"]})
    mapped = enr.map_team_possession(players, possession)
    epl_mean = np.mean([58.4, 61.0, 63.5])
    assert mapped["Team_Possession_Pct"].tolist() == pytest.approx([58.4, epl_mean, 50.0])


def test_enrichment_leaves_no_nulls(enriched: pd.DataFrame) -> None:
    for col in ("Market_Value_EUR_M", "Contract_Expiry_Year", "Team_Possession_Pct"):
        assert enriched[col].notna().all(), col
    assert (enriched["Market_Value_EUR_M"] > 0).all()
    assert enriched["Contract_Expiry_Year"].between(2026, 2029).all()


def test_enrichment_labels_provenance(enriched: pd.DataFrame) -> None:
    assert (enriched[enr.VALUE_SOURCE_COLUMN] == enr.SOURCE_PROXY).all()
    assert (enriched[enr.CONTRACT_SOURCE_COLUMN] == enr.SOURCE_PROXY).all()
    assert (enriched[DATA_ORIGIN_COLUMN] == ORIGIN_FBREF).all()


def test_proxies_are_deterministic(raw_players: pd.DataFrame) -> None:
    a = enr.enrich_real_players(raw_players, fetch_values=None)
    b = enr.enrich_real_players(raw_players.sample(frac=1.0, random_state=3), fetch_values=None)
    key = ["Player", "Born"]
    merged = a.merge(b, on=key, suffixes=("_a", "_b"))
    assert (merged["Contract_Expiry_Year_a"] == merged["Contract_Expiry_Year_b"]).all()
    pd.testing.assert_series_equal(
        enr.proxy_contract_expiry(raw_players), enr.proxy_contract_expiry(raw_players.copy())
    )


def test_contract_proxy_covers_whole_window() -> None:
    players = pd.DataFrame({"Player": [f"P{i}" for i in range(400)], "Born": 2000})
    years = enr.proxy_contract_expiry(players)
    assert set(years) == {2026, 2027, 2028, 2029}
    with pytest.raises(ValueError):
        enr.proxy_contract_expiry(players, (2029, 2026))


def test_proxy_value_rewards_output_and_peak_age() -> None:
    base = dict(League="Premier League", Position="FW", Shots=60, Progressive_Passes=40, Key_Passes=30)
    players = pd.DataFrame([
        {**base, "Player": "elite", "Age": 25, "Minutes_Played": 3000, "Goals": 25},
        {**base, "Player": "average", "Age": 25, "Minutes_Played": 3000, "Goals": 8},
        {**base, "Player": "veteran", "Age": 34, "Minutes_Played": 3000, "Goals": 25},
        {**base, "Player": "bench", "Age": 25, "Minutes_Played": 900, "Goals": 7},
        {**base, "Player": "unused", "Age": 25, "Minutes_Played": 0, "Goals": 0},
    ])
    value = enr.proxy_market_value(players).set_axis(players["Player"])
    assert value["elite"] > value["average"]
    assert value["elite"] > value["veteran"]
    assert value["unused"] == enr.PROXY_VALUE_FLOOR_EUR_M
    assert np.isfinite(value).all()


def test_fetched_values_used_where_available(raw_players: pd.DataFrame) -> None:
    def fake_fetch(players: pd.DataFrame) -> pd.Series:
        return pd.Series([99.0] + [np.nan] * (len(players) - 1), index=players.index)

    out = enr.enrich_real_players(raw_players, fetch_values=fake_fetch)
    assert out.iloc[0]["Market_Value_EUR_M"] == 99.0
    assert out.iloc[0][enr.VALUE_SOURCE_COLUMN] == enr.SOURCE_TRANSFERMARKT
    assert (out.iloc[1:][enr.VALUE_SOURCE_COLUMN] == enr.SOURCE_PROXY).all()
    assert out["Market_Value_EUR_M"].notna().all()


class _Recorder(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def test_transfermarkt_fetch_never_raises(raw_players: pd.DataFrame) -> None:
    # soccerdata installs its own root logging handlers, which bypass pytest's
    # caplog; record directly on the enricher's logger instead.
    pytest.importorskip("soccerdata")
    recorder = _Recorder()
    enr.logger.addHandler(recorder)
    previous = enr.logger.level
    enr.logger.setLevel(logging.INFO)
    try:
        assert enr.fetch_transfermarkt_values(raw_players) is None  # soccerdata ships no reader
    finally:
        enr.logger.removeHandler(recorder)
        enr.logger.setLevel(previous)
    assert any("proxy" in m for m in recorder.messages)


# --------------------------------------------------------------------------- #
# Loading through the standard pipeline
# --------------------------------------------------------------------------- #
@pytest.fixture
def real_dir(tmp_path: Path, enriched: pd.DataFrame) -> Path:
    enriched.to_parquet(tmp_path / f"{real_data_filename('2023-2024')}.parquet", index=False)
    return tmp_path


def test_processed_real_data_passes_loader_validation(enriched: pd.DataFrame) -> None:
    loader = DataLoader()
    df = loader.load(enriched)  # must not raise DataValidationError
    assert loader.last_report.n_invalid == 0
    assert set(df["Position_Group"]) <= {"DF", "MF", "FW"}
    assert "Keeper One" not in set(df["Player"])


def test_load_real_runs_full_pipeline(real_dir: Path) -> None:
    from src.data.loader import LoaderConfig

    loader = DataLoader(LoaderConfig(min_minutes=900))
    df = loader.load_real("2023-2024", data_dir=real_dir)
    assert set(CORE_FEATURES) <= set(df.columns)
    assert loader.last_report.n_unknown_position == 1  # the goalkeeper
    alpha = df.set_index("Player").loc["Alpha Striker"]
    assert alpha["Goals_p90"] == pytest.approx(21 / 2800 * 90)  # Top 5 coefficient 1.0
    mult = 2.0 / (1.0 + np.exp(-0.05 * (58.4 - 50.0)))  # default steepness 0.05
    assert alpha["pAdj_Tackles_Won_p90"] == pytest.approx(12 / 2800 * 90 * mult)
    assert (df[DATA_ORIGIN_COLUMN] == ORIGIN_FBREF).all()


def test_load_real_reads_csv_twin(tmp_path: Path, enriched: pd.DataFrame) -> None:
    enriched.to_csv(tmp_path / f"{real_data_filename('2023-2024')}.csv", index=False)
    assert len(DataLoader().load_real(data_dir=tmp_path)) > 0


def test_load_real_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(RealDataNotFoundError, match="ingest_fbref"):
        DataLoader().load_real(data_dir=tmp_path)


def test_find_real_dataset_prefers_real_over_benchmark(tmp_path: Path, enriched: pd.DataFrame) -> None:
    bench = tmp_path / f"{real_data_filename('2023-2024', ORIGIN_BENCHMARK)}.parquet"
    enriched.to_parquet(bench, index=False)
    found = find_real_dataset(data_dir=tmp_path)
    assert found is not None and found.origin == ORIGIN_BENCHMARK and not found.is_real
    assert find_real_dataset(data_dir=tmp_path, allow_benchmark=False) is None
    enriched.to_parquet(tmp_path / f"{real_data_filename('2023-2024')}.parquet", index=False)
    assert find_real_dataset(data_dir=tmp_path).is_real  # type: ignore[union-attr]


def test_reference_year_and_provenance_helpers(enriched: pd.DataFrame) -> None:
    assert pipeline.dataset_reference_year(enriched) == 2024
    assert pipeline.dataset_reference_year(pd.DataFrame({"x": [1]})) == 2026
    assert pipeline.dataset_origin(enriched) == ORIGIN_FBREF
    assert pipeline.dataset_origin(pd.DataFrame({"x": [1]})) is None
    assert pipeline.proxy_share(enriched, enr.VALUE_SOURCE_COLUMN) == 1.0
    assert pipeline.proxy_share(enriched, "missing") == 0.0


# --------------------------------------------------------------------------- #
# Ingestion script: success path, rate-limit fallback
# --------------------------------------------------------------------------- #
def test_script_success_writes_real_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = load_script()
    monkeypatch.setattr(script, "fetch_fbref_tables", lambda *a, **k: (fbref_tables(), fbref_team_table()))
    origin, parquet = script.run(out_dir=tmp_path)
    assert origin == ORIGIN_FBREF
    assert parquet.name == "real_players_2023_2024.parquet"
    assert (tmp_path / "real_players_2023_2024.csv").is_file()
    df = pd.read_parquet(parquet)
    assert not df.duplicated(["Player", "Born"]).any()
    assert df["Market_Value_EUR_M"].notna().all()


def test_script_falls_back_to_labelled_benchmark_on_rate_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()

    def rate_limited(*args: object, **kwargs: object) -> None:
        raise script.IngestionError("FBref rate limit (HTTP 429)")

    monkeypatch.setattr(script, "fetch_fbref_tables", rate_limited)
    recorder = _Recorder()
    script.logger.addHandler(recorder)
    try:
        origin, parquet = script.run(out_dir=tmp_path)
    finally:
        script.logger.removeHandler(recorder)
    log_text = "\n".join(recorder.messages)
    assert origin == ORIGIN_BENCHMARK
    assert parquet.name == "benchmark_players_2023_2024.parquet"
    assert not (tmp_path / "real_players_2023_2024.parquet").exists()  # never mislabelled as real
    assert "429" in log_text and "NOT real player data" in log_text

    bench = pd.read_parquet(parquet)
    assert (bench[DATA_ORIGIN_COLUMN] == ORIGIN_BENCHMARK).all()
    assert set(REQUIRED_RAW_COLUMNS) <= set(bench.columns)
    assert set(bench["League"]) == {"Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"}
    assert set(bench["Season"]) == {"2023-2024"}
    loaded = DataLoader().load_real(data_dir=tmp_path)
    assert len(loaded) > 300 and pipeline.dataset_reference_year(loaded) == 2024


def test_script_no_fallback_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = load_script()

    def offline(*args: object, **kwargs: object) -> None:
        raise script.IngestionError("network down")

    monkeypatch.setattr(script, "fetch_fbref_tables", offline)
    with pytest.raises(script.IngestionError):
        script.run(out_dir=tmp_path, allow_fallback=False)
    assert script.main(["--out-dir", str(tmp_path), "--no-fallback"]) == 2
    assert script.main(["--out-dir", str(tmp_path)]) == 1  # benchmark written, non-zero to flag it


@pytest.mark.parametrize(
    ("message", "expected"),
    [("429 Client Error: Too Many Requests", "rate limit"),
     ("CAPTCHA detected and could not be solved.", "CAPTCHA"),
     ("Connection reset by peer", "Connection reset")],
)
def test_fetch_classifies_network_failures(monkeypatch: pytest.MonkeyPatch, message: str, expected: str) -> None:
    soccerdata = pytest.importorskip("soccerdata")

    script = load_script()

    class Failing:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ConnectionError(message)

    monkeypatch.setattr(soccerdata, "FBref", Failing)
    with pytest.raises(script.IngestionError, match=expected):
        script.fetch_fbref_tables("2023-2024", ["ENG-Premier League"])


def test_fetch_uses_headless_browser_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    soccerdata = pytest.importorskip("soccerdata")

    script = load_script()
    seen: dict[str, object] = {}

    class Recorder:
        def __init__(self, leagues: object, seasons: object, headless: bool) -> None:
            seen.update(leagues=leagues, seasons=seasons, headless=headless)

        def read_player_season_stats(self, stat_type: str) -> pd.DataFrame:
            return fbref_tables()[stat_type]

        def read_team_season_stats(self, stat_type: str) -> pd.DataFrame:
            return fbref_team_table()

    monkeypatch.setattr(soccerdata, "FBref", Recorder)
    tables, team = script.fetch_fbref_tables("2023-2024", ["ENG-Premier League"])
    assert seen == {"leagues": ["ENG-Premier League"], "seasons": "2324", "headless": True}
    assert set(tables) == set(ft.PLAYER_STAT_TABLES) and len(team) == len(TEAMS)


def test_soccerdata_season_codes() -> None:
    script = load_script()
    assert script.soccerdata_season("2023-2024") == "2324"
    assert script.soccerdata_season("1999-2000") == "9900"
