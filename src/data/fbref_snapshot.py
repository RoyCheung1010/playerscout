"""Offline adapters for FBref season exports scraped before 2026 (Kaggle).

FBref no longer serves its advanced Opta tables (passing, possession,
defense), and ``soccerdata`` 1.9.x only reads ``standard``, ``keeper``,
``shooting``, ``playing_time`` and ``misc``. Full season tables therefore
come from exports scraped before the change. Two layouts are supported and
detected from their columns:

* ``fbref-2017-2024`` (akshankrithick/fbref-2017-2024-for-europes-top-5-leagues,
  e.g. ``cleaned_2023-24.csv``): renamed columns, a mislabelled minutes
  column, take-ons as attempts + success rate.
* ``fbref-big5`` (hubertsidorowicz/football-players-stats-2024-2025,
  ``players_data-2024_2025.csv``): FBref's own column codes, one table per
  stat category merged side by side (duplicate names get ``_stats_<table>``).

Both are mapped onto the raw platform schema and run through the FBref
transforms (transfer deduplication, count finalisation), so the output is
interchangeable with :func:`src.data.fbref_transform.build_raw_player_table`.

Team possession is in neither export. It is parsed from FBref's squad table
in league pages ``soccerdata`` has cached, and otherwise **estimated** from
the players' pass volumes (see :func:`estimate_possession`). Every row records
which in ``Possession_Source``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FBREF_LEAGUES
from src.data.fbref_transform import (
    DedupReport,
    FBrefSchemaError,
    deduplicate_transfers,
    fill_rate_gaps,
    finalize_counts,
    to_number,
)

logger = logging.getLogger(__name__)

POSSESSION_SOURCE_COLUMN = "Possession_Source"
POSSESSION_FBREF = "fbref"
POSSESSION_ESTIMATED = "estimated"


@dataclass(frozen=True)
class SnapshotFormat:
    """How one export's columns map onto the raw schema."""

    name: str
    column_map: dict[str, str]
    signature: tuple[str, ...]  # columns that identify the layout
    take_on_rate_column: str | None = None  # successful take-ons = attempts x rate
    league_prefix: bool = False  # "eng Premier League" -> "Premier League"
    nation_code_last: bool = False  # "eg EGY" -> "EGY"


FBREF_2017_2024 = SnapshotFormat(
    name="fbref-2017-2024",
    # ``Avg Mins per Match`` is mislabelled in this export: it holds *season
    # total* minutes (max 3420 = 38 x 90).
    column_map={
        "player": "Player", "nation": "Nation", "pos": "Position", "squad": "Squad", "comp": "League",
        "age": "Age", "born": "Born", "season": "Season", "Matches Played": "Matches_Played",
        "Avg Mins per Match": "Minutes_Played", "Goals": "Goals", "Assists": "Assists",
        "Total Shots": "Shots", "Passes Completed": "Passes_Completed",
        "Passes Attempted": "Passes_Attempted", "Key passes": "Key_Passes",
        "Progressive Passes": "Progressive_Passes", "Take ons attempted": "Take_Ons_Attempted",
        "% Successful take-ons": "_Take_On_Success_Pct", "Tackles Won": "Tackles_Won",
        "Interceptions": "Interceptions", "Exp NPG": "Non_Penalty_xG",
        "Progressive Carries": "Progressive_Carries", "% Aerial Duels won": "Aerial_Win_Pct",
    },
    signature=("Avg Mins per Match", "Exp NPG"),
    take_on_rate_column="_Take_On_Success_Pct",
)

FBREF_BIG5 = SnapshotFormat(
    name="fbref-big5",
    column_map={
        "Player": "Player", "Nation": "Nation", "Pos": "Position", "Squad": "Squad", "Comp": "League",
        "Age": "Age", "Born": "Born", "MP": "Matches_Played", "Starts": "Starts", "Min": "Minutes_Played",
        "90s": "Nineties", "Gls": "Goals", "Ast": "Assists", "Sh": "Shots", "Cmp": "Passes_Completed",
        "Att": "Passes_Attempted", "KP": "Key_Passes", "PrgP": "Progressive_Passes",
        "Att_stats_possession": "Take_Ons_Attempted", "Succ": "Successful_Take_Ons",
        "TklW": "Tackles_Won", "Int": "Interceptions", "npxG": "Non_Penalty_xG",
        "PrgC": "Progressive_Carries", "Won%": "Aerial_Win_Pct",
    },
    signature=("Att_stats_possession", "npxG", "Won%"),
    league_prefix=True,
    nation_code_last=True,
)

SNAPSHOT_FORMATS: tuple[SnapshotFormat, ...] = (FBREF_2017_2024, FBREF_BIG5)

# Kaggle source per season: (dataset handle, file name).
SEASON_SOURCES: dict[str, tuple[str, str]] = {
    "2023-2024": ("akshankrithick/fbref-2017-2024-for-europes-top-5-leagues", "cleaned_2023-24.csv"),
    "2024-2025": ("hubertsidorowicz/football-players-stats-2024-2025", "players_data-2024_2025.csv"),
}
# Back-compatible alias: the 2023-24 source.
KAGGLE_DATASET = SEASON_SOURCES["2023-2024"][0]

# Kept for callers/tests that refer to the original 2023-24 mapping.
SNAPSHOT_COLUMN_MAP = FBREF_2017_2024.column_map

COUNT_COLUMNS: tuple[str, ...] = (
    "Matches_Played", "Minutes_Played", "Goals", "Assists", "Shots", "Passes_Completed",
    "Passes_Attempted", "Key_Passes", "Progressive_Passes", "Take_Ons_Attempted", "Tackles_Won",
    "Interceptions", "Progressive_Carries", "Non_Penalty_xG",
)
OPTIONAL_COUNT_COLUMNS: tuple[str, ...] = ("Starts", "Nineties", "Successful_Take_Ons")

# FBref's squad table and its player table spell six clubs differently.
# Squad-table name -> player-table name (the latter is what the platform shows).
SQUAD_ALIASES: dict[str, str] = {
    "Frankfurt": "Eint Frankfurt",
    "Newcastle": "Newcastle Utd",
    "Nottingham": "Nott'ham Forest",
    "Paris SG": "Paris S-G",
    "Real Betis": "Betis",
    "Sheffield United": "Sheffield Utd",
}

# Possession = share of passes attempted (FBref's own definition). A club's
# share against the league's average opponent compresses the extremes, so it
# is stretched around 50% by a slope fitted on 2023-24, where FBref's real
# possession is known for all 96 clubs: MAE 2.11 -> 1.31 points, max 4.5.
POSSESSION_CALIBRATION_SLOPE: float = 1.4786


# --------------------------------------------------------------------------- #
# Reading and mapping
# --------------------------------------------------------------------------- #
def read_snapshot(path: str | Path) -> pd.DataFrame:
    """Read the export CSV as-is (validation happens in :func:`snapshot_to_raw`)."""
    return pd.read_csv(path)


def detect_format(snapshot: pd.DataFrame) -> SnapshotFormat:
    """Identify the export layout from its columns.

    Raises:
        FBrefSchemaError: no known layout matches.
    """
    columns = set(map(str, snapshot.columns))
    for fmt in SNAPSHOT_FORMATS:
        if set(fmt.signature) <= columns:
            return fmt
    raise FBrefSchemaError(
        f"unrecognised FBref export: expected one of {[f.signature for f in SNAPSHOT_FORMATS]}. "
        f"Available: {sorted(columns)[:30]}…"
    )


def snapshot_to_raw(snapshot: pd.DataFrame, season: str | None = None) -> pd.DataFrame:
    """Map an export onto schema names; one row per (player, club).

    Raises:
        FBrefSchemaError: unknown layout, a required column is missing, or no
            rows match ``season``.
    """
    fmt = detect_format(snapshot)
    missing = [c for c in fmt.column_map if c not in snapshot.columns]
    if missing:
        raise FBrefSchemaError(
            f"{fmt.name} snapshot is missing {missing}. Available: {sorted(map(str, snapshot.columns))[:30]}…"
        )
    df = snapshot.loc[:, list(fmt.column_map)].rename(columns=fmt.column_map)
    if "Season" in df.columns and season is not None:
        df = df[df["Season"].astype(str) == season]
        if df.empty:
            found = sorted(snapshot["season"].astype(str).unique())
            raise FBrefSchemaError(f"snapshot has no rows for season {season!r} (found {found})")
    elif season is not None:
        df = df.assign(Season=season)

    for col in (*COUNT_COLUMNS, *OPTIONAL_COUNT_COLUMNS):
        if col in df.columns:
            df[col] = to_number(df[col]).fillna(0.0)
    if fmt.take_on_rate_column:
        rate = to_number(df[fmt.take_on_rate_column]).fillna(0.0)
        df["Successful_Take_Ons"] = (df["Take_Ons_Attempted"] * rate / 100.0).round()
        df = df.drop(columns=[fmt.take_on_rate_column])
    for col in ("Player", "Squad", "League", "Position", "Nation"):
        df[col] = df[col].fillna("").astype(str).str.strip()
    if fmt.league_prefix:
        df["League"] = df["League"].str.replace(r"^[a-z]{2,3}\s+", "", regex=True)
    if fmt.nation_code_last:
        df["Nation"] = df["Nation"].str.split().str[-1].fillna("")
    df["Age"] = to_number(df["Age"])
    df["Born"] = to_number(df["Born"])
    df["Aerial_Win_Pct"] = fill_rate_gaps(df, "Aerial_Win_Pct")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Possession
# --------------------------------------------------------------------------- #
def parse_cached_possession(cache_dir: str | Path, season_code: str) -> pd.DataFrame:
    """Team possession from soccerdata's cached FBref league pages.

    Reads ``stats_squads_standard_for`` (sometimes wrapped in an HTML comment)
    from ``players_<league>_<season_code>_standard.html`` for each Top 5
    league found. Returns ``(League, Squad, Team_Possession_Pct)``; empty if
    nothing is cached.
    """
    from lxml import html  # ingestion-only dependency (installed with soccerdata)

    rows: list[tuple[str, str, float]] = []
    for league_id, league_name in FBREF_LEAGUES.items():
        page = Path(cache_dir) / f"players_{league_id}_{season_code}_standard.html"
        if not page.exists():
            logger.info("No cached FBref page for %s (%s)", league_name, page.name)
            continue
        tree = html.parse(str(page))
        table = _find_table(tree, "stats_squads_standard_for")
        if table is None:
            logger.warning("No squad table in %s", page.name)
            continue
        for tr in table.xpath(".//tbody/tr"):
            squad = "".join(tr.xpath(".//*[@data-stat='team']//text()")).strip()
            poss = "".join(tr.xpath(".//*[@data-stat='possession']//text()")).strip()
            if squad and poss:
                rows.append((league_name, SQUAD_ALIASES.get(squad, squad), float(poss)))
    return pd.DataFrame(rows, columns=["League", "Squad", "Team_Possession_Pct"])


def estimate_possession(raw: pd.DataFrame, slope: float = POSSESSION_CALIBRATION_SLOPE) -> pd.DataFrame:
    """Estimate each club's possession from its players' passes attempted.

    ``share = P_club / (P_club + mean P of the league's other clubs)`` is a
    club's share of passes against an average opponent (in a double round
    robin every club faces every other equally), then stretched around 50%
    by ``slope``. Returns ``(League, Squad, Team_Possession_Pct)``.
    """
    team = raw.groupby(["League", "Squad"], as_index=False)["Passes_Attempted"].sum()
    n = team.groupby("League")["Passes_Attempted"].transform("count")
    total = team.groupby("League")["Passes_Attempted"].transform("sum")
    others = (total - team["Passes_Attempted"]) / (n - 1).where(n > 1)
    share = 100.0 * team["Passes_Attempted"] / (team["Passes_Attempted"] + others)
    estimate = (50.0 + slope * (share - 50.0)).fillna(50.0).clip(30.0, 75.0).round(1)
    return team.assign(Team_Possession_Pct=estimate)[["League", "Squad", "Team_Possession_Pct"]]


def combine_possession(cached: pd.DataFrame | None, estimated: pd.DataFrame) -> pd.DataFrame:
    """FBref possession where cached, the estimate for every other club.

    Adds ``Possession_Source`` (``"fbref"`` / ``"estimated"``) per club.
    """
    est = estimated.assign(**{POSSESSION_SOURCE_COLUMN: POSSESSION_ESTIMATED})
    if cached is None or cached.empty:
        return est
    real = cached.assign(**{POSSESSION_SOURCE_COLUMN: POSSESSION_FBREF})
    merged = est.merge(real, on=["League", "Squad"], how="left", suffixes=("_est", ""))
    use_real = merged["Team_Possession_Pct"].notna()
    merged["Team_Possession_Pct"] = merged["Team_Possession_Pct"].where(use_real, merged["Team_Possession_Pct_est"])
    merged[POSSESSION_SOURCE_COLUMN] = np.where(use_real, POSSESSION_FBREF, POSSESSION_ESTIMATED)
    return merged[["League", "Squad", "Team_Possession_Pct", POSSESSION_SOURCE_COLUMN]]


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build_snapshot_dataset(
    snapshot: pd.DataFrame,
    possession: pd.DataFrame | None,
    season: str | None = None,
) -> tuple[pd.DataFrame, DedupReport]:
    """Export -> one season-total row per player in the raw schema (pre-enrichment).

    ``possession`` is FBref's real possession (may be partial or ``None``);
    clubs it does not cover get the pass-share estimate.
    """
    from src.data.real_data_enricher import map_team_possession  # avoid import cycle

    raw = snapshot_to_raw(snapshot, season)
    table = combine_possession(possession, estimate_possession(raw))
    raw = map_team_possession(raw, table)
    sources = table.set_index(["League", "Squad"])[POSSESSION_SOURCE_COLUMN]
    raw[POSSESSION_SOURCE_COLUMN] = [sources.get((lg, sq), POSSESSION_ESTIMATED)
                                     for lg, sq in zip(raw["League"], raw["Squad"])]
    deduped, report = deduplicate_transfers(raw)
    return finalize_counts(deduped), report


def _find_table(tree: object, table_id: str) -> object | None:
    from lxml import html

    found = tree.xpath(f"//table[@id='{table_id}']")  # type: ignore[attr-defined]
    if found:
        return found[0]
    for comment in tree.xpath("//comment()"):  # type: ignore[attr-defined]
        text = comment.text or ""
        if table_id in text:
            inner = html.fromstring(text).xpath(f"//table[@id='{table_id}']")
            if inner:
                return inner[0]
    return None


def season_code(season: str) -> str:
    """``"2023-2024"`` -> soccerdata's cache code ``"2324"``."""
    years = re.findall(r"\d{4}", season)
    if len(years) != 2:
        raise ValueError(f"expected a season like '2023-2024', got {season!r}")
    return f"{int(years[0]) % 100:02d}{int(years[1]) % 100:02d}"


__all__ = [
    "FBREF_2017_2024", "FBREF_BIG5", "KAGGLE_DATASET", "POSSESSION_CALIBRATION_SLOPE",
    "POSSESSION_SOURCE_COLUMN", "SEASON_SOURCES", "SNAPSHOT_COLUMN_MAP", "SNAPSHOT_FORMATS",
    "SQUAD_ALIASES", "build_snapshot_dataset", "combine_possession", "detect_format",
    "estimate_possession", "parse_cached_possession", "read_snapshot", "season_code", "snapshot_to_raw",
]
