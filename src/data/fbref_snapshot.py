"""Offline adapter for a pre-2026 FBref season export (Kaggle snapshot).

FBref no longer serves its advanced Opta tables (passing, possession,
defense), and ``soccerdata`` 1.9.x only reads ``standard``, ``keeper``,
``shooting``, ``playing_time`` and ``misc``. Full 2023-24 season tables
therefore come from a snapshot scraped before the change:

    Kaggle: akshankrithick/fbref-2017-2024-for-europes-top-5-leagues
    file:   cleaned_2023-24.csv  (one row per player per club)

This module maps that export onto the raw platform schema and reuses the
FBref transforms (transfer deduplication, count finalisation), so the output
is interchangeable with :func:`src.data.fbref_transform.build_raw_player_table`.

Team possession is not in the export. It is parsed from FBref's squad table,
which is embedded in the league player-stats pages ``soccerdata`` already
caches (``players_<league>_<season>_standard.html``).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from src.config import FBREF_LEAGUES
from src.data.fbref_transform import (
    DedupReport,
    FBrefSchemaError,
    deduplicate_transfers,
    finalize_counts,
    to_number,
)

logger = logging.getLogger(__name__)

KAGGLE_DATASET = "akshankrithick/fbref-2017-2024-for-europes-top-5-leagues"

# Snapshot column -> raw schema column. ``Avg Mins per Match`` is mislabelled
# in the export: it holds *season total* minutes (max 3420 = 38 x 90).
SNAPSHOT_COLUMN_MAP: dict[str, str] = {
    "player": "Player",
    "nation": "Nation",
    "pos": "Position",
    "squad": "Squad",
    "comp": "League",
    "age": "Age",
    "born": "Born",
    "season": "Season",
    "Matches Played": "Matches_Played",
    "Avg Mins per Match": "Minutes_Played",
    "Goals": "Goals",
    "Assists": "Assists",
    "Total Shots": "Shots",
    "Passes Completed": "Passes_Completed",
    "Passes Attempted": "Passes_Attempted",
    "Key passes": "Key_Passes",
    "Progressive Passes": "Progressive_Passes",
    "Take ons attempted": "Take_Ons_Attempted",
    "% Successful take-ons": "_Take_On_Success_Pct",
    "Tackles Won": "Tackles_Won",
    "Interceptions": "Interceptions",
}

COUNT_COLUMNS: tuple[str, ...] = (
    "Matches_Played", "Minutes_Played", "Goals", "Assists", "Shots", "Passes_Completed",
    "Passes_Attempted", "Key_Passes", "Progressive_Passes", "Take_Ons_Attempted", "Tackles_Won",
    "Interceptions",
)

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


def read_snapshot(path: str | Path) -> pd.DataFrame:
    """Read the export CSV as-is (validation happens in :func:`snapshot_to_raw`)."""
    return pd.read_csv(path)


def snapshot_to_raw(snapshot: pd.DataFrame, season: str | None = None) -> pd.DataFrame:
    """Map the export onto schema names; one row per (player, club).

    Successful take-ons are rebuilt from attempts x success rate, the only
    form the export carries.

    Raises:
        FBrefSchemaError: a required column is missing, or no rows match ``season``.
    """
    missing = [c for c in SNAPSHOT_COLUMN_MAP if c not in snapshot.columns]
    if missing:
        raise FBrefSchemaError(
            f"snapshot is missing {missing}. Available: {sorted(map(str, snapshot.columns))[:30]}…"
        )
    df = snapshot.loc[:, list(SNAPSHOT_COLUMN_MAP)].rename(columns=SNAPSHOT_COLUMN_MAP)
    if season is not None:
        df = df[df["Season"].astype(str) == season]
        if df.empty:
            found = sorted(snapshot["season"].astype(str).unique())
            raise FBrefSchemaError(f"snapshot has no rows for season {season!r} (found {found})")

    for col in (*COUNT_COLUMNS, "_Take_On_Success_Pct"):
        df[col] = to_number(df[col]).fillna(0.0)
    df["Successful_Take_Ons"] = (df["Take_Ons_Attempted"] * df["_Take_On_Success_Pct"] / 100.0).round()
    df = df.drop(columns=["_Take_On_Success_Pct"])
    df["Age"] = to_number(df["Age"])
    df["Born"] = to_number(df["Born"])
    for col in ("Player", "Squad", "League", "Position", "Nation"):
        df[col] = df[col].fillna("").astype(str).str.strip()
    return df.reset_index(drop=True)


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
            logger.warning("No cached FBref page for %s (%s)", league_name, page.name)
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


def build_snapshot_dataset(
    snapshot: pd.DataFrame,
    possession: pd.DataFrame | None,
    season: str | None = None,
) -> tuple[pd.DataFrame, DedupReport]:
    """Export -> one season-total row per player in the raw schema (pre-enrichment)."""
    from src.data.real_data_enricher import map_team_possession  # avoid import cycle

    raw = snapshot_to_raw(snapshot, season)
    if possession is not None and not possession.empty:
        raw = map_team_possession(raw, possession)
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
    "KAGGLE_DATASET", "SNAPSHOT_COLUMN_MAP", "SQUAD_ALIASES", "build_snapshot_dataset",
    "parse_cached_possession", "read_snapshot", "season_code", "snapshot_to_raw",
]
