"""Offline transforms from ``soccerdata.FBref`` tables to the raw platform schema.

Nothing here touches the network: ``scripts/ingest_fbref.py`` fetches the
tables and hands them to :func:`build_raw_player_table`. Keeping the parsing
pure means it is fully unit-tested against FBref-shaped fixtures.

soccerdata returns one frame per stat table, indexed by
``(league, season, team, player)`` with two-level column headers such as
``("Playing Time", "Min")`` or ``("KP", "")``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import FBREF_LEAGUES

# Stat tables pulled per league (soccerdata ``stat_type`` names).
PLAYER_STAT_TABLES: tuple[str, ...] = ("standard", "shooting", "passing", "possession", "defense")

INDEX_COLUMNS: tuple[str, ...] = ("league", "season", "team", "player")

# Flattened FBref column -> raw schema column, per table. Keys are normalised
# with :func:`_normalise` so header case/spacing differences do not matter.
COLUMN_MAP: dict[str, dict[str, str]] = {
    "standard": {
        "nation": "Nation",
        "pos": "Position",
        "age": "Age",
        "born": "Born",
        "playing time_mp": "Matches_Played",
        "playing time_starts": "Starts",
        "playing time_min": "Minutes_Played",
        "playing time_90s": "Nineties",
        "performance_gls": "Goals",
        "performance_ast": "Assists",
    },
    "shooting": {"standard_sh": "Shots"},
    "passing": {
        "total_cmp": "Passes_Completed",
        "total_att": "Passes_Attempted",
        "kp": "Key_Passes",
        "prgp": "Progressive_Passes",
    },
    "possession": {
        "take-ons_att": "Take_Ons_Attempted",
        "take-ons_succ": "Successful_Take_Ons",
    },
    "defense": {
        "tackles_tklw": "Tackles_Won",
        "int": "Interceptions",
    },
}

# Season-total counts: summed across clubs for mid-season transfers.
SUMMED_COLUMNS: tuple[str, ...] = (
    "Matches_Played", "Starts", "Minutes_Played", "Nineties", "Goals", "Assists", "Shots",
    "Passes_Completed", "Passes_Attempted", "Key_Passes", "Progressive_Passes",
    "Take_Ons_Attempted", "Successful_Take_Ons", "Tackles_Won", "Interceptions",
)


class FBrefSchemaError(ValueError):
    """Raised when an FBref table is missing columns the platform needs."""


@dataclass(frozen=True)
class DedupReport:
    n_rows_in: int
    n_players_out: int
    n_transferred: int


# --------------------------------------------------------------------------- #
# Column handling
# --------------------------------------------------------------------------- #
def flatten_fbref_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Join multi-level headers with ``_`` and drop empty levels.

    ``("Playing Time", "Min")`` -> ``"Playing Time_Min"``; ``("KP", "")`` -> ``"KP"``.
    Index levels are reset into ordinary columns. Single-level frames pass
    through unchanged (apart from the index reset).
    """
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "_".join(str(level).strip() for level in col if str(level).strip() and not str(level).startswith("Unnamed"))
            for col in out.columns
        ]
    else:
        out.columns = [str(c).strip() for c in out.columns]
    has_named_index = any(name is not None for name in out.index.names)
    return out.reset_index() if has_named_index else out


def _normalise(name: str) -> str:
    return re.sub(r"\s+", " ", str(name)).strip().lower()


def select_table_columns(flat: pd.DataFrame, table: str) -> pd.DataFrame:
    """Keep the index keys plus the mapped stats of one table, renamed to schema names.

    Raises:
        FBrefSchemaError: a required FBref column is absent.
    """
    lookup = {_normalise(c): c for c in flat.columns}
    missing_keys = [k for k in INDEX_COLUMNS if k not in lookup]
    if missing_keys:
        raise FBrefSchemaError(f"{table}: missing index columns {missing_keys}")
    mapping = COLUMN_MAP[table]
    missing = [src for src in mapping if src not in lookup]
    if missing:
        raise FBrefSchemaError(
            f"{table}: FBref table is missing {missing}. Available: {sorted(lookup)[:25]}…"
        )
    keep = {lookup[k]: k for k in INDEX_COLUMNS}
    keep.update({lookup[src]: dst for src, dst in mapping.items()})
    return flat.loc[:, list(keep)].rename(columns=keep)


def to_number(series: pd.Series) -> pd.Series:
    """FBref numbers can arrive as strings with thousands separators."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    cleaned = series.astype("string").str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def parse_age(series: pd.Series) -> pd.Series:
    """FBref ages look like ``"23-145"`` (years-days); keep whole years."""
    years = series.astype("string").str.extract(r"^\s*(\d+)")[0]
    return pd.to_numeric(years, errors="coerce")


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def merge_player_tables(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Join the five stat tables into one row per (player, club), schema-named.

    The standard table defines the player list; secondary tables are left-joined
    so a player with no logged defensive actions gets 0, not a dropped row.
    League ids become display names (``"ENG-Premier League"`` -> ``"Premier
    League"``) and seasons become ``"2023-2024"``.
    """
    missing = [t for t in PLAYER_STAT_TABLES if t not in tables]
    if missing:
        raise FBrefSchemaError(f"Missing stat tables: {missing}")
    keys = list(INDEX_COLUMNS)
    merged: pd.DataFrame | None = None
    for table in PLAYER_STAT_TABLES:
        part = select_table_columns(flatten_fbref_columns(tables[table]), table)
        part = part.drop_duplicates(subset=keys, keep="first")
        merged = part if merged is None else merged.merge(part, on=keys, how="left")
    assert merged is not None
    for col in SUMMED_COLUMNS:
        merged[col] = to_number(merged[col]).fillna(0.0)
    merged["Age"] = parse_age(merged["Age"])
    merged["Born"] = to_number(merged["Born"])
    merged["Position"] = merged["Position"].fillna("").astype(str)
    merged["League"] = merged["league"].map(lambda v: FBREF_LEAGUES.get(str(v), str(v)))
    merged["Season"] = merged["season"].map(_season_label)
    merged = merged.rename(columns={"player": "Player", "team": "Squad"}).drop(columns=["league", "season"])
    return merged.reset_index(drop=True)


def deduplicate_transfers(players: pd.DataFrame) -> tuple[pd.DataFrame, DedupReport]:
    """Collapse mid-season transfers into one season-total row per player.

    soccerdata's season tables list a transferred player once per club, with no
    aggregate row and no dates. So:

    * counting stats are **summed** across clubs;
    * the player is assigned to the club where they played the **most minutes**
      (season tables cannot say which club was "latest");
    * ``Team_Possession_Pct``, if present, is the **minutes-weighted** average
      of the clubs' possession, since that is the context the stats came from;
    * players are keyed by ``(Player, Born)`` so namesakes are never merged.
    """
    df = players.copy()
    df["_born_key"] = df["Born"].fillna(-1).astype(int)
    keys = ["Player", "_born_key"]
    df = df.sort_values(keys + ["Minutes_Played"], ascending=[True, True, False], kind="mergesort")
    grouped = df.groupby(keys, sort=False)
    summed = _present(df, SUMMED_COLUMNS)

    first = grouped.first()  # metadata from the most-minutes club
    totals = grouped[summed].sum()
    merged = first.drop(columns=summed).join(totals)
    merged["Clubs_In_Season"] = grouped["Squad"].nunique()
    if "Team_Possession_Pct" in df.columns:
        weights = df["Minutes_Played"].clip(lower=0) + 1e-9  # keep 0-minute rows defined
        weighted = (df["Team_Possession_Pct"] * weights).groupby([df[k] for k in keys], sort=False).sum()
        merged["Team_Possession_Pct"] = (weighted / weights.groupby([df[k] for k in keys], sort=False).sum()).round(1)

    out = merged.reset_index().drop(columns=["_born_key"])
    n_transferred = int((out["Clubs_In_Season"] > 1).sum())
    return out, DedupReport(len(players), len(out), n_transferred)


def finalize_counts(players: pd.DataFrame) -> pd.DataFrame:
    """Recompute pass completion from totals and cast counts to integers."""
    out = players.copy()
    attempted = out["Passes_Attempted"].to_numpy(dtype=float)
    completed = out["Passes_Completed"].to_numpy(dtype=float)
    pct = np.zeros_like(attempted)
    np.divide(100.0 * completed, attempted, out=pct, where=attempted > 0)
    out["Pass_Completion_Pct"] = np.round(pct, 1)
    for col in _present(out, SUMMED_COLUMNS):
        if col != "Nineties":
            out[col] = out[col].round().astype(int)
    return out.sort_values(["League", "Squad", "Player"], ignore_index=True)


def _present(df: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    """Columns of ``columns`` that ``df`` has (sources differ, e.g. no ``Starts``)."""
    return [c for c in columns if c in df.columns]


def build_raw_player_table(
    tables: Mapping[str, pd.DataFrame],
    possession: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, DedupReport]:
    """FBref stat tables -> one season-total row per player in the raw schema.

    ``possession`` (from :func:`team_possession`) is joined per club *before*
    deduplication so transferred players get minutes-weighted possession.
    Valuation and contracts are added afterwards by the enricher.
    """
    from src.data.real_data_enricher import map_team_possession  # avoid import cycle

    merged = merge_player_tables(tables)
    if possession is not None:
        merged = map_team_possession(merged, possession)
    deduped, report = deduplicate_transfers(merged)
    return finalize_counts(deduped), report


def team_possession(team_table: pd.DataFrame) -> pd.DataFrame:
    """Team standard table -> ``(League, Squad, Team_Possession_Pct)``."""
    flat = flatten_fbref_columns(team_table)
    lookup = {_normalise(c): c for c in flat.columns}
    needed = ["league", "team", "poss"]
    missing = [k for k in needed if k not in lookup]
    if missing:
        raise FBrefSchemaError(f"team standard table is missing {missing}")
    out = pd.DataFrame({
        "League": flat[lookup["league"]].map(lambda v: FBREF_LEAGUES.get(str(v), str(v))),
        "Squad": flat[lookup["team"]].astype(str),
        "Team_Possession_Pct": to_number(flat[lookup["poss"]]),
    })
    return out.drop_duplicates(["League", "Squad"]).reset_index(drop=True)


def _season_label(value: object) -> str:
    """soccerdata season ``"2324"`` -> ``"2023-2024"``; passes long forms through."""
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) == 4 and (int(digits[2:]) - int(digits[:2])) % 100 == 1:
        start = int(digits[:2])
        century = 2000 if start < 50 else 1900
        return f"{century + start}-{century + start + 1}"
    if len(digits) == 8:
        return f"{digits[:4]}-{digits[4:]}"
    return text
