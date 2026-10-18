"""Single source of truth for schema, league metadata and feature definitions.

Every module (generator, loader, preprocessor, models, UI) imports column names
from here so that a schema change is made in exactly one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Leagues
# --------------------------------------------------------------------------- #
TOP_5_LEAGUES: Final[tuple[str, ...]] = (
    "Premier League",
    "La Liga",
    "Serie A",
    "Bundesliga",
    "Ligue 1",
)
SECONDARY_LEAGUES: Final[tuple[str, ...]] = (
    "Eredivisie",
    "Primeira Liga",
    "Championship",
)
ALL_LEAGUES: Final[tuple[str, ...]] = TOP_5_LEAGUES + SECONDARY_LEAGUES

# League Difficulty Multiplier applied to attacking/creative output when
# comparing players across leagues. Unknown leagues fall back to the default.
LEAGUE_STRENGTH_COEFFICIENTS: Final[dict[str, float]] = {
    **{league: 1.00 for league in TOP_5_LEAGUES},
    **{league: 0.85 for league in SECONDARY_LEAGUES},
}
DEFAULT_LEAGUE_COEFFICIENT: Final[float] = 0.85

# Matches per league season (used to cap plausible minutes).
LEAGUE_MATCHES: Final[dict[str, int]] = {
    "Premier League": 38,
    "La Liga": 38,
    "Serie A": 38,
    "Bundesliga": 34,
    "Ligue 1": 34,
    "Eredivisie": 34,
    "Primeira Liga": 34,
    "Championship": 46,
}

# --------------------------------------------------------------------------- #
# Positions
# --------------------------------------------------------------------------- #
POSITION_GROUPS: Final[tuple[str, ...]] = ("DF", "MF", "FW")

# --------------------------------------------------------------------------- #
# Raw (season-total) schema, mirroring FBref standard/passing/possession tables
# --------------------------------------------------------------------------- #
METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "Player",
    "Squad",
    "League",
    "Position",
    "Age",
    "Minutes_Played",
    "Market_Value_EUR_M",
    "Contract_Expiry_Year",
    "Team_Possession_Pct",
)

# Extra descriptive columns the synthetic DB provides; optional for uploads.
OPTIONAL_METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "Nation",
    "Season",
    "Born",
    "Matches_Played",
    "Starts",
    "Nineties",
)

# Season-total counting stats that the loader converts to per-90.
RAW_COUNT_COLUMNS: Final[tuple[str, ...]] = (
    "Goals",
    "Shots",
    "Assists",
    "Key_Passes",
    "Passes_Completed",
    "Passes_Attempted",
    "Progressive_Passes",
    "Take_Ons_Attempted",
    "Successful_Take_Ons",
    "Tackles_Won",
    "Interceptions",
    "Progressive_Carries",
)

# Season-total volume stats that are not whole counts (expected goals).
RAW_EXPECTED_COLUMNS: Final[tuple[str, ...]] = ("Non_Penalty_xG",)

# Everything the loader converts to per-90.
RAW_VOLUME_COLUMNS: Final[tuple[str, ...]] = RAW_COUNT_COLUMNS + RAW_EXPECTED_COLUMNS

# Rate stats delivered as percentages, not converted to per-90.
RAW_RATE_COLUMNS: Final[tuple[str, ...]] = ("Pass_Completion_Pct", "Aerial_Win_Pct")

REQUIRED_RAW_COLUMNS: Final[tuple[str, ...]] = (
    METADATA_COLUMNS
    + (
        "Goals",
        "Shots",
        "Assists",
        "Key_Passes",
        "Pass_Completion_Pct",
        "Progressive_Passes",
        "Successful_Take_Ons",
        "Tackles_Won",
        "Interceptions",
        "Non_Penalty_xG",
        "Progressive_Carries",
        "Aerial_Win_Pct",
    )
)

# --------------------------------------------------------------------------- #
# Model feature vector
# --------------------------------------------------------------------------- #
CORE_FEATURES: Final[tuple[str, ...]] = (
    "Goals_p90",
    "Shots_p90",
    "Non_Penalty_xG_p90",
    "Assists_p90",
    "Key_Passes_p90",
    "Pass_Completion_Pct",
    "Progressive_Passes_p90",
    "Successful_Take_Ons_p90",
    "Progressive_Carries_p90",
    "pAdj_Tackles_Won_p90",
    "pAdj_Interceptions_p90",
    "Aerial_Win_Pct",
)

FEATURE_LABELS: Final[dict[str, str]] = {
    "Goals_p90": "Clinical Finishing",
    "Shots_p90": "Shot Volume",
    "Non_Penalty_xG_p90": "Chance Quality",
    "Assists_p90": "Final-Ball Output",
    "Key_Passes_p90": "Chance Creation",
    "Pass_Completion_Pct": "Ball Retention",
    "Progressive_Passes_p90": "Line-Breaking Passing",
    "Successful_Take_Ons_p90": "1v1 Dribbling",
    "Progressive_Carries_p90": "Ball Carrying",
    "pAdj_Tackles_Won_p90": "Ball Winning",
    "pAdj_Interceptions_p90": "Reading of Play",
    "Aerial_Win_Pct": "Aerial Dominance",
}

# Feature categories drive both the composite indices and the similarity weights.
FEATURE_CATEGORIES: Final[dict[str, tuple[str, ...]]] = {
    "Scoring": ("Goals_p90", "Shots_p90", "Non_Penalty_xG_p90"),
    "Creation & Passing": (
        "Assists_p90",
        "Key_Passes_p90",
        "Pass_Completion_Pct",
        "Progressive_Passes_p90",
    ),
    "Dribbling": ("Successful_Take_Ons_p90", "Progressive_Carries_p90"),
    "Defending": ("pAdj_Tackles_Won_p90", "pAdj_Interceptions_p90", "Aerial_Win_Pct"),
}

# Per-90 defensive stats corrected for team possession (pAdj sigmoid).
POSSESSION_ADJUSTED_STATS: Final[tuple[str, ...]] = (
    "Tackles_Won_p90",
    "Interceptions_p90",
)

# League scaling applies to every per-90 volume stat (any "_p90" column);
# efficiency percentages such as these are never league-scaled.
PER90_SUFFIX: Final[str] = "_p90"

# Composite 0-100 sub-scores, keyed by the feature category they summarise.
COMPOSITE_SCORES: Final[dict[str, str]] = {
    "Scoring": "Score_Finishing",
    "Creation & Passing": "Score_Creation",
    "Dribbling": "Score_Dribbling",
    "Defending": "Score_Defending",
}

# Suffix for within-position percentile columns (e.g. "Goals_p90_pct").
PERCENTILE_SUFFIX: Final[str] = "_pct"

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
DEFAULT_MIN_MINUTES: Final[int] = 900
# Steepness k of the pAdj sigmoid 2 / (1 + exp(-k * (possession - 50))).
# 0.05 gives ~1.25x at 60% possession; 0.1 over-corrects (~1.46x).
DEFAULT_PADJ_STEEPNESS: Final[float] = 0.05
# Season the data describes; contract "years remaining" are measured from it.
CURRENT_SEASON_END_YEAR: Final[int] = 2026

# --------------------------------------------------------------------------- #
# Real data (FBref ingestion)
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed"
DEFAULT_REAL_SEASON: Final[str] = "2023-2024"
# Real seasons with full advanced stats, oldest first; the newest is the default.
REAL_SEASONS: Final[tuple[str, ...]] = ("2023-2024", "2024-2025")

# soccerdata league id -> display name used throughout the platform.
FBREF_LEAGUES: Final[dict[str, str]] = {
    "ENG-Premier League": "Premier League",
    "ESP-La Liga": "La Liga",
    "GER-Bundesliga": "Bundesliga",
    "ITA-Serie A": "Serie A",
    "FRA-Ligue 1": "Ligue 1",
}

# Provenance column values: where each processed dataset came from.
DATA_ORIGIN_COLUMN: Final[str] = "Data_Origin"
ORIGIN_FBREF: Final[str] = "fbref"
ORIGIN_BENCHMARK: Final[str] = "synthetic-benchmark"


def real_data_filename(season: str, origin: str = ORIGIN_FBREF) -> str:
    """``real_players_2023_2024`` or ``benchmark_players_2023_2024`` (no suffix)."""
    prefix = "real_players" if origin == ORIGIN_FBREF else "benchmark_players"
    return f"{prefix}_{season.replace('-', '_')}"


def season_end_year(season: str) -> int:
    """``"2023-2024"`` / ``"2324"`` / ``"2024"`` -> 2024."""
    digits = "".join(ch for ch in str(season) if ch.isdigit())
    if len(digits) == 8:
        return int(digits[4:])
    if len(digits) == 4 and (int(digits[2:]) - int(digits[:2])) % 100 == 1:
        # soccerdata short form: "2324" -> 2024, "9900" -> 2000.
        end = int(digits[2:])
        return (2000 if end < 50 else 1900) + end
    if len(digits) == 4:
        return int(digits)
    raise ValueError(f"Unrecognised season format: {season!r}")
RANDOM_STATE: Final[int] = 42
