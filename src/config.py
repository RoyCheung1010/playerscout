"""Single source of truth for schema, league metadata and feature definitions.

Every module (generator, loader, preprocessor, models, UI) imports column names
from here so that a schema change is made in exactly one place.
"""

from __future__ import annotations

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
)

# Rate stat delivered as a percentage, not converted to per-90.
RAW_RATE_COLUMNS: Final[tuple[str, ...]] = ("Pass_Completion_Pct",)

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
    )
)

# --------------------------------------------------------------------------- #
# Model feature vector
# --------------------------------------------------------------------------- #
CORE_FEATURES: Final[tuple[str, ...]] = (
    "Goals_p90",
    "Shots_p90",
    "Assists_p90",
    "Key_Passes_p90",
    "Pass_Completion_Pct",
    "Progressive_Passes_p90",
    "Successful_Take_Ons_p90",
    "pAdj_Tackles_Won_p90",
    "pAdj_Interceptions_p90",
)

FEATURE_LABELS: Final[dict[str, str]] = {
    "Goals_p90": "Clinical Finishing",
    "Shots_p90": "Shot Volume",
    "Assists_p90": "Final-Ball Output",
    "Key_Passes_p90": "Chance Creation",
    "Pass_Completion_Pct": "Ball Retention",
    "Progressive_Passes_p90": "Line-Breaking Passing",
    "Successful_Take_Ons_p90": "1v1 Dribbling",
    "pAdj_Tackles_Won_p90": "Ball Winning",
    "pAdj_Interceptions_p90": "Reading of Play",
}

# Feature categories drive both the composite indices and the similarity weights.
FEATURE_CATEGORIES: Final[dict[str, tuple[str, ...]]] = {
    "Scoring": ("Goals_p90", "Shots_p90"),
    "Creation & Passing": (
        "Assists_p90",
        "Key_Passes_p90",
        "Pass_Completion_Pct",
        "Progressive_Passes_p90",
    ),
    "Dribbling": ("Successful_Take_Ons_p90",),
    "Defending": ("pAdj_Tackles_Won_p90", "pAdj_Interceptions_p90"),
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
RANDOM_STATE: Final[int] = 42
