"""Ingestion and contextual-adjustment pipeline for raw player season tables.

Turns a raw FBref-style table (synthetic DB, CSV path or uploaded buffer) into
the model-ready frame used by every downstream module:

1. **Validate** required columns and coerce numerics (bad rows are dropped).
2. **Filter** by ``min_minutes`` and map positions to ``Position_Group``.
3. **Per-90** every raw counting stat: ``stat / Minutes_Played * 90``.
4. **pAdj** defensive actions with the possession sigmoid.
5. **League-scale** every per-90 volume stat (percentages are left untouched).

Each run also produces a :class:`LoadReport` so the UI can explain why rows
disappeared.
"""

from __future__ import annotations

import io
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import IO, Union

import numpy as np
import pandas as pd

from pathlib import Path

from src.config import (
    DATA_ORIGIN_COLUMN,
    DEFAULT_REAL_SEASON,
    ORIGIN_BENCHMARK,
    ORIGIN_FBREF,
    PROCESSED_DATA_DIR,
    DEFAULT_LEAGUE_COEFFICIENT,
    DEFAULT_MIN_MINUTES,
    DEFAULT_PADJ_STEEPNESS,
    LEAGUE_STRENGTH_COEFFICIENTS,
    METADATA_COLUMNS,
    PER90_SUFFIX,
    POSITION_GROUPS,
    POSSESSION_ADJUSTED_STATS,
    RAW_COUNT_COLUMNS,
    RAW_RATE_COLUMNS,
    REQUIRED_RAW_COLUMNS,
    real_data_filename,
)
from src.data.synthetic_generator import GeneratorConfig, generate_synthetic_players

DataSource = Union[pd.DataFrame, str, "os.PathLike[str]", IO[bytes], IO[str], None]

_TEXT_COLUMNS: tuple[str, ...] = ("Player", "Squad", "League", "Position")
_NUMERIC_METADATA: tuple[str, ...] = tuple(c for c in METADATA_COLUMNS if c not in _TEXT_COLUMNS)
_POSITION_SPLIT = re.compile(r"[,/;|\-\s]+")


class DataValidationError(ValueError):
    """Raised when an input table cannot be mapped onto the scouting schema."""


class RealDataNotFoundError(FileNotFoundError):
    """No processed real (or benchmark) dataset exists for the season."""


@dataclass(frozen=True)
class RealDataset:
    """Location and provenance of a processed real-data file."""

    path: Path
    origin: str  # ORIGIN_FBREF or ORIGIN_BENCHMARK
    season: str

    @property
    def is_real(self) -> bool:
        return self.origin == ORIGIN_FBREF


def find_real_dataset(
    season: str = DEFAULT_REAL_SEASON,
    data_dir: Path | str | None = None,
    allow_benchmark: bool = True,
) -> RealDataset | None:
    """Locate the processed file for ``season``: real first, then the benchmark.

    Parquet is preferred; the CSV twin is used if only it exists.
    """
    directory = Path(data_dir) if data_dir is not None else PROCESSED_DATA_DIR
    origins = (ORIGIN_FBREF, ORIGIN_BENCHMARK) if allow_benchmark else (ORIGIN_FBREF,)
    for origin in origins:
        stem = real_data_filename(season, origin)
        for suffix in (".parquet", ".csv"):
            candidate = directory / f"{stem}{suffix}"
            if candidate.is_file():
                return RealDataset(candidate, origin, season)
    return None


@dataclass(frozen=True)
class LoaderConfig:
    """Pipeline settings exposed to the dashboard sidebar."""

    min_minutes: int = DEFAULT_MIN_MINUTES
    apply_league_scaling: bool = True
    league_coefficients: Mapping[str, float] = field(
        default_factory=lambda: dict(LEAGUE_STRENGTH_COEFFICIENTS)
    )
    default_league_coefficient: float = DEFAULT_LEAGUE_COEFFICIENT
    padj_steepness: float = DEFAULT_PADJ_STEEPNESS

    def __post_init__(self) -> None:
        if self.min_minutes < 0:
            raise ValueError("min_minutes must be non-negative")
        if self.default_league_coefficient <= 0:
            raise ValueError("default_league_coefficient must be positive")
        if self.padj_steepness < 0:
            raise ValueError("padj_steepness must be non-negative")
        if any(v <= 0 for v in self.league_coefficients.values()):
            raise ValueError("league coefficients must be positive")


@dataclass
class LoadReport:
    """Row accounting for one pipeline run."""

    n_input: int = 0
    n_invalid: int = 0
    n_unknown_position: int = 0
    n_below_min_minutes: int = 0
    n_output: int = 0
    unknown_leagues: list[str] = field(default_factory=list)

    @property
    def n_dropped(self) -> int:
        return self.n_input - self.n_output


# --------------------------------------------------------------------------- #
# Pure transformations (stateless, reusable and individually testable)
# --------------------------------------------------------------------------- #
def possession_multiplier(
    possession_pct: pd.Series | np.ndarray | float,
    steepness: float = DEFAULT_PADJ_STEEPNESS,
) -> np.ndarray:
    """Sigmoid pAdj factor: ``2 / (1 + exp(-k * (possession - 50)))``.

    Equals 1.0 at 50% possession, rises towards 2.0 for dominant sides (who
    get fewer chances to defend) and falls towards 0.0 for low-possession sides.
    ``steepness`` (k) controls how hard the correction bites; 0 disables it.
    """
    poss = np.asarray(possession_pct, dtype=float)
    return 2.0 / (1.0 + np.exp(-steepness * (poss - 50.0)))


def parse_position_group(positions: pd.Series) -> pd.Series:
    """Map FBref position strings (``"FW,MF"``, ``"DF/MF"``) to their primary group.

    The first listed position wins. Anything that is not DF/MF/FW (e.g. GK,
    blanks) becomes ``NaN`` so the caller can drop it explicitly.
    """
    primary = (
        positions.astype("string")
        .str.strip()
        .str.upper()
        .str.split(_POSITION_SPLIT, n=1, regex=True)
        .str[0]
    )
    return primary.where(primary.isin(POSITION_GROUPS)).astype(object)


def per90(stat: pd.Series, minutes: pd.Series) -> pd.Series:
    """Per-90 rate with a zero-minutes guard (returns 0.0 where minutes <= 0)."""
    stat_arr = stat.to_numpy(dtype=float)
    minutes_arr = minutes.to_numpy(dtype=float)
    out = np.zeros_like(stat_arr)
    np.divide(stat_arr * 90.0, minutes_arr, out=out, where=minutes_arr > 0)
    return pd.Series(out, index=stat.index)


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
class DataLoader:
    """Loads, validates and contextually adjusts a raw player table.

    Example:
        >>> loader = DataLoader(LoaderConfig(min_minutes=900))
        >>> players = loader.load()          # synthetic enterprise DB
        >>> report = loader.last_report
    """

    def __init__(self, config: LoaderConfig | None = None) -> None:
        self.config = config or LoaderConfig()
        self.last_report = LoadReport()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def load(
        self,
        source: DataSource = None,
        generator_config: GeneratorConfig | None = None,
    ) -> pd.DataFrame:
        """Read ``source`` (``None`` = synthetic DB) and run the full pipeline."""
        return self.process(self.read(source, generator_config))

    def load_real(
        self,
        season: str = DEFAULT_REAL_SEASON,
        data_dir: Path | str | None = None,
        allow_benchmark: bool = True,
    ) -> pd.DataFrame:
        """Load the processed FBref dataset for ``season`` through the full pipeline.

        The file is produced by ``scripts/ingest_fbref_snapshot.py`` (or
        ``scripts/ingest_fbref.py``) and goes through
        exactly the same validation, per-90, pAdj and league scaling as any
        other source. If only the synthetic benchmark exists (ingestion fell
        back) it is loaded instead; its rows carry
        ``Data_Origin == "synthetic-benchmark"`` so callers can warn about it.

        Raises:
            RealDataNotFoundError: no processed file exists for the season.
        """
        dataset = find_real_dataset(season, data_dir, allow_benchmark)
        if dataset is None:
            raise RealDataNotFoundError(
                f"No processed data for {season} in {Path(data_dir) if data_dir else PROCESSED_DATA_DIR}. "
                "Run: python scripts/ingest_fbref_snapshot.py --download"
            )
        raw = pd.read_parquet(dataset.path) if dataset.path.suffix == ".parquet" else _read_csv_with_fallback(dataset.path)
        if DATA_ORIGIN_COLUMN not in raw.columns:
            raw[DATA_ORIGIN_COLUMN] = dataset.origin
        return self.process(raw)

    @staticmethod
    def read(source: DataSource = None, generator_config: GeneratorConfig | None = None) -> pd.DataFrame:
        """Return the raw table without any transformation."""
        if source is None:
            return generate_synthetic_players(generator_config)
        if isinstance(source, pd.DataFrame):
            return source.copy()
        if isinstance(source, (str, os.PathLike)):
            return _read_csv_with_fallback(source)
        if hasattr(source, "read"):
            return _read_csv_with_fallback(source)
        raise DataValidationError(f"Unsupported data source type: {type(source).__name__}")

    def process(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Run validation, filtering, per-90, pAdj and league scaling."""
        report = LoadReport(n_input=len(raw))
        df = self.validate(raw)
        report.n_invalid = report.n_input - len(df)

        df["Position_Group"] = parse_position_group(df["Position"])
        unknown_pos = df["Position_Group"].isna()
        report.n_unknown_position = int(unknown_pos.sum())
        df = df.loc[~unknown_pos]

        below = df["Minutes_Played"] < self.config.min_minutes
        report.n_below_min_minutes = int(below.sum())
        df = df.loc[~below].copy()

        df = self._add_per90(df)
        df = self._apply_possession_adjustment(df)
        df, report.unknown_leagues = self._apply_league_scaling(df)

        report.n_output = len(df)
        self.last_report = report
        return df.reset_index(drop=True)

    @staticmethod
    def validate(raw: pd.DataFrame) -> pd.DataFrame:
        """Check the schema, coerce types and drop rows that cannot be used.

        Raises:
            DataValidationError: if required columns are missing.
        """
        df = raw.copy()
        df.columns = [str(c).strip() for c in df.columns]
        if df.columns.duplicated().any():
            dupes = sorted(set(df.columns[df.columns.duplicated()]))
            raise DataValidationError(f"Duplicate columns in input: {dupes}")

        missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
        if missing:
            raise DataValidationError(f"Missing required columns: {missing}")

        for col in _TEXT_COLUMNS:
            df[col] = df[col].astype("string").str.strip()

        numeric_cols = [
            c for c in (*_NUMERIC_METADATA, *RAW_COUNT_COLUMNS, *RAW_RATE_COLUMNS) if c in df.columns
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        required_numeric = [c for c in numeric_cols if c in REQUIRED_RAW_COLUMNS]
        valid = (
            df[required_numeric].notna().all(axis=1)
            & df["Player"].fillna("").ne("").to_numpy(dtype=bool)
            & (df["Minutes_Played"] >= 0)
            & df["Team_Possession_Pct"].between(0, 100)
            & df["Pass_Completion_Pct"].between(0, 100)
            & (df[[c for c in RAW_COUNT_COLUMNS if c in df.columns]].fillna(0) >= 0).all(axis=1)
        )
        return df.loc[valid].copy()

    # ------------------------------------------------------------------ #
    # Pipeline stages
    # ------------------------------------------------------------------ #
    @staticmethod
    def _add_per90(df: pd.DataFrame) -> pd.DataFrame:
        for col in RAW_COUNT_COLUMNS:
            if col in df.columns:
                df[f"{col}{PER90_SUFFIX}"] = per90(df[col].fillna(0), df["Minutes_Played"])
        return df

    def _apply_possession_adjustment(self, df: pd.DataFrame) -> pd.DataFrame:
        multiplier = possession_multiplier(df["Team_Possession_Pct"], self.config.padj_steepness)
        df["Possession_Multiplier"] = multiplier
        for col in POSSESSION_ADJUSTED_STATS:
            df[f"pAdj_{col}"] = df[col] * multiplier
        return df

    def _apply_league_scaling(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        coefficients = self.config.league_coefficients
        known = df["League"].isin(list(coefficients))
        unknown_leagues = sorted(df.loc[~known, "League"].dropna().unique().tolist())

        if self.config.apply_league_scaling:
            coef = df["League"].map(coefficients).astype(float).fillna(self.config.default_league_coefficient)
        else:
            coef = pd.Series(1.0, index=df.index)
        df["League_Coefficient"] = coef

        # Only per-90 volume stats are scaled; rate stats (Pass_Completion_Pct)
        # never carry the _p90 suffix, so they stay untouched by construction.
        for col in [c for c in df.columns if c.endswith(PER90_SUFFIX)]:
            df[col] = df[col] * coef
        return df, unknown_leagues


def _read_csv_with_fallback(source: str | os.PathLike[str] | IO[bytes] | IO[str]) -> pd.DataFrame:
    """Read a CSV, retrying with Latin-1 for Excel-exported files."""
    if hasattr(source, "read"):
        content = source.read()
        if isinstance(content, str):
            return _parse_csv(io.StringIO(content))
        try:
            return _parse_csv(io.StringIO(content.decode("utf-8-sig")))
        except UnicodeDecodeError:
            return _parse_csv(io.StringIO(content.decode("latin-1")))
    try:
        return _parse_csv(source, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return _parse_csv(source, encoding="latin-1")


def _parse_csv(source: str | os.PathLike[str] | IO[str], encoding: str | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(source, encoding=encoding) if encoding else pd.read_csv(source)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError("The uploaded file is empty") from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError(f"Could not parse CSV: {exc}") from exc
