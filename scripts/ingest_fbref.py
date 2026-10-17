"""Ingest real FBref season data into ``data/processed/``.

Usage::

    python scripts/ingest_fbref.py                    # 2023-24, Top 5, headless
    python scripts/ingest_fbref.py --interactive      # visible browser: solve FBref's CAPTCHA once
    python scripts/ingest_fbref.py --no-fallback      # fail instead of writing the benchmark

Pipeline: five player stat tables + the team standard table per league
(``soccerdata.FBref``) -> transfer deduplication -> possession mapping ->
valuation/contract enrichment -> ``real_players_<season>.parquet`` and ``.csv``.

Network safeguard: FBref rate-limits aggressively (HTTP 429) and may serve a
CAPTCHA to automated browsers. Any fetch failure is caught and logged, and the
script falls back to writing ``benchmark_players_<season>.parquet``: a
**synthetic** dataset with the identical schema, labelled
``Data_Origin = "synthetic-benchmark"`` on every row. The benchmark is never
written to the ``real_players`` path, so synthetic players can never be
mistaken for real ones. soccerdata caches every page it downloads under
``~/soccerdata/data/FBref``, so a successful run is fully repeatable offline.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    DATA_ORIGIN_COLUMN,
    DEFAULT_REAL_SEASON,
    FBREF_LEAGUES,
    ORIGIN_BENCHMARK,
    ORIGIN_FBREF,
    PROCESSED_DATA_DIR,
    real_data_filename,
    season_end_year,
)
from src.data.fbref_transform import PLAYER_STAT_TABLES, build_raw_player_table, team_possession  # noqa: E402
from src.data.real_data_enricher import enrich_real_players, primary_position_counts  # noqa: E402
from src.data.synthetic_generator import GeneratorConfig, generate_synthetic_players  # noqa: E402

logger = logging.getLogger("ingest_fbref")


class IngestionError(RuntimeError):
    """FBref data could not be fetched (network, rate limit or CAPTCHA)."""


def soccerdata_season(season: str) -> str:
    """``"2023-2024"`` -> soccerdata's ``"2324"``."""
    end = season_end_year(season)
    return f"{(end - 1) % 100:02d}{end % 100:02d}"


def fetch_fbref_tables(
    season: str, leagues: list[str], interactive: bool = False
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Download the five player tables and the team table for all ``leagues``.

    Raises:
        IngestionError: on any network, rate-limit (429) or CAPTCHA failure.
    """
    try:
        import soccerdata as sd
    except ImportError as exc:
        raise IngestionError("soccerdata is not installed: pip install -r requirements-ingest.txt") from exc

    try:
        fbref = sd.FBref(leagues=leagues, seasons=soccerdata_season(season), headless=not interactive)
        tables: dict[str, pd.DataFrame] = {}
        for stat_type in PLAYER_STAT_TABLES:
            started = time.perf_counter()
            tables[stat_type] = fbref.read_player_season_stats(stat_type=stat_type)
            logger.info("Fetched %-10s %6d rows in %.0fs", stat_type, len(tables[stat_type]),
                        time.perf_counter() - started)
        team_table = fbref.read_team_season_stats(stat_type="standard")
        logger.info("Fetched team standard table (%d clubs)", len(team_table))
        return tables, team_table
    except Exception as exc:  # noqa: BLE001 - soccerdata raises bare Exception/ConnectionError
        message = str(exc)
        if "429" in message or "Too Many Requests" in message:
            reason = "FBref rate limit (HTTP 429)"
        elif "CAPTCHA" in message.upper() or "captcha" in repr(exc.__cause__ or "").lower():
            reason = "FBref CAPTCHA challenge (re-run with --interactive to solve it in a browser)"
        else:
            reason = f"{type(exc).__name__}: {message[:200]}"
        raise IngestionError(reason) from exc


def build_real_dataset(tables: dict[str, pd.DataFrame], team_table: pd.DataFrame) -> pd.DataFrame:
    possession = team_possession(team_table)
    raw, report = build_raw_player_table(tables, possession)
    logger.info("Deduplicated %d club rows -> %d players (%d mid-season transfers)",
                report.n_rows_in, report.n_players_out, report.n_transferred)
    enriched = enrich_real_players(raw)
    logger.info("Positions: %s", primary_position_counts(enriched))
    return enriched


def build_benchmark_dataset(season: str, seed: int = 2324) -> pd.DataFrame:
    """Synthetic Top-5 stand-in with the real dataset's schema, clearly labelled."""
    end = season_end_year(season)
    config = GeneratorConfig(
        seed=seed,
        season=f"{end - 1}-{end}",
        season_end_year=end,
        leagues=tuple(FBREF_LEAGUES.values()),
    )
    df = generate_synthetic_players(config)
    df[DATA_ORIGIN_COLUMN] = ORIGIN_BENCHMARK
    df["Market_Value_Source"] = ORIGIN_BENCHMARK
    df["Contract_Source"] = ORIGIN_BENCHMARK
    return df


def write_outputs(df: pd.DataFrame, out_dir: Path, season: str, origin: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = real_data_filename(season, origin)
    parquet, csv = out_dir / f"{stem}.parquet", out_dir / f"{stem}.csv"
    df.to_parquet(parquet, index=False)
    df.to_csv(csv, index=False)
    return parquet, csv


def run(
    season: str = DEFAULT_REAL_SEASON,
    leagues: list[str] | None = None,
    out_dir: Path = PROCESSED_DATA_DIR,
    interactive: bool = False,
    allow_fallback: bool = True,
) -> tuple[str, Path]:
    """Ingest real data, or fall back to the labelled benchmark. Returns (origin, parquet path)."""
    leagues = leagues or list(FBREF_LEAGUES)
    try:
        tables, team_table = fetch_fbref_tables(season, leagues, interactive=interactive)
        df = build_real_dataset(tables, team_table)
        parquet, csv = write_outputs(df, out_dir, season, ORIGIN_FBREF)
        logger.info("Wrote %d real players to %s (+ %s)", len(df), parquet, csv.name)
        return ORIGIN_FBREF, parquet
    except IngestionError as exc:
        if not allow_fallback:
            raise
        logger.warning("Real data unavailable: %s", exc)
        logger.warning("Falling back to a SYNTHETIC benchmark with the same schema. It is labelled "
                       "'%s' and is NOT real player data.", ORIGIN_BENCHMARK)
        df = build_benchmark_dataset(season)
        parquet, csv = write_outputs(df, out_dir, season, ORIGIN_BENCHMARK)
        logger.warning("Wrote synthetic benchmark (%d players) to %s", len(df), parquet)
        return ORIGIN_BENCHMARK, parquet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--season", default=DEFAULT_REAL_SEASON, help="e.g. 2023-2024 (default)")
    parser.add_argument("--leagues", nargs="+", default=list(FBREF_LEAGUES), choices=list(FBREF_LEAGUES),
                        metavar="LEAGUE", help="soccerdata league ids (default: Top 5)")
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--interactive", action="store_true",
                        help="open a visible browser so you can solve FBref's CAPTCHA manually")
    parser.add_argument("--no-fallback", action="store_true",
                        help="exit with an error instead of writing the synthetic benchmark")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    try:
        origin, path = run(args.season, args.leagues, args.out_dir, args.interactive, not args.no_fallback)
    except IngestionError as exc:
        logger.error("Ingestion failed: %s", exc)
        return 2
    return 0 if origin == ORIGIN_FBREF else 1


if __name__ == "__main__":
    raise SystemExit(main())
