"""Build real season datasets from pre-2026 FBref exports (Kaggle snapshots).

Usage::

    python scripts/ingest_fbref_snapshot.py --download --season all     # every supported season
    python scripts/ingest_fbref_snapshot.py --download --season 2024-2025
    python scripts/ingest_fbref_snapshot.py --csv path/to/export.csv --season 2023-2024 --transfermarkt DIR

Why snapshots: FBref no longer publishes its advanced (Opta) tables, so
``scripts/ingest_fbref.py`` can only fetch standard and shooting stats. The
exports carry every stat the model needs (sources per season in
``src.data.fbref_snapshot.SEASON_SOURCES``). Team possession comes from FBref
league pages ``soccerdata`` has cached (``~/soccerdata/data/FBref``) and is
otherwise estimated from pass volumes, labelled per row.

Pipeline: snapshot CSV -> schema mapping -> possession mapping -> transfer
deduplication -> proxy valuation/contracts -> real Transfermarkt values and
contracts where players match (``src/data/transfermarkt.py``) ->
``real_players_<season>`` ``.parquet`` and ``.csv`` in ``data/processed/``. There is no synthetic
fallback here: on any failure the script exits with code 2 and writes nothing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ORIGIN_FBREF, PROCESSED_DATA_DIR, REAL_SEASONS, real_data_filename  # noqa: E402
from src.data.fbref_snapshot import (  # noqa: E402
    POSSESSION_SOURCE_COLUMN,
    SEASON_SOURCES,
    build_snapshot_dataset,
    parse_cached_possession,
    read_snapshot,
    season_code,
)
from src.data.fbref_transform import FBrefSchemaError  # noqa: E402
from src.data import transfermarkt  # noqa: E402
from src.data.real_data_enricher import enrich_real_players, primary_position_counts  # noqa: E402

logger = logging.getLogger("ingest_fbref_snapshot")

DEFAULT_FBREF_CACHE = Path.home() / "soccerdata" / "data" / "FBref"


def snapshot_filename(season: str) -> str:
    """The export file for ``season`` (e.g. ``"2023-2024"`` -> ``cleaned_2023-24.csv``)."""
    if season not in SEASON_SOURCES:
        raise ValueError(f"no known FBref export for {season}; supported: {sorted(SEASON_SOURCES)}")
    return SEASON_SOURCES[season][1]


def download_snapshot(season: str) -> Path:
    """Fetch just the season's CSV from its public Kaggle dataset."""
    import kagglehub

    filename = snapshot_filename(season)
    handle = SEASON_SOURCES[season][0]
    path = Path(kagglehub.dataset_download(handle, path=filename))
    if not path.exists():
        raise FileNotFoundError(f"{filename} was not downloaded from {handle}")
    return path


def download_transfermarkt() -> Path:
    """Fetch only the two Transfermarkt files needed (the full dataset is ~900 MB)."""
    import kagglehub

    paths = [Path(kagglehub.dataset_download(transfermarkt.KAGGLE_DATASET, path=name))
             for name in transfermarkt.REQUIRED_FILES]
    return paths[0].parent


def add_transfermarkt(df: pd.DataFrame, folder: Path, season: str) -> pd.DataFrame:
    players, valuations = transfermarkt.read_transfermarkt(folder)
    out, report = transfermarkt.attach_transfermarkt(df, players, valuations, season)
    logger.info("Transfermarkt (snapshot %s): matched %d/%d players (%d exact, %d fuzzy); "
                "real values %d, real contracts %d; the rest keep proxies",
                report.snapshot_date.date(), report.n_matched, report.n_players, report.n_exact,
                report.n_fuzzy, report.n_values, report.n_contracts)
    return out


def build(csv: Path, season: str, cache_dir: Path) -> pd.DataFrame:
    snapshot = read_snapshot(csv)
    possession = parse_cached_possession(cache_dir, season_code(season))
    raw, report = build_snapshot_dataset(snapshot, possession, season)
    clubs = raw.drop_duplicates(["League", "Squad"])[POSSESSION_SOURCE_COLUMN].value_counts().to_dict()
    logger.info("Team possession per club: %s (estimates come from pass volumes; "
                "scripts/ingest_fbref.py --interactive caches FBref's real values)", clubs)
    logger.info("Deduplicated %d club rows -> %d players (%d mid-season transfers)",
                report.n_rows_in, report.n_players_out, report.n_transferred)
    enriched = enrich_real_players(raw)
    logger.info("Positions: %s", primary_position_counts(enriched))
    return enriched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="path to the snapshot CSV (e.g. cleaned_2023-24.csv)")
    source.add_argument("--download", action="store_true", help="download the season's Kaggle export(s)")
    parser.add_argument("--season", default=REAL_SEASONS[-1],
                        help=f"one of {', '.join(REAL_SEASONS)} or 'all' (default: {REAL_SEASONS[-1]})")
    parser.add_argument("--fbref-cache", type=Path, default=DEFAULT_FBREF_CACHE,
                        help="soccerdata FBref cache, for team possession")
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--transfermarkt", type=Path, default=None,
                        help="folder with Transfermarkt players.csv + player_valuations.csv "
                             f"({transfermarkt.KAGGLE_DATASET}); downloaded automatically with --download")
    parser.add_argument("--no-transfermarkt", action="store_true",
                        help="keep proxy market values and contracts")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    seasons = list(REAL_SEASONS) if args.season == "all" else [args.season]
    if args.csv is not None and len(seasons) > 1:
        parser.error("--season all needs --download (one --csv holds one season)")

    built: dict[str, pd.DataFrame] = {}
    try:
        tm_folder = None if args.no_transfermarkt else (
            args.transfermarkt or (download_transfermarkt() if args.download else None))
        for season in seasons:
            csv = download_snapshot(season) if args.download else args.csv
            df = build(csv, season, args.fbref_cache)
            if tm_folder is not None:
                df = add_transfermarkt(df, tm_folder, season)
            else:
                logger.warning("No Transfermarkt data: market values and contracts are proxies.")
            built[season] = df
    except (FileNotFoundError, FBrefSchemaError, ImportError, ValueError) as exc:
        logger.error("Snapshot ingestion failed: %s", exc)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for season, df in built.items():
        stem = real_data_filename(season, ORIGIN_FBREF)
        parquet, out_csv = args.out_dir / f"{stem}.parquet", args.out_dir / f"{stem}.csv"
        df.to_parquet(parquet, index=False)
        df.to_csv(out_csv, index=False)
        logger.info("Wrote %d real players (%s) to %s (+ %s)", len(df), season, parquet, out_csv.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
