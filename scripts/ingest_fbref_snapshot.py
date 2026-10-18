"""Build the real 2023-24 dataset from a pre-2026 FBref export (Kaggle snapshot).

Usage::

    python scripts/ingest_fbref_snapshot.py --download              # fetch both Kaggle datasets
    python scripts/ingest_fbref_snapshot.py --csv path/to/cleaned_2023-24.csv --transfermarkt path/to/player-scores

Why a snapshot: FBref no longer publishes its advanced (Opta) tables, so
``scripts/ingest_fbref.py`` can only fetch standard and shooting stats. The
snapshot carries every stat the model needs; team possession comes from the
FBref league pages ``soccerdata`` has cached (``~/soccerdata/data/FBref``).

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

from src.config import DEFAULT_REAL_SEASON, ORIGIN_FBREF, PROCESSED_DATA_DIR, real_data_filename  # noqa: E402
from src.data.fbref_snapshot import (  # noqa: E402
    KAGGLE_DATASET,
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
    """``"2023-2024"`` -> the export's ``cleaned_2023-24.csv``."""
    start = season.split("-")[0]
    return f"cleaned_{start}-{season[-2:]}.csv"


def download_snapshot(season: str) -> Path:
    """Fetch the public Kaggle dataset with ``kagglehub`` and return the season's CSV."""
    import kagglehub

    folder = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    path = folder / snapshot_filename(season)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not in the downloaded dataset ({sorted(p.name for p in folder.iterdir())})")
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
    if possession.empty:
        logger.warning("No cached FBref possession found in %s; Team_Possession_Pct defaults to 50%% "
                       "(pAdj neutral). Run scripts/ingest_fbref.py once to cache the league pages.", cache_dir)
    else:
        logger.info("Possession for %d clubs from cached FBref pages", len(possession))
    raw, report = build_snapshot_dataset(snapshot, possession, season)
    logger.info("Deduplicated %d club rows -> %d players (%d mid-season transfers)",
                report.n_rows_in, report.n_players_out, report.n_transferred)
    enriched = enrich_real_players(raw)
    logger.info("Positions: %s", primary_position_counts(enriched))
    return enriched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="path to the snapshot CSV (e.g. cleaned_2023-24.csv)")
    source.add_argument("--download", action="store_true", help=f"download {KAGGLE_DATASET} via kagglehub")
    parser.add_argument("--season", default=DEFAULT_REAL_SEASON, help="e.g. 2023-2024 (default)")
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

    try:
        csv = download_snapshot(args.season) if args.download else args.csv
        df = build(csv, args.season, args.fbref_cache)
        tm_folder = None if args.no_transfermarkt else (
            args.transfermarkt or (download_transfermarkt() if args.download else None))
        if tm_folder is not None:
            df = add_transfermarkt(df, tm_folder, args.season)
        else:
            logger.warning("No Transfermarkt data: market values and contracts are proxies.")
    except (FileNotFoundError, FBrefSchemaError, ImportError, ValueError) as exc:
        logger.error("Snapshot ingestion failed: %s", exc)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = real_data_filename(args.season, ORIGIN_FBREF)
    parquet, out_csv = args.out_dir / f"{stem}.parquet", args.out_dir / f"{stem}.csv"
    df.to_parquet(parquet, index=False)
    df.to_csv(out_csv, index=False)
    logger.info("Wrote %d real players to %s (+ %s)", len(df), parquet, out_csv.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
