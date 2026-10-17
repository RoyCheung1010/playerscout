"""Fill the metadata FBref does not provide: possession, market value, contracts.

* **Possession** comes from FBref's team standard table and is mapped per club.
* **Market value** is fetched from Transfermarkt when a source is available.
  ``soccerdata`` (1.9.x) ships no Transfermarkt reader, so in practice the
  deterministic **proxy** below is used and every row is labelled with its
  ``Market_Value_Source``.
* **Contract expiry** is not published by FBref at all, so a deterministic
  pseudo-random year is assigned and labelled ``Contract_Source = "proxy"``.

Proxies keep the Moneyball sliders free of nulls. They are estimates, never
facts, and the dashboard says so whenever they are in use.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from src.config import (
    DATA_ORIGIN_COLUMN,
    DEFAULT_LEAGUE_COEFFICIENT,
    LEAGUE_STRENGTH_COEFFICIENTS,
    ORIGIN_FBREF,
    POSITION_GROUPS,
)
from src.features.preprocessor import ecdf_percentile

logger = logging.getLogger(__name__)

VALUE_SOURCE_COLUMN = "Market_Value_Source"
CONTRACT_SOURCE_COLUMN = "Contract_Source"
SOURCE_PROXY = "proxy"
SOURCE_TRANSFERMARKT = "transfermarkt"

# Contract proxy window (inclusive), per the brief.
CONTRACT_PROXY_YEARS: tuple[int, int] = (2026, 2029)

# Position-specific weight of the scoring percentile in the quality score
# (the remainder is the passing percentile).
SCORING_WEIGHT: dict[str, float] = {"FW": 0.7, "MF": 0.4, "DF": 0.2}

# Proxy value curve: a 100th-percentile, peak-age, ever-present top-5 player.
PROXY_VALUE_CEILING_EUR_M = 120.0
PROXY_VALUE_FLOOR_EUR_M = 0.3


# --------------------------------------------------------------------------- #
# Possession
# --------------------------------------------------------------------------- #
def map_team_possession(players: pd.DataFrame, possession: pd.DataFrame) -> pd.DataFrame:
    """Attach ``Team_Possession_Pct`` by ``(League, Squad)``.

    Clubs missing from the team table fall back to their league's mean, then
    to a neutral 50% (where the pAdj multiplier is exactly 1.0).
    """
    table = possession[["League", "Squad", "Team_Possession_Pct"]].drop_duplicates(["League", "Squad"])
    out = players.drop(columns=["Team_Possession_Pct"], errors="ignore").merge(
        table, on=["League", "Squad"], how="left"
    )
    league_mean = table.groupby("League")["Team_Possession_Pct"].mean()
    out["Team_Possession_Pct"] = (
        out["Team_Possession_Pct"]
        .fillna(out["League"].map(league_mean))
        .fillna(50.0)
        .clip(0.0, 100.0)
    )
    unmatched = players.merge(table, on=["League", "Squad"], how="left")["Team_Possession_Pct"].isna()
    if unmatched.any():
        clubs = sorted(players.loc[unmatched.to_numpy(), "Squad"].astype(str).unique())
        logger.warning("No possession data for %d club(s); using league mean: %s", len(clubs), clubs[:10])
    return out


# --------------------------------------------------------------------------- #
# Market value
# --------------------------------------------------------------------------- #
def fetch_transfermarkt_values(players: pd.DataFrame) -> pd.Series | None:
    """Try to fetch Transfermarkt values (EUR M), aligned to ``players``.

    Returns ``None`` (caller falls back to the proxy) when no Transfermarkt
    reader is installed, when the request fails (e.g. HTTP 429), or when
    nothing can be matched. Never raises.
    """
    try:
        import soccerdata as sd
    except ImportError:
        logger.info("soccerdata not installed; using proxy market values.")
        return None
    reader_cls = getattr(sd, "Transfermarkt", None)
    if reader_cls is None:
        logger.info("soccerdata %s has no Transfermarkt reader; using proxy market values.",
                    getattr(sd, "__version__", "?"))
        return None
    try:  # pragma: no cover - depends on a reader soccerdata does not ship today
        reader = reader_cls()
        read = getattr(reader, "read_player_market_values", None) or getattr(reader, "read_players")
        values = read()
        values = values.reset_index()
        values["_key"] = values["player"].map(_name_key)
        lookup = values.drop_duplicates("_key").set_index("_key")["market_value"] / 1e6
        matched = players["Player"].map(_name_key).map(lookup)
        if matched.notna().sum() == 0:
            logger.warning("Transfermarkt returned no matchable players; using proxy values.")
            return None
        return matched
    except Exception as exc:  # noqa: BLE001 - any failure must fall back, not crash
        logger.warning("Transfermarkt fetch failed (%s: %s); using proxy values.", type(exc).__name__, exc)
        return None


def proxy_market_value(players: pd.DataFrame) -> pd.Series:
    """Deterministic EUR M estimate from age, league, minutes and percentiles.

    ``value = ceiling × quality^2.2 × age_curve × league_coef × usage``, where
    *quality* blends within-position percentiles of scoring output
    (goals + shots per 90) and passing output (key + progressive passes per
    90), weighted by position (forwards lean on scoring, defenders on passing).
    """
    minutes = players["Minutes_Played"].astype(float)
    per90 = 90.0 / minutes.where(minutes > 0)
    scoring = ((players["Goals"] + 0.15 * players["Shots"]) * per90).fillna(0.0)
    passing = ((players["Key_Passes"] + 0.35 * players["Progressive_Passes"]) * per90).fillna(0.0)
    group = players["Position"].astype(str).str.upper().str.extract(r"(DF|MF|FW)")[0].fillna("MF")

    scoring_pct = scoring.groupby(group).transform(lambda s: pd.Series(ecdf_percentile(s), index=s.index))
    passing_pct = passing.groupby(group).transform(lambda s: pd.Series(ecdf_percentile(s), index=s.index))
    weight = group.map(SCORING_WEIGHT).fillna(0.5)
    quality = (weight * scoring_pct + (1 - weight) * passing_pct) / 100.0

    age = players["Age"].astype(float).fillna(26.0)
    age_curve = np.exp(-((age - 25.0) ** 2) / (2 * 4.5**2)) * np.where(age < 21, 1.15, 1.0)
    league = players["League"].map(LEAGUE_STRENGTH_COEFFICIENTS).fillna(DEFAULT_LEAGUE_COEFFICIENT)
    usage = 0.4 + 0.6 * np.clip(minutes / 3000.0, 0.0, 1.0)

    value = PROXY_VALUE_CEILING_EUR_M * quality.clip(0, 1) ** 2.2 * age_curve * league * usage
    return value.clip(PROXY_VALUE_FLOOR_EUR_M, None).round(1)


# --------------------------------------------------------------------------- #
# Contracts
# --------------------------------------------------------------------------- #
def proxy_contract_expiry(players: pd.DataFrame, years: tuple[int, int] = CONTRACT_PROXY_YEARS) -> pd.Series:
    """Stable pseudo-random expiry year per player within ``years`` (inclusive).

    Seeded by a hash of ``(Player, Born)``, so re-running the pipeline never
    reshuffles contracts and results are reproducible across machines.
    """
    low, high = years
    if high < low:
        raise ValueError("contract year window is empty")
    span = high - low + 1
    born = players["Born"] if "Born" in players.columns else pd.Series("", index=players.index)
    keys = players["Player"].astype(str) + "|" + born.astype(str)
    return keys.map(lambda k: low + int(hashlib.sha256(k.encode("utf-8")).hexdigest(), 16) % span).astype(int)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def enrich_real_players(
    players: pd.DataFrame,
    fetch_values: Callable[[pd.DataFrame], pd.Series | None] | None = fetch_transfermarkt_values,
) -> pd.DataFrame:
    """Add market value, contract expiry and provenance columns.

    Fetched values are used where available and the proxy fills any gaps, so
    ``Market_Value_EUR_M`` and ``Contract_Expiry_Year`` are never null.
    """
    out = players.copy()
    proxy = proxy_market_value(out)
    fetched = fetch_values(out) if fetch_values is not None else None
    if fetched is not None:
        fetched = pd.to_numeric(fetched.reindex(out.index), errors="coerce")
        use_fetched = fetched.notna() & (fetched > 0)
        out["Market_Value_EUR_M"] = fetched.where(use_fetched, proxy).round(1)
        out[VALUE_SOURCE_COLUMN] = np.where(use_fetched, SOURCE_TRANSFERMARKT, SOURCE_PROXY)
    else:
        out["Market_Value_EUR_M"] = proxy
        out[VALUE_SOURCE_COLUMN] = SOURCE_PROXY

    out["Contract_Expiry_Year"] = proxy_contract_expiry(out)
    out[CONTRACT_SOURCE_COLUMN] = SOURCE_PROXY
    if "Team_Possession_Pct" not in out.columns:
        out["Team_Possession_Pct"] = 50.0
    out[DATA_ORIGIN_COLUMN] = ORIGIN_FBREF
    return out


def primary_position_counts(players: pd.DataFrame) -> dict[str, int]:
    """Players per primary outfield group (for ingestion logs)."""
    group = players["Position"].astype(str).str.upper().str.extract(r"^(DF|MF|FW|GK)")[0]
    return {g: int((group == g).sum()) for g in (*POSITION_GROUPS, "GK")}


def _name_key(name: object) -> str:
    import unicodedata

    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return " ".join(text.lower().replace("-", " ").split())
