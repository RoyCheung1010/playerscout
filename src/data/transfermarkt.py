"""Real market values and contracts from the public Transfermarkt dataset.

Source (Kaggle, updated weekly): ``davidcariboo/player-scores``. Two files
are used:

* ``players.csv``: one row per player, with date of birth, the *currently
  listed* contract expiry and ``last_season`` (the latest season Transfermarkt
  tracked the player in a covered competition);
* ``player_valuations.csv``: dated market-value history, each row tagged with
  the player's domestic league at the time.

Rules:

* **Market value** = the player's latest valuation within the season window
  (default 1 Jan 2023 to 30 Jun 2024), so values describe the same season as
  the stats.
* **Contract expiry** is the contract listed *at the snapshot date*:
  Transfermarkt keeps no contract history. It is only used for players still
  tracked (``last_season`` at least the snapshot season), since an untracked
  player's listed contract may be stale. Contract years then count from the
  snapshot year, recorded per row in ``Contract_Reference_Year``.
* Anything unmatched keeps the enricher's proxy, labelled ``"proxy"``.

Matching FBref players to Transfermarkt is done in two passes, both requiring
the same **birth year**: (1) identical normalised name (or, if no one of
that name shares the birth year, a unique namesake born one year either
side); (2) a conservative fuzzy name rule restricted to players valued in
the **same league** during the season. Ambiguous matches are left unmatched
rather than guessed.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from src.data.real_data_enricher import SOURCE_TRANSFERMARKT

logger = logging.getLogger(__name__)

KAGGLE_DATASET = "davidcariboo/player-scores"
REQUIRED_FILES: tuple[str, ...] = ("players.csv", "player_valuations.csv")

# Transfermarkt competition id -> platform league name.
TM_LEAGUES: dict[str, str] = {
    "GB1": "Premier League",
    "ES1": "La Liga",
    "L1": "Bundesliga",
    "IT1": "Serie A",
    "FR1": "Ligue 1",
}

CONTRACT_REFERENCE_COLUMN = "Contract_Reference_Year"
TM_ID_COLUMN = "Transfermarkt_ID"
TM_MATCH_COLUMN = "Transfermarkt_Match"
FUZZY_RATIO = 0.85


@dataclass(frozen=True)
class SeasonWindow:
    """Valuations dated inside ``[start, end]`` describe the season."""

    start: pd.Timestamp
    end: pd.Timestamp

    @classmethod
    def for_season(cls, season: str) -> "SeasonWindow":
        first, second = (int(y) for y in season.split("-"))
        return cls(pd.Timestamp(first, 1, 1), pd.Timestamp(second, 6, 30))


@dataclass(frozen=True)
class MatchReport:
    n_players: int
    n_exact: int
    n_fuzzy: int
    n_values: int
    n_contracts: int
    snapshot_date: pd.Timestamp

    @property
    def n_matched(self) -> int:
        return self.n_exact + self.n_fuzzy


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def read_transfermarkt(folder: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ``players.csv`` and ``player_valuations.csv`` (only needed columns)."""
    folder = Path(folder)
    missing = [f for f in REQUIRED_FILES if not (folder / f).exists()]
    if missing:
        raise FileNotFoundError(f"Transfermarkt files missing in {folder}: {missing}")
    players = pd.read_csv(
        folder / "players.csv",
        usecols=["player_id", "name", "date_of_birth", "contract_expiration_date", "last_season"],
    )
    valuations = pd.read_csv(
        folder / "player_valuations.csv",
        usecols=lambda c: c in {"player_id", "date", "market_value_in_eur",
                                "player_club_domestic_competition_id", "current_club_name"},
    )
    return players, valuations


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #
def name_key(name: object) -> str:
    """Accent-free, lower-case, punctuation-free name (``"Álex Grimaldo"`` -> ``"alex grimaldo"``).

    Apostrophes are dropped (``"N'Dicka"`` -> ``"ndicka"``, matching
    Transfermarkt's ``"Ndicka"``); hyphens and dots separate tokens.
    """
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    for ch in "'’":
        text = text.replace(ch, "")
    for ch in "-.":
        text = text.replace(ch, " ")
    return " ".join(text.split())


_CLUB_STOPWORDS = frozenset({
    "fc", "cf", "ac", "as", "ss", "sc", "rc", "cd", "ud", "sd", "afc", "ssc", "acf", "us", "ogc", "vfb",
    "vfl", "tsg", "sv", "fsv", "bv", "rb", "de", "la", "del", "club", "calcio", "football", "futbol",
    "spa", "sad", "1899", "1900", "1904", "1907", "1909", "04", "05", "09", "1846", "1848",
})


def clubs_match(ours: object, theirs: object) -> bool:
    """Loose club-name agreement across sources (``"Wolves"`` ~ ``"Wolverhampton
    Wanderers"``, ``"Paris S-G"`` ~ ``"Paris Saint-Germain"``). Unknown -> False."""
    if pd.isna(ours) or pd.isna(theirs):
        return False
    a = [t for t in name_key(ours).split() if t not in _CLUB_STOPWORDS]
    b = [t for t in name_key(theirs).split() if t not in _CLUB_STOPWORDS]
    return any(x.startswith(y) or y.startswith(x) or (len(x) >= 4 and len(y) >= 4 and x[:4] == y[:4])
               for x in a for y in b)


def surname_only(a: str, b: str) -> bool:
    """Two multi-token keys sharing only a surname of 5+ letters, with first
    names that are not variants of each other (``"carlos romero"`` vs ``"ivan romero"``)."""
    ta, tb = a.split(), b.split()
    if len(ta) < 2 or len(tb) < 2 or ta[-1] != tb[-1] or len(ta[-1]) < 5:
        return False
    first_related = ta[0].startswith(tb[0]) or tb[0].startswith(ta[0]) or ta[0][:3] == tb[0][:3]
    return not (first_related or set(ta) <= set(tb) or set(tb) <= set(ta))


def names_compatible(a: str, b: str) -> bool:
    """Conservative fuzzy rule for two keys (already same birth year and league).

    True when the token sets are equal in any order (``"kim min jae"`` vs
    ``"min jae kim"``), one name's tokens are contained in the other's and they
    share the surname (``"ohis felix uduokhai"`` vs ``"felix uduokhai"``), the
    surnames match and one first name abbreviates the other (``"gio reyna"``
    vs ``"giovanni reyna"``), or the strings are near-identical.
    """
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return False
    sa, sb = set(ta), set(tb)
    if sa == sb:
        return True
    same_surname = ta[-1] == tb[-1] and len(ta[-1]) >= 3
    if same_surname and (sa <= sb or sb <= sa):
        return True
    if same_surname and (ta[0].startswith(tb[0]) or tb[0].startswith(ta[0])) and min(len(ta[0]), len(tb[0])) >= 3:
        return True
    if same_surname and ta[0][:3] == tb[0][:3]:  # "alex grimaldo" vs "alejandro grimaldo"
        return True
    # Mononyms: "gabriel" vs "gabriel magalhaes", "nacho" vs "nacho fernandez".
    short, long_ = (ta, tb) if len(ta) < len(tb) else (tb, ta)
    if len(short) == 1 and len(short[0]) >= 4 and short[0] in (long_[0], long_[-1]):
        return True
    # Same surname, different first name or nickname ("valentin castellanos"
    # vs "taty castellanos"). Common surnames make this unsafe on its own, so
    # match_players only accepts it when the club matches too.
    if surname_only(a, b):
        return True
    return SequenceMatcher(None, a, b).ratio() >= FUZZY_RATIO


# --------------------------------------------------------------------------- #
# Season facts per Transfermarkt player
# --------------------------------------------------------------------------- #
def season_facts(players: pd.DataFrame, valuations: pd.DataFrame, window: SeasonWindow) -> pd.DataFrame:
    """One row per Transfermarkt player valued in a Top 5 league in the window.

    Columns: ``player_id, key, born, league, value_eur_m, contract_year, last_season``.
    ``league`` is the league of the latest in-window valuation.
    """
    v = valuations.copy()
    v["date"] = pd.to_datetime(v["date"], errors="coerce")
    v = v[(v["date"] >= window.start) & (v["date"] <= window.end)]
    v = v[v["player_club_domestic_competition_id"].isin(TM_LEAGUES)]
    v = v.sort_values(["player_id", "date"]).groupby("player_id", sort=False).tail(1)
    latest = pd.DataFrame({
        "player_id": v["player_id"].to_numpy(),
        "league": v["player_club_domestic_competition_id"].map(TM_LEAGUES).to_numpy(),
        # Club at that valuation (optional column): confirms surname-only matches.
        "club": (v["current_club_name"] if "current_club_name" in v.columns
                 else pd.Series(pd.NA, index=v.index)).to_numpy(),
        "value_eur_m": (pd.to_numeric(v["market_value_in_eur"], errors="coerce") / 1e6).round(1).to_numpy(),
    })
    p = players.copy()
    p["key"] = p["name"].map(name_key)
    p["born"] = pd.to_datetime(p["date_of_birth"], errors="coerce").dt.year
    p["contract_year"] = pd.to_datetime(p["contract_expiration_date"], errors="coerce").dt.year
    cols = ["player_id", "key", "born", "contract_year", "last_season"]
    return latest.merge(p[cols], on="player_id", how="inner")


def snapshot_date(valuations: pd.DataFrame) -> pd.Timestamp:
    """Latest valuation date: when the dataset (and its contracts) was captured."""
    return pd.to_datetime(valuations["date"], errors="coerce").max()


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def match_players(ours: pd.DataFrame, facts: pd.DataFrame) -> pd.DataFrame:
    """Align ``ours`` (Player, Born, League) to Transfermarkt rows.

    Returns a frame indexed like ``ours`` with ``player_id`` (NaN if
    unmatched), ``method`` ("exact"/"fuzzy"/None) and the matched facts.
    """
    keys = ours["Player"].map(name_key)
    born = pd.to_numeric(ours["Born"], errors="coerce")
    by_born_key = {k: g for k, g in facts.groupby(["born", "key"])}
    by_born_league = {k: g for k, g in facts.groupby(["born", "league"])}

    chosen: list[object] = []
    methods: list[str | None] = []
    for idx in ours.index:
        b, k, league = born[idx], keys[idx], ours.at[idx, "League"]
        pid, method = None, None
        exact = by_born_key.get((b, k))
        if exact is not None:
            if len(exact) > 1:
                exact = exact[exact["league"] == league]
            if len(exact) == 1:
                pid, method = exact["player_id"].iloc[0], "exact"
        if pid is None and exact is None:
            # Birth years differ by one between sources around New Year
            # (Leif Davis: 31 Dec 1999 vs 2000). Exact name, unique candidate only.
            near = [g for year in (b - 1, b + 1) if (g := by_born_key.get((year, k))) is not None]
            near_df = pd.concat(near) if near else None
            if near_df is not None and len(near_df) == 1:
                pid, method = near_df["player_id"].iloc[0], "exact"
        if pid is None and (b, league) in by_born_league:
            pool = by_born_league[(b, league)]
            hits = pool[pool["key"].map(lambda c: names_compatible(k, c))]
            squad = ours.at[idx, "Squad"] if "Squad" in ours.columns else pd.NA
            if len(hits) == 1 and (not surname_only(k, hits["key"].iloc[0])
                                   or clubs_match(squad, hits["club"].iloc[0])):
                pid, method = hits["player_id"].iloc[0], "fuzzy"
        chosen.append(pid)
        methods.append(method)

    out = pd.DataFrame({"player_id": chosen, "method": methods}, index=ours.index)
    # Never map two of our players onto one Transfermarkt player.
    claimed = out["player_id"].dropna()
    clashes = claimed[claimed.duplicated(keep=False)].index
    out.loc[clashes, ["player_id", "method"]] = None
    facts_by_id = facts.set_index("player_id")
    return out.join(facts_by_id[["value_eur_m", "contract_year", "last_season"]], on="player_id")


# --------------------------------------------------------------------------- #
# Applying
# --------------------------------------------------------------------------- #
def attach_transfermarkt(
    enriched: pd.DataFrame,
    players: pd.DataFrame,
    valuations: pd.DataFrame,
    season: str,
) -> tuple[pd.DataFrame, MatchReport]:
    """Overwrite proxy values/contracts with Transfermarkt facts where matched.

    ``enriched`` comes from :func:`src.data.real_data_enricher.enrich_real_players`.
    Unmatched rows keep their proxies. Adds ``Transfermarkt_ID``,
    ``Transfermarkt_Match`` and ``Contract_Reference_Year`` columns.
    """
    from src.data.real_data_enricher import CONTRACT_SOURCE_COLUMN, VALUE_SOURCE_COLUMN

    window = SeasonWindow.for_season(season)
    taken = snapshot_date(valuations)
    snapshot_year = int(taken.year)
    matched = match_players(enriched, season_facts(players, valuations, window))

    out = enriched.copy()
    out[TM_ID_COLUMN] = matched["player_id"].astype("Int64")
    out[TM_MATCH_COLUMN] = matched["method"].fillna("unmatched")

    has_value = matched["value_eur_m"].notna() & (matched["value_eur_m"] > 0)
    out.loc[has_value, "Market_Value_EUR_M"] = matched.loc[has_value, "value_eur_m"].clip(lower=0.1)
    out.loc[has_value, VALUE_SOURCE_COLUMN] = SOURCE_TRANSFERMARKT

    tracked = pd.to_numeric(matched["last_season"], errors="coerce") >= snapshot_year - 1
    has_contract = matched["contract_year"].notna() & tracked
    out.loc[has_contract, "Contract_Expiry_Year"] = matched.loc[has_contract, "contract_year"].astype(int)
    out.loc[has_contract, CONTRACT_SOURCE_COLUMN] = SOURCE_TRANSFERMARKT
    out["Contract_Expiry_Year"] = out["Contract_Expiry_Year"].astype(int)

    # Real contracts are "as listed at the snapshot", so years remaining count
    # from the snapshot year. One reference year per dataset keeps the filter
    # consistent; the few proxy contracts are counted from it too.
    season_end = int(season.split("-")[1])
    out[CONTRACT_REFERENCE_COLUMN] = snapshot_year if has_contract.any() else season_end

    report = MatchReport(
        n_players=len(out),
        n_exact=int((matched["method"] == "exact").sum()),
        n_fuzzy=int((matched["method"] == "fuzzy").sum()),
        n_values=int(has_value.sum()),
        n_contracts=int(has_contract.sum()),
        snapshot_date=taken,
    )
    return out, report
