"""Synthetic multi-league player database mirroring FBref/Opta season tables.

The generator produces *raw season totals* (not per-90 values) so that the
downstream loader exercises the exact same per-90, possession-adjustment and
league-coefficient pipeline that a real FBref export would go through.

Realism is driven by a small latent model per player:

* **Archetype** - a hidden tactical role (e.g. deep-lying playmaker, poacher)
  that sets the expected per-90 rate of every stat.
* **Talent** - a latent quality score, correlated with club strength, that
  scales output volume and feeds market value.
* **Club context** - stronger clubs keep more possession, which inflates
  passing volume and suppresses raw defensive actions (the bias the pAdj
  sigmoid later corrects).
* **League context** - weaker leagues inflate per-90 volume slightly (the
  bias the league coefficient later corrects).

Counts are sampled from Poisson/Binomial distributions so that, for example,
goals are a thinned subset of shots and successful take-ons a subset of
attempts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

from src.config import (
    CURRENT_SEASON_END_YEAR,
    LEAGUE_MATCHES,
    LEAGUE_STRENGTH_COEFFICIENTS,
    RANDOM_STATE,
    TOP_5_LEAGUES,
)

# --------------------------------------------------------------------------- #
# Static reference data
# --------------------------------------------------------------------------- #
# Clubs are listed roughly strongest-first; the order seeds club strength.
LEAGUE_CLUBS: Final[dict[str, tuple[str, ...]]] = {
    "Premier League": (
        "Manchester City", "Arsenal", "Liverpool", "Chelsea", "Newcastle United",
        "Aston Villa", "Tottenham Hotspur", "Brighton & Hove Albion", "Brentford",
        "Everton",
    ),
    "La Liga": (
        "Real Madrid", "Barcelona", "Atlético Madrid", "Athletic Club",
        "Real Sociedad", "Villarreal", "Real Betis", "Girona", "Sevilla", "Valencia",
    ),
    "Serie A": (
        "Inter", "Napoli", "Juventus", "Milan", "Atalanta", "Roma", "Lazio",
        "Bologna", "Fiorentina", "Torino",
    ),
    "Bundesliga": (
        "Bayern Munich", "Bayer Leverkusen", "Borussia Dortmund", "RB Leipzig",
        "VfB Stuttgart", "Eintracht Frankfurt", "SC Freiburg", "Mainz 05",
        "Borussia Mönchengladbach", "Wolfsburg",
    ),
    "Ligue 1": (
        "Paris Saint-Germain", "Monaco", "Marseille", "Lille", "Lyon", "Nice",
        "Lens", "Rennes", "Strasbourg", "Nantes",
    ),
    "Eredivisie": (
        "PSV Eindhoven", "Feyenoord", "Ajax", "AZ Alkmaar", "FC Twente",
        "FC Utrecht", "Go Ahead Eagles", "NEC Nijmegen", "Sparta Rotterdam",
        "Heerenveen",
    ),
    "Primeira Liga": (
        "Benfica", "Sporting CP", "Porto", "Braga", "Vitória de Guimarães",
        "Famalicão", "Moreirense", "Estoril", "Rio Ave", "Gil Vicente",
    ),
    "Championship": (
        "Southampton", "Ipswich Town", "Sheffield United", "Middlesbrough",
        "West Bromwich Albion", "Norwich City", "Coventry City", "Watford",
        "Bristol City", "Stoke City",
    ),
}

# Nation code -> (first names, surnames). Codes follow FBref's 3-letter style.
NAME_POOLS: Final[dict[str, tuple[tuple[str, ...], tuple[str, ...]]]] = {
    "ENG": (
        ("Jack", "Harry", "Oliver", "James", "Callum", "Mason", "Jordan", "Reece",
         "Tom", "Ben", "Lewis", "Ryan", "Jacob", "Charlie", "Kyle"),
        ("Walker", "Hughes", "Turner", "Clarke", "Wright", "Mitchell", "Barnes",
         "Phillips", "Cooper", "Ward", "Fletcher", "Stones", "Bennett", "Holt", "Price"),
    ),
    "ESP": (
        ("Pablo", "Álvaro", "Sergio", "Iker", "Dani", "Mikel", "Unai", "Jorge",
         "Rodrigo", "Marcos", "Adrián", "Hugo", "Javier", "Raúl", "Carlos"),
        ("García", "Fernández", "Ruiz", "Moreno", "Navarro", "Iglesias", "Torres",
         "Herrera", "Castillo", "Ortega", "Delgado", "Vidal", "Romero", "Soler", "Gil"),
    ),
    "ITA": (
        ("Lorenzo", "Matteo", "Federico", "Alessandro", "Davide", "Nicolò",
         "Giacomo", "Andrea", "Marco", "Riccardo", "Simone", "Gianluca", "Luca",
         "Stefano", "Tommaso"),
        ("Rossi", "Bianchi", "Esposito", "Romano", "Colombo", "Ricci", "Marino",
         "Greco", "Bruno", "Gallo", "Conti", "Mancini", "Lombardi", "Barbieri", "Ferri"),
    ),
    "GER": (
        ("Lukas", "Jonas", "Leon", "Felix", "Niklas", "Florian", "Julian", "Maximilian",
         "Kai", "Timo", "Jannik", "Moritz", "Tobias", "Paul", "Elias"),
        ("Müller", "Schmidt", "Schneider", "Fischer", "Weber", "Wagner", "Becker",
         "Hoffmann", "Koch", "Richter", "Klein", "Wolf", "Neumann", "Braun", "Krüger"),
    ),
    "FRA": (
        ("Lucas", "Théo", "Hugo", "Antoine", "Mathis", "Kylian", "Rayan", "Enzo",
         "Maxence", "Adrien", "Bastien", "Clément", "Jules", "Noah", "Yanis"),
        ("Martin", "Bernard", "Dubois", "Lefèvre", "Moreau", "Laurent", "Girard",
         "Roux", "Fontaine", "Chevalier", "Lambert", "Faure", "Mercier", "Blanc", "Garnier"),
    ),
    "NED": (
        ("Daan", "Sem", "Luuk", "Milan", "Jesse", "Bram", "Thijs", "Ruben", "Stijn",
         "Jurriën", "Teun", "Wout", "Joey", "Kees", "Jorrel"),
        ("de Jong", "Jansen", "de Vries", "van Dijk", "Bakker", "Visser", "Smit",
         "Meijer", "Mulder", "de Boer", "Bos", "Vos", "Peters", "Hendriks", "van Leeuwen"),
    ),
    "POR": (
        ("João", "Diogo", "Rúben", "Gonçalo", "Bernardo", "Tiago", "Rafael", "Nuno",
         "Pedro", "Vitinha", "Francisco", "Martim", "André", "Bruno", "Samuel"),
        ("Silva", "Santos", "Ferreira", "Pereira", "Oliveira", "Costa", "Rodrigues",
         "Martins", "Sousa", "Gomes", "Lopes", "Marques", "Almeida", "Ribeiro", "Pinto"),
    ),
    "BRA": (
        ("Gabriel", "Lucas", "Matheus", "Vinícius", "Rodrygo", "Bruno", "Thiago",
         "Éder", "Danilo", "Gustavo", "Igor", "Murilo", "Wesley", "Yuri", "Caio"),
        ("Souza", "Lima", "Carvalho", "Araújo", "Barbosa", "Rocha", "Dias", "Teixeira",
         "Moura", "Cardoso", "Nascimento", "Freitas", "Correia", "Monteiro", "Batista"),
    ),
    "ARG": (
        ("Nicolás", "Facundo", "Lautaro", "Julián", "Enzo", "Alexis", "Thiago",
         "Valentín", "Leandro", "Exequiel", "Nahuel", "Gonzalo", "Santiago",
         "Franco", "Matías"),
        ("González", "Rodríguez", "Gómez", "Díaz", "Álvarez", "Sosa", "Benítez",
         "Acosta", "Medina", "Molina", "Paredes", "Ojeda", "Ríos", "Ledesma", "Correa"),
    ),
    "BEL": (
        ("Arthur", "Louis", "Jules", "Victor", "Lucas", "Amadou", "Charles", "Romeo",
         "Maxim", "Zeno", "Aster", "Thomas", "Hans", "Dodi", "Leandro"),
        ("Peeters", "Janssens", "Maes", "Jacobs", "Mertens", "Willems", "Claes",
         "Goossens", "Wouters", "De Smet", "Dubois", "Lambert", "Hermans", "Vermeulen",
         "Lemmens"),
    ),
    "SEN": (
        ("Moussa", "Ismaïla", "Cheikh", "Pape", "Idrissa", "Abdou", "Mamadou",
         "Krépin", "Habib", "Youssouf", "Nicolas", "Lamine", "Iliman", "Bamba", "Ousmane"),
        ("Diallo", "Ndiaye", "Sarr", "Diatta", "Mendy", "Gueye", "Cissé", "Fall",
         "Koné", "Diop", "Faye", "Seck", "Ba", "Sow", "Camara"),
    ),
    "CRO": (
        ("Luka", "Ivan", "Josip", "Mateo", "Marko", "Lovro", "Martin", "Petar",
         "Nikola", "Borna", "Duje", "Ante", "Mario", "Dominik", "Kristijan"),
        ("Horvat", "Kovačević", "Babić", "Marić", "Jurić", "Novak", "Knežević",
         "Vuković", "Perić", "Pavlović", "Šimić", "Radić", "Lovrić", "Matić", "Barišić"),
    ),
}

# Probability that a player at a club in a given league is a home national.
LEAGUE_HOME_NATION: Final[dict[str, str]] = {
    "Premier League": "ENG",
    "La Liga": "ESP",
    "Serie A": "ITA",
    "Bundesliga": "GER",
    "Ligue 1": "FRA",
    "Eredivisie": "NED",
    "Primeira Liga": "POR",
    "Championship": "ENG",
}
HOME_NATION_SHARE: Final[float] = 0.45


@dataclass(frozen=True)
class Archetype:
    """Hidden tactical role defining expected per-90 output at average talent.

    ``position_variants`` holds FBref-style position strings (first token is
    the primary bucket) with their sampling probabilities.
    """

    name: str
    group: str
    goals_per_shot: float
    shots_p90: float
    key_passes_p90: float
    assists_per_key_pass: float
    pass_completion_pct: float
    passes_attempted_p90: float
    progressive_passes_p90: float
    take_ons_attempted_p90: float
    take_on_success_rate: float
    tackles_won_p90: float
    interceptions_p90: float
    position_variants: tuple[tuple[str, float], ...]
    base_value_eur_m: float


ARCHETYPES: Final[tuple[Archetype, ...]] = (
    # ----------------------------- Defenders ----------------------------- #
    Archetype("Ball-Playing Centre-Back", "DF", 0.08, 0.55, 0.30, 0.09, 90.0, 68.0,
              6.5, 0.40, 0.60, 1.20, 1.45, (("DF", 1.0),), 14.0),
    Archetype("Stopper Centre-Back", "DF", 0.09, 0.60, 0.20, 0.08, 83.0, 50.0,
              3.0, 0.25, 0.55, 1.65, 1.85, (("DF", 1.0),), 10.0),
    Archetype("Attacking Full-Back", "DF", 0.06, 0.65, 1.25, 0.12, 79.0, 52.0,
              5.5, 1.90, 0.50, 1.50, 1.10, (("DF", 0.75), ("DF,MF", 0.25)), 13.0),
    # ---------------------------- Midfielders ---------------------------- #
    Archetype("Deep-Lying Playmaker", "MF", 0.06, 0.90, 1.45, 0.08, 89.0, 72.0,
              8.5, 0.90, 0.62, 1.20, 1.05, (("MF", 1.0),), 16.0),
    Archetype("Defensive Destroyer", "MF", 0.05, 0.60, 0.65, 0.07, 86.0, 58.0,
              5.0, 0.80, 0.58, 2.30, 1.60, (("MF", 0.75), ("MF,DF", 0.25)), 12.0),
    Archetype("Box-to-Box Engine", "MF", 0.10, 1.30, 1.10, 0.10, 84.0, 52.0,
              5.0, 1.80, 0.55, 1.60, 0.90, (("MF", 1.0),), 15.0),
    Archetype("Advanced Creator", "MF", 0.11, 2.10, 2.40, 0.11, 82.0, 46.0,
              4.6, 2.60, 0.52, 0.80, 0.40, (("MF", 0.6), ("MF,FW", 0.4)), 20.0),
    # ----------------------------- Forwards ------------------------------ #
    Archetype("Penalty-Box Poacher", "FW", 0.15, 2.90, 0.85, 0.12, 72.0, 22.0,
              1.2, 1.40, 0.45, 0.35, 0.18, (("FW", 1.0),), 22.0),
    Archetype("Pressing Target Forward", "FW", 0.12, 2.40, 1.10, 0.10, 68.0, 28.0,
              1.6, 1.10, 0.42, 0.75, 0.30, (("FW", 1.0),), 15.0),
    Archetype("Inside Forward", "FW", 0.12, 2.50, 1.70, 0.12, 76.0, 34.0,
              2.8, 4.60, 0.47, 0.55, 0.25, (("FW", 0.55), ("FW,MF", 0.45)), 24.0),
    Archetype("Touchline Dribbler", "FW", 0.09, 1.70, 1.95, 0.13, 75.0, 33.0,
              3.0, 7.00, 0.48, 0.60, 0.30, (("FW,MF", 0.6), ("FW", 0.4)), 20.0),
)

# Per-archetype profile for the stats added in v1.4, at average talent:
# (progressive carries per 90, aerial duel win %, npxG per shot).
# They are drawn from a separate random stream (see ``_extra_rng``) so that
# adding them left every pre-existing column identical for a given seed.
EXTRA_PROFILES: Final[dict[str, tuple[float, float, float]]] = {
    "Ball-Playing Centre-Back": (1.40, 63.0, 0.09),
    "Stopper Centre-Back": (0.55, 68.0, 0.10),
    "Attacking Full-Back": (2.30, 45.0, 0.06),
    "Deep-Lying Playmaker": (1.30, 48.0, 0.06),
    "Defensive Destroyer": (0.90, 56.0, 0.06),
    "Box-to-Box Engine": (2.00, 50.0, 0.09),
    "Advanced Creator": (2.40, 37.0, 0.10),
    "Penalty-Box Poacher": (0.80, 42.0, 0.15),
    "Pressing Target Forward": (0.90, 56.0, 0.12),
    "Inside Forward": (3.30, 30.0, 0.11),
    "Touchline Dribbler": (3.80, 28.0, 0.09),
}

# Outfield squad slots generated per club, by position group.
GROUP_SLOTS: Final[dict[str, int]] = {"DF": 5, "MF": 5, "FW": 4}

# Position-group value multipliers (attackers command a premium).
GROUP_VALUE_FACTOR: Final[dict[str, float]] = {"DF": 0.85, "MF": 1.0, "FW": 1.2}


@dataclass(frozen=True)
class GeneratorConfig:
    """Tunable knobs for the synthetic database."""

    seed: int = RANDOM_STATE
    season: str = "2025-2026"
    season_end_year: int = CURRENT_SEASON_END_YEAR
    clubs_per_league: int = 10
    leagues: tuple[str, ...] = field(default_factory=lambda: tuple(LEAGUE_CLUBS))
    # Share of each squad that are regular starters (rest are rotation/fringe).
    regular_share: float = 0.68
    # Share of forwards seeded as generational outliers (Haaland/Mbappé tier).
    elite_outlier_share: float = 0.02
    # Per-90 volume inflation in leagues with coefficient < 1.
    secondary_league_inflation: float = 1.12
    # Keep the hidden archetype column (useful to validate clustering).
    include_ground_truth: bool = False

    def __post_init__(self) -> None:
        unknown = set(self.leagues) - set(LEAGUE_CLUBS)
        if unknown:
            raise ValueError(f"Unknown leagues requested: {sorted(unknown)}")
        if not 1 <= self.clubs_per_league <= 10:
            raise ValueError("clubs_per_league must be between 1 and 10")
        if not 0.0 <= self.regular_share <= 1.0:
            raise ValueError("regular_share must be within [0, 1]")
        if not 0.0 <= self.elite_outlier_share <= 1.0:
            raise ValueError("elite_outlier_share must be within [0, 1]")


@dataclass(frozen=True)
class _Club:
    name: str
    league: str
    strength: float  # latent z-score, higher is stronger
    possession_pct: float


class SyntheticPlayerGenerator:
    """Builds a reproducible FBref-style season table of outfield players.

    Example:
        >>> df = SyntheticPlayerGenerator().generate()
        >>> len(df) >= 500
        True
    """

    def __init__(self, config: GeneratorConfig | None = None) -> None:
        self.config = config or GeneratorConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._extra_rng = np.random.default_rng([self.config.seed, 14])
        self._used_names: set[str] = set()
        self._archetypes_by_group: dict[str, list[Archetype]] = {
            group: [a for a in ARCHETYPES if a.group == group] for group in GROUP_SLOTS
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate(self) -> pd.DataFrame:
        """Return one row per player with raw season totals and metadata."""
        # Reset state so repeated calls on the same instance are identical.
        self._rng = np.random.default_rng(self.config.seed)
        self._extra_rng = np.random.default_rng([self.config.seed, 14])
        self._used_names = set()

        rows: list[dict[str, object]] = []
        for league in self.config.leagues:
            for club in self._build_clubs(league):
                for group, n_slots in GROUP_SLOTS.items():
                    for slot in range(n_slots):
                        rows.append(self._build_player(club, group, slot, n_slots))

        df = pd.DataFrame(rows)
        if not self.config.include_ground_truth:
            df = df.drop(columns=["Generator_Archetype"])
        return df.sort_values(["League", "Squad", "Player"], ignore_index=True)

    # ------------------------------------------------------------------ #
    # Clubs
    # ------------------------------------------------------------------ #
    def _build_clubs(self, league: str) -> list[_Club]:
        names = LEAGUE_CLUBS[league][: self.config.clubs_per_league]
        n = len(names)
        base_strength = np.linspace(1.6, -1.3, n) if n > 1 else np.zeros(1)
        strengths = base_strength + self._rng.normal(0.0, 0.25, n)
        # Strong sides dominate the ball; clip to the plausible top-flight range.
        possession = np.clip(50.0 + 6.5 * strengths + self._rng.normal(0.0, 1.5, n), 36.0, 68.0)
        return [
            _Club(name, league, float(s), round(float(p), 1))
            for name, s, p in zip(names, strengths, possession)
        ]

    # ------------------------------------------------------------------ #
    # Players
    # ------------------------------------------------------------------ #
    def _build_player(self, club: _Club, group: str, slot: int, n_slots: int) -> dict[str, object]:
        rng = self._rng
        cfg = self.config
        archetype = self._sample_archetype(group)

        # Talent correlates with club strength; regulars are better than fringe.
        is_regular = slot < round(n_slots * cfg.regular_share)
        talent = 0.45 * club.strength + rng.normal(0.0, 0.85) + (0.25 if is_regular else -0.25)
        is_elite = group == "FW" and rng.random() < cfg.elite_outlier_share
        if is_elite:
            talent += 1.1

        age = self._sample_age(is_regular)
        matches_played, starts, minutes = self._sample_minutes(club.league, is_regular, age)
        nineties = minutes / 90.0

        league_inflation = (
            1.0
            if LEAGUE_STRENGTH_COEFFICIENTS.get(club.league, 1.0) >= 1.0
            else cfg.secondary_league_inflation
        )
        attack_mult = float(np.exp(0.22 * talent)) * league_inflation
        if is_elite:
            attack_mult *= 1.12

        # Possession context: dominant teams pass more, defend (raw) less.
        poss_ratio = club.possession_pct / 50.0
        defensive_exposure = ((100.0 - club.possession_pct) / 50.0) ** 0.9

        def rate(base: float, mult: float = 1.0, noise: float = 0.18) -> float:
            """Player-specific per-90 rate with multiplicative log-normal noise."""
            return max(base * mult * float(np.exp(rng.normal(0.0, noise))), 0.0)

        # --- Shooting -------------------------------------------------------
        shots = int(rng.poisson(rate(archetype.shots_p90, attack_mult) * nineties))
        finishing = float(np.clip(archetype.goals_per_shot * np.exp(0.12 * talent + rng.normal(0, 0.15)), 0.02, 0.35))
        goals = int(rng.binomial(shots, finishing))

        # --- Creation -------------------------------------------------------
        key_passes = int(rng.poisson(rate(archetype.key_passes_p90, attack_mult) * nineties))
        assist_rate = float(np.clip(archetype.assists_per_key_pass * np.exp(rng.normal(0, 0.2)), 0.02, 0.3))
        assists = int(rng.binomial(key_passes, assist_rate))

        # --- Passing --------------------------------------------------------
        passes_attempted = int(rng.poisson(rate(archetype.passes_attempted_p90, poss_ratio, 0.12) * nineties))
        completion = float(np.clip(
            archetype.pass_completion_pct + 2.2 * talent + 10.0 * (poss_ratio - 1.0) + rng.normal(0, 2.5),
            55.0, 95.0,
        )) / 100.0
        passes_completed = int(rng.binomial(passes_attempted, completion))
        pass_completion_pct = round(100.0 * passes_completed / passes_attempted, 1) if passes_attempted else 0.0
        progressive_passes = int(rng.poisson(
            rate(archetype.progressive_passes_p90, float(np.exp(0.15 * talent)) * poss_ratio) * nineties
        ))

        # --- Carrying -------------------------------------------------------
        take_ons_attempted = int(rng.poisson(rate(archetype.take_ons_attempted_p90, attack_mult, 0.22) * nineties))
        take_on_success = float(np.clip(archetype.take_on_success_rate + 0.03 * talent + rng.normal(0, 0.05), 0.2, 0.8))
        successful_take_ons = int(rng.binomial(take_ons_attempted, take_on_success))

        # --- Defending (raw, possession-biased) -----------------------------
        def_mult = defensive_exposure * float(np.exp(0.08 * talent)) * league_inflation
        tackles_won = int(rng.poisson(rate(archetype.tackles_won_p90, def_mult) * nineties))
        interceptions = int(rng.poisson(rate(archetype.interceptions_p90, def_mult) * nineties))

        # --- v1.4 stats (separate stream: earlier columns stay seed-stable) --
        xrng = self._extra_rng
        carries_p90, aerial_base, xg_per_shot = EXTRA_PROFILES[archetype.name]
        progressive_carries = int(xrng.poisson(
            carries_p90 * attack_mult * float(np.exp(xrng.normal(0.0, 0.22))) * nineties
        ))
        aerial_win_pct = round(float(np.clip(aerial_base + 2.5 * talent + xrng.normal(0.0, 6.0), 10.0, 90.0)), 1)
        # Shot quality tracks the finishing profile, without the finishing luck.
        npxg = round(shots * xg_per_shot * float(np.exp(0.08 * talent + xrng.normal(0.0, 0.10))), 1)

        contract_expiry = self._sample_contract_expiry(age)
        market_value = self._market_value(archetype, club, talent, age, contract_expiry, minutes)
        nation = self._sample_nation(club.league)

        return {
            "Player": self._unique_name(nation),
            "Nation": nation,
            "Position": self._sample_position(archetype),
            "Squad": club.name,
            "League": club.league,
            "Season": cfg.season,
            "Age": age,
            "Born": cfg.season_end_year - age - int(rng.integers(0, 2)),
            "Matches_Played": matches_played,
            "Starts": starts,
            "Minutes_Played": minutes,
            "Nineties": round(nineties, 1),
            "Goals": goals,
            "Shots": shots,
            "Assists": assists,
            "Key_Passes": key_passes,
            "Passes_Completed": passes_completed,
            "Passes_Attempted": passes_attempted,
            "Pass_Completion_Pct": pass_completion_pct,
            "Progressive_Passes": progressive_passes,
            "Take_Ons_Attempted": take_ons_attempted,
            "Successful_Take_Ons": successful_take_ons,
            "Tackles_Won": tackles_won,
            "Interceptions": interceptions,
            "Non_Penalty_xG": npxg,
            "Progressive_Carries": progressive_carries,
            "Aerial_Win_Pct": aerial_win_pct,
            "Team_Possession_Pct": club.possession_pct,
            "Market_Value_EUR_M": market_value,
            "Contract_Expiry_Year": contract_expiry,
            "Generator_Archetype": archetype.name,
        }

    def _sample_archetype(self, group: str) -> Archetype:
        options = self._archetypes_by_group[group]
        return options[int(self._rng.integers(0, len(options)))]

    def _sample_position(self, archetype: Archetype) -> str:
        labels = [label for label, _ in archetype.position_variants]
        probs = np.array([p for _, p in archetype.position_variants], dtype=float)
        return str(self._rng.choice(labels, p=probs / probs.sum()))

    def _sample_age(self, is_regular: bool) -> int:
        mean = 26.5 if is_regular else 23.5
        return int(np.clip(round(self._rng.normal(mean, 3.8)), 17, 37))

    def _sample_minutes(self, league: str, is_regular: bool, age: int) -> tuple[int, int, int]:
        rng = self._rng
        n_matches = LEAGUE_MATCHES.get(league, 38)
        if is_regular:
            availability = rng.beta(7.0, 2.2)  # injuries / rotation
            matches_played = max(int(round(n_matches * availability)), 1)
            start_share = float(np.clip(rng.normal(0.85, 0.1), 0.4, 1.0))
            mins_per_start, mins_per_sub = rng.normal(82, 5), rng.normal(22, 6)
        else:
            availability = rng.beta(2.0, 3.5)
            # Veterans and teenagers on the fringe tend to see fewer minutes.
            if age <= 19 or age >= 34:
                availability *= 0.7
            matches_played = int(round(n_matches * availability))
            start_share = float(np.clip(rng.normal(0.35, 0.15), 0.0, 1.0))
            mins_per_start, mins_per_sub = rng.normal(72, 8), rng.normal(18, 6)

        starts = int(round(matches_played * start_share))
        subs = matches_played - starts
        minutes = int(round(starts * np.clip(mins_per_start, 60, 90) + subs * np.clip(mins_per_sub, 5, 45)))
        return matches_played, starts, max(min(minutes, n_matches * 90), 0)

    def _sample_contract_expiry(self, age: int) -> int:
        # Younger players are usually tied down for longer.
        start = self.config.season_end_year + 1
        max_years = 5 if age <= 24 else 4 if age <= 29 else 2
        return start + int(self._rng.integers(0, max_years))

    def _market_value(
        self,
        archetype: Archetype,
        club: _Club,
        talent: float,
        age: int,
        contract_expiry: int,
        minutes: int,
    ) -> float:
        rng = self._rng
        league_factor = 1.0 if club.league in TOP_5_LEAGUES else 0.4
        # Value peaks around 24-26 and decays sharply after 30.
        age_factor = float(np.exp(-((age - 25.0) ** 2) / (2 * 4.5**2)))
        if age < 21:
            age_factor *= 1.15  # resale premium for prospects
        years_left = contract_expiry - self.config.season_end_year
        contract_factor = 0.55 + 0.15 * min(years_left, 3)
        usage_factor = 0.6 + 0.4 * min(minutes / 2500.0, 1.0)
        value = (
            archetype.base_value_eur_m
            * GROUP_VALUE_FACTOR[archetype.group]
            * league_factor
            * age_factor
            * contract_factor
            * usage_factor
            * float(np.exp(0.65 * talent + 0.25 * club.strength + rng.normal(0.0, 0.3)))
        )
        # Selling-league clubs rarely command fees above this ceiling.
        value_cap = 200.0 if club.league in TOP_5_LEAGUES else 60.0
        return round(float(np.clip(value, 0.2, value_cap)), 1)

    def _sample_nation(self, league: str) -> str:
        home = LEAGUE_HOME_NATION.get(league)
        if home and self._rng.random() < HOME_NATION_SHARE:
            return home
        nations = list(NAME_POOLS)
        return nations[int(self._rng.integers(0, len(nations)))]

    def _unique_name(self, nation: str) -> str:
        first_names, surnames = NAME_POOLS[nation]
        for _ in range(50):
            name = f"{self._rng.choice(first_names)} {self._rng.choice(surnames)}"
            if name not in self._used_names:
                self._used_names.add(name)
                return name
        # Pool exhausted for this nation: disambiguate with a middle initial.
        base = f"{self._rng.choice(first_names)} {self._rng.choice(surnames)}"
        for letter in "ABCDEFGHIJKLMNOPRSTVW":
            first, last = base.split(" ", 1)
            name = f"{first} {letter}. {last}"
            if name not in self._used_names:
                self._used_names.add(name)
                return name
        name = f"{base} {len(self._used_names)}"
        self._used_names.add(name)
        return name


def generate_synthetic_players(config: GeneratorConfig | None = None) -> pd.DataFrame:
    """Convenience wrapper used by the loader and Streamlit cache layer."""
    return SyntheticPlayerGenerator(config).generate()
