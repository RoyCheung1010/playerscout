"""Position-aware tactical role names for clusters.

The feature-pair lookup in :mod:`src.models.clustering` names a cluster from
its two highest *scaled* centroid features, whatever the position. On real
data that misfires: with robust scaling a defender's goals have a median of 0
and a tiny IQR, so any centre-back who scores a couple of headers becomes
"Clinical Finishing", and clusters below average on everything are still
named after their two least-bad traits.

Here a cluster is described by its **percentile profile**: the mean
within-position percentile of its members on each feature, re-centred so 0 is
the position median and ±1 the extremes. Percentiles are bounded, so no
single feature can swamp the description. Each position group has a
catalogue of football roles, each a template of expected deviations (+1 well
above the norm, -1 well below, omitted = no expectation). Clusters are named
by **cosine similarity** between profile and template; the Hungarian
algorithm hands out distinct roles where that costs little fit, and genuine
look-alike clusters share a role name qualified by what sets them apart.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.config import FEATURE_LABELS, PERCENTILE_SUFFIX

G, SH, XG, A, KP, PC, PP, TO, PCR, TK, INT, AER = (
    "Goals_p90", "Shots_p90", "Non_Penalty_xG_p90", "Assists_p90", "Key_Passes_p90", "Pass_Completion_Pct",
    "Progressive_Passes_p90", "Successful_Take_Ons_p90", "Progressive_Carries_p90", "pAdj_Tackles_Won_p90",
    "pAdj_Interceptions_p90", "Aerial_Win_Pct",
)
_ALL_LOW = {G: -0.6, SH: -0.6, XG: -0.6, A: -0.6, KP: -0.6, PP: -0.6, TO: -0.6, PCR: -0.6, TK: -0.6, INT: -0.6}

ROLE_TEMPLATES: dict[str, dict[str, Mapping[str, float]]] = {
    "FW": {
        "Penalty-Box Striker": {G: 1.0, SH: 1.0, XG: 1.0, KP: -0.5, TO: -0.5, PP: -0.5, PCR: -0.5, A: -0.2},
        "Target Forward": {AER: 1.0, SH: 0.5, XG: 0.5, G: 0.3, TO: -0.6, PCR: -0.6, PC: -0.3},
        "Complete Forward": {G: 0.8, SH: 0.8, XG: 0.8, A: 0.8, KP: 0.8, TO: 0.5, PCR: 0.5, PP: 0.3},
        "Dribbling Winger": {TO: 1.0, PCR: 0.9, KP: 0.6, A: 0.4, PP: 0.3, G: -0.2, TK: -0.2, AER: -0.3},
        "Creative Playmaker": {KP: 1.0, A: 0.8, PP: 0.8, PC: 0.4, G: -0.3, XG: -0.3},
        "Pressing Forward": {TK: 1.0, INT: 0.8, G: -0.3, XG: -0.3, KP: -0.2},
        "Low-Involvement Forward": _ALL_LOW,
    },
    "MF": {
        "Deep-Lying Playmaker": {PC: 1.0, PP: 1.0, INT: 0.3, G: -0.4, SH: -0.3, XG: -0.3, TO: -0.2},
        "Ball-Winning Midfielder": {TK: 1.0, INT: 1.0, AER: 0.3, KP: -0.4, G: -0.4, SH: -0.3},
        "Box-to-Box Midfielder": {TK: 0.6, PP: 0.5, PCR: 0.5, SH: 0.6, G: 0.5, XG: 0.5, TO: 0.4},
        "Progressive Playmaker": {PP: 1.0, KP: 0.9, A: 0.7, PCR: 0.4, PC: 0.3, SH: 0.2},
        "Advanced Attacking Midfielder": {SH: 0.6, G: 0.5, XG: 0.5, KP: 0.5, TO: 0.1, PCR: 0.1,
                                          PC: -0.6, PP: -0.4, TK: -0.4, INT: -0.4},
        "Goal-Scoring Midfielder": {G: 1.0, SH: 1.0, XG: 1.0, A: 0.3, PC: -0.2},
        "Ball-Carrying Midfielder": {PCR: 1.0, TO: 1.0, KP: 0.4, SH: 0.3, PP: 0.3, PC: -0.2, AER: -0.4},
        "Low-Involvement Midfielder": _ALL_LOW,
    },
    "DF": {
        "Ball-Playing Centre-Back": {PC: 1.0, PP: 0.9, PCR: 0.3, INT: 0.3, AER: 0.3, TO: -0.5, KP: -0.5, A: -0.4},
        "Stopper Centre-Back": {AER: 1.0, INT: 0.9, TK: 0.3, KP: -0.7, TO: -0.7, PCR: -0.5, A: -0.6, PP: -0.3},
        "Attacking Full-Back": {KP: 1.0, A: 0.9, TO: 0.8, PCR: 0.8, PP: 0.4, SH: 0.3, AER: -0.5},
        "Defensive Full-Back": {TK: 1.0, TO: 0.4, KP: 0.2, INT: 0.2, PC: -0.4, G: -0.2, AER: -0.2},
        "Set-Piece Threat Centre-Back": {G: 1.0, SH: 0.9, XG: 1.0, AER: 0.7, KP: -0.4, TO: -0.5, A: -0.3},
        "Positional Centre-Back": {PC: 0.4, AER: 0.4, KP: -0.6, TO: -0.6, PCR: -0.5, A: -0.5, SH: -0.5,
                                   PP: -0.4, TK: -0.4, G: -0.2},
        "Low-Involvement Defender": _ALL_LOW,
    },
}

# Below this cosine similarity a cluster does not resemble any template.
MIN_TEMPLATE_FIT: float = 0.3
# The Hungarian (distinct-role) choice is kept unless it costs more fit than this.
MAX_ASSIGNMENT_LOSS: float = 0.15
# A namesake cluster must beat the others by this much (percentile scale /50)
# on some feature to be qualified by it; otherwise it is the "Lower Volume" one.
MIN_TRAIT_EDGE: float = 0.05


def percentile_profiles(
    cohort: pd.DataFrame, labels: Sequence[int] | np.ndarray, feature_names: Sequence[str]
) -> np.ndarray | None:
    """Mean within-position percentile per cluster and feature, mapped to [-1, 1].

    Returns ``None`` when the cohort lacks the ``*_pct`` columns.
    """
    columns = [f"{f}{PERCENTILE_SUFFIX}" for f in feature_names]
    if not set(columns) <= set(cohort.columns):
        return None
    labels = np.asarray(labels)
    k = int(labels.max()) + 1 if len(labels) else 0
    means = (
        cohort[columns].reset_index(drop=True).groupby(pd.Series(labels, name="cluster")).mean()
        .reindex(range(k)).fillna(50.0)
    )
    return (means.to_numpy(dtype=float) - 50.0) / 50.0


def template_matrix(position_group: str, feature_names: Sequence[str]) -> tuple[list[str], np.ndarray]:
    """``(role names, templates as rows over feature_names)`` for a position."""
    templates = ROLE_TEMPLATES[position_group]
    index = {name: i for i, name in enumerate(feature_names)}
    matrix = np.zeros((len(templates), len(feature_names)))
    for row, weights in enumerate(templates.values()):
        for feature, weight in weights.items():
            if feature in index:
                matrix[row, index[feature]] = weight
    return list(templates), matrix


def template_fit(profiles: np.ndarray, templates: np.ndarray) -> np.ndarray:
    """Cosine similarity of every profile (rows) to every template (columns)."""
    def unit(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return np.divide(m, norms, out=np.zeros_like(m), where=norms > 0)

    return unit(np.asarray(profiles, dtype=float)) @ unit(templates).T


def positional_role_labels(
    profiles: np.ndarray, feature_names: Sequence[str], position_group: str
) -> dict[int, str] | None:
    """Unique role name per cluster, or ``None`` if the position has no catalogue."""
    if position_group not in ROLE_TEMPLATES:
        return None
    profiles = np.asarray(profiles, dtype=float)
    names, templates = template_matrix(position_group, feature_names)
    fit = template_fit(profiles, templates)
    best = fit.argmax(axis=1)

    rows, cols = linear_sum_assignment(-fit)
    choice = best.copy()
    for cluster, role in zip(rows, cols):
        if fit[cluster, role] >= fit[cluster, best[cluster]] - MAX_ASSIGNMENT_LOSS:
            choice[cluster] = role

    labels: dict[int, str] = {}
    for cluster, role in enumerate(choice):
        if fit[cluster, role] >= MIN_TEMPLATE_FIT:
            labels[cluster] = names[role]
        else:
            labels[cluster] = _hybrid_name(profiles[cluster], feature_names)
    return _qualify_duplicates(labels, profiles, feature_names)


def _hybrid_name(profile: np.ndarray, feature_names: Sequence[str]) -> str:
    first, second = (feature_names[i] for i in np.argsort(profile)[::-1][:2])
    return f"Hybrid {_label(first)} & {_label(second)} Profile"


def _qualify_duplicates(
    labels: dict[int, str], profiles: np.ndarray, feature_names: Sequence[str]
) -> dict[int, str]:
    """Clusters sharing a role get "(+ trait)", the feature where each beats its
    namesakes most, or "(Lower Volume)" when it is not ahead anywhere."""
    result = dict(labels)
    for role in set(labels.values()):
        members = [c for c, r in labels.items() if r == role]
        if len(members) < 2:
            continue
        for cluster in members:
            others = profiles[[m for m in members if m != cluster]].mean(axis=0)
            gaps = profiles[cluster] - others
            edge = int(np.argmax(gaps))
            qualifier = f"+ {_label(feature_names[edge])}" if gaps[edge] >= MIN_TRAIT_EDGE else "Lower Volume"
            result[cluster] = f"{role} ({qualifier})"
    seen: dict[str, int] = {}
    for cluster in sorted(result):  # last resort: numbering
        seen[result[cluster]] = seen.get(result[cluster], 0) + 1
        if seen[result[cluster]] > 1:
            result[cluster] = f"{result[cluster]} #{seen[result[cluster]]}"
    return result


def _label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)
