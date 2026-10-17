"""Dark-theme Matplotlib scouting visuals.

Every public function accepts an optional ``ax`` and returns the
:class:`matplotlib.figure.Figure` it drew on, ready for ``st.pyplot(fig)``.

Figures are created with :class:`matplotlib.figure.Figure` directly rather than
``plt.subplots``, so they never enter pyplot's global figure registry and are
garbage-collected like any other object - no ``plt.close`` bookkeeping, no
leaks across Streamlit reruns.

Colour design:

* The 10-slot cluster palette was validated for the ``#0e1117`` surface
  (lightness band, chroma floor, adjacent-pair colour-vision-deficiency
  separation, contrast). Scatter plots put every pair side by side, which no
  10-colour palette can survive, so clusters also get a distinct marker shape
  and centroids carry direct role labels: identity is never colour alone.
* Colour follows the cluster id, never its rank or size, so a cluster keeps
  its colour when ``k`` changes elsewhere.
* Text always uses ink tokens; series colours are reserved for marks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import wraps
from typing import ParamSpec, TypeVar

import textwrap

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.projections.polar import PolarAxes
from scipy.stats import chi2

from src.config import CORE_FEATURES, PERCENTILE_SUFFIX
from src.models.pca import TacticalPCA


# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScoutingTheme:
    """Colour tokens for the dark scouting aesthetic."""

    background: str = "#0e1117"
    panel: str = "#161b22"
    grid: str = "#30363d"
    ink: str = "#c9d1d9"         # body text, tick labels
    ink_strong: str = "#ffffff"  # titles, emphasis
    ink_muted: str = "#8b949e"   # subtitles, captions
    target: str = "#e3b341"      # gold - the player being replaced
    candidate: str = "#58a6ff"   # electric blue - replacement candidates
    # Validated categorical order (dark surface); assigned by cluster id.
    clusters: tuple[str, ...] = (
        "#3987e5", "#d95926", "#199e70", "#c98500", "#d55181",
        "#008300", "#9085e9", "#e66767", "#3a9cc2", "#958c22",
    )
    # Secondary encoding so clusters stay distinguishable without colour.
    markers: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "p", "h", "<", ">")
    heatmap_cmap: str = "RdBu_r"


THEME = ScoutingTheme()

# Largest k for which centroids are labelled with full role names on the map.
CENTROID_FULL_LABEL_MAX = 5

RC_PARAMS: dict[str, object] = {
    "figure.facecolor": THEME.background,
    "axes.facecolor": THEME.background,
    "savefig.facecolor": THEME.background,
    "axes.edgecolor": THEME.grid,
    "axes.labelcolor": THEME.ink,
    "axes.titlecolor": THEME.ink_strong,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": THEME.grid,
    "grid.alpha": 0.6,
    "grid.linewidth": 0.6,
    "xtick.color": THEME.ink,
    "ytick.color": THEME.ink,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "text.color": THEME.ink,
    "legend.facecolor": THEME.panel,
    "legend.edgecolor": THEME.grid,
    "legend.labelcolor": THEME.ink,
    "legend.fontsize": 8.5,
    "lines.linewidth": 2.0,
    "font.family": "DejaVu Sans",
}

P = ParamSpec("P")
R = TypeVar("R")


def themed(func: Callable[P, R]) -> Callable[P, R]:
    """Run ``func`` inside the scouting rc context (no global rcParams mutation)."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with mpl.rc_context(RC_PARAMS):
            return func(*args, **kwargs)

    return wrapper


def apply_global_theme() -> None:
    """Opt-in: apply the theme to every Matplotlib figure in the process."""
    mpl.rcParams.update(RC_PARAMS)


def cluster_color(cluster_id: int) -> str:
    return THEME.clusters[int(cluster_id) % len(THEME.clusters)]


def cluster_marker(cluster_id: int) -> str:
    return THEME.markers[int(cluster_id) % len(THEME.markers)]


def format_metric(name: str) -> str:
    """``"pAdj_Tackles_Won_p90"`` -> ``"pAdj Tackles Won p90"``."""
    if name.endswith(PERCENTILE_SUFFIX):
        name = name[: -len(PERCENTILE_SUFFIX)]
    return name.replace("_", " ").strip()


def _new_figure(figsize: tuple[float, float], polar: bool = False) -> tuple[Figure, Axes]:
    fig = Figure(figsize=figsize, dpi=110)
    ax = fig.add_subplot(projection="polar" if polar else None)
    return fig, ax


def _resolve_axes(ax: Axes | None, figsize: tuple[float, float], polar: bool = False) -> tuple[Figure, Axes]:
    if ax is None:
        return _new_figure(figsize, polar)
    if polar and not isinstance(ax, PolarAxes):
        raise ValueError("plot_radar_comparison needs a polar Axes (projection='polar')")
    fig = ax.get_figure()
    if fig is None:
        raise ValueError("The supplied Axes is not attached to a Figure")
    return fig, ax  # type: ignore[return-value]


def _wrap(text: str, width: int, max_lines: int = 2) -> str:
    """Wrap onto at most ``max_lines`` lines, truncating only what still overflows."""
    lines = textwrap.wrap(text, width) or [""]
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [_truncate(" ".join(lines[max_lines - 1 :]), width)]
    return "\n".join(lines)


def _truncate(text: str, width: int = 34) -> str:
    return text if len(text) <= width else text[: width - 1].rstrip() + "…"


# --------------------------------------------------------------------------- #
# 1. PCA tactical cluster map
# --------------------------------------------------------------------------- #
@themed
def plot_pca_tactical_map(
    pca_df: pd.DataFrame,
    pca_model: TacticalPCA,
    cluster_labels: np.ndarray,
    role_names: Mapping[int, str],
    target_player: str | None = None,
    candidate_matches: pd.DataFrame | None = None,
    show_centroids: bool = True,
    ax: Axes | None = None,
    show_ellipses: bool = True,
    ellipse_confidence: float = 0.68,
    max_candidates: int = 5,
    legend_loc: str = "right",
) -> Figure:
    """Scatter of ``PC1`` vs ``PC2`` coloured (and shaped) by tactical role.

    Args:
        pca_df: Row-aligned with ``cluster_labels``; needs ``PC1``, ``PC2`` and
            ``Player`` (``Squad`` is used to disambiguate if present).
        pca_model: Fitted model, used for the explained-variance subtitle.
        cluster_labels: Integer cluster id per row.
        role_names: Cluster id -> role name.
        target_player: Name of the player to highlight with a gold star.
        candidate_matches: Similarity results (``Rank``, ``Player``,
            ``Match_Score_Pct``); the first ``max_candidates`` are ringed.
        show_centroids: Mark each cluster's mean PC position with an ``X``.
        show_ellipses: Draw a covariance ellipse per cluster.
        ellipse_confidence: Probability mass inside each ellipse (Gaussian).
        legend_loc: ``"right"`` (beside the plot) or ``"below"`` (two
            columns under it, keeping the plot wide in narrow containers).

    Raises:
        ValueError: missing columns, misaligned labels, or bad confidence.
    """
    missing = [c for c in ("PC1", "PC2", "Player") if c not in pca_df.columns]
    if missing:
        raise ValueError(f"pca_df is missing columns: {missing}")
    labels = np.asarray(cluster_labels)
    if len(labels) != len(pca_df):
        raise ValueError(f"{len(labels)} cluster labels for {len(pca_df)} rows")
    if not 0.0 < ellipse_confidence < 1.0:
        raise ValueError("ellipse_confidence must be in (0, 1)")
    if legend_loc not in ("right", "below"):
        raise ValueError("legend_loc must be 'right' or 'below'")

    fig, ax = _resolve_axes(ax, (11, 7.5) if legend_loc == "right" else (10, 9.2))
    data = pca_df.reset_index(drop=True)
    xy = data[["PC1", "PC2"]].to_numpy(dtype=float)

    cluster_ids = sorted(int(c) for c in np.unique(labels))
    # Past this many clusters, full role names on the map collide; centroids
    # then carry just their id and the legend supplies the role name.
    compact_centroid_labels = len(cluster_ids) > CENTROID_FULL_LABEL_MAX
    handles: list[Line2D] = []
    for cid in cluster_ids:
        members = xy[labels == cid]
        color, marker = cluster_color(cid), cluster_marker(cid)
        role = role_names.get(cid, f"Cluster {cid}")
        ax.scatter(
            members[:, 0], members[:, 1], s=26, c=color, marker=marker, alpha=0.78,
            edgecolors=THEME.background, linewidths=0.5, zorder=2,
        )
        if show_ellipses and len(members) >= 3:
            _draw_confidence_ellipse(ax, members, color, ellipse_confidence)
        if show_centroids and len(members):
            cx, cy = members.mean(axis=0)
            ax.scatter(cx, cy, s=150, marker="X", c=color, edgecolors=THEME.ink_strong,
                       linewidths=1.2, zorder=5)
            ax.annotate(
                str(cid) if compact_centroid_labels else _truncate(role, 28),
                (cx, cy), xytext=(0, 11), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5, color=THEME.ink_strong, zorder=6,
                bbox={"boxstyle": "round,pad=0.25", "fc": THEME.panel, "ec": color, "lw": 0.8, "alpha": 0.9},
            )
        handles.append(Line2D([], [], ls="", marker=marker, color=color, markersize=7,
                              label=f"{cid}: {_truncate(role, 48)} ({len(members)})"))

    callouts: list[_Callout] = []
    if target_player is not None:
        target_idx = _find_player(data, target_player)
        if target_idx is not None:
            tx, ty = xy[target_idx]
            ax.scatter(tx, ty, s=220, marker="*", c=THEME.target, edgecolors=THEME.background,
                       linewidths=1.0, zorder=8)
            callouts.append(_Callout((tx, ty), f"Target: {data.at[target_idx, 'Player']}", THEME.target, True))
            handles.append(Line2D([], [], ls="", marker="*", color=THEME.target, markersize=12,
                                  label=f"Target: {target_player}"))

    if candidate_matches is not None and not candidate_matches.empty:
        shown = 0
        ordered = candidate_matches.sort_values("Rank") if "Rank" in candidate_matches else candidate_matches
        for i, (_, match) in enumerate(ordered.head(max_candidates).iterrows()):
            idx = _find_player(data, str(match["Player"]), match.get("Squad"))
            if idx is None:
                continue
            px, py = xy[idx]
            ax.scatter(px, py, s=140, marker="o", facecolors="none", edgecolors=THEME.candidate,
                       linewidths=2.0, zorder=7)
            rank = int(match["Rank"]) if "Rank" in match else i + 1
            score = f" · {float(match['Match_Score_Pct']):.0f}%" if "Match_Score_Pct" in match else ""
            callouts.append(_Callout((px, py), f"#{rank} {match['Player']}{score}", THEME.candidate))
            shown += 1
        if shown:
            handles.append(Line2D([], [], ls="", marker="o", markerfacecolor="none",
                                  markeredgecolor=THEME.candidate, markeredgewidth=2, markersize=10,
                                  label="Top replacement targets"))

    _draw_callout_lane(ax, xy, callouts)

    ratio = pca_model.explained_variance_ratio
    ax.set_title("Tactical Role Map", loc="left", pad=22)
    ax.text(
        0.0, 1.015,
        f"Explained Variance: PC1 ({ratio[0]:.1%}) + PC2 ({ratio[1]:.1%}) = {pca_model.cumulative_variance:.1%}",
        transform=ax.transAxes, fontsize=9, color=THEME.ink_muted, va="bottom",
    )
    ax.set_xlabel(f"PC1 ({ratio[0]:.1%} of variance)")
    ax.set_ylabel(f"PC2 ({ratio[1]:.1%} of variance)")
    ax.axhline(0, color=THEME.grid, lw=0.8, zorder=1)
    ax.axvline(0, color=THEME.grid, lw=0.8, zorder=1)
    if legend_loc == "right":
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True,
                  title="Tactical roles", title_fontsize=9)
    else:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2,
                  frameon=True, title="Tactical roles", title_fontsize=9)
    fig.tight_layout()
    return fig


def _draw_confidence_ellipse(ax: Axes, points: np.ndarray, color: str, confidence: float) -> None:
    cov = np.cov(points, rowvar=False)
    if not np.isfinite(cov).all():
        return
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    if eigvals.max() <= 0:
        return
    scale = np.sqrt(chi2.ppf(confidence, df=2))
    width, height = 2 * scale * np.sqrt(eigvals[::-1])
    angle = float(np.degrees(np.arctan2(eigvecs[1, -1], eigvecs[0, -1])))
    ax.add_patch(Ellipse(points.mean(axis=0), width, height, angle=angle, facecolor=color,
                         alpha=0.08, edgecolor=color, lw=1.2, ls="--", zorder=1))


def _find_player(data: pd.DataFrame, name: str, squad: object = None) -> int | None:
    hits = data["Player"].astype(str) == name
    if squad is not None and not pd.isna(squad) and "Squad" in data.columns:
        narrowed = hits & (data["Squad"].astype(str) == str(squad))
        hits = narrowed if narrowed.any() else hits
    positions = np.flatnonzero(hits.to_numpy())
    return int(positions[0]) if positions.size else None


@dataclass(frozen=True)
class _Callout:
    point: tuple[float, float]
    text: str
    color: str
    bold: bool = False


def _draw_callout_lane(ax: Axes, points: np.ndarray, callouts: Sequence[_Callout]) -> None:
    """Stack callout labels in a margin lane beside the point cloud.

    The axis is widened on the side nearest the highlighted players and labels
    are stacked there, ordered by the height of their points so leader lines
    never cross. This keeps every label legible however tightly the targets
    cluster, which per-point offsets cannot guarantee.
    """
    if len(points) == 0:
        return
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)
    pad_x, pad_y = 0.05 * x_span, 0.06 * y_span
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    if not callouts:
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        return

    anchors = np.array([c.point for c in callouts], dtype=float)
    lane = 0.45 * x_span
    on_left = anchors[:, 0].mean() < np.median(points[:, 0])
    if on_left:
        ax.set_xlim(x_min - pad_x - lane, x_max + pad_x)
        lane_x, ha = x_min - pad_x - 0.97 * lane, "left"
    else:
        ax.set_xlim(x_min - pad_x, x_max + pad_x + lane)
        lane_x, ha = x_max + pad_x + 0.97 * lane, "right"

    n = len(callouts)
    gap = min(0.09 * y_span, y_span / max(n - 1, 1))
    half = gap * (n - 1) / 2
    centre = float(np.clip(anchors[:, 1].mean(), y_min + half, y_max - half))
    slots = centre + half - gap * np.arange(n)
    for slot, i in zip(slots, np.argsort(-anchors[:, 1], kind="stable")):
        callout = callouts[int(i)]
        ax.annotate(
            callout.text, callout.point, xytext=(lane_x, slot), textcoords="data",
            ha=ha, va="center", fontsize=8.5, color=THEME.ink_strong, zorder=9,
            fontweight="bold" if callout.bold else "normal",
            bbox={"boxstyle": "round,pad=0.3", "fc": THEME.panel, "ec": callout.color, "lw": 1.2},
            arrowprops={"arrowstyle": "-", "color": callout.color, "lw": 0.9, "alpha": 0.85,
                        "shrinkA": 0, "shrinkB": 7},
        )


# --------------------------------------------------------------------------- #
# 2. Dual-player radar
# --------------------------------------------------------------------------- #
@themed
def plot_radar_comparison(
    player_1_data: pd.Series | Mapping[str, float],
    player_1_name: str,
    player_2_data: pd.Series | Mapping[str, float] | None = None,
    player_2_name: str | None = None,
    metrics: Sequence[str] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Polar percentile radar for one player, or two head-to-head.

    For each metric the ``<metric>_pct`` value is used when present (so a full
    processed player row can be passed directly), otherwise ``<metric>`` itself
    is read as a 0-100 percentile. Values are clipped to [0, 100]; missing or
    NaN values plot at 0 and are flagged in the caption.

    Raises:
        ValueError: fewer than 3 metrics, a metric absent from the data, or a
            non-polar ``ax``.
    """
    metric_list = list(metrics) if metrics is not None else list(CORE_FEATURES)
    if len(metric_list) < 3:
        raise ValueError("A radar needs at least 3 metrics")

    v1, gaps1 = _radar_values(player_1_data, metric_list, player_1_name)
    series = [(player_1_name, v1, THEME.target, 0.35)]
    gaps = list(gaps1)
    if player_2_data is not None:
        name_2 = player_2_name or "Comparison"
        v2, gaps2 = _radar_values(player_2_data, metric_list, name_2)
        series.append((name_2, v2, THEME.candidate, 0.30))
        gaps += gaps2

    owns_figure = ax is None
    fig, ax = _resolve_axes(ax, (8.5, 7.8), polar=True)
    n = len(metric_list)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=7.5, color=THEME.ink_muted)
    ax.set_rlabel_position(180 / n)
    ax.set_xticks(angles)
    ax.set_xticklabels([format_metric(m) for m in metric_list], fontsize=9, color=THEME.ink)
    ax.tick_params(axis="x", pad=10)
    ax.grid(color=THEME.grid, alpha=0.7, lw=0.8)
    ax.spines["polar"].set_color(THEME.grid)
    ax.set_facecolor(THEME.background)

    for name, values, color, alpha in series:
        closed = np.concatenate([values, values[:1]])  # close the polygon loop
        ax.fill(closed_angles, closed, color=color, alpha=alpha, zorder=2)
        ax.plot(closed_angles, closed, color=color, lw=2.0, zorder=3, label=name)
        ax.scatter(angles, values, s=24, color=color, edgecolors=THEME.background,
                   linewidths=0.8, zorder=4)

    title = player_1_name if len(series) == 1 else f"{player_1_name} vs {series[1][0]}"
    ax.set_title(title, color=THEME.ink_strong, pad=28, fontsize=13, fontweight="bold")
    caption = "Within-position percentiles (0-100)"
    if gaps:
        caption += f" · no data for: {', '.join(sorted(set(gaps)))}"
    ax.text(0.5, -0.1, caption, transform=ax.transAxes, ha="center", fontsize=8.5,
            color=THEME.ink_muted)
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.12), frameon=True)
    if owns_figure:
        # Room for the long spoke labels at 9 and 3 o'clock.
        fig.subplots_adjust(top=0.84, bottom=0.1, left=0.17, right=0.83)
    return fig


def _radar_values(
    data: pd.Series | Mapping[str, float], metrics: list[str], who: str
) -> tuple[np.ndarray, list[str]]:
    lookup = data if isinstance(data, pd.Series) else pd.Series(dict(data), dtype=object)
    values: list[float] = []
    gaps: list[str] = []
    missing: list[str] = []
    for metric in metrics:
        key = f"{metric}{PERCENTILE_SUFFIX}"
        if key in lookup.index:
            raw = lookup[key]
        elif metric in lookup.index:
            raw = lookup[metric]
        else:
            missing.append(metric)
            continue
        value = pd.to_numeric(raw, errors="coerce")
        if pd.isna(value):
            gaps.append(format_metric(metric))
            value = 0.0
        values.append(float(np.clip(value, 0.0, 100.0)))
    if missing:
        raise ValueError(f"{who}: no value for metrics {missing}")
    return np.array(values), gaps


# --------------------------------------------------------------------------- #
# 3. Cluster DNA heatmap
# --------------------------------------------------------------------------- #
@themed
def plot_cluster_dna_heatmap(
    centroids: np.ndarray,
    feature_names: Sequence[str],
    role_names: Mapping[int, str],
    ax: Axes | None = None,
    vlim: float = 2.0,
) -> Figure:
    """Diverging heatmap of each cluster's centroid in scaled feature space.

    Colour saturates at ``±vlim``; the printed numbers are always the true
    values, so an off-scale +3.1 still reads correctly.

    Raises:
        ValueError: centroid matrix does not match ``feature_names``.
    """
    matrix = np.asarray(centroids, dtype=float)
    names = list(feature_names)
    if matrix.ndim != 2 or matrix.shape[1] != len(names) or matrix.shape[0] == 0:
        raise ValueError(f"centroids shape {matrix.shape} does not match {len(names)} features")
    if vlim <= 0:
        raise ValueError("vlim must be positive")

    n_clusters, n_features = matrix.shape
    fig, ax = _resolve_axes(ax, (11, 1.8 + 0.66 * n_clusters))
    image = ax.imshow(
        matrix, cmap=THEME.heatmap_cmap, norm=TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim),
        aspect="auto", interpolation="nearest",
    )
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(range(n_features))
    ax.set_xticklabels([format_metric(f) for f in names], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels(
        [_wrap(f"Cluster {i}: {role_names.get(i, 'Unlabelled')}", 36) for i in range(n_clusters)],
        fontsize=8.5,
    )
    # Cell gaps in the surface colour separate adjacent fills.
    ax.set_xticks(np.arange(-0.5, n_features), minor=True)
    ax.set_yticks(np.arange(-0.5, n_clusters), minor=True)
    ax.grid(which="minor", color=THEME.background, linewidth=2, alpha=1.0)
    ax.tick_params(which="both", length=0)

    for i in range(n_clusters):
        for j in range(n_features):
            value = matrix[i, j]
            strong = abs(value) >= 0.5 * vlim
            ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=8.5,
                    color=THEME.ink_strong if strong else THEME.background,
                    fontweight="bold" if strong else "normal")

    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Centroid deviation (scaled units)", color=THEME.ink, fontsize=9)
    cbar.ax.tick_params(colors=THEME.ink, labelsize=8)
    cbar.outline.set_edgecolor(THEME.grid)
    ax.set_title("Cluster DNA — Centroid Signatures", loc="left", pad=12)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 4. Elbow + silhouette diagnostics
# --------------------------------------------------------------------------- #
@themed
def plot_clustering_diagnostics(
    diagnostics: Mapping[str, Sequence[float]],
    selected_k: int | None = None,
) -> Figure:
    """Side-by-side elbow (inertia) and silhouette curves over k.

    Two panels rather than one dual-axis chart: the measures have unrelated
    scales. The best silhouette k is annotated; ``selected_k`` gets a dashed
    marker line on both panels.

    Raises:
        ValueError: missing keys, mismatched lengths or empty diagnostics.
    """
    required = ("k", "inertias", "silhouette_scores")
    missing = [key for key in required if key not in diagnostics]
    if missing:
        raise ValueError(f"diagnostics missing keys: {missing}")
    k = np.asarray(diagnostics["k"], dtype=float)
    inertia = np.asarray(diagnostics["inertias"], dtype=float)
    silhouette = np.asarray(diagnostics["silhouette_scores"], dtype=float)
    if not (len(k) == len(inertia) == len(silhouette)) or len(k) == 0:
        raise ValueError("diagnostics lists must be non-empty and equal length")

    fig = Figure(figsize=(12, 5.0), dpi=110)
    ax_elbow, ax_sil = fig.subplots(1, 2)

    panels = (
        (ax_elbow, inertia, "Elbow Curve", "Inertia (within-cluster SS)", THEME.clusters[0]),
        (ax_sil, silhouette, "Silhouette Score", "Mean silhouette (higher = better)", THEME.clusters[2]),
    )
    for ax, values, title, ylabel, color in panels:
        ax.plot(k, values, color=color, lw=2.0, marker="o", markersize=7,
                markeredgecolor=THEME.background, markeredgewidth=1.5, zorder=3)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(k)
        ax.set_xticklabels([str(int(v)) for v in k])
        if selected_k is not None and k.min() <= selected_k <= k.max():
            ax.axvline(selected_k, color=THEME.target, ls="--", lw=1.5, zorder=2)
            ax.text(selected_k, 1.0, f" k = {selected_k} ", transform=ax.get_xaxis_transform(),
                    color=THEME.ink_strong, fontsize=8.5, va="top",
                    ha="right" if selected_k == k.max() else "left")

    finite = np.isfinite(silhouette)
    if finite.any():
        best = int(np.argmax(np.where(finite, silhouette, -np.inf)))
        ax_sil.scatter(k[best], silhouette[best], s=150, facecolors="none",
                       edgecolors=THEME.ink_strong, linewidths=1.5, zorder=4)
        ax_sil.annotate(f"best k = {int(k[best])} ({silhouette[best]:.3f})",
                        (k[best], silhouette[best]), xytext=(10, -14), textcoords="offset points",
                        fontsize=8.5, color=THEME.ink_strong)

    fig.suptitle("Choosing k: Cluster Quality Diagnostics", x=0.01, ha="left",
                 color=THEME.ink_strong, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
