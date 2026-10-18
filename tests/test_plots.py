"""Headless rendering tests for the dark-theme visual suite."""

from __future__ import annotations

import gc
import weakref
from collections.abc import Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.image import AxesImage  # noqa: E402

from src.config import CORE_FEATURES  # noqa: E402
from src.models.clustering import TacticalClusterer  # noqa: E402
from src.models.pca import TacticalPCA  # noqa: E402
from src.models.similarity import ScoutingSimilarityEngine  # noqa: E402
from src.visualization import plots  # noqa: E402

FEATURES = list(CORE_FEATURES)


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def forward_cohort(processed_players: pd.DataFrame) -> dict[str, object]:
    """Forwards run through the real scaler -> PCA -> K-Means -> search chain."""
    from src.features.preprocessor import ScoutingPreprocessor

    fw = processed_players[processed_players["Position_Group"] == "FW"].reset_index(drop=True)
    X, _ = ScoutingPreprocessor().scale_features(fw, "robust")
    pca = TacticalPCA().fit(X)
    pca_df = pd.concat([fw[["Player", "Squad"]], pca.to_frame(pca.transform(X))], axis=1)
    clusterer = TacticalClusterer(4)
    labels = clusterer.fit_predict(X)
    roles = clusterer.role_labels(FEATURES)
    target = str(fw.sort_values("Market_Value_EUR_M").iloc[-1]["Player"])
    matches = ScoutingSimilarityEngine().search(target, fw, X, top_n=10)
    return {
        "fw": fw, "X": X, "pca": pca, "pca_df": pca_df, "labels": labels, "roles": roles,
        "centroids": clusterer.centroids_, "target": target, "matches": matches,
    }


def render(fig: Figure) -> None:
    """Force a full draw so layout and text errors surface."""
    fig.canvas.draw()


def texts(fig: Figure) -> list[str]:
    return [t.get_text() for ax in fig.axes for t in ax.texts] + [t.get_text() for t in fig.texts]


# --------------------------------------------------------------------------- #
# PCA tactical map
# --------------------------------------------------------------------------- #
def test_pca_map_without_target_or_candidates(forward_cohort: dict[str, object]) -> None:
    c = forward_cohort
    fig = plots.plot_pca_tactical_map(c["pca_df"], c["pca"], c["labels"], c["roles"])
    assert isinstance(fig, Figure)
    render(fig)
    assert not any(t.startswith(("Target:", "#")) for t in texts(fig))


def test_pca_map_with_target_and_candidates(forward_cohort: dict[str, object]) -> None:
    c = forward_cohort
    fig = plots.plot_pca_tactical_map(
        c["pca_df"], c["pca"], c["labels"], c["roles"],
        target_player=c["target"], candidate_matches=c["matches"],
    )
    assert isinstance(fig, Figure)
    render(fig)
    labels = texts(fig)
    assert f"Target: {c['target']}" in labels
    callouts = [t for t in labels if t.startswith("#")]
    assert len(callouts) == 5  # top 5 of the 10 matches
    assert callouts[0].startswith("#") and "%" in callouts[0]


def test_pca_map_subtitle_reports_explained_variance(forward_cohort: dict[str, object]) -> None:
    c = forward_cohort
    fig = plots.plot_pca_tactical_map(c["pca_df"], c["pca"], c["labels"], c["roles"])
    pca: TacticalPCA = c["pca"]  # type: ignore[assignment]
    expected = f"= {pca.cumulative_variance:.1%}"
    assert any(t.startswith("Explained Variance: PC1") and t.endswith(expected) for t in texts(fig))


def test_pca_map_centroids_toggle(forward_cohort: dict[str, object]) -> None:
    c = forward_cohort
    with_c = plots.plot_pca_tactical_map(c["pca_df"], c["pca"], c["labels"], c["roles"])
    without = plots.plot_pca_tactical_map(c["pca_df"], c["pca"], c["labels"], c["roles"],
                                          show_centroids=False, show_ellipses=False)
    assert len(with_c.axes[0].collections) > len(without.axes[0].collections)
    assert not without.axes[0].patches


def test_pca_map_compact_centroid_labels_for_large_k(forward_cohort: dict[str, object]) -> None:
    c = forward_cohort
    clusterer = TacticalClusterer(9)
    labels = clusterer.fit_predict(c["X"])  # type: ignore[arg-type]
    roles = clusterer.role_labels(FEATURES)
    fig = plots.plot_pca_tactical_map(c["pca_df"], c["pca"], labels, roles)
    assert {str(i) for i in range(9)} <= set(texts(fig))


def test_pca_map_tolerates_unknown_target_and_candidates(forward_cohort: dict[str, object]) -> None:
    c = forward_cohort
    ghosts = pd.DataFrame({"Rank": [1], "Player": ["Nobody"], "Match_Score_Pct": [99.0]})
    fig = plots.plot_pca_tactical_map(c["pca_df"], c["pca"], c["labels"], c["roles"],
                                      target_player="Nobody", candidate_matches=ghosts)
    render(fig)
    assert not any(t.startswith(("Target:", "#")) for t in texts(fig))


def test_pca_map_empty_candidates(forward_cohort: dict[str, object]) -> None:
    c = forward_cohort
    empty = c["matches"].iloc[0:0]  # type: ignore[union-attr]
    fig = plots.plot_pca_tactical_map(c["pca_df"], c["pca"], c["labels"], c["roles"],
                                      target_player=c["target"], candidate_matches=empty)
    render(fig)


def test_pca_map_draws_on_supplied_axes(forward_cohort: dict[str, object]) -> None:
    c = forward_cohort
    fig, ax = plt.subplots()
    assert plots.plot_pca_tactical_map(c["pca_df"], c["pca"], c["labels"], c["roles"], ax=ax) is fig


def test_pca_map_tiny_clusters() -> None:
    """Clusters of one or two players get no ellipse but still render."""
    pca_df = pd.DataFrame({"Player": list("abcd"), "PC1": [0, 1, 5, 6.0], "PC2": [0, 1, 5, 4.0]})
    pca = TacticalPCA().fit(np.random.default_rng(0).normal(size=(10, 9)))
    fig = plots.plot_pca_tactical_map(pca_df, pca, np.array([0, 0, 1, 2]), {0: "A", 1: "B", 2: "C"})
    render(fig)
    assert not fig.axes[0].patches


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda df: df.drop(columns=["PC2"]), "missing"),
        (lambda df: df.iloc[:-1], "cluster labels"),
    ],
)
def test_pca_map_invalid_inputs(forward_cohort: dict[str, object], mutate, match: str) -> None:
    c = forward_cohort
    with pytest.raises(ValueError, match=match):
        plots.plot_pca_tactical_map(mutate(c["pca_df"]), c["pca"], c["labels"], c["roles"])


def test_pca_map_bad_confidence(forward_cohort: dict[str, object]) -> None:
    c = forward_cohort
    with pytest.raises(ValueError, match="ellipse_confidence"):
        plots.plot_pca_tactical_map(c["pca_df"], c["pca"], c["labels"], c["roles"], ellipse_confidence=1.5)


# --------------------------------------------------------------------------- #
# Radar
# --------------------------------------------------------------------------- #
def radar_lines(fig: Figure) -> list:
    return [line for line in fig.axes[0].lines if not line.get_label().startswith("_")]


def test_radar_single_player(processed_players: pd.DataFrame) -> None:
    row = processed_players.iloc[0]
    fig = plots.plot_radar_comparison(row, str(row["Player"]))
    assert isinstance(fig, Figure)
    render(fig)
    assert [line.get_label() for line in radar_lines(fig)] == [row["Player"]]


def test_radar_head_to_head(processed_players: pd.DataFrame) -> None:
    a, b = processed_players.iloc[0], processed_players.iloc[1]
    fig = plots.plot_radar_comparison(a, "A", b, "B")
    render(fig)
    assert [line.get_label() for line in radar_lines(fig)] == ["A", "B"]
    assert fig.axes[0].get_title() == "A vs B"


def test_radar_uses_percentile_columns_and_closes_polygon(processed_players: pd.DataFrame) -> None:
    row = processed_players.iloc[0]
    fig = plots.plot_radar_comparison(row, "A")
    ydata = np.asarray(radar_lines(fig)[0].get_ydata())
    assert len(ydata) == len(FEATURES) + 1
    assert ydata[0] == ydata[-1]
    np.testing.assert_allclose(ydata[:-1], [row[f"{f}_pct"] for f in FEATURES])


def test_radar_clips_out_of_range_values() -> None:
    wild = {f: v for f, v in zip(FEATURES, [-50, 150, 1e6, 50, -1, 101, 0, 100, 42, 7, 99, 63])}
    fig = plots.plot_radar_comparison(wild, "Wild")
    render(fig)
    ydata = np.asarray(radar_lines(fig)[0].get_ydata())
    assert ydata.min() >= 0.0 and ydata.max() <= 100.0
    assert ydata[0] == 0.0 and ydata[1] == 100.0


def test_radar_nan_values_plot_at_zero_and_are_flagged() -> None:
    data = {f: 50.0 for f in FEATURES}
    data["Goals_p90"] = float("nan")
    fig = plots.plot_radar_comparison(data, "Gappy")
    assert np.asarray(radar_lines(fig)[0].get_ydata())[0] == 0.0
    assert any("no data for: Goals p90" in t for t in texts(fig))


def test_radar_custom_metrics() -> None:
    fig = plots.plot_radar_comparison({"a": 10, "b": 20, "c": 30}, "X", metrics=["a", "b", "c"])
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert labels == ["a", "b", "c"]


def test_radar_metric_labels_are_formatted() -> None:
    fig = plots.plot_radar_comparison({f: 50 for f in FEATURES}, "X")
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert "Goals p90" in labels and "pAdj Tackles Won p90" in labels


def test_radar_guide_circles() -> None:
    fig = plots.plot_radar_comparison({f: 50 for f in FEATURES}, "X")
    assert list(fig.axes[0].get_yticks()) == [25, 50, 75, 100]


def test_radar_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="no value"):
        plots.plot_radar_comparison({"Goals_p90": 50}, "X")
    with pytest.raises(ValueError, match="at least 3"):
        plots.plot_radar_comparison({"a": 1, "b": 2}, "X", metrics=["a", "b"])
    _, flat_ax = plt.subplots()
    with pytest.raises(ValueError, match="polar"):
        plots.plot_radar_comparison({f: 50 for f in FEATURES}, "X", ax=flat_ax)


def test_radar_on_supplied_polar_axes() -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection="polar")
    assert plots.plot_radar_comparison({f: 50 for f in FEATURES}, "X", ax=ax) is fig


# --------------------------------------------------------------------------- #
# Cluster DNA heatmap
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [2, 5, 10])
def test_heatmap_dimensions(k: int) -> None:
    centroids = np.random.default_rng(k).normal(size=(k, len(FEATURES)))
    roles = {i: f"Role {i}" for i in range(k)}
    fig = plots.plot_cluster_dna_heatmap(centroids, FEATURES, roles)
    assert isinstance(fig, Figure)
    render(fig)
    image = next(a for a in fig.axes[0].get_children() if isinstance(a, AxesImage))
    assert image.get_array().shape == (k, len(FEATURES))
    assert len(fig.axes[0].texts) == k * len(FEATURES)
    assert len(fig.axes[0].get_yticklabels()) == k


def test_heatmap_annotations_show_true_values_beyond_colour_range() -> None:
    centroids = np.zeros((2, len(FEATURES)))
    centroids[0, 0] = 3.14
    centroids[1, 1] = -0.04
    fig = plots.plot_cluster_dna_heatmap(centroids, FEATURES, {0: "A", 1: "B"})
    cell_texts = [t.get_text() for t in fig.axes[0].texts]
    assert "+3.1" in cell_texts and "-0.0" in cell_texts
    image = next(a for a in fig.axes[0].get_children() if isinstance(a, AxesImage))
    assert image.norm.vmin == -2.0 and image.norm.vmax == 2.0


def test_heatmap_text_contrast() -> None:
    centroids = np.zeros((1, len(FEATURES)))
    centroids[0, 0] = 1.8
    fig = plots.plot_cluster_dna_heatmap(centroids, FEATURES, {0: "A"})
    strong, weak = fig.axes[0].texts[0], fig.axes[0].texts[1]
    assert strong.get_color() == plots.THEME.ink_strong
    assert weak.get_color() == plots.THEME.background


def test_heatmap_role_labels(forward_cohort: dict[str, object]) -> None:
    c = forward_cohort
    fig = plots.plot_cluster_dna_heatmap(c["centroids"], FEATURES, c["roles"])
    render(fig)
    labels = [t.get_text().replace("\n", " ") for t in fig.axes[0].get_yticklabels()]
    roles: dict[int, str] = c["roles"]  # type: ignore[assignment]
    assert labels[0].startswith("Cluster 0: ")
    assert roles[0][:20] in labels[0]


def test_heatmap_invalid_shape() -> None:
    with pytest.raises(ValueError):
        plots.plot_cluster_dna_heatmap(np.zeros((3, 4)), FEATURES, {})
    with pytest.raises(ValueError):
        plots.plot_cluster_dna_heatmap(np.zeros((0, len(FEATURES))), FEATURES, {})


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def diagnostics(forward_cohort: dict[str, object]) -> dict[str, list[float]]:
    return TacticalClusterer.compute_diagnostics(forward_cohort["X"], range(2, 11))  # type: ignore[arg-type]


def dashed_lines(ax) -> list:
    return [line for line in ax.lines if line.get_linestyle() == "--"]


def test_diagnostics_two_panels_over_k_2_to_10(diagnostics: dict[str, list[float]]) -> None:
    fig = plots.plot_clustering_diagnostics(diagnostics)
    assert isinstance(fig, Figure)
    render(fig)
    assert len(fig.axes) == 2
    for ax in fig.axes:
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), np.arange(2, 11))
        assert not dashed_lines(ax)
    assert fig.axes[0].get_title(loc="left") == "Elbow Curve"
    assert fig.axes[1].get_title(loc="left") == "Silhouette Score"


@pytest.mark.parametrize("k", [2, 5, 10])
def test_diagnostics_mark_selected_k(diagnostics: dict[str, list[float]], k: int) -> None:
    fig = plots.plot_clustering_diagnostics(diagnostics, selected_k=k)
    render(fig)
    for ax in fig.axes:
        marks = dashed_lines(ax)
        assert len(marks) == 1
        assert list(marks[0].get_xdata()) == [k, k]


def test_diagnostics_ignore_out_of_range_k(diagnostics: dict[str, list[float]]) -> None:
    fig = plots.plot_clustering_diagnostics(diagnostics, selected_k=15)
    assert all(not dashed_lines(ax) for ax in fig.axes)


def test_diagnostics_annotate_best_silhouette(diagnostics: dict[str, list[float]]) -> None:
    fig = plots.plot_clustering_diagnostics(diagnostics)
    best_k = int(diagnostics["k"][int(np.argmax(diagnostics["silhouette_scores"]))])
    assert any(t.startswith(f"best k = {best_k} ") for t in texts(fig))


def test_diagnostics_handle_nan_silhouette() -> None:
    diag = {"k": [2.0, 3.0, 4.0], "inertias": [10.0, 6.0, 5.0], "silhouette_scores": [0.3, float("nan"), 0.4]}
    fig = plots.plot_clustering_diagnostics(diag, selected_k=3)
    render(fig)
    assert any(t.startswith("best k = 4 ") for t in texts(fig))
    all_nan = {"k": [2.0], "inertias": [1.0], "silhouette_scores": [float("nan")]}
    render(plots.plot_clustering_diagnostics(all_nan))


@pytest.mark.parametrize(
    "diag",
    [
        {"k": [2.0], "inertias": [1.0]},
        {"k": [2.0, 3.0], "inertias": [1.0], "silhouette_scores": [0.1, 0.2]},
        {"k": [], "inertias": [], "silhouette_scores": []},
    ],
)
def test_diagnostics_invalid_inputs(diag: dict[str, list[float]]) -> None:
    with pytest.raises(ValueError):
        plots.plot_clustering_diagnostics(diag)


# --------------------------------------------------------------------------- #
# Theme & memory
# --------------------------------------------------------------------------- #
def test_figures_use_dark_canvas(forward_cohort: dict[str, object], diagnostics) -> None:
    c = forward_cohort
    figs = [
        plots.plot_pca_tactical_map(c["pca_df"], c["pca"], c["labels"], c["roles"]),
        plots.plot_radar_comparison({f: 50 for f in FEATURES}, "X"),
        plots.plot_cluster_dna_heatmap(c["centroids"], FEATURES, c["roles"]),
        plots.plot_clustering_diagnostics(diagnostics),
    ]
    for fig in figs:
        assert matplotlib.colors.to_hex(fig.get_facecolor()) == "#0e1117"


def test_theme_does_not_leak_into_global_rcparams() -> None:
    before = matplotlib.rcParams["axes.facecolor"]
    plots.plot_radar_comparison({f: 50 for f in FEATURES}, "X")
    assert matplotlib.rcParams["axes.facecolor"] == before


def test_cluster_colours_follow_id_not_rank() -> None:
    assert plots.cluster_color(3) == plots.THEME.clusters[3]
    assert plots.cluster_color(13) == plots.cluster_color(3)
    assert len(set(plots.THEME.clusters)) == 10
    assert len(set(plots.THEME.markers)) == 10


def test_no_figures_registered_with_pyplot(forward_cohort: dict[str, object], diagnostics) -> None:
    """Default figures bypass pyplot's registry, so Streamlit reruns cannot leak them."""
    c = forward_cohort
    plt.close("all")
    plots.plot_pca_tactical_map(c["pca_df"], c["pca"], c["labels"], c["roles"],
                                target_player=c["target"], candidate_matches=c["matches"])
    plots.plot_radar_comparison({f: 50 for f in FEATURES}, "X")
    plots.plot_cluster_dna_heatmap(c["centroids"], FEATURES, c["roles"])
    plots.plot_clustering_diagnostics(diagnostics)
    assert plt.get_fignums() == []


def test_figures_are_garbage_collected(diagnostics) -> None:
    fig = plots.plot_clustering_diagnostics(diagnostics)
    render(fig)
    ref = weakref.ref(fig)
    del fig
    gc.collect()
    assert ref() is None


def test_supplied_axes_figures_are_closed_by_fixture() -> None:
    fig, _ = plt.subplots()
    assert plt.fignum_exists(fig.number)
    plt.close(fig)
    assert not plt.fignum_exists(fig.number)
