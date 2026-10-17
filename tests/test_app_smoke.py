"""Smoke tests: the pipeline helpers behind app.py, and app.py itself run headless."""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import pipeline
from src.config import COMPOSITE_SCORES, REQUIRED_RAW_COLUMNS
from src.data.loader import DataLoader, DataValidationError
from src.exceptions import InsufficientDataError

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")
NO_MATCH = "No players match the specified recruitment constraints. Try broadening age or budget filters."


# --------------------------------------------------------------------------- #
# Pipeline helpers (no Streamlit)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def fw_model(processed_players: pd.DataFrame) -> pipeline.CohortModel:
    cohort = pipeline.filter_cohort(processed_players, "FW", None)
    return pipeline.build_cohort_model(cohort, 5, "robust")


@pytest.fixture(scope="module")
def target(fw_model: pipeline.CohortModel) -> tuple[str, str]:
    return next(iter(pipeline.player_options(fw_model.cohort).values()))


def test_filter_cohort(processed_players: pd.DataFrame) -> None:
    cohort = pipeline.filter_cohort(processed_players, "MF", ["Serie A", "Eredivisie"])
    assert set(cohort["Position_Group"]) == {"MF"}
    assert set(cohort["League"]) == {"Serie A", "Eredivisie"}
    assert cohort.index.equals(pd.RangeIndex(len(cohort)))


def test_build_cohort_model(fw_model: pipeline.CohortModel) -> None:
    cohort = fw_model.cohort
    assert {"Cluster", "Assigned_Role", "PC1", "PC2"} <= set(cohort.columns)
    assert fw_model.X.shape == (len(cohort), 9)
    assert set(cohort["Cluster"]) == set(range(5))
    assert set(cohort["Assigned_Role"]) == set(fw_model.role_names.values())
    assert fw_model.centroids.shape == (5, 9)


def test_build_cohort_model_rejects_tiny_cohort(processed_players: pd.DataFrame) -> None:
    tiny = processed_players.head(4)
    with pytest.raises(InsufficientDataError):
        pipeline.build_cohort_model(tiny, 5, "standard")


def test_player_options_disambiguate_duplicates(fw_model: pipeline.CohortModel) -> None:
    cohort = fw_model.cohort.copy()
    other_club = int(np.flatnonzero(cohort["Squad"] != cohort.loc[0, "Squad"])[0])
    cohort.loc[other_club, "Player"] = cohort.loc[0, "Player"]
    options = pipeline.player_options(cohort)
    namesakes = {label: club for label, (name, club) in options.items() if name == cohort.loc[0, "Player"]}
    assert len(namesakes) == 2
    assert all(label == f"{cohort.loc[0, 'Player']} ({club})" for label, club in namesakes.items())
    assert len(options) == len(cohort)


def test_player_options_unique_for_namesakes_at_same_club(fw_model: pipeline.CohortModel) -> None:
    cohort = fw_model.cohort.copy()
    same_club = cohort[cohort["Squad"] == cohort.loc[0, "Squad"]].index[:3]
    cohort.loc[same_club, "Player"] = "Twin Name"
    cohort.loc[same_club, "Age"] = 24  # identical age forces the counter
    options = pipeline.player_options(cohort)
    assert len(options) == len(cohort)
    twins = [label for label in options if label.startswith("Twin Name")]
    assert len(twins) == 3 and len(set(twins)) == 3


def test_player_options_ordered_by_value(fw_model: pipeline.CohortModel) -> None:
    first_name, first_squad = next(iter(pipeline.player_options(fw_model.cohort).values()))
    row = pipeline.get_player_row(fw_model.cohort, first_name, first_squad)
    assert row["Market_Value_EUR_M"] == fw_model.cohort["Market_Value_EUR_M"].max()


def test_shortlist_and_formatting(fw_model: pipeline.CohortModel, target: tuple[str, str]) -> None:
    shortlist = pipeline.run_shortlist(fw_model, *target, "cosine", {}, pipeline.RecruitmentFilters())
    assert len(shortlist) == 10
    table = pipeline.format_shortlist(shortlist)
    assert list(table.columns) == list(pipeline.SHORTLIST_DISPLAY.values())
    assert target[0] not in set(table["Player"])


def test_shortlist_respects_constraints(fw_model: pipeline.CohortModel, target: tuple[str, str]) -> None:
    filters = pipeline.RecruitmentFilters(max_age=23, max_market_value=15.0, contract_years_max=2)
    shortlist = pipeline.run_shortlist(fw_model, *target, "euclidean", {"Scoring": 2.0}, filters)
    assert (shortlist["Age"] <= 23).all()
    assert (shortlist["Market_Value_EUR_M"] <= 15.0).all()
    assert (shortlist["Contract_Expiry_Year"] <= 2028).all()


def test_impossible_constraints_give_empty_shortlist(fw_model: pipeline.CohortModel, target: tuple[str, str]) -> None:
    filters = pipeline.RecruitmentFilters(max_age=18, max_market_value=0.1)
    assert pipeline.run_shortlist(fw_model, *target, "cosine", {}, filters).empty


def test_filters_describe() -> None:
    assert pipeline.RecruitmentFilters().describe() == "No constraints"
    text = pipeline.RecruitmentFilters(21, 5.0, 2).describe()
    assert "Age ≤ 21" in text and "€5.0M" in text and "2 yr" in text


def test_cluster_summary(fw_model: pipeline.CohortModel) -> None:
    summary = pipeline.cluster_summary(fw_model)
    assert len(summary) == 5
    assert summary["Players"].sum() == len(fw_model.cohort)
    assert summary["Share (%)"].sum() == pytest.approx(100.0, abs=0.5)
    assert summary["Top Traits"].str.contains(r" \+ ").all()


def test_ground_truth_report(fw_model: pipeline.CohortModel) -> None:
    report = pipeline.ground_truth_report(fw_model)
    assert report is not None and 0.0 <= report.attrs["nmi"] <= 1.0
    no_truth = pipeline.CohortModel(
        fw_model.cohort.drop(columns=[pipeline.GROUND_TRUTH_COLUMN]), fw_model.X, fw_model.pca,
        fw_model.centroids, fw_model.role_names, fw_model.scaler_type,
    )
    assert pipeline.ground_truth_report(no_truth) is None


def test_diagnostics_and_best_k(fw_model: pipeline.CohortModel) -> None:
    diag = pipeline.compute_diagnostics(fw_model.X)
    assert diag["k"] == [float(k) for k in range(2, 11)]
    best = pipeline.best_k(diag)
    assert best in range(2, 11)
    assert pipeline.best_k({"k": [2.0], "silhouette_scores": [float("nan")]}) is None
    assert pipeline.best_k({}) is None


def test_export(fw_model: pipeline.CohortModel, target: tuple[str, str]) -> None:
    scores = pipeline.match_scores(fw_model, *target, "cosine", {})
    export = pipeline.build_export(fw_model, scores)
    required = {"Cluster", "Assigned_Role", "PC1", "PC2", "Match_Score_Pct", *COMPOSITE_SCORES.values()}
    assert required <= set(export.columns)
    assert len(export) == len(fw_model.cohort)
    assert export["Match_Score_Pct"].is_monotonic_decreasing
    assert export.iloc[0]["Player"] == target[0]  # the target matches itself at 100%
    round_trip = pd.read_csv(io.StringIO(export.to_csv(index=False)))
    assert round_trip.shape == export.shape


def test_export_without_scores(fw_model: pipeline.CohortModel) -> None:
    assert "Match_Score_Pct" not in pipeline.build_export(fw_model).columns


def test_export_filename() -> None:
    name = pipeline.export_filename("FW", datetime(2026, 10, 18, 9, 30, 5))
    assert name == "scoutintel_FW_shortlist_20261018_093005.csv"


@pytest.mark.parametrize(("score", "tier"), [(95, "Elite"), (70, "Strong"), (50, "Average"),
                                             (25, "Below Average"), (5, "Weak")])
def test_score_badge(score: float, tier: str) -> None:
    assert pipeline.score_badge(score)[0] == tier


def test_overall_percentile(fw_model: pipeline.CohortModel) -> None:
    row = fw_model.cohort.iloc[0]
    assert 0.0 <= pipeline.overall_percentile(row) <= 100.0
    assert pipeline.overall_percentile(pd.Series(dtype=float)) == 0.0


def test_bad_upload_reports_missing_columns() -> None:
    """The error app.py shows for a CSV missing required columns."""
    bad = pd.DataFrame({"Player": ["A"], "Squad": ["B"]}).to_csv(index=False).encode()
    with pytest.raises(DataValidationError, match="Missing required columns") as info:
        DataLoader().load(io.BytesIO(bad))
    assert "Minutes_Played" in str(info.value)


def test_template_has_required_schema() -> None:
    template = DataLoader.read().head(5)
    assert set(REQUIRED_RAW_COLUMNS) <= set(template.columns)


# --------------------------------------------------------------------------- #
# app.py end-to-end (headless Streamlit)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    return at


def count(at: AppTest, element_type: str) -> int:
    total = 0

    def walk(node: object) -> None:
        nonlocal total
        if getattr(node, "type", None) == element_type:
            total += 1
        for child in getattr(node, "children", {}).values():
            walk(child)

    walk(at._tree)
    return total


def test_app_renders_cleanly(app: AppTest) -> None:
    assert not app.exception
    assert not app.error
    assert len(app.tabs) == 4
    assert count(app, "image") == 5  # radar, overlay radar, PCA map, heatmap, diagnostics
    assert count(app, "download_button") == 1
    assert app.selectbox(key="target_label").value is not None


def test_app_default_controls(app: AppTest) -> None:
    assert app.radio(key="position").value == "FW"
    assert app.slider(key="k").value == 5
    assert app.slider(key="max_age").value == 40
    assert app.slider(key="max_value").value == app.slider(key="max_value").max


def test_target_selection_persists_across_reruns(app: AppTest) -> None:
    options = app.selectbox(key="target_label").options
    app.selectbox(key="target_label").set_value(options[3]).run()
    app.slider(key="k").set_value(7).run()  # change model, same cohort
    assert app.selectbox(key="target_label").value == options[3]
    assert not app.exception


def test_impossible_constraints_show_warning(app: AppTest) -> None:
    app.slider(key="max_age").set_value(18)
    app.slider(key="max_value").set_value(0.5)
    app.run()
    assert NO_MATCH in [w.value for w in app.warning]
    assert not app.exception
    app.slider(key="max_age").set_value(40)
    app.slider(key="max_value").set_value(app.slider(key="max_value").max)
    app.run()


def test_zero_weights_show_error_not_crash(app: AppTest) -> None:
    for category in ("Scoring", "Creation & Passing", "Dribbling", "Defending"):
        app.slider(key=f"w_{category}").set_value(0.0)
    app.run()
    assert any("weight" in e.value for e in app.error)
    assert not app.exception
    for category in ("Scoring", "Creation & Passing", "Dribbling", "Defending"):
        app.slider(key=f"w_{category}").set_value(1.0)
    app.run()


@pytest.mark.parametrize("position", ["DF", "MF", "FW"])
def test_every_cohort_and_metric(app: AppTest, position: str) -> None:
    app.radio(key="position").set_value(position)
    app.radio(key="metric").set_value(app.radio(key="metric").options[1])
    app.selectbox(key="scaler").set_value(app.selectbox(key="scaler").options[1])
    app.slider(key="k").set_value(10)
    app.run()
    assert not app.exception and not app.error
    assert count(app, "image") == 5


def test_empty_league_selection_warns(app: AppTest) -> None:
    app.multiselect(key="leagues").set_value([]).run()
    assert any("at least one league" in w.value for w in app.warning)
    assert not app.exception
    app.multiselect(key="leagues").set_value(app.multiselect(key="leagues").options).run()


def test_tiny_cohort_warns_instead_of_crashing(app: AppTest) -> None:
    app.slider(key="k").set_value(10)
    app.multiselect(key="leagues").set_value(["Eredivisie"])
    app.run()
    assert not app.exception
    app.multiselect(key="leagues").set_value(app.multiselect(key="leagues").options).run()


def test_upload_mode_without_file_shows_template() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.radio(key="source").set_value("Upload Custom CSV").run()
    assert not at.exception
    assert any("Upload an FBref-style" in i.value for i in at.info)
    assert count(at, "download_button") == 1  # the CSV template


# --------------------------------------------------------------------------- #
# Real Data source
# --------------------------------------------------------------------------- #
REAL_SOURCE = "Real Data (2023-24 Top 5 European Leagues)"


def _benchmark_frame() -> pd.DataFrame:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ingest_fbref_for_app_tests", Path(APP_PATH).parent / "scripts" / "ingest_fbref.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.build_benchmark_dataset("2023-2024")


def _real_source_app(monkeypatch: pytest.MonkeyPatch, data_dir: Path) -> AppTest:
    import src.data.loader as loader_module

    monkeypatch.setattr(loader_module, "PROCESSED_DATA_DIR", data_dir)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.radio(key="source").set_value(REAL_SOURCE).run()
    return at


def test_real_source_without_data_shows_instructions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _real_source_app(monkeypatch, tmp_path)
    assert not at.exception
    assert any("Real data has not been ingested yet" in i.value for i in at.info)
    assert any("ingest_fbref.py" in i.value for i in at.info)


def test_real_source_benchmark_is_flagged_as_synthetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _benchmark_frame().to_parquet(tmp_path / "benchmark_players_2023_2024.parquet", index=False)
    at = _real_source_app(monkeypatch, tmp_path)
    assert not at.exception and not at.error
    assert any("Synthetic benchmark, not real players" in w.value for w in at.warning)
    assert count(at, "image") == 5
    # Contracts count from the 2023-24 season end, not the synthetic DB's 2026.
    assert "2024" in (at.select_slider(key="contract_years").help or "")


def test_real_source_with_fbref_file_shows_estimate_notice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data.real_data_enricher import CONTRACT_SOURCE_COLUMN, VALUE_SOURCE_COLUMN

    frame = _benchmark_frame()
    frame["Data_Origin"] = "fbref"  # real-format file (test fixture only)
    frame[VALUE_SOURCE_COLUMN] = "proxy"
    frame[CONTRACT_SOURCE_COLUMN] = "proxy"
    frame.to_parquet(tmp_path / "real_players_2023_2024.parquet", index=False)
    at = _real_source_app(monkeypatch, tmp_path)
    assert not at.exception and not at.error
    assert any("model estimates" in i.value for i in at.info)
    assert not any("Synthetic benchmark" in w.value for w in at.warning)
    assert count(at, "image") == 5
