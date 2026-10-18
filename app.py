"""ScoutIntel - Player Recruitment & Tactical Clustering dashboard.

Run with:  streamlit run app.py

This module is the UI layer only: widgets, caching and layout. All analytics
live in ``src/`` (see ``src/pipeline.py`` for the orchestration it calls).

Caching tiers
-------------
1. ``load_dataset``      - ingestion + preprocessing; keyed on data source,
                           min minutes, pAdj steepness and league scaling.
2. ``get_cohort_model``  - scaling, PCA and K-Means for one cohort.
3. ``get_diagnostics``   - elbow/silhouette sweep (independent of chosen k).
4. ``render_*``          - charts cached as PNG bytes keyed on their inputs.
Similarity search (~10 ms) runs live on every interaction.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import asdict

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

from src import pipeline
from src.config import (
    CORE_FEATURES,
    DEFAULT_MIN_MINUTES,
    DEFAULT_PADJ_STEEPNESS,
    DEFAULT_REAL_SEASON,
    ORIGIN_BENCHMARK,
    POSITION_GROUPS,
    RANDOM_STATE,
)
from src.data.loader import (
    DataLoader,
    DataValidationError,
    LoaderConfig,
    RealDataNotFoundError,
    find_real_dataset,
)
from src.data.real_data_enricher import CONTRACT_SOURCE_COLUMN, SOURCE_TRANSFERMARKT, VALUE_SOURCE_COLUMN
from src.data.synthetic_generator import GeneratorConfig
from src.exceptions import InsufficientDataError
from src.features.preprocessor import ScalerType, ScoutingPreprocessor
from src.models.similarity import Metric
from src.visualization import plots

SYNTHETIC = "Synthetic Enterprise DB"
REAL = "Real Data (2023-24 Top 5 European Leagues)"
UPLOAD = "Upload Custom CSV"
SCALERS: dict[str, ScalerType] = {"RobustScaler (IQR)": "robust", "StandardScaler (Z-score)": "standard"}
METRICS: dict[str, Metric] = {
    "Cosine Similarity (Style / Profile Clone)": "cosine",
    "Euclidean Distance (Volume Output Match)": "euclidean",
}
WEIGHT_SLIDERS: dict[str, str] = {
    "Scoring": "Scoring",
    "Creation & Passing": "Creation & Passing",
    "Dribbling & Ball Carrying": "Dribbling",
    "Defending": "Defending",
}
NO_MATCH_MESSAGE = (
    "No players match the specified recruitment constraints. Try broadening age or budget filters."
)
PNG_DPI = 140

st.set_page_config(
    page_title="ScoutIntel | Tactical Recruitment Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Cached computation
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading and normalising player database…", max_entries=8)
def load_dataset(
    source_kind: str,
    file_bytes: bytes | None,
    seed: int,
    min_minutes: int,
    padj_steepness: float,
    apply_league_scaling: bool,
    real_file_stamp: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Ingest, adjust and percentile-rank the full database (all positions).

    ``real_file_stamp`` (path + modification time) is only a cache key: a
    re-run of the ingestion script invalidates the cached real dataset.
    """
    loader = DataLoader(LoaderConfig(
        min_minutes=min_minutes,
        padj_steepness=padj_steepness,
        apply_league_scaling=apply_league_scaling,
    ))
    if source_kind == UPLOAD:
        df = loader.load(io.BytesIO(file_bytes or b""))
    elif source_kind == REAL:
        df = loader.load_real(DEFAULT_REAL_SEASON)
    else:
        df = loader.load(generator_config=GeneratorConfig(seed=seed, include_ground_truth=True))
    report = asdict(loader.last_report)
    report["n_dropped"] = loader.last_report.n_dropped
    if df.empty:
        return df, report
    return ScoutingPreprocessor().transform(df), report


@st.cache_data(show_spinner="Fitting tactical role model…", max_entries=32)
def get_cohort_model(
    cohort_key: str, n_clusters: int, scaler_type: ScalerType, _cohort: pd.DataFrame
) -> pipeline.CohortModel:
    """Scale + PCA + K-Means. ``_cohort`` is identified by ``cohort_key``."""
    return pipeline.build_cohort_model(_cohort, n_clusters, scaler_type)


@st.cache_data(show_spinner="Sweeping k = 2…10…", max_entries=32)
def get_diagnostics(cohort_key: str, scaler_type: ScalerType, _X: np.ndarray) -> dict[str, list[float]]:
    return pipeline.compute_diagnostics(_X)


def fig_to_png(fig: Figure) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=PNG_DPI, facecolor=fig.get_facecolor())
    return buffer.getvalue()


@st.cache_data(max_entries=64)
def render_pca_map(
    model_key: str, target: str, shortlist_key: tuple[str, ...],
    _model: pipeline.CohortModel, _shortlist: pd.DataFrame,
) -> bytes:
    return fig_to_png(plots.plot_pca_tactical_map(
        _model.pca_frame, _model.pca, _model.labels, _model.role_names,
        target_player=target, candidate_matches=_shortlist if len(_shortlist) else None,
        legend_loc="below",
    ))


@st.cache_data(max_entries=128)
def render_radar(
    cohort_key: str, name_1: str, squad_1: str, name_2: str | None, squad_2: str | None,
    _row_1: pd.Series, _row_2: pd.Series | None,
) -> bytes:
    return fig_to_png(plots.plot_radar_comparison(_row_1, name_1, _row_2, name_2))


@st.cache_data(max_entries=32)
def render_heatmap(model_key: str, _model: pipeline.CohortModel) -> bytes:
    return fig_to_png(plots.plot_cluster_dna_heatmap(_model.centroids, list(CORE_FEATURES), _model.role_names))


@st.cache_data(max_entries=64)
def render_diagnostics(cohort_key: str, scaler_type: str, k: int, _diagnostics: dict[str, list[float]]) -> bytes:
    return fig_to_png(plots.plot_clustering_diagnostics(_diagnostics, selected_k=k))


# --------------------------------------------------------------------------- #
# Small UI helpers
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    st.markdown(
        """
        <style>
          .block-container {padding-top: 3.2rem;}
          .si-header {padding: 0.2rem 0 0.8rem 0; border-bottom: 1px solid #30363d; margin-bottom: 0.8rem;}
          .si-title {font-size: 1.9rem; font-weight: 700; color: #ffffff; letter-spacing: -0.01em;}
          .si-title span {color: #e3b341;}
          .si-sub {color: #8b949e; font-size: 0.95rem; margin-top: 0.15rem;}
          div[data-testid="stMetric"] {background: #161b22; border: 1px solid #30363d;
                                       border-radius: 8px; padding: 0.7rem 0.9rem;}
          div[data-testid="stMetricValue"] {font-size: 1.35rem;}
        </style>
        <div class="si-header">
          <div class="si-title">Scout<span>Intel</span> · Tactical Recruitment Platform</div>
          <div class="si-sub">Player similarity, tactical role clustering and moneyball replacement
          search across Europe's top 5 and secondary leagues.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def keep_valid_selection(key: str, options: list[str]) -> None:
    """Reset a persisted selectbox value that no longer exists in ``options``."""
    if options and st.session_state.get(key) not in options:
        st.session_state[key] = options[0]


def show_pipeline_status(report: dict[str, object], source: str, cohort_size: int | None) -> None:
    with st.expander("System Architecture & Pipeline Status", icon=":material/account_tree:"):
        cols = st.columns(5)
        cols[0].metric("Rows ingested", f"{report['n_input']:,}")
        cols[1].metric("Invalid rows dropped", f"{report['n_invalid']:,}")
        cols[2].metric("Non-outfield / unknown pos.", f"{report['n_unknown_position']:,}")
        cols[3].metric("Below minutes threshold", f"{report['n_below_min_minutes']:,}")
        cols[4].metric("Players in model", f"{report['n_output']:,}",
                       delta=f"{cohort_size:,} in cohort" if cohort_size is not None else None,
                       delta_color="off")
        if report.get("unknown_leagues"):
            st.caption(f"Leagues without a strength coefficient (default applied): "
                       f"{', '.join(map(str, report['unknown_leagues']))}")
        st.markdown(
            f"**Source:** {source}  \n"
            "**Pipeline:** validate → minutes filter → per-90 → possession adjustment (sigmoid) → "
            "league strength scaling → within-position percentiles → composite scores → "
            "scaling → PCA → K-Means role clustering → weighted similarity search"
        )


def describe_source(source: str, seed: int, file_bytes: bytes | None, real_stamp: str | None) -> str:
    if source == SYNTHETIC:
        return f"{SYNTHETIC} (seed {seed})"
    if source == REAL:
        path = (real_stamp or "").split("|")[0]
        return f"{REAL} — {path}"
    return f"Uploaded CSV ({len(file_bytes or b'') / 1024:,.0f} KB)"


def show_real_data_missing() -> None:
    st.info(
        "**Real data has not been ingested yet.**\n\n"
        "Build it once from the FBref 2023-24 Top 5 snapshot (under a minute):\n\n"
        "```bash\npip install -r requirements-ingest.txt\npython scripts/ingest_fbref_snapshot.py --download\n```\n\n"
        "Team possession is read from FBref pages cached by `scripts/ingest_fbref.py`; without them "
        "it defaults to a neutral 50%. Switch to the Synthetic Enterprise DB in the meantime.",
        icon=":material/cloud_download:",
    )


def show_provenance_banner(df: pd.DataFrame) -> None:
    """Make it impossible to mistake estimated or synthetic data for fact."""
    if pipeline.dataset_origin(df) == ORIGIN_BENCHMARK:
        st.warning(
            "**Synthetic benchmark, not real players.** FBref could not be reached when the ingestion "
            "script ran (rate limit, CAPTCHA or network), so it wrote a synthetic stand-in with the same "
            "schema. Run `python scripts/ingest_fbref_snapshot.py --download` to build the real data.",
            icon=":material/warning:",
        )
        return
    value_proxy = pipeline.proxy_share(df, VALUE_SOURCE_COLUMN)
    contract_proxy = pipeline.proxy_share(df, CONTRACT_SOURCE_COLUMN)
    value_tm = pipeline.source_share(df, VALUE_SOURCE_COLUMN, SOURCE_TRANSFERMARKT)
    contract_tm = pipeline.source_share(df, CONTRACT_SOURCE_COLUMN, SOURCE_TRANSFERMARKT)
    if not (value_tm or contract_tm):
        if value_proxy or contract_proxy:
            st.info(
                f"Real FBref performance data. **Market values ({value_proxy:.0%}) and contract expiries "
                f"({contract_proxy:.0%}) are model estimates**, not Transfermarkt figures: values are a "
                "proxy from age, league, minutes and percentiles; contracts are assigned deterministically "
                "within 2026-2029. Treat budget and contract filters as indicative.",
                icon=":material/info:",
            )
        return
    contract_year = pipeline.contract_reference_year(df)
    notes = [
        f"**Market values:** {value_tm:.0%} Transfermarkt, as valued at the end of the season",
        f"**Contracts:** {contract_tm:.0%} Transfermarkt, as listed in {contract_year} "
        "(years remaining count from then; the club may differ from the 2023-24 one)",
    ]
    if value_proxy or contract_proxy:
        notes.append(f"The rest ({value_proxy:.0%} of values, {contract_proxy:.0%} of contracts) are "
                     "model estimates for players Transfermarkt could not match or no longer tracks.")
    st.info("Real FBref performance data.\n\n" + "\n\n".join(notes), icon=":material/info:")


def style_value_diff(value: float) -> str:
    if pd.isna(value) or value == 0:
        return ""
    return "color: #3fb950; font-weight: 600" if value > 0 else "color: #f85149"


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar_data_controls() -> tuple[str, bytes | None, int, int, float, bool]:
    with st.sidebar:
        st.header("Recruitment Filters")
        with st.expander("Data Source & Scope", expanded=True, icon=":material/database:"):
            source = st.radio("Data source", [SYNTHETIC, REAL, UPLOAD], key="source")
            file_bytes: bytes | None = None
            seed = RANDOM_STATE
            if source == UPLOAD:
                uploaded = st.file_uploader("Player season CSV (FBref-style)", type=["csv"])
                file_bytes = uploaded.getvalue() if uploaded is not None else None
            elif source == SYNTHETIC:
                seed = int(st.number_input("Synthetic DB seed", 0, 10_000, RANDOM_STATE, step=1))
            min_minutes = st.slider("Minimum minutes played", 0, 3000, DEFAULT_MIN_MINUTES, step=90)
        with st.expander("Contextual Adjustments", icon=":material/tune:"):
            padj = st.slider("Possession adjustment steepness (k)", 0.0, 0.15, DEFAULT_PADJ_STEEPNESS,
                             step=0.01, help="0 disables pAdj. 0.05 ≈ 1.25× at 60% possession.")
            league_scaling = st.toggle("Apply league strength coefficients", value=True)
    return source, file_bytes, seed, min_minutes, padj, league_scaling


def sidebar_model_controls(df: pd.DataFrame, reference_year: int) -> dict[str, object]:
    leagues_available = sorted(df["League"].dropna().unique().tolist())
    groups_available = [g for g in POSITION_GROUPS[::-1] if g in set(df["Position_Group"])]  # FW, MF, DF
    with st.sidebar:
        with st.expander("Cohort", expanded=True, icon=":material/groups:"):
            position = st.radio("Position group", groups_available, horizontal=True, key="position")
            leagues = st.multiselect("Leagues", leagues_available, default=leagues_available, key="leagues")

        with st.expander("Recruitment Constraints (Moneyball)", expanded=True, icon=":material/payments:"):
            max_age = st.slider("Max age", 18, 40, 40, key="max_age")
            value_cap = float(max(np.ceil(df["Market_Value_EUR_M"].max()), 1.0))
            max_value = st.slider("Max market value (€M)", 0.0, value_cap, value_cap, step=0.5, key="max_value")
            years = (df["Contract_Expiry_Year"] - reference_year).astype(int)
            year_options = list(range(max(int(years.min()), 0), max(int(years.max()), 0) + 1))
            contract_years: int | None = None
            if len(year_options) > 1:
                contract_years = st.select_slider(
                    "Max contract years remaining", options=year_options, value=year_options[-1],
                    key="contract_years", help=f"Years left on the contract, counted from {reference_year}.",
                )

        with st.expander("Model Hyperparameters", expanded=True, icon=":material/model_training:"):
            k = st.slider("Number of tactical clusters (k)", 2, 10, 5, key="k")
            scaler_label = st.selectbox("Feature scaler", list(SCALERS), key="scaler")
            metric_label = st.radio("Similarity metric", list(METRICS), key="metric")

        with st.expander("Feature Weight Multipliers", icon=":material/balance:"):
            weights = {
                category: st.slider(label, 0.0, 3.0, 1.0, step=0.1, key=f"w_{category}")
                for label, category in WEIGHT_SLIDERS.items()
            }

    # A slider left at its maximum means "no constraint", not an explicit cap.
    filters = pipeline.RecruitmentFilters(
        max_age=None if max_age >= 40 else int(max_age),
        max_market_value=None if max_value >= value_cap else float(max_value),
        contract_years_max=None if contract_years in (None, year_options[-1]) else int(contract_years),
    )
    return {
        "position": position,
        "leagues": leagues,
        "filters": filters,
        "k": int(k),
        "scaler": SCALERS[scaler_label],
        "metric_label": metric_label,
        "metric": METRICS[metric_label],
        "weights": weights,
        "reference_year": reference_year,
    }


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def tab_profile(model: pipeline.CohortModel, target: pd.Series, cohort_key: str) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Age", int(target["Age"]))
    c2.metric("Squad & League", str(target["Squad"]), delta=str(target["League"]), delta_color="off",
              delta_arrow="off")
    c3.metric("Market Value", f"€{target['Market_Value_EUR_M']:,.1f}M",
              delta=f"Contract to {int(target['Contract_Expiry_Year'])}", delta_color="off", delta_arrow="off")
    c4.metric("Assigned Tactical Role", f"Cluster {int(target['Cluster'])}",
              delta=str(target["Assigned_Role"]), delta_color="off", delta_arrow="off")

    left, right = st.columns([5, 7], gap="large")
    with left:
        overall = pipeline.overall_percentile(target)
        tier, color = pipeline.score_badge(overall)
        st.subheader("Player DNA")
        st.metric("Overall percentile rating", f"{overall:.0f} / 100",
                  help=f"Mean within-{target['Position_Group']} percentile across the 9 core metrics.")
        st.badge(tier, color=color)
        for column, label in pipeline.COMPOSITE_DISPLAY.items():
            score = float(target[column])
            sub_tier, _ = pipeline.score_badge(score)
            st.progress(min(max(score / 100.0, 0.0), 1.0), text=f"**{label}** · {score:.0f} · {sub_tier}")
        st.caption(f"Scores are percentiles against all {target['Position_Group']} players in the database "
                   f"({model.cohort.shape[0]} in the active cohort).")
    with right:
        st.image(render_radar(cohort_key, str(target["Player"]), str(target["Squad"]), None, None, target, None),
                 width="stretch")


def tab_replacements(
    model: pipeline.CohortModel, target: pd.Series, shortlist: pd.DataFrame,
    controls: dict[str, object], cohort_key: str,
) -> None:
    filters: pipeline.RecruitmentFilters = controls["filters"]  # type: ignore[assignment]
    st.subheader(f"Replacements for {target['Player']}")
    st.caption(f"**Metric:** {controls['metric_label']}  ·  **Constraints:** {filters.describe()}  ·  "
               f"**Pool:** {target['Position_Group']} in {len(controls['leagues'])} league(s)")
    if controls["metric"] == "euclidean":
        st.caption("Euclidean scores reward like-for-like output volume; elite targets naturally score lower. "
                   "Switch to Cosine to find stylistic twins.")

    if shortlist.empty:
        st.warning(NO_MATCH_MESSAGE)
        return

    table = pipeline.format_shortlist(shortlist)
    st.dataframe(
        table.style.map(style_value_diff, subset=["Value Diff (€M)"]),
        hide_index=True,
        width="stretch",
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Market Value (€M)": st.column_config.NumberColumn(format="€%.1fM"),
            "Value Diff (€M)": st.column_config.NumberColumn(
                format="€%+.1fM", help="Target value minus candidate value; positive = money saved."),
            "Contract Expiry": st.column_config.NumberColumn(format="%d"),
            "Match Score (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
        },
    )

    st.divider()
    st.subheader("Head-to-Head Tactical Overlay")
    labels = [f"#{r.Rank} {r.Player} ({r.Squad})" for r in shortlist.itertuples()]
    keep_valid_selection("candidate_label", labels)
    choice = st.selectbox("Compare against", labels, key="candidate_label")
    candidate_row = shortlist.iloc[labels.index(choice)]
    candidate = pipeline.get_player_row(model.cohort, candidate_row["Player"], candidate_row["Squad"])

    saved = float(target["Market_Value_EUR_M"] - candidate["Market_Value_EUR_M"])
    age_gap = int(candidate["Age"] - target["Age"])
    same_role = candidate["Cluster"] == target["Cluster"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Match score", f"{candidate_row['Match_Score_Pct']:.1f}%")
    k2.metric("Candidate value", f"€{candidate['Market_Value_EUR_M']:,.1f}M",
              delta=f"€{saved:,.1f}M saved" if saved >= 0 else f"€{-saved:,.1f}M premium",
              delta_color="normal" if saved >= 0 else "inverse")
    k3.metric("Age", int(candidate["Age"]), delta=f"{age_gap:+d} yrs vs target", delta_color="inverse")
    k4.metric("Role alignment", "Same role" if same_role else "Different role",
              delta=str(candidate["Assigned_Role"]), delta_color="off", delta_arrow="off")

    _, mid, _ = st.columns([2, 5, 2])
    with mid:
        st.image(
            render_radar(cohort_key, str(target["Player"]), str(target["Squad"]),
                         str(candidate["Player"]), str(candidate["Squad"]), target, candidate),
            width="stretch",
        )


def tab_clusters(
    model: pipeline.CohortModel, target: pd.Series, shortlist: pd.DataFrame, model_key: str
) -> None:
    left, right = st.columns([8, 4], gap="medium")
    # Callouts print match %, so scores belong in the cache key, not just names.
    shortlist_key = tuple(
        f"{p}|{s}|{m:.2f}" for p, s, m in zip(shortlist["Player"], shortlist["Squad"], shortlist["Match_Score_Pct"])
    )
    with left:
        st.image(render_pca_map(model_key, str(target["Player"]), shortlist_key, model, shortlist), width="stretch")
    with right:
        st.subheader("Cluster Summary")
        target_cluster = int(target["Cluster"])
        for _, row in pipeline.cluster_summary(model).iterrows():
            with st.container(border=True):
                marker = " · ★ target" if row["Cluster"] == target_cluster else ""
                st.markdown(f"**{row['Cluster']} · {row['Tactical Role']}**{marker}")
                st.caption(f"{row['Players']} players ({row['Share (%)']:.0f}%) · "
                           f"median €{row['Median Value (€M)']:,.1f}M  \n"
                           f"Signature: {row['Top Traits']}")
        st.caption(f"PCA map retains {model.pca.cumulative_variance:.1%} of the variance. "
                   f"Target sits in cluster {int(target['Cluster'])}.")

    st.image(render_heatmap(model_key, model), width="stretch")

    with st.expander("Cohort Explorer", icon=":material/search:"):
        roles = {f"{cid}: {role}": cid for cid, role in sorted(model.role_names.items())}
        picked = st.selectbox("Playing style", list(roles), key="explorer_cluster")
        members = model.cohort[model.cohort["Cluster"] == roles[picked]]
        columns = ["Player", "Squad", "League", "Age", "Market_Value_EUR_M", "Contract_Expiry_Year",
                   *pipeline.COMPOSITE_DISPLAY]
        st.dataframe(
            members[columns].sort_values("Market_Value_EUR_M", ascending=False)
            .rename(columns={"Market_Value_EUR_M": "Market Value (€M)", "Contract_Expiry_Year": "Contract Expiry",
                             **pipeline.COMPOSITE_DISPLAY}),
            hide_index=True, width="stretch",
            column_config={"Market Value (€M)": st.column_config.NumberColumn(format="€%.1fM"),
                           "Contract Expiry": st.column_config.NumberColumn(format="%d"),
                           **{label: st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f")
                              for label in pipeline.COMPOSITE_DISPLAY.values()}},
        )


def tab_diagnostics(
    model: pipeline.CohortModel, diagnostics: dict[str, list[float]], k: int,
    export: pd.DataFrame, position: str, cohort_key: str, scaler: str,
) -> None:
    st.subheader("Model Diagnostics")
    optimal = pipeline.best_k(diagnostics)
    current_sil = dict(zip(map(int, diagnostics["k"]), diagnostics["silhouette_scores"])).get(k)
    m1, m2, m3 = st.columns(3)
    m1.metric("Active k", k)
    m2.metric("Silhouette at active k", f"{current_sil:.3f}" if current_sil is not None else "n/a")
    m3.metric("Silhouette-optimal k", optimal if optimal is not None else "n/a")
    st.image(render_diagnostics(cohort_key, scaler, k, diagnostics), width="stretch")

    st.subheader("PCA Factor Loadings")
    st.caption("Eigenvector weights: a large positive value pushes players high on that metric to the right (PC1) "
               "or top (PC2) of the tactical map.")
    loadings = model.pca.get_loadings_df(list(CORE_FEATURES))
    loadings.index = [f"{pc} ({ratio:.1%})" for pc, ratio in zip(loadings.index, model.pca.explained_variance_ratio)]
    st.dataframe(
        loadings.style.format("{:+.2f}").background_gradient(cmap="RdBu_r", vmin=-0.7, vmax=0.7, axis=None),
        width="stretch",
    )

    truth = pipeline.ground_truth_report(model)
    if truth is not None:
        st.subheader("Ground-Truth Validation")
        st.caption("The synthetic DB records each player's hidden generator archetype. Normalised Mutual "
                   "Information measures how well the clusters recover it (0 = unrelated, 1 = identical).")
        st.metric("NMI vs. hidden archetypes", f"{truth.attrs['nmi']:.3f}")
        st.dataframe(truth, width="stretch")

    st.subheader("Export Scouting Report")
    st.caption(f"{len(export):,} players with clusters, roles, PCA coordinates, composite scores and match % "
               "against the current target.")
    st.download_button(
        "Download clustered shortlist (CSV)",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name=pipeline.export_filename(position),
        mime="text/csv",
        icon=":material/download:",
        type="primary",
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    inject_css()
    source, file_bytes, seed, min_minutes, padj, league_scaling = sidebar_data_controls()

    if source == UPLOAD and file_bytes is None:
        st.info("Upload an FBref-style season CSV in the sidebar to begin. The template below shows the "
                "expected schema.", icon=":material/upload_file:")
        template = DataLoader.read(generator_config=GeneratorConfig(clubs_per_league=1)).head(5)
        st.dataframe(template, hide_index=True, width="stretch")
        st.download_button("Download CSV template", template.to_csv(index=False).encode("utf-8"),
                           file_name="scoutintel_template.csv", mime="text/csv")
        st.stop()

    real_stamp: str | None = None
    if source == REAL:
        dataset = find_real_dataset(DEFAULT_REAL_SEASON)
        if dataset is None:
            show_real_data_missing()
            st.stop()
        real_stamp = f"{dataset.path}|{dataset.path.stat().st_mtime_ns}"

    try:
        df, report = load_dataset(source, file_bytes, seed, min_minutes, padj, league_scaling, real_stamp)
    except RealDataNotFoundError:
        show_real_data_missing()
        st.stop()
    except DataValidationError as exc:
        st.error(f"Could not load the uploaded file: {exc}", icon=":material/error:")
        st.stop()

    source_desc = describe_source(source, seed, file_bytes, real_stamp)
    if df.empty:
        show_pipeline_status(report, source_desc, None)
        st.error("No players survive validation and the minutes threshold. Lower the minimum minutes "
                 "or check the uploaded data (see the drop log above).")
        st.stop()
    if source == UPLOAD and int(report["n_invalid"]) > 0:
        st.warning(f"{report['n_invalid']} uploaded rows were dropped as invalid (non-numeric or "
                   "out-of-range values). See the pipeline status for the full drop log.")

    if source == REAL:
        show_provenance_banner(df)
    controls = sidebar_model_controls(df, pipeline.contract_reference_year(df))
    if not controls["leagues"]:
        show_pipeline_status(report, source_desc, 0)
        st.warning("Select at least one league in the sidebar.")
        st.stop()

    cohort = pipeline.filter_cohort(df, controls["position"], controls["leagues"])
    show_pipeline_status(report, source_desc, len(cohort))

    if file_bytes:
        data_key = hashlib.sha256(file_bytes).hexdigest()[:16]
    elif source == REAL:
        data_key = f"real-{hashlib.sha256((real_stamp or '').encode()).hexdigest()[:12]}"
    else:
        data_key = f"synthetic-{seed}"
    cohort_key = f"{data_key}|{min_minutes}|{padj}|{league_scaling}|{controls['position']}|" + \
        ",".join(sorted(controls["leagues"]))
    model_key = f"{cohort_key}|k={controls['k']}|{controls['scaler']}"

    try:
        model = get_cohort_model(cohort_key, controls["k"], controls["scaler"], cohort)
        diagnostics = get_diagnostics(cohort_key, controls["scaler"], model.X)
    except InsufficientDataError as exc:
        st.warning(f"{exc} Broaden the league selection, lower the minutes threshold or reduce k.")
        st.stop()

    tabs = st.tabs([
        ":material/person: Target Player & Tactical DNA",
        ":material/swap_horiz: Moneyball Replacement Engine",
        ":material/hub: Tactical Role Clusters",
        ":material/analytics: Model Diagnostics & Export",
    ])

    options = pipeline.player_options(model.cohort)
    labels = list(options)
    with tabs[0]:
        keep_valid_selection("target_label", labels)
        target_label = st.selectbox(
            "Target player", labels, key="target_label",
            help="Searchable. Ordered by market value; duplicate names show their squad.",
        )
    player, squad = options[target_label]
    target = pipeline.get_player_row(model.cohort, player, squad)

    try:
        shortlist = pipeline.run_shortlist(model, player, squad, controls["metric"], controls["weights"],
                                           controls["filters"], reference_year=controls["reference_year"])
        scores = pipeline.match_scores(model, player, squad, controls["metric"], controls["weights"])
    except ValueError as exc:  # e.g. every feature weight set to zero
        st.error(f"Similarity search unavailable: {exc}")
        shortlist = pd.DataFrame(columns=list(pipeline.SHORTLIST_DISPLAY) + ["Position_Group"])
        scores = None

    with tabs[0]:
        tab_profile(model, target, cohort_key)
    with tabs[1]:
        tab_replacements(model, target, shortlist, controls, cohort_key)
    with tabs[2]:
        tab_clusters(model, target, shortlist, model_key)
    with tabs[3]:
        tab_diagnostics(model, diagnostics, controls["k"], pipeline.build_export(model, scores),
                        controls["position"], cohort_key, controls["scaler"])


main()
