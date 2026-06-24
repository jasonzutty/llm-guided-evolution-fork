from pathlib import Path
import subprocess
import shutil
import math
import importlib.util

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="LLM-Guided Evolution Analysis Dashboard",
    layout="wide",
)

st.title("LLM-Guided Evolution Analysis Dashboard")
st.caption("Upload Slurm logs and/or pickle files, run the analysis pipeline, and view generated tables, reports, and visuals.")

APP_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = APP_DIR.parent

UPLOAD_DIR = APP_DIR / "uploads"
RESULTS_DIR = ANALYSIS_DIR / "results" / "analysis"
SCRIPTS_DIR = ANALYSIS_DIR / "scripts"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="ignore")


def run_command(cmd: list[str], required: bool = True) -> None:
    st.write("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        st.text(result.stdout)

    if result.returncode != 0:
        st.error(result.stderr)
        if required:
            st.stop()


def load_script_module(script_name: str):
    path = SCRIPTS_DIR / script_name
    if not path.exists():
        st.warning(f"Missing script: {path}")
        return None

    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_optional_visual_script(script_name: str, function_name: str, *args):
    module = load_script_module(script_name)
    if module is None:
        return

    if not hasattr(module, function_name):
        st.warning(f"{script_name} does not contain function {function_name}.")
        return

    try:
        getattr(module, function_name)(*args)
        st.success(f"Generated output using {script_name}.")
    except Exception as e:
        st.warning(f"{script_name} could not run: {e}")


def finite_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df["fitness_1_num"].apply(lambda x: pd.notna(x) and math.isfinite(x))
        & df["fitness_2_num"].apply(lambda x: pd.notna(x) and math.isfinite(x))
    )


def display_image_if_exists(path: Path, caption: str):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"{caption} not generated yet.")


# -------------------------
# Sidebar upload
# -------------------------

st.sidebar.header("Upload Files")

uploaded_files = st.sidebar.file_uploader(
    "Upload Slurm, log, or pickle files",
    type=["out", "log", "pkl"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.sidebar.write("Uploaded:", [f.name for f in uploaded_files])

if st.sidebar.button("Analyze Uploaded Files"):
    if not uploaded_files:
        st.sidebar.warning("Upload at least one file first.")
    else:
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        for uploaded_file in uploaded_files:
            file_path = UPLOAD_DIR / uploaded_file.name
            with file_path.open("wb") as f:
                f.write(uploaded_file.getbuffer())

        st.sidebar.success(f"Saved {len(uploaded_files)} uploaded file(s).")

        # -------------------------
        # Core pipeline scripts
        # -------------------------

        core_commands = [
            [
                "python3", str(SCRIPTS_DIR / "summarize_slurm.py"),
                "--input", str(UPLOAD_DIR),
                "--output", str(RESULTS_DIR / "slurm_summary.csv"),
            ],
            [
                "python3", str(SCRIPTS_DIR / "inventory_runs.py"),
                "--input", str(UPLOAD_DIR),
                "--output", str(RESULTS_DIR / "run_inventory.csv"),
            ],
            [
                "python3", str(SCRIPTS_DIR / "extract_metrics.py"),
                "--input", str(UPLOAD_DIR),
                "--output", str(RESULTS_DIR / "run_metrics.csv"),
            ],
            [
                "python3", str(SCRIPTS_DIR / "entity_metrics_summary.py"),
                "--input", str(RESULTS_DIR / "run_metrics.csv"),
                "--output", str(RESULTS_DIR / "entity_metrics_summary.csv"),
                "--report-output", str(RESULTS_DIR / "entity_metrics_report.md"),
            ],
            [
                "python3", str(SCRIPTS_DIR / "build_failure_report.py"),
                "--input", str(RESULTS_DIR / "slurm_summary.csv"),
                "--counts-output", str(RESULTS_DIR / "failure_counts.csv"),
                "--report-output", str(RESULTS_DIR / "failure_report.md"),
            ],
            [
                "python3", str(SCRIPTS_DIR / "compare_runs.py"),
                "--slurm-input", str(RESULTS_DIR / "slurm_summary.csv"),
                "--inventory-input", str(RESULTS_DIR / "run_inventory.csv"),
                "--metrics-input", str(RESULTS_DIR / "run_metrics.csv"),
                "--output", str(RESULTS_DIR / "run_comparison.csv"),
            ],
            [
                "python3", str(SCRIPTS_DIR / "export_dashboard_data.py"),
                "--slurm-input", str(RESULTS_DIR / "slurm_summary.csv"),
                "--comparison-input", str(RESULTS_DIR / "run_comparison.csv"),
                "--failure-input", str(RESULTS_DIR / "failure_counts.csv"),
                "--metrics-input", str(RESULTS_DIR / "run_metrics.csv"),
                "--output-dir", str(RESULTS_DIR),
            ],
        ]

        st.subheader("Pipeline Run")
        for cmd in core_commands:
            run_command(cmd, required=True)

        # -------------------------
        # Visualization/report scripts
        # -------------------------

        st.subheader("Visualization Script Run")

        run_optional_visual_script(
            "plot_controller_activity.py",
            "generate_controller_activity_chart",
            str(RESULTS_DIR / "slurm_summary.csv"),
            str(RESULTS_DIR / "controller_activity_summary.png"),
        )

        run_optional_visual_script(
            "plot_entity_status.py",
            "generate_entity_status_chart",
            str(RESULTS_DIR / "entity_metrics_summary.csv"),
            str(RESULTS_DIR / "entity_status_summary.png"),
        )

        run_optional_visual_script(
            "plot_fitness_quality.py",
            "generate_fitness_quality_chart",
            str(RESULTS_DIR / "entity_metrics_summary.csv"),
            str(RESULTS_DIR / "fitness_quality_summary.png"),
        )

        run_optional_visual_script(
            "generate_leaderboard.py",
            "generate_leaderboard",
            str(RESULTS_DIR / "entity_metrics_summary.csv"),
            str(RESULTS_DIR / "top_10_leaderboard.csv"),
        )

        # These scripts are written as standalone scripts with hardcoded analysis/results/analysis paths.
        # They run after the core outputs exist.
        run_command(["python3", str(SCRIPTS_DIR / "generation_timeline.py")], required=False)

        # Build Pareto report from first uploaded pickle, if any exists.
        uploaded_pkls = sorted(UPLOAD_DIR.glob("*.pkl"))
        if uploaded_pkls:
            run_command(
                [
                    "python3", str(SCRIPTS_DIR / "global_data_fitness_report.py"),
                    "--input", str(uploaded_pkls[0]),
                    "--output", str(RESULTS_DIR / "global_data_pareto_report.md"),
                ],
                required=False,
            )

            run_command(["python3", str(SCRIPTS_DIR / "pareto_summary_card.py")], required=False)
        else:
            st.info("No .pkl uploaded, so Pareto report scripts were skipped.")

        st.session_state["analysis_complete"] = True
        st.success("Analysis complete. Dashboard outputs updated.")
        st.rerun()


# -------------------------
# Load generated outputs
# -------------------------

slurm_summary = read_csv(RESULTS_DIR / "slurm_summary.csv")
failure_counts = read_csv(RESULTS_DIR / "failure_counts.csv")
entity_summary = read_csv(RESULTS_DIR / "entity_metrics_summary.csv")
run_comparison = read_csv(RESULTS_DIR / "run_comparison.csv")
run_inventory = read_csv(RESULTS_DIR / "run_inventory.csv")
leaderboard = read_csv(RESULTS_DIR / "top_10_leaderboard.csv")

st.caption(f"Reading outputs from: `{RESULTS_DIR}`")


# -------------------------
# Tabs
# -------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Run Health",
    "Entity Fitness",
    "Generated Visuals",
    "Reports",
    "Raw Tables",
])


# -------------------------
# Tab 1: Run Health
# -------------------------

with tab1:
    st.header("Run Health")

    if not slurm_summary.empty:
        col1, col2, col3 = st.columns(3)

        total_runs = len(slurm_summary)
        failed_runs = (
            (slurm_summary["status_guess"] == "failed").sum()
            if "status_guess" in slurm_summary.columns
            else 0
        )
        completed_runs = (
            slurm_summary["status_guess"].isin(["completed", "completed_or_unknown"]).sum()
            if "status_guess" in slurm_summary.columns
            else 0
        )

        col1.metric("Total Runs", int(total_runs))
        col2.metric("Failed Runs", int(failed_runs))
        col3.metric("Completed Runs", int(completed_runs))

        st.dataframe(slurm_summary, use_container_width=True)
    else:
        st.info("Upload a Slurm `.out` or `.log` file to populate run health.")

    st.subheader("Failure Breakdown")

    if not failure_counts.empty:
        st.dataframe(failure_counts, use_container_width=True)

        if {"group_type", "group_value", "count"}.issubset(failure_counts.columns):
            failure_df = failure_counts[failure_counts["group_type"] == "failure_reason"]

            if not failure_df.empty:
                st.bar_chart(failure_df.set_index("group_value")["count"])
            else:
                st.info("No failure reasons found.")
    else:
        st.info("No failure data available.")

    st.subheader("Controller / Child Job Activity")

    if not slurm_summary.empty:
        cols = ["job_id", "submitted_child_jobs", "completed_child_jobs", "wait_events"]
        available = [c for c in cols if c in slurm_summary.columns]

        if len(available) > 1:
            controller_df = slurm_summary[available].copy()

            for c in available:
                if c != "job_id":
                    controller_df[c] = pd.to_numeric(controller_df[c], errors="coerce").fillna(0)

            st.dataframe(controller_df, use_container_width=True)

            numeric_cols = [c for c in available if c != "job_id"]
            st.bar_chart(controller_df.set_index("job_id")[numeric_cols])
        else:
            st.info("No controller activity columns found.")


# -------------------------
# Tab 2: Entity Fitness
# -------------------------

with tab2:
    st.header("Entity Fitness")

    finite_df = pd.DataFrame()

    if not entity_summary.empty and {"fitness_1", "fitness_2"}.issubset(entity_summary.columns):
        df = entity_summary.copy()

        df["fitness_1_num"] = pd.to_numeric(df["fitness_1"], errors="coerce")
        df["fitness_2_num"] = pd.to_numeric(df["fitness_2"], errors="coerce")

        finite = finite_mask(df)

        infinite = (
            df["fitness_1"].astype(str).isin(["inf", "-inf"])
            | df["fitness_2"].astype(str).isin(["inf", "-inf"])
        )

        missing = ~(finite | infinite)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Entities", int(len(df)))
        col2.metric("Finite Fitness", int(finite.sum()))
        col3.metric("Infinite Fitness", int(infinite.sum()))
        col4.metric("Missing / Malformed", int(missing.sum()))

        quality_df = pd.DataFrame({
            "category": ["finite", "infinite", "missing/malformed"],
            "count": [int(finite.sum()), int(infinite.sum()), int(missing.sum())],
        })

        st.bar_chart(quality_df.set_index("category")["count"])

        finite_df = df[finite].copy()

        st.subheader("Top Individuals")

        if not leaderboard.empty:
            st.dataframe(leaderboard, use_container_width=True)
        elif not finite_df.empty:
            finite_df = finite_df.sort_values("fitness_1_num", ascending=False)

            show_cols = [
                "entity_id",
                "job_id",
                "results_job",
                "entity_status",
                "fitness_1",
                "fitness_2",
                "most_common_mutate_type",
            ]
            show_cols = [c for c in show_cols if c in finite_df.columns]

            st.dataframe(finite_df[show_cols].head(10), use_container_width=True)
        else:
            st.info("No finite fitness rows available.")

        st.subheader("Pareto Scatter")

        if not finite_df.empty:
            fig, ax = plt.subplots(figsize=(9, 6))

            ax.scatter(
                finite_df["fitness_2_num"],
                finite_df["fitness_1_num"],
                alpha=0.7,
                label="Finite entities",
            )

            ax.set_xlabel("fitness_2")
            ax.set_ylabel("fitness_1")
            ax.set_title("Pareto Scatter: Finite Fitness Points")
            ax.legend()

            st.pyplot(fig)
            st.caption("Only finite fitness rows are plotted. Infinite or missing fitness values are excluded.")
        else:
            st.info("No finite fitness points available for Pareto scatter.")
    else:
        st.info("Upload a `.pkl` file to populate entity fitness, leaderboard, and Pareto views.")


# -------------------------
# Tab 3: Generated Visuals
# -------------------------

with tab3:
    st.header("Generated Visuals From Visualization Scripts")

    col1, col2 = st.columns(2)

    with col1:
        display_image_if_exists(
            RESULTS_DIR / "controller_activity_summary.png",
            "Controller Activity Summary",
        )

        display_image_if_exists(
            RESULTS_DIR / "entity_status_summary.png",
            "Entity Status Summary",
        )

        display_image_if_exists(
            RESULTS_DIR / "pareto_kpi_graphs" / "pareto_kpi_bar.png",
            "Pareto KPI Summary",
        )

    with col2:
        display_image_if_exists(
            RESULTS_DIR / "fitness_quality_summary.png",
            "Fitness Quality Summary",
        )

        display_image_if_exists(
            RESULTS_DIR / "generation_timeline_graphs" / "generation_completed_count_bar.png",
            "Generation Completed Count",
        )

        display_image_if_exists(
            RESULTS_DIR / "generation_timeline_graphs" / "generation_cumulative_timeline.png",
            "Generation Cumulative Timeline",
        )

        display_image_if_exists(
            RESULTS_DIR / "generation_timeline_graphs" / "generation_duration_proxy.png",
            "Generation Duration Proxy",
        )


# -------------------------
# Tab 4: Reports
# -------------------------

with tab4:
    st.header("Generated Markdown Reports")

    report_files = [
        RESULTS_DIR / "failure_report.md",
        RESULTS_DIR / "entity_metrics_report.md",
        RESULTS_DIR / "global_data_pareto_report.md",
        RESULTS_DIR / "pareto_kpi_summary.md",
        RESULTS_DIR / "generation_timeline_report.md",
    ]

    for report_path in report_files:
        st.subheader(report_path.name)
        text = read_text(report_path)
        if text:
            st.markdown(text)
        else:
            st.info(f"Report not found: `{report_path}`")


# -------------------------
# Tab 5: Raw Tables
# -------------------------

with tab5:
    st.header("Raw Pipeline Tables")

    st.subheader("Run Inventory")
    if not run_inventory.empty:
        st.dataframe(run_inventory, use_container_width=True)
    else:
        st.info("No run inventory available.")

    st.subheader("Run Comparison")
    if not run_comparison.empty:
        st.dataframe(run_comparison, use_container_width=True)
    else:
        st.info("No run comparison available.")

    st.subheader("Run Metrics")
    run_metrics = read_csv(RESULTS_DIR / "run_metrics.csv")
    if not run_metrics.empty:
        st.dataframe(run_metrics, use_container_width=True)
    else:
        st.info("No run metrics available.")

    st.subheader("Entity Metrics Summary")
    if not entity_summary.empty:
        st.dataframe(entity_summary, use_container_width=True)
    else:
        st.info("No entity metrics summary available.")

    st.subheader("Dashboard Exports")
    dashboard_runs = read_csv(RESULTS_DIR / "dashboard_runs.csv")
    dashboard_failures = read_csv(RESULTS_DIR / "dashboard_failures.csv")
    dashboard_metrics = read_csv(RESULTS_DIR / "dashboard_metrics.csv")

    if not dashboard_runs.empty:
        st.write("dashboard_runs.csv")
        st.dataframe(dashboard_runs, use_container_width=True)

    if not dashboard_failures.empty:
        st.write("dashboard_failures.csv")
        st.dataframe(dashboard_failures, use_container_width=True)

    if not dashboard_metrics.empty:
        st.write("dashboard_metrics.csv")
        st.dataframe(dashboard_metrics, use_container_width=True)