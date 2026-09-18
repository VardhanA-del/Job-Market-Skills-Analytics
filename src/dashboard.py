"""Shared Streamlit data access, filtering, and chart presentation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import PROCESSED_DIR, load_processed_data
from src.preprocessing import build_processed_data


COLOR_SEQUENCE = ["#0F766E", "#2563EB", "#7C3AED", "#EA580C", "#DC2626", "#0891B2"]


@st.cache_data(show_spinner="Preparing the analytics dataset…")
def load_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not (PROCESSED_DIR / "clean_jobs.parquet").exists():
        build_processed_data()
    return load_processed_data()


def configure_page(title: str, icon: str = "📊") -> None:
    st.set_page_config(page_title=title, page_icon=icon, layout="wide")


def page_intro(title: str, description: str) -> None:
    st.title(title)
    st.caption(description)


def render_filters(
    jobs: pd.DataFrame,
    skills: pd.DataFrame,
    key_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Render supported global filters and return filtered jobs and skills."""
    st.sidebar.header("Filters")
    result = jobs.copy()

    role_options = sorted(result["normalized_job_role"].dropna().unique())
    roles = st.sidebar.multiselect("Job role", role_options, key=f"{key_prefix}_roles")
    if roles:
        result = result[result["normalized_job_role"].isin(roles)]

    location_options = result["primary_location"].value_counts().head(300).index.tolist()
    locations = st.sidebar.multiselect("Primary location", location_options, key=f"{key_prefix}_locations")
    if locations:
        result = result[result["primary_location"].isin(locations)]

    exp_options = [x for x in ["Fresher", "Entry Level", "Mid Level", "Senior", "Not specified"] if x in set(result["experience_category"])]
    experience = st.sidebar.multiselect("Experience level", exp_options, key=f"{key_prefix}_experience")
    if experience:
        result = result[result["experience_category"].isin(experience)]

    company_options = result["company_name"].value_counts().head(500).index.tolist()
    companies = st.sidebar.multiselect("Company (top 500)", company_options, key=f"{key_prefix}_companies")
    if companies:
        result = result[result["company_name"].isin(companies)]

    skill_options = skills[skills["job_id"].isin(result["job_id"])]["skill"].value_counts().head(150).index.tolist()
    selected_skills = st.sidebar.multiselect("Skills (match all)", skill_options, key=f"{key_prefix}_skills")
    if selected_skills:
        selected = skills[skills["skill"].isin(selected_skills)]
        ids = selected.groupby("job_id")["skill"].nunique()
        result = result[result["job_id"].isin(ids[ids >= len(selected_skills)].index)]

    require_salary = st.sidebar.checkbox("Require disclosed INR salary", value=False, key=f"{key_prefix}_salary_required")
    salary_values = jobs["average_salary_lpa"].dropna()
    if require_salary:
        result = result[result["average_salary_lpa"].notna()]
    if not salary_values.empty:
        lower = float(max(0, salary_values.quantile(0.01)))
        upper = float(salary_values.quantile(0.99))
        selected_range = st.sidebar.slider(
            "Average salary (LPA, 1st–99th percentile)", lower, upper, (lower, upper),
            key=f"{key_prefix}_salary", help="Moving this range excludes missing salaries; INR listings only.",
        )
        if selected_range != (lower, upper):
            result = result[result["average_salary_lpa"].between(*selected_range)]

    if jobs["company_rating"].notna().any():
        require_rating = st.sidebar.checkbox("Require a company rating", value=False, key=f"{key_prefix}_rating_required")
        rating_range = st.sidebar.slider("Company rating", 1.0, 5.0, (1.0, 5.0), 0.1, key=f"{key_prefix}_rating")
        if require_rating or rating_range != (1.0, 5.0):
            result = result[result["company_rating"].between(*rating_range)]

    filtered_skills = skills[skills["job_id"].isin(result["job_id"])].copy()
    st.sidebar.caption(f"{result['job_id'].nunique():,} listings match. Locations and companies are limited to frequent filter options for responsiveness.")
    return result, filtered_skills


def empty_state() -> None:
    st.warning("No listings match these filters. Broaden one or more sidebar selections.")
    st.stop()


def bar_chart(frame: pd.DataFrame, x: str, y: str, title: str, horizontal: bool = False, **kwargs):
    if frame.empty:
        return None
    if horizontal:
        fig = px.bar(frame.sort_values(x), x=x, y=y, orientation="h", title=title, color_discrete_sequence=COLOR_SEQUENCE, **kwargs)
    else:
        fig = px.bar(frame, x=x, y=y, title=title, color_discrete_sequence=COLOR_SEQUENCE, **kwargs)
    fig.update_layout(margin=dict(l=20, r=20, t=55, b=20), legend_title_text="")
    return fig

