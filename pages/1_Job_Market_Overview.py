"""Job-market overview page."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard import COLOR_SEQUENCE, bar_chart, configure_page, empty_state, load_dashboard_data, page_intro, render_filters
from src.metrics import (
    get_median_salary, get_top_companies, get_top_locations, get_top_roles, get_top_skills,
    get_total_companies, get_total_jobs, get_total_locations,
)
from src.salary_analysis import valid_inr_salaries
from src.utils import format_lpa


configure_page("Job Market & Skills Analytics", "💼")
page_intro(
    "Job Market & Skills Analytics",
    "A dataset-grounded view of Indian job listings. Findings describe this source snapshot, not the entire labor market.",
)

jobs, skills, _ = load_dashboard_data()
filtered, filtered_skills = render_filters(jobs, skills, "overview")
if filtered.empty:
    empty_state()

top_roles = get_top_roles(filtered)
top_skills = get_top_skills(filtered_skills)
kpis = [
    ("Job listings", f"{get_total_jobs(filtered):,}"),
    ("Companies", f"{get_total_companies(filtered):,}"),
    ("Primary locations", f"{get_total_locations(filtered):,}"),
    ("Role groups", f"{filtered['normalized_job_role'].nunique():,}"),
    ("Median disclosed salary", format_lpa(get_median_salary(filtered))),
    ("Median experience", f"{filtered['average_experience'].median():.1f} yrs" if filtered["average_experience"].notna().any() else "N/A"),
    ("Most in-demand skill", top_skills.iloc[0]["skill"] if not top_skills.empty else "N/A"),
    ("Most in-demand role", top_roles.iloc[0]["normalized_job_role"] if not top_roles.empty else "N/A"),
]
for start in (0, 4):
    columns = st.columns(4)
    for column, (label, value) in zip(columns, kpis[start:start + 4]):
        column.metric(label, value)

left, right = st.columns(2)
with left:
    fig = bar_chart(top_roles, "job_count", "normalized_job_role", "Most in-demand role groups", horizontal=True)
    if fig:
        st.plotly_chart(fig, width="stretch")
with right:
    locations = get_top_locations(filtered)
    fig = bar_chart(locations, "job_count", "primary_location", "Listings by primary location", horizontal=True)
    if fig:
        st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
with left:
    companies = get_top_companies(filtered)
    fig = bar_chart(companies, "job_count", "company_name", "Top hiring companies", horizontal=True)
    if fig:
        st.plotly_chart(fig, width="stretch")
with right:
    experience = filtered.groupby("experience_category")["job_id"].nunique().rename("job_count").reset_index()
    order = ["Fresher", "Entry Level", "Mid Level", "Senior", "Not specified"]
    experience["experience_category"] = pd.Categorical(experience["experience_category"], order, ordered=True)
    experience = experience.sort_values("experience_category")
    fig = bar_chart(experience, "experience_category", "job_count", "Listings by experience level")
    if fig:
        st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
with left:
    salaries = valid_inr_salaries(filtered)
    if salaries.empty:
        st.info("No disclosed INR salaries are available for the current filters.")
    else:
        p99 = salaries["average_salary_lpa"].quantile(0.99)
        chart_data = salaries[salaries["average_salary_lpa"] <= p99]
        fig = px.histogram(chart_data, x="average_salary_lpa", nbins=40, title="Average salary distribution (INR, display capped at 99th percentile)", color_discrete_sequence=COLOR_SEQUENCE)
        fig.update_layout(xaxis_title="Average annual salary (LPA)", yaxis_title="Listings", margin=dict(l=20, r=20, t=55, b=20))
        st.plotly_chart(fig, width="stretch")
with right:
    recency = filtered["posting_age_days"].dropna().value_counts().sort_index().rename_axis("days_ago").reset_index(name="job_count")
    if recency.empty:
        st.info("The source has no reliable posting dates for the current filters.")
    else:
        fig = px.line(recency, x="days_ago", y="job_count", markers=True, title="Listings by relative posting age", color_discrete_sequence=COLOR_SEQUENCE)
        fig.update_layout(xaxis_title="Days ago (relative label in source)", yaxis_title="Listings", margin=dict(l=20, r=20, t=55, b=20))
        st.plotly_chart(fig, width="stretch")

with st.expander("Methodology and scope"):
    st.markdown(
        """
        Counts use unique job IDs after deduplication. Salary KPIs use disclosed, positive INR annualized values only;
        missing, unpaid, zero, and USD listings are excluded from INR comparisons. Experience stages use minimum
        requested experience: strict Fresher = minimum 0 and maximum ≤2 years; Entry Level = minimum ≤2 years;
        Mid Level = minimum 3–5 years; Senior = minimum >5 years. Relative upload labels are shown as age, not
        fabricated calendar dates.
        """
    )

