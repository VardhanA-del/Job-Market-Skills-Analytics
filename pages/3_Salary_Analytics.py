"""Salary distribution and comparison analytics."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.dashboard import COLOR_SEQUENCE, bar_chart, configure_page, empty_state, load_dashboard_data, page_intro, render_filters
from src.salary_analysis import salary_by_group, salary_by_skill, valid_inr_salaries


configure_page("Salary Analytics", "💰")
page_intro("Salary Analytics", "Currency-aware salary comparisons using positive, disclosed INR values; missing salaries are never treated as zero.")
jobs, skills, _ = load_dashboard_data()
filtered, filtered_skills = render_filters(jobs, skills, "salary")
if filtered.empty:
    empty_state()
salaries = valid_inr_salaries(filtered)
if salaries.empty:
    st.info("No disclosed INR salaries are available for this selection.")
    st.stop()

cols = st.columns(4)
cols[0].metric("Salary sample", f"{len(salaries):,}")
cols[1].metric("Median salary", f"₹{salaries['average_salary_lpa'].median():.1f} LPA")
cols[2].metric("25th percentile", f"₹{salaries['average_salary_lpa'].quantile(.25):.1f} LPA")
cols[3].metric("75th percentile", f"₹{salaries['average_salary_lpa'].quantile(.75):.1f} LPA")

left, right = st.columns(2)
with left:
    p99 = salaries["average_salary_lpa"].quantile(.99)
    fig = px.histogram(salaries[salaries["average_salary_lpa"] <= p99], x="average_salary_lpa", nbins=50, title="Salary distribution (display capped at 99th percentile)", color_discrete_sequence=COLOR_SEQUENCE)
    fig.update_xaxes(title="Average annual salary (LPA)")
    st.plotly_chart(fig, width="stretch")
with right:
    top_roles = salaries["normalized_job_role"].value_counts().head(12).index
    box = salaries[salaries["normalized_job_role"].isin(top_roles) & salaries["average_salary_lpa"].le(p99)]
    fig = px.box(box, x="normalized_job_role", y="average_salary_lpa", points=False, title="Salary spread across major roles (display capped at 99th percentile)", color_discrete_sequence=COLOR_SEQUENCE)
    fig.update_xaxes(title="", tickangle=-35)
    fig.update_yaxes(title="Average annual salary (LPA)")
    st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
with left:
    by_role = salary_by_group(filtered, "normalized_job_role", min_n=30, n=15)
    fig = bar_chart(by_role, "median_salary_lpa", "normalized_job_role", "Highest-paying role groups (minimum 30 salaries)", horizontal=True, hover_data=["sample_size"])
    if fig:
        fig.update_xaxes(title="Median annual salary (LPA)")
        st.plotly_chart(fig, width="stretch")
with right:
    by_location = salary_by_group(filtered, "primary_location", min_n=30, n=15)
    fig = bar_chart(by_location, "median_salary_lpa", "primary_location", "Highest-paying primary locations (minimum 30 salaries)", horizontal=True, hover_data=["sample_size"])
    if fig:
        fig.update_xaxes(title="Median annual salary (LPA)")
        st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
with left:
    by_exp = salaries.groupby("minimum_experience").agg(median_salary_lpa=("average_salary_lpa", "median"), sample_size=("job_id", "nunique")).reset_index()
    by_exp = by_exp[by_exp["sample_size"] >= 20]
    fig = px.scatter(by_exp, x="minimum_experience", y="median_salary_lpa", size="sample_size", trendline=None, title="Median salary vs minimum experience", color_discrete_sequence=COLOR_SEQUENCE)
    fig.update_xaxes(title="Minimum experience (years)")
    fig.update_yaxes(title="Median annual salary (LPA)")
    st.plotly_chart(fig, width="stretch")
with right:
    by_skill = salary_by_skill(filtered, filtered_skills, min_n=30, n=15)
    fig = bar_chart(by_skill, "median_salary_lpa", "skill", "Highest-paying skills (minimum 30 salaries)", horizontal=True, hover_data=["sample_size"])
    if fig:
        fig.update_xaxes(title="Median annual salary (LPA)")
        st.plotly_chart(fig, width="stretch")

by_company = salary_by_group(filtered, "company_name", min_n=20, n=15)
fig = bar_chart(by_company, "median_salary_lpa", "company_name", "Highest-paying companies (minimum 20 disclosed salaries)", horizontal=True, hover_data=["sample_size"])
if fig:
    fig.update_xaxes(title="Median annual salary (LPA)")
    st.plotly_chart(fig, width="stretch")

with st.expander("Salary methodology"):
    st.write("Lac/Lakh/LPA values are multiplied by 100,000; reliably marked monthly values are multiplied by 12. Mixed ranges retain explicit rupee endpoints. Not disclosed, unpaid, and zero values become missing. USD records remain in native units and are excluded from INR/LPA charts because no historical exchange-rate context is provided. Median comparisons enforce minimum sample sizes shown in chart titles.")

