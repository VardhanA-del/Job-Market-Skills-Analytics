"""Company hiring, role, skill, rating, salary, and experience analytics."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.dashboard import COLOR_SEQUENCE, bar_chart, configure_page, empty_state, load_dashboard_data, page_intro, render_filters
from src.metrics import get_top_companies
from src.salary_analysis import salary_by_group


configure_page("Company Analytics", "🏢")
page_intro("Company Analytics", "Hiring patterns within the source snapshot, with minimum samples for salary comparisons.")
jobs, skills, _ = load_dashboard_data()
filtered, filtered_skills = render_filters(jobs, skills, "company")
if filtered.empty:
    empty_state()

top = get_top_companies(filtered, 20)
top_names = top.head(12)["company_name"]
left, right = st.columns(2)
with left:
    fig = bar_chart(top, "job_count", "company_name", "Companies with the most listings", horizontal=True)
    st.plotly_chart(fig, width="stretch")
with right:
    rated = filtered["company_rating"].dropna()
    if rated.empty:
        st.info("No company ratings are available for this selection.")
    else:
        fig = px.histogram(filtered, x="company_rating", nbins=20, title="Company rating distribution (listing-weighted)", color_discrete_sequence=COLOR_SEQUENCE)
        fig.update_xaxes(title="Aggregate company rating")
        st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
with left:
    company_roles = filtered[filtered["company_name"].isin(top_names)]
    top_roles = company_roles["normalized_job_role"].value_counts().head(10).index
    role_matrix = company_roles[company_roles["normalized_job_role"].isin(top_roles)].pivot_table(
        index="company_name", columns="normalized_job_role", values="job_id", aggfunc="nunique", fill_value=0
    )
    if not role_matrix.empty:
        fig = px.imshow(role_matrix, text_auto=True, aspect="auto", title="Company vs role group", color_continuous_scale="Blues")
        st.plotly_chart(fig, width="stretch")
with right:
    company_skills = filtered_skills.merge(filtered[["job_id", "company_name"]], on="job_id")
    company_skills = company_skills[company_skills["company_name"].isin(top_names)]
    top_skill_names = company_skills["skill"].value_counts().head(10).index
    skill_matrix = company_skills[company_skills["skill"].isin(top_skill_names)].pivot_table(
        index="company_name", columns="skill", values="job_id", aggfunc="nunique", fill_value=0
    )
    if not skill_matrix.empty:
        fig = px.imshow(skill_matrix, text_auto=True, aspect="auto", title="Company vs requested skill", color_continuous_scale="Teal")
        st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
with left:
    salary = salary_by_group(filtered, "company_name", min_n=20, n=15)
    if salary.empty:
        st.info("No companies meet the 20-salary minimum.")
    else:
        fig = bar_chart(salary, "median_salary_lpa", "company_name", "Median salary by company (minimum 20)", horizontal=True, hover_data=["sample_size"])
        fig.update_xaxes(title="Median annual salary (LPA)")
        st.plotly_chart(fig, width="stretch")
with right:
    exp = filtered[filtered["company_name"].isin(top_names)].groupby("company_name").agg(
        median_min_experience=("minimum_experience", "median"), experience_sample=("minimum_experience", "count")
    ).reset_index()
    fig = bar_chart(exp, "median_min_experience", "company_name", "Experience demand at top hiring companies", horizontal=True, hover_data=["experience_sample"])
    if fig:
        fig.update_xaxes(title="Median minimum experience (years)")
        st.plotly_chart(fig, width="stretch")

st.caption("Ratings and review counts come from the source platform. The rating histogram is listing-weighted because the dataset repeats company-level ratings across listings.")

