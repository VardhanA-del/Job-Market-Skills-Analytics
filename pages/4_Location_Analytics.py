"""Location demand, salary, skill, company, and opportunity-score analytics."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.dashboard import bar_chart, configure_page, empty_state, load_dashboard_data, page_intro, render_filters
from src.metrics import opportunity_score


configure_page("Location Analytics", "📍")
page_intro("Location Analytics", "Each comma-separated source location is analyzed once per listing; filters use the listing's primary location.")
jobs, skills, locations = load_dashboard_data()
filtered, filtered_skills = render_filters(jobs, skills, "location")
if filtered.empty:
    empty_state()

loc_rows = locations[locations["job_id"].isin(filtered["job_id"])].merge(
    filtered[["job_id", "company_name", "normalized_job_role", "average_salary_lpa", "minimum_experience"]],
    on="job_id", how="inner",
)
counts = loc_rows.groupby("location")["job_id"].nunique().nlargest(20).rename("job_count").reset_index()
left, right = st.columns(2)
with left:
    fig = bar_chart(counts, "job_count", "location", "Top locations for job opportunities", horizontal=True)
    st.plotly_chart(fig, width="stretch")
with right:
    salary = loc_rows.groupby("location").agg(
        median_salary_lpa=("average_salary_lpa", "median"), salary_sample=("average_salary_lpa", "count")
    ).reset_index()
    salary = salary[salary["salary_sample"] >= 30].nlargest(20, "median_salary_lpa")
    if salary.empty:
        st.info("No locations meet the 30-salary minimum for this selection.")
    else:
        fig = bar_chart(salary, "median_salary_lpa", "location", "Median salary by location (minimum 30)", horizontal=True, hover_data=["salary_sample"])
        fig.update_xaxes(title="Median annual salary (LPA)")
        st.plotly_chart(fig, width="stretch")

top_locations = counts.head(10)["location"]
left, right = st.columns(2)
with left:
    companies = loc_rows[loc_rows["location"].isin(top_locations)].groupby(["location", "company_name"])["job_id"].nunique().rename("job_count").reset_index()
    top_company_names = companies.groupby("company_name")["job_count"].sum().nlargest(12).index
    company_matrix = companies[companies["company_name"].isin(top_company_names)].pivot(index="location", columns="company_name", values="job_count").fillna(0)
    if not company_matrix.empty:
        fig = px.imshow(company_matrix, text_auto=True, aspect="auto", title="Hiring companies across top locations", color_continuous_scale="Blues")
        st.plotly_chart(fig, width="stretch")
with right:
    loc_skill = locations[locations["job_id"].isin(filtered["job_id"])].merge(filtered_skills, on="job_id")
    top_skill_names = filtered_skills["skill"].value_counts().head(10).index
    loc_skill = loc_skill[loc_skill["location"].isin(top_locations) & loc_skill["skill"].isin(top_skill_names)]
    skill_matrix = loc_skill.pivot_table(index="location", columns="skill", values="job_id", aggfunc="nunique", fill_value=0)
    if not skill_matrix.empty:
        fig = px.imshow(skill_matrix, text_auto=True, aspect="auto", title="Skills across top locations", color_continuous_scale="Teal")
        st.plotly_chart(fig, width="stretch")

experience = loc_rows[loc_rows["location"].isin(top_locations)].groupby("location").agg(
    median_minimum_experience=("minimum_experience", "median"), sample_size=("minimum_experience", "count")
).reset_index()
fig = bar_chart(experience, "median_minimum_experience", "location", "Typical minimum experience across top locations", horizontal=True, hover_data=["sample_size"])
if fig:
    fig.update_xaxes(title="Median minimum experience (years)")
    st.plotly_chart(fig, width="stretch")

score_input = loc_rows.rename(columns={"location": "analysis_location"})
score = opportunity_score(score_input, "analysis_location", min_salary_n=30).head(20)
if not score.empty:
    st.subheader("Job Opportunity Score")
    st.dataframe(
        score[["analysis_location", "opportunity_score", "job_count", "median_salary_lpa", "salary_sample"]].style.format(
            {"opportunity_score": "{:.1f}", "median_salary_lpa": "{:.1f}"}
        ), width="stretch", hide_index=True,
    )
    st.caption("Score = 60% listing-volume percentile + 40% median-salary percentile, among locations with at least 30 disclosed INR salaries. It is a within-dataset ranking, not a real-world market index.")

