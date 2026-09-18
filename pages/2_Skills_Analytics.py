"""Skills demand, role, salary, and co-occurrence analytics."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard import COLOR_SEQUENCE, bar_chart, configure_page, empty_state, load_dashboard_data, page_intro, render_filters
from src.metrics import get_top_skills, role_skill_matrix
from src.salary_analysis import salary_by_skill
from src.skills_processing import skill_cooccurrence


configure_page("Skills Analytics", "🧰")
page_intro("Skills Analytics", "Demand, combinations, role fit, location patterns, and salary associations—counted once per skill per listing.")
jobs, skills, locations = load_dashboard_data()
filtered, filtered_skills = render_filters(jobs, skills, "skills")
if filtered.empty:
    empty_state()

top = get_top_skills(filtered_skills, 25)
left, right = st.columns([1.05, 0.95])
with left:
    fig = bar_chart(top, "job_count", "skill", "Most requested skills", horizontal=True, hover_data={"share_of_jobs": ":.1%"})
    if fig:
        st.plotly_chart(fig, width="stretch")
with right:
    matrix = role_skill_matrix(filtered, filtered_skills, roles=10, skill_count=12)
    if matrix.empty:
        st.info("Not enough data for the role-skill matrix.")
    else:
        fig = px.imshow(matrix, text_auto=True, aspect="auto", title="Skill demand by role group", color_continuous_scale="Teal")
        fig.update_layout(xaxis_title="Skill", yaxis_title="Role group", margin=dict(l=20, r=20, t=55, b=20))
        st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
with left:
    pairs = skill_cooccurrence(filtered_skills, top_n=20).head(20)
    if pairs.empty:
        st.info("Not enough skill pairs for the current filters.")
    else:
        pairs["combination"] = pairs["skill_1"] + " + " + pairs["skill_2"]
        fig = bar_chart(pairs, "job_count", "combination", "Frequent skill combinations", horizontal=True)
        st.plotly_chart(fig, width="stretch")
with right:
    paid_skills = salary_by_skill(filtered, filtered_skills, min_n=20, n=20)
    if paid_skills.empty:
        st.info("At least 20 disclosed INR salaries per skill are required for comparison.")
    else:
        fig = bar_chart(paid_skills, "median_salary_lpa", "skill", "Highest median salary by skill (minimum 20 listings)", horizontal=True, hover_data=["sample_size"])
        fig.update_xaxes(title="Median annual salary (LPA)")
        st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
with left:
    merged = filtered_skills.merge(filtered[["job_id", "experience_category"]], on="job_id")
    exp_skills = merged[merged["skill"].isin(top.head(12)["skill"])]
    exp_matrix = exp_skills.pivot_table(index="experience_category", columns="skill", values="job_id", aggfunc="nunique", fill_value=0)
    if not exp_matrix.empty:
        order = [x for x in ["Fresher", "Entry Level", "Mid Level", "Senior", "Not specified"] if x in exp_matrix.index]
        fig = px.imshow(exp_matrix.reindex(order), text_auto=True, aspect="auto", title="Top skills by experience level", color_continuous_scale="Blues")
        st.plotly_chart(fig, width="stretch")
with right:
    top_locations = filtered["primary_location"].value_counts().head(10).index
    loc_skills = filtered_skills.merge(filtered[["job_id", "primary_location"]], on="job_id")
    loc_skills = loc_skills[loc_skills["primary_location"].isin(top_locations) & loc_skills["skill"].isin(top.head(10)["skill"])]
    loc_matrix = loc_skills.pivot_table(index="primary_location", columns="skill", values="job_id", aggfunc="nunique", fill_value=0)
    if not loc_matrix.empty:
        fig = px.imshow(loc_matrix, text_auto=True, aspect="auto", title="Top skills by primary location", color_continuous_scale="Purples")
        st.plotly_chart(fig, width="stretch")

analyst = filtered[filtered["normalized_job_role"].eq("Data Analyst")]
if analyst["job_id"].nunique() >= 30:
    st.subheader("Data Analyst skill focus")
    analyst_skills = filtered_skills[filtered_skills["job_id"].isin(analyst["job_id"])]
    core = ["SQL", "Python", "Excel", "Power BI", "Tableau", "Statistics", "ETL", "AWS", "Azure", "Snowflake"]
    comparison = analyst_skills[analyst_skills["skill"].isin(core)].groupby("skill")["job_id"].nunique().reindex(core, fill_value=0).rename("job_count").reset_index()
    comparison["share_of_data_analyst_jobs"] = comparison["job_count"] / analyst["job_id"].nunique()
    fig = bar_chart(comparison, "job_count", "skill", "Core technology demand in Data Analyst listings", horizontal=True, hover_data={"share_of_data_analyst_jobs": ":.1%"})
    st.plotly_chart(fig, width="stretch")
    sql_count = int(comparison.loc[comparison["skill"].eq("SQL"), "job_count"].sum())
    python_count = int(comparison.loc[comparison["skill"].eq("Python"), "job_count"].sum())
    pbi_count = int(comparison.loc[comparison["skill"].eq("Power BI"), "job_count"].sum())
    tableau_count = int(comparison.loc[comparison["skill"].eq("Tableau"), "job_count"].sum())
    st.caption(f"In the filtered Data Analyst listings: SQL appears in {sql_count:,} jobs vs Python in {python_count:,}; Power BI appears in {pbi_count:,} vs Tableau in {tableau_count:,}.")

with st.expander("How skill analysis works"):
    st.write("Comma/semicolon/pipe-separated tags are normalized conservatively, then core technologies are also detected in descriptions. Variants such as python3, Microsoft Excel, PowerBI, and Amazon Web Services map to canonical labels. Each skill and each pair is counted at most once per job.")

