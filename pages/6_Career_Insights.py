"""Dataset-derived role explorer, skill recommender, and fresher analysis."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard import COLOR_SEQUENCE, bar_chart, configure_page, empty_state, load_dashboard_data, page_intro, render_filters
from src.metrics import get_top_companies, get_top_locations, get_top_roles, get_top_skills
from src.skills_processing import skill_cooccurrence
from src.utils import format_lpa


configure_page("Career Insights", "🧭")
page_intro("Career Insights", "Explore a target role and get transparent, frequency-based skill priorities from actual listings.")
jobs, skills, _ = load_dashboard_data()
filtered, filtered_skills = render_filters(jobs, skills, "career")
if filtered.empty:
    empty_state()

role_counts = get_top_roles(filtered, 50)
role_options = role_counts[role_counts["job_count"] >= 10]["normalized_job_role"].tolist()
target = st.selectbox("Target role", role_options, index=role_options.index("Data Analyst") if "Data Analyst" in role_options else 0)
role_jobs = filtered[filtered["normalized_job_role"].eq(target)]
role_skills = filtered_skills[filtered_skills["job_id"].isin(role_jobs["job_id"])]
top_skills = get_top_skills(role_skills, 20)

cols = st.columns(4)
cols[0].metric("Matching listings", f"{role_jobs['job_id'].nunique():,}")
cols[1].metric("Median disclosed salary", format_lpa(role_jobs["average_salary_lpa"].median()))
cols[2].metric("Median minimum experience", f"{role_jobs['minimum_experience'].median():.1f} yrs" if role_jobs["minimum_experience"].notna().any() else "N/A")
cols[3].metric("Top skill", top_skills.iloc[0]["skill"] if not top_skills.empty else "N/A")

left, right = st.columns(2)
with left:
    fig = bar_chart(top_skills.head(15), "job_count", "skill", f"Skills requested for {target}", horizontal=True, hover_data={"share_of_jobs": ":.1%"})
    if fig:
        st.plotly_chart(fig, width="stretch")
with right:
    pairs = skill_cooccurrence(role_skills, top_n=20).head(15)
    if pairs.empty:
        st.info("Not enough skill combinations for this role.")
    else:
        pairs["combination"] = pairs["skill_1"] + " + " + pairs["skill_2"]
        fig = bar_chart(pairs, "job_count", "combination", "Common skill combinations", horizontal=True)
        st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
with left:
    locations = get_top_locations(role_jobs, 10)
    fig = bar_chart(locations, "job_count", "primary_location", "Top hiring locations", horizontal=True)
    if fig:
        st.plotly_chart(fig, width="stretch")
with right:
    companies = get_top_companies(role_jobs, 10)
    fig = bar_chart(companies, "job_count", "company_name", "Top hiring companies", horizontal=True)
    if fig:
        st.plotly_chart(fig, width="stretch")

st.subheader("Skill gap recommender")
known = st.multiselect("Skills you already know", top_skills["skill"].tolist())
recommendations = top_skills[~top_skills["skill"].isin(known)].head(8).copy()
if recommendations.empty:
    st.success("Your selections cover the most frequently requested skills shown for this role.")
else:
    recommendations["priority"] = range(1, len(recommendations) + 1)
    st.dataframe(
        recommendations[["priority", "skill", "job_count", "share_of_jobs"]].style.format({"share_of_jobs": "{:.1%}"}),
        width="stretch", hide_index=True,
    )
    st.caption("Recommendations are the most frequent missing skills in matching listings; they are not claims about personal aptitude or guaranteed outcomes.")

st.divider()
st.subheader("Fresher and entry-level analysis")
strict_fresher = filtered[filtered["minimum_experience"].eq(0) & filtered["maximum_experience"].le(2)]
entry = filtered[filtered["minimum_experience"].le(2)]
mode = st.radio("Candidate scope", ["Strict fresher (0–2 years maximum)", "Entry level (minimum requirement ≤2 years)"], horizontal=True)
early_jobs = strict_fresher if mode.startswith("Strict") else entry
early_skills = filtered_skills[filtered_skills["job_id"].isin(early_jobs["job_id"])]
if early_jobs.empty:
    st.info("No listings match this experience rule and the active filters.")
else:
    cols = st.columns(3)
    cols[0].metric("Listings", f"{early_jobs['job_id'].nunique():,}")
    cols[1].metric("Median disclosed salary", format_lpa(early_jobs["average_salary_lpa"].median()))
    cols[2].metric("Top requested skill", get_top_skills(early_skills, 1).iloc[0]["skill"] if not early_skills.empty else "N/A")
    a, b, c = st.columns(3)
    with a:
        fig = bar_chart(get_top_roles(early_jobs, 10), "job_count", "normalized_job_role", "Entry roles", horizontal=True)
        if fig: st.plotly_chart(fig, width="stretch")
    with b:
        fig = bar_chart(get_top_skills(early_skills, 10), "job_count", "skill", "Skills to prioritize", horizontal=True)
        if fig: st.plotly_chart(fig, width="stretch")
    with c:
        fig = bar_chart(get_top_locations(early_jobs, 10), "job_count", "primary_location", "Hiring locations", horizontal=True)
        if fig: st.plotly_chart(fig, width="stretch")
    st.caption("The strict fresher rule requires minimum experience = 0 and maximum experience ≤2. The broader entry-level rule includes any listing whose minimum requirement is ≤2 years, even if its upper range is higher.")

