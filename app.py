"""Streamlit application router."""

import streamlit as st


pages = [
    st.Page("pages/1_Job_Market_Overview.py", title="Job Market Overview", icon="📊", default=True),
    st.Page("pages/2_Skills_Analytics.py", title="Skills Analytics", icon="🧰"),
    st.Page("pages/3_Salary_Analytics.py", title="Salary Analytics", icon="💰"),
    st.Page("pages/4_Location_Analytics.py", title="Location Analytics", icon="📍"),
    st.Page("pages/5_Company_Analytics.py", title="Company Analytics", icon="🏢"),
    st.Page("pages/6_Career_Insights.py", title="Career Insights", icon="🧭"),
]

st.navigation(pages).run()

