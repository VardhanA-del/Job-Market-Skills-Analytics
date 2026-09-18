"""Consistent KPI and grouped-analysis functions used across dashboard pages."""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_total_jobs(jobs: pd.DataFrame) -> int:
    return int(jobs["job_id"].nunique()) if "job_id" in jobs else len(jobs)


def get_total_companies(jobs: pd.DataFrame) -> int:
    return int(jobs["company_name"].nunique(dropna=True)) if "company_name" in jobs else 0


def get_total_locations(jobs: pd.DataFrame) -> int:
    return int(jobs["primary_location"].nunique(dropna=True)) if "primary_location" in jobs else 0


def get_median_salary(jobs: pd.DataFrame) -> float:
    values = jobs.get("average_salary_lpa", pd.Series(dtype=float)).dropna()
    return float(values.median()) if not values.empty else np.nan


def get_top_roles(jobs: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    if jobs.empty or "normalized_job_role" not in jobs:
        return pd.DataFrame(columns=["normalized_job_role", "job_count"])
    return jobs.groupby("normalized_job_role", dropna=True)["job_id"].nunique().nlargest(n).rename("job_count").reset_index()


def get_top_locations(jobs: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    if jobs.empty or "primary_location" not in jobs:
        return pd.DataFrame(columns=["primary_location", "job_count"])
    return jobs.groupby("primary_location", dropna=True)["job_id"].nunique().nlargest(n).rename("job_count").reset_index()


def get_top_companies(jobs: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    if jobs.empty or "company_name" not in jobs:
        return pd.DataFrame(columns=["company_name", "job_count"])
    return jobs.groupby("company_name", dropna=True)["job_id"].nunique().nlargest(n).rename("job_count").reset_index()


def get_top_skills(skill_rows: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    if skill_rows.empty:
        return pd.DataFrame(columns=["skill", "job_count", "share_of_jobs"])
    total_jobs = skill_rows["job_id"].nunique()
    result = skill_rows.groupby("skill")["job_id"].nunique().nlargest(n).rename("job_count").reset_index()
    result["share_of_jobs"] = result["job_count"] / total_jobs if total_jobs else 0
    return result


def grouped_job_counts(jobs: pd.DataFrame, field: str, n: int = 20) -> pd.DataFrame:
    if jobs.empty or field not in jobs:
        return pd.DataFrame(columns=[field, "job_count"])
    return jobs.groupby(field, dropna=True)["job_id"].nunique().nlargest(n).rename("job_count").reset_index()


def filter_related_rows(rows: pd.DataFrame, jobs: pd.DataFrame) -> pd.DataFrame:
    """Restrict an exploded table to the current set of job IDs."""
    if rows.empty or jobs.empty:
        return rows.iloc[0:0].copy()
    return rows[rows["job_id"].isin(jobs["job_id"])].copy()


def role_skill_matrix(jobs: pd.DataFrame, skills: pd.DataFrame, roles: int = 10, skill_count: int = 12) -> pd.DataFrame:
    if jobs.empty or skills.empty:
        return pd.DataFrame()
    top_roles = get_top_roles(jobs, roles)["normalized_job_role"]
    top_skills = get_top_skills(skills, skill_count)["skill"]
    merged = skills.merge(jobs[["job_id", "normalized_job_role"]], on="job_id", how="inner")
    subset = merged[merged["normalized_job_role"].isin(top_roles) & merged["skill"].isin(top_skills)]
    return subset.pivot_table(index="normalized_job_role", columns="skill", values="job_id", aggfunc="nunique", fill_value=0)


def opportunity_score(jobs: pd.DataFrame, field: str, min_salary_n: int = 20) -> pd.DataFrame:
    """Rank groups using 60% listing volume and 40% median INR salary percentile."""
    if jobs.empty or field not in jobs:
        return pd.DataFrame()
    grouped = jobs.groupby(field, dropna=True).agg(
        job_count=("job_id", "nunique"),
        median_salary_lpa=("average_salary_lpa", "median"),
        salary_sample=("average_salary_lpa", "count"),
    ).reset_index()
    grouped = grouped[grouped["salary_sample"] >= min_salary_n].copy()
    if grouped.empty:
        return grouped
    grouped["volume_percentile"] = grouped["job_count"].rank(pct=True)
    grouped["salary_percentile"] = grouped["median_salary_lpa"].rank(pct=True)
    grouped["opportunity_score"] = 100 * (0.60 * grouped["volume_percentile"] + 0.40 * grouped["salary_percentile"])
    return grouped.sort_values("opportunity_score", ascending=False, ignore_index=True)

