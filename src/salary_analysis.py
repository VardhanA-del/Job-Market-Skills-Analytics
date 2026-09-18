"""Salary-specific analyses with sample-size safeguards."""

from __future__ import annotations

import pandas as pd


def valid_inr_salaries(jobs: pd.DataFrame) -> pd.DataFrame:
    """Return only positive, disclosed INR annual salaries."""
    if "average_salary_lpa" not in jobs:
        return jobs.iloc[0:0].copy()
    return jobs[jobs["average_salary_lpa"].gt(0) & jobs["average_salary_lpa"].notna()].copy()


def salary_by_group(jobs: pd.DataFrame, field: str, min_n: int = 20, n: int = 20) -> pd.DataFrame:
    salaries = valid_inr_salaries(jobs)
    if salaries.empty or field not in salaries:
        return pd.DataFrame(columns=[field, "median_salary_lpa", "mean_salary_lpa", "sample_size"])
    result = salaries.groupby(field, dropna=True).agg(
        median_salary_lpa=("average_salary_lpa", "median"),
        mean_salary_lpa=("average_salary_lpa", "mean"),
        sample_size=("job_id", "nunique"),
    ).reset_index()
    return result[result["sample_size"] >= min_n].nlargest(n, "median_salary_lpa").reset_index(drop=True)


def salary_by_skill(jobs: pd.DataFrame, skills: pd.DataFrame, min_n: int = 20, n: int = 20) -> pd.DataFrame:
    salaries = valid_inr_salaries(jobs)[["job_id", "average_salary_lpa"]]
    merged = skills.merge(salaries, on="job_id", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["skill", "median_salary_lpa", "sample_size"])
    result = merged.groupby("skill").agg(
        median_salary_lpa=("average_salary_lpa", "median"),
        mean_salary_lpa=("average_salary_lpa", "mean"),
        sample_size=("job_id", "nunique"),
    ).reset_index()
    return result[result["sample_size"] >= min_n].nlargest(n, "median_salary_lpa").reset_index(drop=True)

