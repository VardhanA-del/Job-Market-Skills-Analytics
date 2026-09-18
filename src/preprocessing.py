"""End-to-end preprocessing pipeline for job-market data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_cleaning import (
    clean_html_text, clean_text, experience_category, normalize_column_name,
    normalize_job_role, normalize_location, parse_experience, parse_posting_age_days,
    parse_salary,
)
from src.data_loader import DATA_DIR, PROCESSED_DIR, load_raw_jobs
from src.skills_processing import explode_skills


REQUIRED_COLUMNS = {"title", "job_id", "company_name", "location"}


def _deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    exact = int(frame.duplicated().sum())
    if "job_id" not in frame:
        result = frame.drop_duplicates().copy()
        return result, {"exact_duplicates_removed": exact, "duplicate_job_ids_removed": 0}
    completeness = frame.notna().sum(axis=1)
    result = (
        frame.assign(_completeness=completeness)
        .sort_values(["job_id", "_completeness"], ascending=[True, False])
        .drop_duplicates("job_id", keep="first")
        .drop(columns="_completeness")
        .copy()
    )
    return result, {"exact_duplicates_removed": exact, "duplicate_job_ids_removed": int(len(frame) - len(result))}


def _split_locations(value: object) -> list[str]:
    if pd.isna(value):
        return ["Not specified"]
    tokens = [normalize_location(token) for token in str(value).split(",")]
    return list(dict.fromkeys(token for token in tokens if token)) or ["Not specified"]


def preprocess_jobs(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Clean raw jobs and return job, skill, location tables plus a quality report."""
    original_rows = len(raw)
    frame = raw.copy()
    frame.columns = [normalize_column_name(column) for column in frame.columns]
    missing_required = REQUIRED_COLUMNS - set(frame.columns)
    if missing_required:
        raise ValueError(f"Required columns are unavailable: {sorted(missing_required)}")
    frame, duplicate_report = _deduplicate(frame)

    text_columns = ["title", "company_name", "currency", "salary", "experience", "location", "tags_and_skills"]
    for column in text_columns:
        if column in frame:
            frame[column] = frame[column].map(clean_text)
    if "job_description" in frame:
        frame["job_description"] = frame["job_description"].map(clean_html_text)
    else:
        frame["job_description"] = pd.NA

    frame["original_job_title"] = frame["title"]
    frame["normalized_job_role"] = frame["title"].map(normalize_job_role)
    frame["skills_raw"] = frame.get("tags_and_skills", pd.Series(pd.NA, index=frame.index))
    frame["original_location"] = frame["location"]
    location_lists = frame["location"].map(_split_locations)
    frame["primary_location"] = location_lists.str[0]
    frame["location_count"] = location_lists.str.len().astype("int16")

    parsed_salary = frame["salary"].map(parse_salary)
    frame["minimum_salary"] = [item[0] for item in parsed_salary]
    frame["maximum_salary"] = [item[1] for item in parsed_salary]
    frame["average_salary"] = frame[["minimum_salary", "maximum_salary"]].mean(axis=1)
    frame["salary_disclosed"] = frame["average_salary"].notna()
    inr = frame.get("currency", pd.Series("INR", index=frame.index)).eq("INR")
    frame["minimum_salary_lpa"] = frame["minimum_salary"].where(inr) / 100_000
    frame["maximum_salary_lpa"] = frame["maximum_salary"].where(inr) / 100_000
    frame["average_salary_lpa"] = frame["average_salary"].where(inr) / 100_000

    parsed_experience = frame.get("experience", pd.Series(pd.NA, index=frame.index)).map(parse_experience)
    raw_min = pd.to_numeric(frame.get("minimum_experience"), errors="coerce")
    raw_max = pd.to_numeric(frame.get("maximum_experience"), errors="coerce")
    parsed_min = pd.Series([item[0] for item in parsed_experience], index=frame.index)
    parsed_max = pd.Series([item[1] for item in parsed_experience], index=frame.index)
    frame["minimum_experience"] = raw_min.combine_first(parsed_min)
    frame["maximum_experience"] = raw_max.combine_first(parsed_max)
    frame["average_experience"] = frame[["minimum_experience", "maximum_experience"]].mean(axis=1)
    frame["experience_category"] = [
        experience_category(a, b) for a, b in zip(frame["minimum_experience"], frame["maximum_experience"])
    ]

    frame["posting_age_days"] = frame.get("job_uploaded", pd.Series(pd.NA, index=frame.index)).map(parse_posting_age_days)
    frame["company_rating"] = pd.to_numeric(frame.get("aggregate_rating"), errors="coerce").where(lambda x: x.between(1, 5))
    frame["reviews_count"] = pd.to_numeric(frame.get("reviews_count"), errors="coerce").where(lambda x: x.ge(0))
    frame["company_name"] = frame["company_name"].map(clean_text)
    frame["job_id"] = pd.to_numeric(frame["job_id"], errors="coerce").astype("Int64")

    skills = explode_skills(frame)
    location_records = [
        (job_id, location)
        for job_id, locations in zip(frame["job_id"], location_lists)
        for location in locations
    ]
    locations = pd.DataFrame(location_records, columns=["job_id", "location"]).drop_duplicates(ignore_index=True)

    keep = [
        "job_id", "original_job_title", "normalized_job_role", "company_name", "company_id",
        "currency", "salary", "minimum_salary", "maximum_salary", "average_salary",
        "minimum_salary_lpa", "maximum_salary_lpa", "average_salary_lpa", "salary_disclosed",
        "experience", "minimum_experience", "maximum_experience", "average_experience",
        "experience_category", "original_location", "primary_location", "location_count",
        "company_rating", "reviews_count", "job_uploaded", "posting_age_days", "skills_raw",
        "job_description",
    ]
    for column in keep:
        if column not in frame:
            frame[column] = pd.NA
    jobs = frame[keep].reset_index(drop=True)
    report = {
        "raw_rows": original_rows,
        "clean_rows": len(jobs),
        **duplicate_report,
        "salary_disclosed_rows": int(jobs["salary_disclosed"].sum()),
        "salary_disclosed_inr_rows": int(jobs["average_salary_lpa"].notna().sum()),
        "usd_rows": int(jobs["currency"].eq("USD").sum()),
        "jobs_with_experience": int(jobs["minimum_experience"].notna().sum()),
        "jobs_with_skills": int(jobs["job_id"].isin(skills["job_id"]).sum()),
        "skill_rows": len(skills),
        "location_rows": len(locations),
    }
    return jobs, skills, locations, report


def build_processed_data(data_dir: Path = DATA_DIR, output_dir: Path = PROCESSED_DIR) -> dict:
    """Load, clean, and write cache-friendly Parquet outputs without touching raw files."""
    raw, source = load_raw_jobs(data_dir)
    jobs, skills, locations, report = preprocess_jobs(raw)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs.to_parquet(output_dir / "clean_jobs.parquet", index=False)
    skills.to_parquet(output_dir / "job_skills.parquet", index=False)
    locations.to_parquet(output_dir / "job_locations.parquet", index=False)
    metadata = {**source, **report}
    (output_dir / "processing_report.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    print(json.dumps(build_processed_data(), indent=2))

