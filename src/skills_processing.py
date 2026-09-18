"""Skill extraction, normalization, and co-occurrence analysis."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
import re
from typing import Any, Iterable

import pandas as pd


SKILL_ALIASES = {
    "python3": "Python", "python 3": "Python", "python": "Python",
    "structured query language": "SQL", "sql": "SQL", "ms sql": "SQL",
    "powerbi": "Power BI", "power bi": "Power BI", "ms power bi": "Power BI",
    "tableau": "Tableau", "ms excel": "Excel", "microsoft excel": "Excel", "excel": "Excel",
    "postgres": "PostgreSQL", "postgresql": "PostgreSQL", "machine learning": "Machine Learning",
    "ml": "Machine Learning", "amazon web services": "AWS", "aws": "AWS",
    "microsoft azure": "Azure", "azure": "Azure", "google cloud platform": "GCP", "gcp": "GCP",
    "pyspark": "PySpark", "apache spark": "Spark", "spark": "Spark", "apache hadoop": "Hadoop",
    "hadoop": "Hadoop", "snowflake": "Snowflake", "dbt": "dbt", "etl": "ETL",
    "statistics": "Statistics", "statistical analysis": "Statistics", "deep learning": "Deep Learning",
    "natural language processing": "NLP", "nlp": "NLP", "numpy": "NumPy", "pandas": "Pandas",
    "docker": "Docker", "kubernetes": "Kubernetes", "git": "Git", "github": "GitHub",
    "r programming": "R", "r language": "R", "artificial intelligence": "AI",
    "m.s office": "MS Office", "ms office": "MS Office", "microsoft office": "MS Office",
    "devops": "DevOps", "azure devops": "Azure DevOps", "communication skills": "Communication",
}

CORE_SKILL_PATTERNS = {
    "Python": r"(?<![\w])python(?:\s*3)?(?![\w])",
    "SQL": r"(?<![\w])sql(?![\w])|structured query language",
    "Excel": r"(?<![\w])(?:ms |microsoft )?excel(?![\w])",
    "Power BI": r"(?<![\w])power\s*bi(?![\w])",
    "Tableau": r"(?<![\w])tableau(?![\w])",
    "R": r"\br programming\b|\br language\b",
    "Machine Learning": r"\bmachine learning\b",
    "Pandas": r"\bpandas\b", "NumPy": r"\bnumpy\b", "AWS": r"\baws\b|amazon web services",
    "Azure": r"\bazure\b", "GCP": r"\bgcp\b|google cloud platform",
    "Spark": r"\b(?:apache )?spark\b", "Hadoop": r"\bhadoop\b", "Snowflake": r"\bsnowflake\b",
    "dbt": r"(?<![\w])dbt(?![\w])", "ETL": r"(?<![\w])etl(?![\w])", "Statistics": r"\bstatistic(?:s|al)\b",
    "Deep Learning": r"\bdeep learning\b", "NLP": r"(?<![\w])nlp(?![\w])|natural language processing",
    "Docker": r"\bdocker\b", "Kubernetes": r"\bkubernetes\b", "Git": r"(?<![\w])git(?![\w])",
}


def normalize_skill(value: Any) -> str | None:
    """Return a conservative canonical label for one skill token."""
    if pd.isna(value):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip(" .,-_/|")
    if len(text) < 2:
        return None
    canonical = SKILL_ALIASES.get(text.casefold())
    if canonical:
        return canonical
    if len(text) > 80:
        return None
    return text.upper() if len(text) <= 4 and text.isalpha() else text.title()


def extract_skills(skill_text: Any, description: Any = None) -> list[str]:
    """Extract unique skills from tags and augment core technologies from description."""
    found: list[str] = []
    if pd.notna(skill_text):
        for token in re.split(r"[,;|\n]+", str(skill_text)):
            normalized = normalize_skill(token)
            if normalized:
                found.append(normalized)
    combined = " ".join(str(item) for item in (skill_text, description) if pd.notna(item)).casefold()
    for skill, pattern in CORE_SKILL_PATTERNS.items():
        if re.search(pattern, combined, flags=re.I):
            found.append(skill)
    return list(dict.fromkeys(found))


def explode_skills(jobs: pd.DataFrame) -> pd.DataFrame:
    """Create one unique job-skill row per listing."""
    records: list[tuple[int, str]] = []
    for row in jobs[["job_id", "skills_raw", "job_description"]].itertuples(index=False):
        records.extend((row.job_id, skill) for skill in extract_skills(row.skills_raw, row.job_description))
    return pd.DataFrame(records, columns=["job_id", "skill"]).drop_duplicates(ignore_index=True)


def skill_cooccurrence(skill_rows: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Count unordered skill pairs once per listing, restricted to frequent skills."""
    if skill_rows.empty:
        return pd.DataFrame(columns=["skill_1", "skill_2", "job_count"])
    top = set(skill_rows["skill"].value_counts().head(top_n).index)
    counter: Counter[tuple[str, str]] = Counter()
    for skills in skill_rows.loc[skill_rows["skill"].isin(top)].groupby("job_id")["skill"]:
        counter.update(combinations(sorted(set(skills[1])), 2))
    return pd.DataFrame(
        [(a, b, count) for (a, b), count in counter.items()],
        columns=["skill_1", "skill_2", "job_count"],
    ).sort_values("job_count", ascending=False, ignore_index=True)

