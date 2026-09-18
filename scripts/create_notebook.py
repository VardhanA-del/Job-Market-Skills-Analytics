"""Generate a self-contained Google Colab-compatible EDA notebook."""

from pathlib import Path

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


cells = [
    md(
        """# Job Market & Skills Analytics — Colab Exploratory Analysis

This notebook is **self-contained**: it does not import the project's local `src` package. In Google Colab, upload the dataset workbook directly into `/content`, then run all cells in order.

The notebook discovers the uploaded filename automatically, audits every supported table, cleans the selected job-listing table, and performs role, skill, salary, experience, location, company, Data Analyst, and fresher analysis. Results describe this dataset snapshot—not the entire Indian labor market."""
    ),
    md(
        """## 1. Environment setup

Colab normally includes Pandas, NumPy, and Plotly. This cell installs only packages that are missing. When run locally, the notebook falls back to `dataset/`."""
    ),
    code(
        """from pathlib import Path
import importlib.util
import subprocess
import sys

REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "plotly": "plotly",
    "openpyxl": "openpyxl",
    "pyarrow": "pyarrow",
}
missing = [package for module, package in REQUIRED_PACKAGES.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

import html
import json
import math
import re
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "colab" if importlib.util.find_spec("google.colab") else "notebook_connected"
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 120)

IN_COLAB = importlib.util.find_spec("google.colab") is not None
DATA_DIR = Path("/content") if IN_COLAB else Path("dataset")
OUTPUT_DIR = Path("/content/job_market_outputs") if IN_COLAB else Path("data/processed/notebook")

print("Running in Colab:", IN_COLAB)
print("Dataset directory:", DATA_DIR.resolve())
print("Output directory:", OUTPUT_DIR.resolve())"""
    ),
    md(
        """## 2. Dataset discovery

Upload the workbook using Colab's **Files** panel before running this cell. If no supported file is present, Colab will open its upload dialog. The original file is read only and never overwritten."""
    ),
    code(
        """SUPPORTED_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xlsm", ".json", ".jsonl", ".parquet"}

def discover_files(data_dir=DATA_DIR):
    return sorted(
        path for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )

dataset_files = discover_files()
if not dataset_files and IN_COLAB:
    from google.colab import files
    print("No supported dataset found in /content. Select the dataset file to upload.")
    files.upload()
    dataset_files = discover_files()

if not dataset_files:
    raise FileNotFoundError(f"No CSV, TSV, Excel, JSON, or Parquet files found in {DATA_DIR.resolve()}")

inventory = pd.DataFrame([
    {"file": path.name, "type": path.suffix.lower(), "size_mb": path.stat().st_size / 1_000_000}
    for path in dataset_files
])
display(inventory.style.format({"size_mb": "{:.2f}"}))"""
    ),
    md(
        """## 3. Load and inspect every table

Each sheet or file is profiled first. The most job-like table is selected using its schema and row count rather than a hardcoded filename."""
    ),
    code(
        """def load_tables(path):
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        workbook = pd.ExcelFile(path)
        return {sheet: pd.read_excel(path, sheet_name=sheet) for sheet in workbook.sheet_names}
    if suffix == ".csv":
        return {path.stem: pd.read_csv(path, low_memory=False)}
    if suffix == ".tsv":
        return {path.stem: pd.read_csv(path, sep="\\t", low_memory=False)}
    if suffix in {".json", ".jsonl"}:
        return {path.stem: pd.read_json(path, lines=suffix == ".jsonl")}
    if suffix == ".parquet":
        return {path.stem: pd.read_parquet(path)}
    return {}

tables = {}
for path in dataset_files:
    for table_name, frame in load_tables(path).items():
        tables[(path.name, table_name)] = frame

table_summary = pd.DataFrame([
    {"file": file_name, "table": table_name, "rows": len(frame), "columns": frame.shape[1]}
    for (file_name, table_name), frame in tables.items()
]).sort_values(["rows", "columns"], ascending=False)
display(table_summary)

EXPECTED_FIELDS = {
    "title", "jobtitle", "jobid", "companyname", "company", "location",
    "salary", "experience", "tagsandskills", "skills", "jobdescription",
}

def schema_score(frame):
    compact = {re.sub(r"[^a-z0-9]", "", str(column).lower()) for column in frame.columns}
    return len(compact & EXPECTED_FIELDS)

candidates = [
    (schema_score(frame), len(frame), file_name, table_name, frame)
    for (file_name, table_name), frame in tables.items()
    if schema_score(frame) >= 2
]
if not candidates:
    raise ValueError("No job-listing table could be identified from the uploaded files.")

_, _, source_file, source_table, raw = max(candidates, key=lambda item: (item[0], item[1]))
print(f"Selected source: {source_file} — {source_table}")
print("Shape:", raw.shape)
display(raw.head())"""
    ),
    md(
        """## 4. Schema, missing values, duplicates, and samples

This audit establishes which analyses are reliable before any transformation is performed."""
    ),
    code(
        """schema = pd.DataFrame({
    "column": raw.columns,
    "dtype": raw.dtypes.astype(str).values,
    "missing": raw.isna().sum().values,
    "missing_pct": raw.isna().mean().values,
    "unique_values": raw.nunique(dropna=True).values,
})
display(schema.style.format({"missing_pct": "{:.1%}"}))

compact_columns = {re.sub(r"[^a-z0-9]", "", str(column).lower()): column for column in raw.columns}
job_id_column = compact_columns.get("jobid")
print("Exact duplicate rows:", int(raw.duplicated().sum()))
if job_id_column:
    print("Duplicate job IDs beyond the first:", int(raw[job_id_column].duplicated().sum()))
display(raw.sample(min(5, len(raw)), random_state=42))"""
    ),
    md(
        """## 5. Reusable cleaning functions

Rules are deliberately conservative. Original values remain available beside normalized analytical fields. Missing salaries are never replaced with zero."""
    ),
    code(
        """MISSING_TEXT = {"", "nan", "none", "null", "n/a", "na", "not available"}

def normalize_column_name(name):
    text = re.sub(r"([a-z0-9])([A-Z])", r"\\1_\\2", str(name).strip())
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()

def clean_text(value):
    if pd.isna(value):
        return pd.NA
    text = re.sub(r"\\s+", " ", str(value)).strip()
    return pd.NA if text.casefold() in MISSING_TEXT else text

def clean_html_text(value):
    if pd.isna(value):
        return pd.NA
    text = re.sub(r"<\\s*br\\s*/?>|</p>|</li>", " ", str(value), flags=re.I)
    return clean_text(html.unescape(re.sub(r"<[^>]+>", " ", text)))

def salary_number(token, unit=None):
    number = float(token.replace(",", ""))
    unit = (unit or "").casefold()
    if unit.startswith(("lac", "lakh")) or unit == "lpa":
        number *= 100_000
    elif unit.startswith(("cr", "crore")):
        number *= 10_000_000
    elif unit in {"k", "thousand"}:
        number *= 1_000
    return number

def parse_salary(value):
    if pd.isna(value):
        return np.nan, np.nan
    text = str(value).strip().lower()
    if any(term in text for term in ("not disclosed", "unpaid", "negotiable", "not specified")):
        return np.nan, np.nan
    matches = re.findall(
        r"(?<![a-z])(\\d[\\d,]*(?:\\.\\d+)?)\\s*(lacs?|lakhs?|lpa|crores?|cr|thousand|k)?",
        text,
    )[:2]
    if not matches:
        return np.nan, np.nan
    fallback_unit = next((unit for _, unit in reversed(matches) if unit), None)
    numbers = []
    for token, unit in matches:
        inferred = unit or (fallback_unit if "," not in token and float(token) < 1_000 else None)
        numbers.append(salary_number(token, inferred))
    minimum, maximum = (numbers[0], numbers[0]) if len(numbers) == 1 else sorted(numbers[:2])
    if "/month" in text or "per month" in text or re.search(r"\\bpm\\b", text):
        minimum, maximum = minimum * 12, maximum * 12
    return (np.nan, np.nan) if maximum <= 0 else (float(minimum), float(maximum))

def parse_experience(value):
    if pd.isna(value):
        return np.nan, np.nan
    text = str(value).strip().lower()
    if "fresher" in text or "no experience" in text:
        return 0.0, 0.0
    values = [float(item) for item in re.findall(r"\\d+(?:\\.\\d+)?", text)]
    if not values:
        return np.nan, np.nan
    if len(values) == 1:
        return values[0], np.nan if "+" in text else values[0]
    return min(values[:2]), max(values[:2])

def experience_category(minimum, maximum):
    if pd.isna(minimum):
        return "Not specified"
    if minimum == 0 and pd.notna(maximum) and maximum <= 2:
        return "Fresher"
    if minimum <= 2:
        return "Entry Level"
    if minimum <= 5:
        return "Mid Level"
    return "Senior"

TITLE_RULES = [
    (r"\\bdata\\s+analyst\\b|\\banalyst.*data\\b", "Data Analyst"),
    (r"\\bbusiness\\s+analyst\\b", "Business Analyst"),
    (r"\\b(?:bi|business intelligence)\\s+analyst\\b", "BI Analyst"),
    (r"\\bdata\\s+scientist\\b", "Data Scientist"),
    (r"\\bdata\\s+engineer\\b|\\betl\\s+(?:engineer|developer)\\b", "Data Engineer"),
    (r"\\bmachine learning\\s+(?:engineer|developer)\\b|\\bml\\s+engineer\\b", "Machine Learning Engineer"),
    (r"\\bai\\s*[/&-]?\\s*ml\\b|artificial intelligence|\\bai engineer\\b", "AI / ML Engineer"),
    (r"\\bdevops\\b|site reliability|\\bsre\\b", "DevOps / SRE"),
    (r"\\bcloud\\s+(?:engineer|architect|consultant)\\b", "Cloud Engineer"),
    (r"\\bsoftware\\b|\\bdeveloper\\b|\\bprogrammer\\b|application (?:lead|engineer)", "Software Development"),
    (r"\\bproduct manager\\b|\\bproduct management\\b", "Product Management"),
    (r"\\bproject manager\\b|\\bprogram manager\\b", "Project / Program Management"),
    (r"\\bquality\\b|\\bqa\\b|\\btesting\\b|\\btester\\b", "Quality / Testing"),
    (r"\\bsales\\b|business development|relationship manager", "Sales / Business Development"),
    (r"\\bmarketing\\b|\\bseo\\b|\\bbrand manager\\b", "Marketing"),
    (r"\\brecruit|human resources|\\bhr\\b|talent acquisition", "HR / Recruitment"),
    (r"\\baccountant\\b|\\bfinance\\b|financial analyst|\\btax\\b|\\baudit", "Finance / Accounting"),
    (r"customer (?:support|care|service)|\\bbpo\\b|voice process|call center", "Customer Support / BPO"),
    (r"\\boperations?\\b", "Operations"),
    (r"supply chain|\\blogistics\\b|\\bprocurement\\b|\\bpurchase\\b", "Supply Chain / Logistics"),
    (r"\\bmechanical\\b|maintenance engineer", "Mechanical / Maintenance"),
    (r"\\belectrical\\b|\\belectronics\\b", "Electrical / Electronics"),
    (r"\\bcivil\\b|construction|site engineer", "Civil / Construction"),
    (r"\\bdoctor\\b|\\bnurse\\b|\\bmedical\\b|\\bpharma", "Healthcare / Pharma"),
    (r"\\bteacher\\b|\\bfaculty\\b|\\btrainer\\b|\\bprofessor\\b", "Education / Training"),
    (r"\\bdesigner\\b|\\bgraphic\\b|\\bux\\b|\\bui\\b|creative", "Design / Creative"),
    (r"\\blegal\\b|\\blawyer\\b|\\bcounsel\\b", "Legal"),
]

def normalize_job_role(title):
    text = "" if pd.isna(title) else str(title).casefold()
    return next((role for pattern, role in TITLE_RULES if re.search(pattern, text, flags=re.I)), "Other")

LOCATION_ALIASES = {
    "bangalore": "Bengaluru", "bengaluru": "Bengaluru", "gurgaon": "Gurugram",
    "gurugram": "Gurugram", "bombay": "Mumbai", "new delhi": "Delhi",
    "delhi ncr": "Delhi NCR", "remote": "Remote", "work from home": "Remote",
}

def normalize_location(value):
    if pd.isna(value):
        return "Not specified"
    text = re.sub(r"\\([^)]*\\)", "", str(value))
    text = re.sub(r"^hybrid\\s*[-:]\\s*", "", text, flags=re.I)
    text = re.sub(r"\\s+", " ", text).strip(" ,-\\t")
    return LOCATION_ALIASES.get(text.casefold(), text.title()) if text else "Not specified"
"""
    ),
    md(
        """## 6. Skills extraction and normalization

Skills are split from tags and supplemented with explicit core-technology matches in descriptions. Each skill is counted at most once per job."""
    ),
    code(
        """SKILL_ALIASES = {
    "python3": "Python", "python 3": "Python", "python": "Python",
    "sql": "SQL", "structured query language": "SQL", "ms sql": "SQL",
    "powerbi": "Power BI", "power bi": "Power BI", "ms power bi": "Power BI",
    "tableau": "Tableau", "ms excel": "Excel", "microsoft excel": "Excel", "excel": "Excel",
    "postgres": "PostgreSQL", "postgresql": "PostgreSQL", "machine learning": "Machine Learning",
    "ml": "Machine Learning", "amazon web services": "AWS", "aws": "AWS",
    "microsoft azure": "Azure", "azure": "Azure", "google cloud platform": "GCP", "gcp": "GCP",
    "apache spark": "Spark", "spark": "Spark", "apache hadoop": "Hadoop", "hadoop": "Hadoop",
    "snowflake": "Snowflake", "dbt": "dbt", "etl": "ETL", "statistics": "Statistics",
    "statistical analysis": "Statistics", "deep learning": "Deep Learning",
    "natural language processing": "NLP", "nlp": "NLP", "numpy": "NumPy", "pandas": "Pandas",
    "docker": "Docker", "kubernetes": "Kubernetes", "git": "Git", "r programming": "R",
    "r language": "R", "devops": "DevOps", "azure devops": "Azure DevOps",
    "communication skills": "Communication",
}

CORE_PATTERNS = {
    "Python": r"(?<![\\w])python(?:\\s*3)?(?![\\w])",
    "SQL": r"(?<![\\w])sql(?![\\w])|structured query language",
    "Excel": r"(?<![\\w])(?:ms |microsoft )?excel(?![\\w])",
    "Power BI": r"(?<![\\w])power\\s*bi(?![\\w])", "Tableau": r"\\btableau\\b",
    "Machine Learning": r"\\bmachine learning\\b", "Pandas": r"\\bpandas\\b", "NumPy": r"\\bnumpy\\b",
    "AWS": r"\\baws\\b|amazon web services", "Azure": r"\\bazure\\b",
    "GCP": r"\\bgcp\\b|google cloud platform", "Spark": r"\\b(?:apache )?spark\\b",
    "Hadoop": r"\\bhadoop\\b", "Snowflake": r"\\bsnowflake\\b", "dbt": r"(?<![\\w])dbt(?![\\w])",
    "ETL": r"(?<![\\w])etl(?![\\w])", "Statistics": r"\\bstatistic(?:s|al)\\b",
    "Deep Learning": r"\\bdeep learning\\b", "NLP": r"(?<![\\w])nlp(?![\\w])|natural language processing",
    "Docker": r"\\bdocker\\b", "Kubernetes": r"\\bkubernetes\\b", "Git": r"(?<![\\w])git(?![\\w])",
}

def normalize_skill(value):
    if pd.isna(value):
        return None
    text = re.sub(r"\\s+", " ", str(value)).strip(" .,-_/|")
    if len(text) < 2 or len(text) > 80:
        return None
    canonical = SKILL_ALIASES.get(text.casefold())
    return canonical or (text.upper() if len(text) <= 4 and text.isalpha() else text.title())

def extract_skills(skill_text, description=None):
    found = []
    if pd.notna(skill_text):
        for token in re.split(r"[,;|\\n]+", str(skill_text)):
            normalized = normalize_skill(token)
            if normalized:
                found.append(normalized)
    combined = " ".join(str(value) for value in (skill_text, description) if pd.notna(value)).casefold()
    found.extend(skill for skill, pattern in CORE_PATTERNS.items() if re.search(pattern, combined, flags=re.I))
    return list(dict.fromkeys(found))

def skill_cooccurrence(skill_rows, top_n=20):
    top = set(skill_rows["skill"].value_counts().head(top_n).index)
    counter = Counter()
    for _, values in skill_rows[skill_rows["skill"].isin(top)].groupby("job_id")["skill"]:
        counter.update(combinations(sorted(set(values)), 2))
    return pd.DataFrame(
        [(left, right, count) for (left, right), count in counter.items()],
        columns=["skill_1", "skill_2", "job_count"],
    ).sort_values("job_count", ascending=False, ignore_index=True)"""
    ),
    md(
        """## 7. Build clean job, skill, and location tables

The most complete row is retained for each stable job ID. Processed outputs are written to a separate output folder; the uploaded source remains unchanged."""
    ),
    code(
        """jobs = raw.copy()
jobs.columns = [normalize_column_name(column) for column in jobs.columns]
required = {"title", "job_id", "company_name", "location"}
missing_required = required - set(jobs.columns)
if missing_required:
    raise ValueError(f"Required fields are missing: {sorted(missing_required)}")

raw_row_count = len(jobs)
exact_duplicates = int(jobs.duplicated().sum())
jobs = (
    jobs.assign(_completeness=jobs.notna().sum(axis=1))
    .sort_values(["job_id", "_completeness"], ascending=[True, False])
    .drop_duplicates("job_id", keep="first")
    .drop(columns="_completeness")
    .copy()
)

for column in ["title", "company_name", "currency", "salary", "experience", "location", "tags_and_skills"]:
    if column in jobs:
        jobs[column] = jobs[column].map(clean_text)
if "job_description" not in jobs:
    jobs["job_description"] = pd.NA
jobs["job_description"] = jobs["job_description"].map(clean_html_text)

jobs["original_job_title"] = jobs["title"]
jobs["normalized_job_role"] = jobs["title"].map(normalize_job_role)
jobs["original_location"] = jobs["location"]
location_lists = jobs["location"].map(
    lambda value: list(dict.fromkeys(normalize_location(token) for token in str(value).split(",")))
    if pd.notna(value) else ["Not specified"]
)
jobs["primary_location"] = location_lists.str[0]

salary_ranges = jobs["salary"].map(parse_salary)
jobs["minimum_salary"] = [value[0] for value in salary_ranges]
jobs["maximum_salary"] = [value[1] for value in salary_ranges]
jobs["average_salary"] = jobs[["minimum_salary", "maximum_salary"]].mean(axis=1)
is_inr = jobs.get("currency", pd.Series("INR", index=jobs.index)).eq("INR")
jobs["minimum_salary_lpa"] = jobs["minimum_salary"].where(is_inr) / 100_000
jobs["maximum_salary_lpa"] = jobs["maximum_salary"].where(is_inr) / 100_000
jobs["average_salary_lpa"] = jobs["average_salary"].where(is_inr) / 100_000

experience_ranges = jobs.get("experience", pd.Series(pd.NA, index=jobs.index)).map(parse_experience)
parsed_min = pd.Series([value[0] for value in experience_ranges], index=jobs.index)
parsed_max = pd.Series([value[1] for value in experience_ranges], index=jobs.index)
source_min = pd.to_numeric(jobs.get("minimum_experience"), errors="coerce")
source_max = pd.to_numeric(jobs.get("maximum_experience"), errors="coerce")
jobs["minimum_experience"] = source_min.combine_first(parsed_min)
jobs["maximum_experience"] = source_max.combine_first(parsed_max)
jobs["average_experience"] = jobs[["minimum_experience", "maximum_experience"]].mean(axis=1)
jobs["experience_category"] = [
    experience_category(minimum, maximum)
    for minimum, maximum in zip(jobs["minimum_experience"], jobs["maximum_experience"])
]

jobs["company_rating"] = pd.to_numeric(jobs.get("aggregate_rating"), errors="coerce")
jobs["company_rating"] = jobs["company_rating"].where(jobs["company_rating"].between(1, 5))
jobs["job_id"] = pd.to_numeric(jobs["job_id"], errors="coerce").astype("Int64")
jobs["skills_raw"] = jobs.get("tags_and_skills", pd.Series(pd.NA, index=jobs.index))

skill_records = []
for row in jobs[["job_id", "skills_raw", "job_description"]].itertuples(index=False):
    skill_records.extend((row.job_id, skill) for skill in extract_skills(row.skills_raw, row.job_description))
job_skills = pd.DataFrame(skill_records, columns=["job_id", "skill"]).drop_duplicates(ignore_index=True)

location_records = [
    (job_id, location)
    for job_id, values in zip(jobs["job_id"], location_lists)
    for location in values
]
job_locations = pd.DataFrame(location_records, columns=["job_id", "location"]).drop_duplicates(ignore_index=True)

print(f"Raw rows: {raw_row_count:,}")
print(f"Clean unique jobs: {len(jobs):,}")
print(f"Exact duplicate rows found: {exact_duplicates:,}")
print(f"Duplicate job-ID rows removed: {raw_row_count - len(jobs):,}")
print(f"Unique job-skill rows: {len(job_skills):,}")
print(f"Unique job-location rows: {len(job_locations):,}")"""
    ),
    md(
        """## 8. Save processed outputs

Parquet is compact and preserves numeric types. CSV exports are also provided for portability. Re-running this cell overwrites only generated files in the output directory—not the uploaded dataset."""
    ),
    code(
        """OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
jobs.to_parquet(OUTPUT_DIR / "clean_jobs.parquet", index=False)
job_skills.to_parquet(OUTPUT_DIR / "job_skills.parquet", index=False)
job_locations.to_parquet(OUTPUT_DIR / "job_locations.parquet", index=False)

report = {
    "source_file": source_file,
    "source_table": source_table,
    "raw_rows": raw_row_count,
    "clean_rows": len(jobs),
    "exact_duplicates": exact_duplicates,
    "duplicate_job_ids_removed": raw_row_count - len(jobs),
    "usable_inr_salaries": int(jobs["average_salary_lpa"].notna().sum()),
    "jobs_with_skills": int(jobs["job_id"].isin(job_skills["job_id"]).sum()),
}
(OUTPUT_DIR / "processing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
display(pd.Series(report, name="value").to_frame())
print("Saved to:", OUTPUT_DIR.resolve())"""
    ),
    md(
        """## 9. Job role demand

Broad role groups make highly varied titles comparable while leaving every original title intact."""
    ),
    code(
        """top_roles = (
    jobs.groupby("normalized_job_role")["job_id"].nunique()
    .nlargest(20).rename("job_count").reset_index()
)
display(top_roles)
fig = px.bar(
    top_roles.sort_values("job_count"), x="job_count", y="normalized_job_role",
    orientation="h", title="Most in-demand role groups",
    labels={"job_count": "Unique listings", "normalized_job_role": "Role group"},
)
fig.show()"""
    ),
    md(
        """## 10. Skills demand and combinations

Demand counts represent unique listings. Co-occurrence counts an unordered pair at most once per job."""
    ),
    code(
        """top_skills = job_skills.groupby("skill")["job_id"].nunique().nlargest(25).rename("job_count").reset_index()
top_skills["share_of_skilled_jobs"] = top_skills["job_count"] / job_skills["job_id"].nunique()
display(top_skills.style.format({"share_of_skilled_jobs": "{:.1%}"}))
px.bar(
    top_skills.sort_values("job_count"), x="job_count", y="skill", orientation="h",
    title="Most requested skills", labels={"job_count": "Unique listings", "skill": "Skill"},
).show()

pairs = skill_cooccurrence(job_skills, top_n=25).head(25).copy()
pairs["combination"] = pairs["skill_1"] + " + " + pairs["skill_2"]
display(pairs)
px.bar(
    pairs.sort_values("job_count"), x="job_count", y="combination", orientation="h",
    title="Frequent skill combinations", labels={"job_count": "Listings", "combination": "Combination"},
).show()"""
    ),
    md(
        """## 11. Salary analysis

Only positive, disclosed INR salaries are compared. USD listings remain unconverted. Median is emphasized because the distribution is skewed; the histogram display is capped at the 99th percentile without deleting source rows."""
    ),
    code(
        """salary_jobs = jobs[jobs["average_salary_lpa"].gt(0) & jobs["average_salary_lpa"].notna()].copy()
display(salary_jobs["average_salary_lpa"].describe(percentiles=[.01, .25, .5, .75, .95, .99]).to_frame())

p99 = salary_jobs["average_salary_lpa"].quantile(.99)
px.histogram(
    salary_jobs[salary_jobs["average_salary_lpa"] <= p99],
    x="average_salary_lpa", nbins=50,
    title="Average annual salary in LPA (display capped at 99th percentile)",
    labels={"average_salary_lpa": "Average salary (LPA)"},
).show()

role_salary = salary_jobs.groupby("normalized_job_role").agg(
    median_salary_lpa=("average_salary_lpa", "median"),
    mean_salary_lpa=("average_salary_lpa", "mean"),
    sample_size=("job_id", "nunique"),
).reset_index()
role_salary = role_salary[role_salary["sample_size"] >= 30].sort_values("median_salary_lpa", ascending=False)
display(role_salary)
px.bar(
    role_salary.head(15).sort_values("median_salary_lpa"),
    x="median_salary_lpa", y="normalized_job_role", orientation="h",
    hover_data=["sample_size"], title="Median salary by role (minimum 30 disclosed salaries)",
).show()"""
    ),
    md(
        """## 12. Salary by experience, location, and skill

Minimum sample thresholds reduce unstable comparisons. These are associations within listing data and should not be interpreted as causal effects."""
    ),
    code(
        """experience_salary = salary_jobs.groupby("minimum_experience").agg(
    median_salary_lpa=("average_salary_lpa", "median"), sample_size=("job_id", "nunique")
).reset_index()
experience_salary = experience_salary[experience_salary["sample_size"] >= 20]
px.scatter(
    experience_salary, x="minimum_experience", y="median_salary_lpa", size="sample_size",
    title="Median salary vs minimum experience",
    labels={"minimum_experience": "Minimum experience (years)", "median_salary_lpa": "Median salary (LPA)"},
).show()

location_salary = salary_jobs.groupby("primary_location").agg(
    median_salary_lpa=("average_salary_lpa", "median"), sample_size=("job_id", "nunique")
).reset_index()
location_salary = location_salary[location_salary["sample_size"] >= 30].nlargest(20, "median_salary_lpa")
display(location_salary)

skill_salary = job_skills.merge(salary_jobs[["job_id", "average_salary_lpa"]], on="job_id")
skill_salary = skill_salary.groupby("skill").agg(
    median_salary_lpa=("average_salary_lpa", "median"), sample_size=("job_id", "nunique")
).reset_index()
skill_salary = skill_salary[skill_salary["sample_size"] >= 30].nlargest(20, "median_salary_lpa")
display(skill_salary)"""
    ),
    md(
        """## 13. Location and company analysis

Primary-location rankings count each job once. Company counts reflect this source snapshot rather than total corporate hiring."""
    ),
    code(
        """top_locations = jobs.groupby("primary_location")["job_id"].nunique().nlargest(20).rename("job_count").reset_index()
top_companies = jobs.groupby("company_name")["job_id"].nunique().nlargest(20).rename("job_count").reset_index()

display(top_locations)
px.bar(
    top_locations.sort_values("job_count"), x="job_count", y="primary_location", orientation="h",
    title="Top primary locations", labels={"job_count": "Unique listings"},
).show()

display(top_companies)
px.bar(
    top_companies.sort_values("job_count"), x="job_count", y="company_name", orientation="h",
    title="Companies with the most listings", labels={"job_count": "Unique listings"},
).show()"""
    ),
    md(
        """## 14. Data Analyst focus

This section directly compares SQL vs Python, Power BI vs Tableau, frequently paired skills, entry-level demand, and salary-associated skills for the normalized Data Analyst group."""
    ),
    code(
        """analyst_jobs = jobs[jobs["normalized_job_role"].eq("Data Analyst")].copy()
analyst_skills = job_skills[job_skills["job_id"].isin(analyst_jobs["job_id"])].copy()
core = ["SQL", "Python", "Excel", "Power BI", "Tableau", "Statistics", "ETL", "AWS", "Azure", "Snowflake"]
core_demand = (
    analyst_skills[analyst_skills["skill"].isin(core)]
    .groupby("skill")["job_id"].nunique().reindex(core, fill_value=0)
    .rename("job_count").reset_index()
)
core_demand["share_of_data_analyst_jobs"] = core_demand["job_count"] / max(len(analyst_jobs), 1)
print("Data Analyst listings:", len(analyst_jobs))
display(core_demand.style.format({"share_of_data_analyst_jobs": "{:.1%}"}))
px.bar(core_demand.sort_values("job_count"), x="job_count", y="skill", orientation="h", title="Core skill demand for Data Analyst listings").show()

print("Most common Data Analyst skill pairs")
display(skill_cooccurrence(analyst_skills, top_n=20).head(15))

analyst_entry = analyst_jobs[analyst_jobs["minimum_experience"].le(2)]
analyst_entry_skills = job_skills[job_skills["job_id"].isin(analyst_entry["job_id"])]
print("Entry-level Data Analyst listings:", len(analyst_entry))
display(analyst_entry_skills.groupby("skill")["job_id"].nunique().nlargest(15).rename("job_count").reset_index())

analyst_skill_salary = analyst_skills.merge(
    analyst_jobs[["job_id", "average_salary_lpa"]].dropna(), on="job_id"
).groupby("skill").agg(
    median_salary_lpa=("average_salary_lpa", "median"), sample_size=("job_id", "nunique")
).reset_index()
display(analyst_skill_salary[analyst_skill_salary["sample_size"] >= 10].nlargest(15, "median_salary_lpa"))"""
    ),
    md(
        """## 15. Fresher and entry-level analysis

Strict fresher jobs require minimum experience of 0 and maximum experience no greater than 2 years. A broader entry-level view includes any listing whose minimum requirement is no more than 2 years."""
    ),
    code(
        """strict_fresher = jobs[jobs["minimum_experience"].eq(0) & jobs["maximum_experience"].le(2)]
entry_level = jobs[jobs["minimum_experience"].le(2)]
fresher_skills = job_skills[job_skills["job_id"].isin(strict_fresher["job_id"])]

print(f"Strict fresher listings: {len(strict_fresher):,}")
print(f"Broader entry-level listings: {len(entry_level):,}")
print("Top fresher role groups")
display(strict_fresher.groupby("normalized_job_role")["job_id"].nunique().nlargest(15).rename("job_count").reset_index())
print("Top fresher skills")
display(fresher_skills.groupby("skill")["job_id"].nunique().nlargest(15).rename("job_count").reset_index())
print("Top fresher locations")
display(strict_fresher.groupby("primary_location")["job_id"].nunique().nlargest(15).rename("job_count").reset_index())
print("Top fresher companies")
display(strict_fresher.groupby("company_name")["job_id"].nunique().nlargest(15).rename("job_count").reset_index())"""
    ),
    md(
        """## 16. Key findings and limitations

The following summary is calculated from the processed data. Preserve the sample-size caveats when using these results in a portfolio or presentation."""
    ),
    code(
        """summary = {
    "unique_jobs": int(jobs["job_id"].nunique()),
    "unique_companies": int(jobs["company_name"].nunique(dropna=True)),
    "unique_primary_locations": int(jobs["primary_location"].nunique(dropna=True)),
    "usable_inr_salary_rows": int(salary_jobs["job_id"].nunique()),
    "median_disclosed_inr_salary_lpa": float(salary_jobs["average_salary_lpa"].median()),
    "top_named_role": top_roles.loc[top_roles["normalized_job_role"].ne("Other"), "normalized_job_role"].iloc[0],
    "top_skill": top_skills.iloc[0]["skill"],
    "top_location": top_locations.iloc[0]["primary_location"],
    "top_company": top_companies.iloc[0]["company_name"],
    "strict_fresher_jobs": len(strict_fresher),
}
display(pd.Series(summary, name="value").to_frame())

print("Limitations:")
print("• One supplied source snapshot; results are not the complete Indian labor market.")
print("• Missing/unpaid salaries are excluded, and USD salaries are not converted.")
print("• Relative upload labels cannot support a reliable calendar trend.")
print("• Role and skill normalization simplify free text and should be reviewed for specialized use.")
print("• Salary and skill patterns are associations, not evidence of causation.")"""
    ),
    md(
        """## 17. Optional: download generated outputs from Colab

Run this cell when you want a ZIP archive containing the processed Parquet files and processing report."""
    ),
    code(
        """if IN_COLAB:
    import shutil
    from google.colab import files
    archive = shutil.make_archive("/content/job_market_outputs", "zip", OUTPUT_DIR)
    print("Created:", archive)
    # Uncomment the next line to download immediately:
    # files.download(archive)
else:
    print("Processed outputs are available at:", OUTPUT_DIR.resolve())"""
    ),
]

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
notebook.metadata.language_info = {"name": "python", "version": "3.10"}
notebook.metadata.colab = {
    "name": "exploratory_analysis.ipynb",
    "provenance": [],
    "toc_visible": True,
}

output = Path("notebooks/exploratory_analysis.ipynb")
output.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, output)
print(output)
