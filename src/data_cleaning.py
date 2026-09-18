"""Reusable field-cleaning and parsing functions."""

from __future__ import annotations

import html
import re
from typing import Any

import numpy as np
import pandas as pd


MISSING_TEXT = {"", "nan", "none", "null", "n/a", "na", "not available"}


def normalize_column_name(name: Any) -> str:
    """Convert camelCase and punctuation-heavy labels to snake_case."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name).strip())
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def clean_text(value: Any) -> Any:
    """Trim and collapse whitespace while retaining missing values."""
    if pd.isna(value):
        return pd.NA
    text = re.sub(r"\s+", " ", str(value)).strip()
    return pd.NA if text.casefold() in MISSING_TEXT else text


def clean_html_text(value: Any) -> Any:
    """Convert simple HTML job descriptions into compact plain text."""
    if pd.isna(value):
        return pd.NA
    text = re.sub(r"<\s*br\s*/?>|</p>|</li>", " ", str(value), flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return clean_text(html.unescape(text))


def _salary_number(token: str, unit: str | None) -> float:
    value = float(token.replace(",", ""))
    normalized_unit = (unit or "").casefold()
    if normalized_unit.startswith(("lac", "lakh")) or normalized_unit == "lpa":
        value *= 100_000
    elif normalized_unit.startswith(("cr", "crore")):
        value *= 10_000_000
    elif normalized_unit in {"k", "thousand"}:
        value *= 1_000
    return value


def parse_salary(value: Any) -> tuple[float, float]:
    """Parse a salary label into annual minimum and maximum native-currency values.

    Lakh/Lac/LPA amounts are multiplied by 100,000. Mixed ranges such as
    ``70,000-2 Lacs PA`` retain the explicit 70,000 endpoint. Reliably identified
    monthly values are annualized by multiplying by 12. Non-disclosed, unpaid, and
    zero-only values return missing values.
    """
    if pd.isna(value):
        return np.nan, np.nan
    text = str(value).strip().lower()
    if any(term in text for term in ("not disclosed", "unpaid", "negotiable", "not specified")):
        return np.nan, np.nan
    matches = re.findall(
        r"(?<![a-z])(\d[\d,]*(?:\.\d+)?)\s*(lacs?|lakhs?|lpa|crores?|cr|thousand|k)?",
        text,
    )
    if not matches:
        return np.nan, np.nan
    matches = matches[:2]
    fallback_unit = next((unit for _, unit in reversed(matches) if unit), None)
    numbers: list[float] = []
    for token, unit in matches:
        # A unit at the end of a simple range applies to both endpoints. In a
        # mixed range (70,000-2 Lacs), the comma-bearing endpoint is explicit.
        inferred_unit = unit or (fallback_unit if "," not in token and float(token) < 1_000 else None)
        numbers.append(_salary_number(token, inferred_unit))
    if len(numbers) == 1:
        minimum = maximum = numbers[0]
    else:
        minimum, maximum = sorted(numbers[:2])
    if "/month" in text or "per month" in text or re.search(r"\bpm\b", text):
        minimum, maximum = minimum * 12, maximum * 12
    if maximum <= 0:
        return np.nan, np.nan
    return float(minimum), float(maximum)


def parse_experience(value: Any) -> tuple[float, float]:
    """Parse experience labels such as ``0-2 Yrs``, ``5+ years``, and ``Fresher``."""
    if pd.isna(value):
        return np.nan, np.nan
    text = str(value).strip().lower()
    if "fresher" in text or "no experience" in text:
        return 0.0, 0.0
    numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return np.nan, np.nan
    if len(numbers) == 1:
        maximum = np.nan if "+" in text else numbers[0]
        return numbers[0], maximum
    return min(numbers[0], numbers[1]), max(numbers[0], numbers[1])


def experience_category(minimum: Any, maximum: Any) -> str:
    """Assign a documented career stage using the minimum requested experience."""
    if pd.isna(minimum):
        return "Not specified"
    minimum = float(minimum)
    if minimum == 0 and pd.notna(maximum) and float(maximum) <= 2:
        return "Fresher"
    if minimum <= 2:
        return "Entry Level"
    if minimum <= 5:
        return "Mid Level"
    return "Senior"


def parse_posting_age_days(value: Any) -> float:
    """Parse relative posting labels without inventing an absolute date."""
    if pd.isna(value):
        return np.nan
    text = str(value).casefold()
    if any(term in text for term in ("just now", "few hours", "today")):
        return 0.0
    match = re.search(r"(\d+)\s+days?\s+ago", text)
    return float(match.group(1)) if match else np.nan


TITLE_RULES: list[tuple[str, str]] = [
    (r"\bdata\s+analyst\b|\banalyst.*data\b", "Data Analyst"),
    (r"\bbusiness\s+analyst\b", "Business Analyst"),
    (r"\b(?:bi|business intelligence)\s+analyst\b", "BI Analyst"),
    (r"\bdata\s+scientist\b", "Data Scientist"),
    (r"\bdata\s+engineer\b|\betl\s+(?:engineer|developer)\b", "Data Engineer"),
    (r"\bmachine learning\s+(?:engineer|developer)\b|\bml\s+engineer\b", "Machine Learning Engineer"),
    (r"\bai\s*[/&-]?\s*ml\b|\bartificial intelligence\b|\bai engineer\b", "AI / ML Engineer"),
    (r"\bdevops\b|site reliability|\bsre\b", "DevOps / SRE"),
    (r"\bcloud\s+(?:engineer|architect|consultant)\b", "Cloud Engineer"),
    (r"\bsoftware\b|\bdeveloper\b|\bprogrammer\b|application (?:lead|engineer)", "Software Development"),
    (r"\bproduct manager\b|\bproduct management\b", "Product Management"),
    (r"\bproject manager\b|\bprogram manager\b", "Project / Program Management"),
    (r"\bquality\b|\bqa\b|\btesting\b|\btester\b", "Quality / Testing"),
    (r"\bsales\b|business development|relationship manager", "Sales / Business Development"),
    (r"\bmarketing\b|\bseo\b|\bbrand manager\b", "Marketing"),
    (r"\brecruit|human resources|\bhr\b|talent acquisition", "HR / Recruitment"),
    (r"\baccountant\b|\bfinance\b|financial analyst|\btax\b|\baudit", "Finance / Accounting"),
    (r"customer (?:support|care|service)|\bbpo\b|voice process|call center", "Customer Support / BPO"),
    (r"\boperations?\b|operations manager", "Operations"),
    (r"supply chain|\blogistics\b|\bprocurement\b|\bpurchase\b", "Supply Chain / Logistics"),
    (r"\bmechanical\b|maintenance engineer", "Mechanical / Maintenance"),
    (r"\belectrical\b|\belectronics\b", "Electrical / Electronics"),
    (r"\bcivil\b|construction|site engineer", "Civil / Construction"),
    (r"\bdoctor\b|\bnurse\b|\bmedical\b|\bpharma", "Healthcare / Pharma"),
    (r"\bteacher\b|\bfaculty\b|\btrainer\b|\bprofessor\b", "Education / Training"),
    (r"\bdesigner\b|\bgraphic\b|\bux\b|\bui\b|creative", "Design / Creative"),
    (r"\blegal\b|\blawyer\b|\bcounsel\b", "Legal"),
]


def normalize_job_role(title: Any) -> str:
    """Map detailed titles into transparent, broad analytical role groups."""
    if pd.isna(title):
        return "Other"
    text = str(title).casefold()
    for pattern, role in TITLE_RULES:
        if re.search(pattern, text, flags=re.I):
            return role
    return "Other"


LOCATION_ALIASES = {
    "bangalore": "Bengaluru", "bengaluru": "Bengaluru", "gurgaon": "Gurugram",
    "gurugram": "Gurugram", "bombay": "Mumbai", "new delhi": "Delhi",
    "delhi ncr": "Delhi NCR", "remote": "Remote", "work from home": "Remote",
}


def normalize_location(value: Any) -> str:
    """Normalize one location token while retaining meaningful city granularity."""
    if pd.isna(value):
        return "Not specified"
    text = re.sub(r"\([^)]*\)", "", str(value))
    text = re.sub(r"^hybrid\s*[-:]\s*", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" ,-\t")
    if not text:
        return "Not specified"
    return LOCATION_ALIASES.get(text.casefold(), text.title())
