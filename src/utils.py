"""Small formatting and safety helpers."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def format_lpa(value: Any) -> str:
    return "N/A" if value is None or pd.isna(value) else f"₹{float(value):,.1f} LPA"


def format_number(value: Any) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{int(value):,}"


def safe_first(frame: pd.DataFrame, column: str, default: str = "N/A") -> str:
    if frame.empty or column not in frame or frame[column].isna().all():
        return default
    return str(frame.iloc[0][column])

