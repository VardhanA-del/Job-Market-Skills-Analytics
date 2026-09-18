"""Dataset discovery and loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd


DATA_DIR = Path("dataset")
PROCESSED_DIR = Path("data/processed")
SUPPORTED_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".xlsm", ".json", ".jsonl", ".parquet"}


def discover_dataset_files(data_dir: Path = DATA_DIR) -> list[Path]:
    """Return every supported tabular file without modifying the data directory."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    return sorted(path for path in data_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)


def iter_tables(path: Path) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield every table in a supported file as ``(table_name, dataframe)``."""
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        book = pd.ExcelFile(path)
        for sheet in book.sheet_names:
            yield sheet, pd.read_excel(path, sheet_name=sheet)
    elif suffix == ".csv":
        yield path.stem, pd.read_csv(path, low_memory=False)
    elif suffix == ".tsv":
        yield path.stem, pd.read_csv(path, sep="\t", low_memory=False)
    elif suffix in {".json", ".jsonl"}:
        yield path.stem, pd.read_json(path, lines=suffix == ".jsonl")
    elif suffix == ".parquet":
        yield path.stem, pd.read_parquet(path)


def load_raw_jobs(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load the most job-like table discovered in ``dataset/``.

    Selection is schema-based instead of filename-based. The chosen table must have
    a title-like column and is ranked by recognized job-market fields and row count.
    """
    candidates: list[tuple[int, int, Path, str, pd.DataFrame]] = []
    expected = {
        "title", "jobtitle", "jobid", "companyname", "company", "location",
        "salary", "experience", "tagsandskills", "skills", "jobdescription",
    }
    for path in discover_dataset_files(data_dir):
        for table_name, frame in iter_tables(path):
            normalized = {"".join(ch.lower() for ch in str(c) if ch.isalnum()) for c in frame.columns}
            score = len(normalized & expected)
            if {"title", "jobtitle"} & normalized:
                candidates.append((score, len(frame), path, table_name, frame))
    if not candidates:
        raise ValueError("No job-listing table with a title column was found in dataset/.")
    _, _, path, table_name, frame = max(candidates, key=lambda item: (item[0], item[1]))
    return frame, {"source_file": path.as_posix(), "source_table": table_name}


def load_processed_data(processed_dir: Path = PROCESSED_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load cleaned jobs, exploded skills, and exploded locations."""
    paths = [processed_dir / name for name in ("clean_jobs.parquet", "job_skills.parquet", "job_locations.parquet")]
    missing = [path.as_posix() for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Processed files are missing: {', '.join(missing)}")
    return tuple(pd.read_parquet(path) for path in paths)  # type: ignore[return-value]

