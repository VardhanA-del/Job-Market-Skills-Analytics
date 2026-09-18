"""Read-only profiler for tabular files in the local dataset directory."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


DATA_DIR = Path("dataset")


def _load_tables(path: Path) -> dict[str, pd.DataFrame]:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        book = pd.ExcelFile(path)
        return {sheet: pd.read_excel(path, sheet_name=sheet) for sheet in book.sheet_names}
    if suffix == ".csv":
        return {path.stem: pd.read_csv(path)}
    if suffix == ".tsv":
        return {path.stem: pd.read_csv(path, sep="\t")}
    if suffix in {".json", ".jsonl"}:
        return {path.stem: pd.read_json(path, lines=suffix == ".jsonl")}
    return {}


def profile() -> dict:
    result: dict = {"files": [], "tables": {}}
    for path in sorted(DATA_DIR.rglob("*")):
        if not path.is_file():
            continue
        result["files"].append(
            {"path": path.as_posix(), "suffix": path.suffix.lower(), "bytes": path.stat().st_size}
        )
        for sheet, frame in _load_tables(path).items():
            key = f"{path.name}::{sheet}"
            samples = frame.head(5).copy()
            for column in samples.select_dtypes(include=["datetime", "datetimetz"]).columns:
                samples[column] = samples[column].astype(str)
            result["tables"][key] = {
                "shape": list(frame.shape),
                "columns": list(map(str, frame.columns)),
                "dtypes": {str(k): str(v) for k, v in frame.dtypes.items()},
                "missing": {str(k): int(v) for k, v in frame.isna().sum().items()},
                "exact_duplicate_rows": int(frame.duplicated().sum()),
                "unique_counts": {str(k): int(v) for k, v in frame.nunique(dropna=True).items()},
                "sample_rows": samples.where(samples.notna(), None).to_dict(orient="records"),
                "example_values": {
                    str(column): [str(value)[:250] for value in frame[column].dropna().unique()[:12]]
                    for column in frame.columns
                },
            }
    return result


if __name__ == "__main__":
    print(json.dumps(profile(), indent=2, ensure_ascii=False, default=str))
