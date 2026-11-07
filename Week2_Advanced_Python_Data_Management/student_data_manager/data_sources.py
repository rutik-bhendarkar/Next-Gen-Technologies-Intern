"""CSV and JSON adapters, using an abstract base class and inheritance."""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class DataSource(ABC):
    """Abstract adapter shared by file-based data sources."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @abstractmethod
    def read_records(self) -> list[dict[str, Any]]:
        """Read tabular records from the configured file."""

    def _ensure_exists(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Input file does not exist: {self.path}")
        if not self.path.is_file():
            raise ValueError(f"Input path is not a file: {self.path}")


class CSVDataSource(DataSource):
    """Read rows from a CSV file."""

    def read_records(self) -> list[dict[str, Any]]:
        self._ensure_exists()
        with self.path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                raise ValueError("CSV file is empty or has no header row")
            return [dict(row) for row in reader]


class JSONDataSource(DataSource):
    """Read an array of objects from a JSON file."""

    def read_records(self) -> list[dict[str, Any]]:
        self._ensure_exists()
        with self.path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, list):
            raise ValueError("JSON input must be a list of objects")
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError("Every item in the JSON list must be an object")
        return payload


def source_for(path: str | Path) -> DataSource:
    """Factory function selecting a source adapter based on file extension."""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return CSVDataSource(path)
    if suffix == ".json":
        return JSONDataSource(path)
    raise ValueError("Unsupported file type. Please use .csv or .json")
