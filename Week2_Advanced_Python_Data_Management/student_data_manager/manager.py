"""Encapsulated coordinator for importing, filtering, and reporting student data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .data_sources import source_for
from .models import Student
from .processing import build_students, filter_students, sort_students


class DataManager:
    """Owns the active student collection and its source/import diagnostics."""

    def __init__(self) -> None:
        self.__students: list[Student] = []
        self.__errors: list[dict[str, str]] = []
        self.__source_path: str | None = None
        self.__raw_count = 0

    def load(self, path: str | Path) -> None:
        records: list[dict[str, Any]] = source_for(path).read_records()
        students, errors = build_students(records)
        self.__students = students
        self.__errors = errors
        self.__source_path = str(path)
        self.__raw_count = len(records)

    @property
    def students(self) -> list[Student]:
        """Expose a copy, not the internal collection."""
        return list(self.__students)

    @property
    def errors(self) -> list[dict[str, str]]:
        return [error.copy() for error in self.__errors]

    @property
    def source_path(self) -> str | None:
        return self.__source_path

    @property
    def raw_count(self) -> int:
        return self.__raw_count

    def get_students(
        self,
        department: str | None = None,
        min_average: float | None = None,
        sort_by: str = "average",
        descending: bool = True,
    ) -> list[Student]:
        filtered = filter_students(self.__students, department=department, min_average=min_average)
        return sort_students(filtered, sort_by=sort_by, descending=descending)
