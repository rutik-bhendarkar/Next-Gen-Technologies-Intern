"""Data cleaning and processing helpers: functions, lambdas, and comprehensions."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .models import Student

REQUIRED_FIELDS = ("student_id", "name", "python", "sql", "statistics")


def find_missing_fields(record: dict[str, Any]) -> list[str]:
    """Return required fields that are missing or blank using a comprehension."""
    normalized = {str(key).strip().lower(): value for key, value in record.items()}
    return [field for field in REQUIRED_FIELDS if normalized.get(field) is None or str(normalized.get(field)).strip() == ""]


def build_students(records: Iterable[dict[str, Any]]) -> tuple[list[Student], list[dict[str, str]]]:
    """Convert raw records to validated objects; report invalid rows rather than crashing."""
    students: list[Student] = []
    errors: list[dict[str, str]] = []
    for index, record in enumerate(records, start=2):
        missing = find_missing_fields(record)
        if missing:
            errors.append({"row": str(index), "reason": f"Missing required fields: {', '.join(missing)}"})
            continue
        try:
            students.append(Student.from_mapping(record))
        except (TypeError, ValueError) as exc:
            errors.append({"row": str(index), "reason": str(exc)})
    return students, errors


def sort_students(students: Iterable[Student], sort_by: str = "average", descending: bool = True) -> list[Student]:
    """Sort student objects using a lambda key function."""
    allowed = {
        "average": lambda student: student.average,
        "name": lambda student: student.name.casefold(),
        "student_id": lambda student: student.student_id.casefold(),
    }
    if sort_by not in allowed:
        raise ValueError(f"sort_by must be one of: {', '.join(allowed)}")
    return sorted(students, key=allowed[sort_by], reverse=descending)


def filter_students(
    students: Iterable[Student],
    department: str | None = None,
    min_average: float | None = None,
) -> list[Student]:
    """Filter records using a combined comprehension."""
    filtered = [
        student for student in students
        if (department is None or student.department.casefold() == department.casefold())
        and (min_average is None or student.average >= min_average)
    ]
    return filtered


def grade_distribution(students: Iterable[Student]) -> dict[str, int]:
    """Count each grade and include zero-count grades for stable reports."""
    counts = Counter(student.grade for student in students)
    grade_order = ("A+", "A", "B", "C", "D", "F")
    return {grade: counts.get(grade, 0) for grade in grade_order}
