"""Summary report generation and exporting."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from .models import Student
from .processing import grade_distribution, sort_students


class ReportGenerator:
    """Create JSON, Markdown, and CSV reports from validated Student objects."""

    def __init__(self, students: Iterable[Student], source_path: str, raw_count: int, errors: list[dict[str, str]] | None = None, valid_count: int | None = None) -> None:
        self.students = list(students)
        self.source_path = source_path
        self.raw_count = raw_count
        self.errors = list(errors or [])
        self.valid_count = len(self.students) if valid_count is None else valid_count

    def build_summary(self, top_n: int = 5) -> dict[str, Any]:
        total = len(self.students)
        subject_averages = {
            subject: round(mean(student.scores[subject] for student in self.students), 2) if total else 0.0
            for subject in Student.SUBJECTS
        }
        averages = [student.average for student in self.students]
        department_counts: dict[str, int] = defaultdict(int)
        for student in self.students:
            department_counts[student.department] += 1

        top_students = [student.to_dict() for student in sort_students(self.students, "average", True)[:max(0, top_n)]]
        low_students = [student.to_dict() for student in sort_students(self.students, "average", False)[:min(3, total)]]

        return {
            "project": "Week 2 - Advanced Python Data Management",
            "source_file": self.source_path,
            "records_read": self.raw_count,
            "valid_records": self.valid_count,
            "selected_records": total,
            "skipped_records": len(self.errors),
            "overall_average": round(mean(averages), 2) if averages else 0.0,
            "overall_median": round(median(averages), 2) if averages else 0.0,
            "subject_averages": subject_averages,
            "grade_distribution": grade_distribution(self.students),
            "department_counts": dict(sorted(department_counts.items())),
            "top_students": top_students,
            "students_needing_support": low_students,
            "import_errors": self.errors,
        }

    @staticmethod
    def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
        lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
        lines.extend("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in rows)
        return "\n".join(lines)

    def write_reports(self, output_dir: str | Path, top_n: int = 5) -> dict[str, str]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        summary = self.build_summary(top_n=top_n)

        json_path = output / "report.json"
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        top_rows = [
            [item["student_id"], item["name"], item["department"], f'{item["average"]:.2f}', item["grade"]]
            for item in summary["top_students"]
        ]
        grade_rows = [[grade, count] for grade, count in summary["grade_distribution"].items()]
        subject_labels = {"python": "Python", "sql": "SQL", "statistics": "Statistics"}
        subject_rows = [[subject_labels.get(subject, subject.title()), average] for subject, average in summary["subject_averages"].items()]

        markdown = [
            "# Student Data Management Report",
            "",
            f"- **Input file:** `{self.source_path}`",
            f"- **Records read:** {self.raw_count}",
            f"- **Valid records:** {self.valid_count}",
            f"- **Records selected for this report:** {len(self.students)}",
            f"- **Skipped records:** {len(self.errors)}",
            f"- **Overall average:** {summary['overall_average']:.2f}",
            f"- **Median student average:** {summary['overall_median']:.2f}",
            "",
            "## Subject averages",
            self._markdown_table(["Subject", "Average score"], subject_rows),
            "",
            "## Grade distribution",
            self._markdown_table(["Grade", "Students"], grade_rows),
            "",
            "## Top students",
            self._markdown_table(["Student ID", "Name", "Department", "Average", "Grade"], top_rows),
            "",
            "## Department counts",
            self._markdown_table(["Department", "Students"], [[name, count] for name, count in summary["department_counts"].items()]),
            "",
            "## Import validation",
            "No invalid rows were skipped." if not self.errors else self._markdown_table(["Row", "Reason"], [[item["row"], item["reason"]] for item in self.errors]),
            "",
            "## Notes",
            "Grades are calculated from the arithmetic mean of Python, SQL, and Statistics scores. "
            "This report is generated from the supplied sample dataset and is intended for a learning project.",
        ]
        markdown_path = output / "report.md"
        markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")

        results_path = output / "student_results.csv"
        fields = ["student_id", "name", "email", "department", "year", "python", "sql", "statistics", "average", "grade"]
        with results_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for student in sort_students(self.students, "average", True):
                writer.writerow(student.to_dict())

        return {"json": str(json_path), "markdown": str(markdown_path), "csv": str(results_path)}
