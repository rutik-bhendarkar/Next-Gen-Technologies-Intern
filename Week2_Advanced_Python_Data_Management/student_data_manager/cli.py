"""Command-line interface for the Week 2 data management application."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .manager import DataManager
from .reports import ReportGenerator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import student records from CSV/JSON, validate them, and produce reports."
    )
    parser.add_argument("--input", required=True, help="Path to a .csv or .json file")
    parser.add_argument("--output", default="outputs", help="Directory where reports are written (default: outputs)")
    parser.add_argument("--department", help="Optional department filter, e.g. IT")
    parser.add_argument("--min-average", type=float, help="Optional minimum student average")
    parser.add_argument("--sort-by", choices=("average", "name", "student_id"), default="average")
    parser.add_argument("--ascending", action="store_true", help="Sort in ascending rather than descending order")
    parser.add_argument("--top", type=int, default=5, help="Number of top students in the report (default: 5)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_average is not None and not 0 <= args.min_average <= 100:
        print("ERROR: --min-average must be between 0 and 100.", file=sys.stderr)
        return 2
    if args.top < 0:
        print("ERROR: --top cannot be negative.", file=sys.stderr)
        return 2

    manager = DataManager()
    try:
        manager.load(args.input)
        filtered_students = manager.get_students(
            department=args.department,
            min_average=args.min_average,
            sort_by=args.sort_by,
            descending=not args.ascending,
        )
        reports = ReportGenerator(
            filtered_students,
            source_path=manager.source_path or str(Path(args.input)),
            raw_count=manager.raw_count,
            errors=manager.errors,
            valid_count=len(manager.students),
        ).write_reports(args.output, top_n=args.top)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("=" * 62)
    print("WEEK 2 - ADVANCED PYTHON DATA MANAGEMENT")
    print("=" * 62)
    print(f"Input file       : {manager.source_path}")
    print(f"Records read     : {manager.raw_count}")
    print(f"Valid records    : {len(manager.students)}")
    print(f"Skipped records  : {len(manager.errors)}")
    print(f"Records selected : {len(filtered_students)}")
    if manager.errors:
        print("\nImport issues:")
        for item in manager.errors:
            print(f"  Row {item['row']}: {item['reason']}")
    print("\nReports generated:")
    for label, path in reports.items():
        print(f"  {label.title():<10}: {path}")
    print("\nStatus: SUCCESS")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
