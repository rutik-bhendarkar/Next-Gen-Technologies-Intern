from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from src.math_operations import describe_numbers, probability_from_data

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "student_performance.csv"
OUT = ROOT / "outputs"


def run() -> dict:
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    scores = df["exam_score"].to_numpy(dtype=float)
    hours = df["study_hours_per_day"].to_numpy(dtype=float)
    attendance = df["attendance_percent"].to_numpy(dtype=float)
    result = {
        "dataset": {"rows": int(df.shape[0]), "columns": int(df.shape[1]), "file": "data/student_performance.csv"},
        "exam_score_statistics": describe_numbers(scores),
        "study_hours_statistics": describe_numbers(hours),
        "attendance_statistics": describe_numbers(attendance),
        "probability_score_at_least_85": probability_from_data(scores, lambda x: x >= 85),
        "probability_attendance_at_least_90": probability_from_data(attendance, lambda x: x >= 90),
        "interpretation": [
            "The exam score mean and median summarize the centre of the sample.",
            "The mode identifies the most frequently observed exam score; ties are returned together.",
            "Population and sample variance use different denominators (n versus n-1).",
            "Empirical probability is the observed proportion in this small educational dataset, not a guarantee about future students."
        ]
    }
    (OUT/"results_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["WEEK 5 - MATHEMATICS FOR MACHINE LEARNING", "="*48,
             f"Rows: {df.shape[0]} | Columns: {df.shape[1]}"]
    for key, stats in [("Exam score", result["exam_score_statistics"]), ("Study hours", result["study_hours_statistics"]), ("Attendance", result["attendance_statistics"])]:
        lines += ["", key, "-"*len(key)]
        for k, v in stats.items(): lines.append(f"{k}: {v}")
    lines += ["", f"P(score >= 85): {result['probability_score_at_least_85']:.4f}",
              f"P(attendance >= 90): {result['probability_attendance_at_least_90']:.4f}"]
    (OUT/"console_results.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines))
    return result

if __name__ == "__main__":
    run()
