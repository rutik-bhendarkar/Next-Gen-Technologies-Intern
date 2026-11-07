"""Automated tests for the Week 2 student data management application."""

import json
import tempfile
import unittest
from pathlib import Path

from student_data_manager.data_sources import CSVDataSource, JSONDataSource, source_for
from student_data_manager.manager import DataManager
from student_data_manager.models import Student
from student_data_manager.processing import build_students, filter_students, sort_students
from student_data_manager.reports import ReportGenerator


class ModelTests(unittest.TestCase):
    def test_average_and_grade(self):
        student = Student("S1", "Test Student", scores={"python": 90, "sql": 80, "statistics": 100})
        self.assertAlmostEqual(student.average, 90)
        self.assertEqual(student.grade, "A+")

    def test_score_range_is_validated(self):
        with self.assertRaises(ValueError):
            Student("S1", "Test Student", scores={"python": 101, "sql": 80, "statistics": 100})

    def test_scores_property_returns_copy(self):
        student = Student("S1", "Test Student", scores={"python": 90, "sql": 80, "statistics": 100})
        scores = student.scores
        scores["python"] = 0
        self.assertEqual(student.scores["python"], 90)


class DataSourceTests(unittest.TestCase):
    def test_csv_source_reads_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.csv"
            path.write_text("student_id,name,python,sql,statistics\nS1,Alex,80,90,70\n", encoding="utf-8")
            self.assertEqual(len(CSVDataSource(path).read_records()), 1)
            self.assertIsInstance(source_for(path), CSVDataSource)

    def test_json_source_reads_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.json"
            path.write_text(json.dumps([{"student_id": "S1", "name": "Alex"}]), encoding="utf-8")
            self.assertEqual(len(JSONDataSource(path).read_records()), 1)
            self.assertIsInstance(source_for(path), JSONDataSource)

    def test_invalid_json_shape_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.json"
            path.write_text(json.dumps({"student_id": "S1"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                JSONDataSource(path).read_records()


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.students = [
            Student("S1", "Alex", department="IT", scores={"python": 90, "sql": 90, "statistics": 90}),
            Student("S2", "Blair", department="CSE", scores={"python": 60, "sql": 60, "statistics": 60}),
        ]

    def test_sorting(self):
        result = sort_students(self.students, "average", descending=True)
        self.assertEqual(result[0].student_id, "S1")

    def test_filtering(self):
        result = filter_students(self.students, department="IT", min_average=80)
        self.assertEqual([student.student_id for student in result], ["S1"])

    def test_bad_record_is_reported(self):
        students, errors = build_students([{"student_id": "S1", "name": ""}])
        self.assertEqual(students, [])
        self.assertEqual(len(errors), 1)


class IntegrationTests(unittest.TestCase):
    def test_sample_csv_generates_reports(self):
        project_root = Path(__file__).resolve().parents[1]
        manager = DataManager()
        manager.load(project_root / "data" / "students.csv")
        self.assertEqual(manager.raw_count, 12)
        self.assertEqual(len(manager.students), 12)
        self.assertEqual(len(manager.errors), 0)
        with tempfile.TemporaryDirectory() as temp_dir:
            files = ReportGenerator(manager.students, str(manager.source_path), manager.raw_count, manager.errors).write_reports(temp_dir)
            self.assertTrue(Path(files["json"]).exists())
            self.assertTrue(Path(files["markdown"]).exists())
            self.assertTrue(Path(files["csv"]).exists())
            payload = json.loads(Path(files["json"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["valid_records"], 12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
