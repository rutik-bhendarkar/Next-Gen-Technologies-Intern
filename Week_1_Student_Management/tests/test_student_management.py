import json
import tempfile
import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import Student, StudentManager


class StudentTests(unittest.TestCase):
    def test_total_percentage_and_grade(self):
        marks = {subject: 90 for subject in Student.SUBJECTS}
        student = Student("T001", "Test Student", 21, "B.Tech", marks)
        self.assertEqual(student.total_marks, 450)
        self.assertEqual(student.percentage, 90)
        self.assertEqual(student.grade, "A+")

    def test_grade_boundaries(self):
        boundaries = [
            (90, "A+"), (80, "A"), (70, "B"),
            (60, "C"), (50, "D"), (49.9, "F")
        ]
        for mark, expected_grade in boundaries:
            marks = {subject: mark for subject in Student.SUBJECTS}
            student = Student("T002", "Boundary Test", 21, "B.Tech", marks)
            self.assertEqual(student.grade, expected_grade)

    def test_json_save_and_load(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "students.json"
            manager = StudentManager(path)
            marks = {subject: 85 for subject in Student.SUBJECTS}
            manager.add_student(Student("T003", "Save Test", 20, "B.Tech", marks))

            loaded = StudentManager(path)
            self.assertIn("T003", loaded.students)
            self.assertEqual(loaded.students["T003"].name, "Save Test")
            self.assertEqual(loaded.students["T003"].grade, "A")

    def test_search(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "students.json"
            manager = StudentManager(path)
            marks = {subject: 75 for subject in Student.SUBJECTS}
            manager.add_student(Student("T004", "Anita Singh", 22, "B.Tech", marks))
            self.assertEqual(len(manager.search("T004")), 1)
            self.assertEqual(len(manager.search("anita")), 1)
            self.assertEqual(len(manager.search("missing")), 0)

    def test_update_and_delete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "students.json"
            manager = StudentManager(path)
            marks = {subject: 65 for subject in Student.SUBJECTS}
            manager.add_student(Student("T005", "Old Name", 20, "B.Tech", marks))

            new_marks = {subject: 95 for subject in Student.SUBJECTS}
            manager.update_student("T005", {"name": "New Name", "marks": new_marks})
            self.assertEqual(manager.students["T005"].name, "New Name")
            self.assertEqual(manager.students["T005"].grade, "A+")

            manager.delete_student("T005")
            self.assertNotIn("T005", manager.students)

    def test_invalid_duplicate_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "students.json"
            manager = StudentManager(path)
            marks = {subject: 80 for subject in Student.SUBJECTS}
            student = Student("T006", "Duplicate Test", 20, "B.Tech", marks)
            manager.add_student(student)
            with self.assertRaises(ValueError):
                manager.add_student(student)


if __name__ == "__main__":
    unittest.main()
