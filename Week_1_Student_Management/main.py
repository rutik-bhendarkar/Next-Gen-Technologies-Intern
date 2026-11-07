import json
from pathlib import Path


class Student:
    """Represents one student and handles mark/grade calculations."""

    SUBJECTS = ("Python", "Mathematics", "Statistics", "Data Science", "AI/ML")

    def __init__(self, student_id, name, age, course, marks):
        self.student_id = student_id
        self.name = name
        self.age = age
        self.course = course
        self.marks = marks

    @property
    def total_marks(self):
        return sum(self.marks.values())

    @property
    def percentage(self):
        return self.total_marks / len(self.SUBJECTS)

    @property
    def grade(self):
        percentage = self.percentage
        if percentage >= 90:
            return "A+"
        if percentage >= 80:
            return "A"
        if percentage >= 70:
            return "B"
        if percentage >= 60:
            return "C"
        if percentage >= 50:
            return "D"
        return "F"

    def to_dict(self):
        return {
            "student_id": self.student_id,
            "name": self.name,
            "age": self.age,
            "course": self.course,
            "marks": self.marks,
            "total_marks": self.total_marks,
            "percentage": round(self.percentage, 2),
            "grade": self.grade,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            student_id=str(data["student_id"]),
            name=str(data["name"]),
            age=int(data["age"]),
            course=str(data["course"]),
            marks={subject: float(data["marks"][subject]) for subject in cls.SUBJECTS},
        )


class StudentManager:
    """Manages student records and persists them as JSON."""

    def __init__(self, file_path="data/students.json"):
        self.file_path = Path(file_path)
        self.students = {}
        self.load_data()

    def load_data(self):
        try:
            if not self.file_path.exists():
                self.students = {}
                return
            with self.file_path.open("r", encoding="utf-8") as file:
                raw_data = json.load(file)
            self.students = {
                str(item["student_id"]): Student.from_dict(item)
                for item in raw_data
            }
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as error:
            print(f"Warning: Could not load saved data ({error}). Starting with empty records.")
            self.students = {}

    def save_data(self):
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with self.file_path.open("w", encoding="utf-8") as file:
            json.dump([student.to_dict() for student in self.students.values()], file, indent=4)

    def add_student(self, student):
        if student.student_id in self.students:
            raise ValueError("Student ID already exists.")
        self.students[student.student_id] = student
        self.save_data()

    def get_all(self):
        return list(self.students.values())

    def search(self, query):
        query = query.strip().lower()
        return [
            student
            for student in self.students.values()
            if query in student.student_id.lower() or query in student.name.lower()
        ]

    def update_student(self, student_id, updates):
        student_id = str(student_id)
        if student_id not in self.students:
            raise KeyError("Student not found.")
        student = self.students[student_id]
        for field, value in updates.items():
            if field == "marks":
                student.marks = value
            else:
                setattr(student, field, value)
        self.save_data()

    def delete_student(self, student_id):
        student_id = str(student_id)
        if student_id not in self.students:
            raise KeyError("Student not found.")
        del self.students[student_id]
        self.save_data()

    def top_student(self):
        return max(self.students.values(), key=lambda student: student.percentage, default=None)

    def bottom_student(self):
        return min(self.students.values(), key=lambda student: student.percentage, default=None)


def get_non_empty(prompt):
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("Input cannot be empty. Please try again.")


def get_int(prompt, minimum=None, maximum=None):
    while True:
        try:
            value = int(input(prompt).strip())
            if minimum is not None and value < minimum:
                raise ValueError(f"Value must be at least {minimum}.")
            if maximum is not None and value > maximum:
                raise ValueError(f"Value must be at most {maximum}.")
            return value
        except ValueError as error:
            print(f"Invalid input: {error}")


def get_marks(existing=None):
    marks = {}
    for subject in Student.SUBJECTS:
        default = ""
        if existing and subject in existing:
            default = f" [{existing[subject]:g}]"
        while True:
            try:
                raw = input(f"Enter {subject} marks{default} (0-100): ").strip()
                if raw == "" and existing:
                    marks[subject] = float(existing[subject])
                    break
                value = float(raw)
                if not 0 <= value <= 100:
                    raise ValueError("Marks must be between 0 and 100.")
                marks[subject] = value
                break
            except ValueError as error:
                print(f"Invalid marks: {error}")
    return marks


def display_student(student):
    print("-" * 58)
    print(f"ID         : {student.student_id}")
    print(f"Name       : {student.name}")
    print(f"Age        : {student.age}")
    print(f"Course     : {student.course}")
    print("Marks      :")
    for subject, marks in student.marks.items():
        print(f"  {subject:<14}: {marks:g}")
    print(f"Total Marks: {student.total_marks:g}/500")
    print(f"Percentage : {student.percentage:.2f}%")
    print(f"Grade      : {student.grade}")


def add_student(manager):
    print("\n--- Add Student ---")
    student_id = get_non_empty("Enter student ID: ")
    name = get_non_empty("Enter student name: ")
    age = get_int("Enter age (1-100): ", 1, 100)
    course = get_non_empty("Enter course: ")
    marks = get_marks()
    try:
        manager.add_student(Student(student_id, name, age, course, marks))
        print("Student added successfully and saved to students.json.")
    except ValueError as error:
        print(f"Error: {error}")


def view_students(manager):
    print("\n--- Student Records ---")
    students = manager.get_all()
    if not students:
        print("No student records found.")
        return
    for student in students:
        display_student(student)
    print("-" * 58)
    print(f"Total students: {len(students)}")


def search_student(manager):
    print("\n--- Search Student ---")
    query = get_non_empty("Enter student ID or name: ")
    results = manager.search(query)
    if not results:
        print("No matching student found.")
        return
    print(f"Found {len(results)} student(s).")
    for student in results:
        display_student(student)


def update_student(manager):
    print("\n--- Update Student ---")
    student_id = get_non_empty("Enter student ID to update: ")
    results = manager.search(student_id)
    student = next((item for item in results if item.student_id == student_id), None)
    if student is None:
        print("Student not found.")
        return

    print("Press Enter to keep the current text value.")
    name = input(f"Name [{student.name}]: ").strip() or student.name
    age_raw = input(f"Age [{student.age}]: ").strip()
    course = input(f"Course [{student.course}]: ").strip() or student.course

    try:
        age = student.age if age_raw == "" else int(age_raw)
        if not 1 <= age <= 100:
            raise ValueError("Age must be between 1 and 100.")
        print("Enter new marks, or press Enter to keep each current mark.")
        marks = get_marks(student.marks)
        manager.update_student(student_id, {"name": name, "age": age, "course": course, "marks": marks})
        print("Student updated successfully and saved.")
    except ValueError as error:
        print(f"Invalid update: {error}")


def delete_student(manager):
    print("\n--- Delete Student ---")
    student_id = get_non_empty("Enter student ID to delete: ")
    try:
        student = manager.students.get(student_id)
        if student is None:
            raise KeyError("Student not found.")
        confirm = input(f"Delete {student.name} ({student.student_id})? [y/N]: ").strip().lower()
        if confirm == "y":
            manager.delete_student(student_id)
            print("Student deleted successfully and saved.")
        else:
            print("Delete operation cancelled.")
    except KeyError as error:
        print(f"Error: {error.args[0]}")


def show_top_bottom(manager):
    print("\n--- Performance Summary ---")
    top = manager.top_student()
    bottom = manager.bottom_student()
    if not top:
        print("No student records found.")
        return
    print(f"Top Student    : {top.name} ({top.percentage:.2f}%, Grade {top.grade})")
    print(f"Lowest Student : {bottom.name} ({bottom.percentage:.2f}%, Grade {bottom.grade})")


def main():
    manager = StudentManager()

    while True:
        print("\n" + "=" * 58)
        print("        STUDENT MANAGEMENT SYSTEM - WEEK 1")
        print("=" * 58)
        print("1. Add student")
        print("2. View students")
        print("3. Search student")
        print("4. Update student")
        print("5. Delete student")
        print("6. Performance summary")
        print("7. Exit")
        print("=" * 58)

        try:
            choice = input("Enter your choice (1-7): ").strip()
            if choice == "1":
                add_student(manager)
            elif choice == "2":
                view_students(manager)
            elif choice == "3":
                search_student(manager)
            elif choice == "4":
                update_student(manager)
            elif choice == "5":
                delete_student(manager)
            elif choice == "6":
                show_top_bottom(manager)
            elif choice == "7":
                print("Thank you for using the Student Management System.")
                break
            else:
                print("Invalid choice. Please select a number from 1 to 7.")
        except (KeyboardInterrupt, EOFError):
            print("\nProgram terminated safely.")
            break
        except OSError as error:
            print(f"File operation error: {error}")


if __name__ == "__main__":
    main()
