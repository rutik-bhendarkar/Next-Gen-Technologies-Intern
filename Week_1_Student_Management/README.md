# Student Management System - Week 1 AI/ML Internship

## 1. Project Overview

This project is the Week 1 practical assignment for the AI/ML internship. It demonstrates core Python programming concepts by implementing a command-line Student Management System.

## 2. Objectives

The project demonstrates:

- Variables and data types
- Operators
- Conditional statements
- Loops
- Functions
- Lists, tuples, sets and dictionaries
- String manipulation
- Exception handling
- File handling
- Basic Object-Oriented Programming

## 3. Features

1. Add student
2. View student records
3. Search student by ID or name
4. Update student information and marks
5. Delete a student
6. Calculate total marks, percentage and grade
7. Show highest and lowest performers
8. Save records to a JSON file
9. Load existing records automatically when the program starts
10. Validate invalid input without crashing the program

## 4. Technologies

- Python 3.10 or newer
- Python standard library
- JSON for persistent storage

No third-party packages are required.

## 5. Grade Rules

| Percentage | Grade |
|---:|:---|
| 90-100 | A+ |
| 80-89.99 | A |
| 70-79.99 | B |
| 60-69.99 | C |
| 50-59.99 | D |
| Below 50 | F |

Five subjects are used, each out of 100:

- Python
- Mathematics
- Statistics
- Data Science
- AI/ML

Percentage = Total Marks / 5

## 6. Project Structure

```text
Week_1_Student_Management/
|
|-- main.py
|-- requirements.txt
|-- run.bat
|-- README.md
|
|-- data/
|   `-- students.json
|
|-- tests/
|   `-- test_student_management.py
|
`-- screenshots/
    |-- 01_menu_and_records.png
    |-- 02_search_and_summary.png
    |-- 03_update_record.png
    |-- 04_delete_record.png
    `-- 05_invalid_input_handling.png
```

## 7. How to Run on Windows

1. Install Python 3.10+.
2. Open Command Prompt or PowerShell.
3. Navigate to this project folder.
4. Run:

```bash
python main.py
```

You can also double-click `run.bat` on Windows.

## 8. How to Use

When the program starts, choose one of the menu options:

```text
1. Add student
2. View students
3. Search student
4. Update student
5. Delete student
6. Performance summary
7. Exit
```

Student records are automatically saved to `data/students.json` after add, update and delete operations.

## 9. Testing

Automated unit tests are included in `tests/test_student_management.py`.

Run them from the project folder:

```bash
python -m unittest discover -s tests -v
```

A successful run should show all tests passing.

## 10. Week 1 Learning Outcome

After completing this project, the learner has applied Python fundamentals to a complete CRUD-style application with input validation, file persistence, calculations, and basic OOP.

## 11. Future Improvements

Possible extensions for later weeks include a GUI, database storage using SQL, authentication, analytics, and integration with data-science or machine-learning components.
