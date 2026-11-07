# Week 2: Advanced Python Data Management

**AI/ML Internship - 24-Week Practical Task Plan**  
**Assignment:** Build a data management application that reads CSV/JSON files, processes the information, and generates useful reports.

**Prepared for:** Rutik Bhendarkar

## What this project does

- Reads student records from **CSV or JSON**.
- Validates required fields, year values, and scores (0-100).
- Uses classes and objects to represent people and students.
- Demonstrates **inheritance** (`Student` extends `Person`; CSV/JSON adapters extend `DataSource`).
- Demonstrates **encapsulation** through private/internal state and safe properties.
- Uses reusable functions, a factory function, a lambda-based sort key, list/dictionary comprehensions, and the standard library.
- Filters records by department and minimum average.
- Calculates student averages and grades, subject averages, median, grade distribution, department counts, top students, and students needing support.
- Exports `report.json`, `report.md`, and `student_results.csv`.
- Includes a sample dataset in both CSV and JSON, automated tests, and demo evidence.

The application uses only Python's standard library, so no third-party installation is required.

## Requirements

- Python 3.10 or newer (tested with Python 3.12).
- No `pip install` is required.

## Project structure

```text
Week2_Advanced_Python_Data_Management/
├── student_data_manager/
│   ├── __init__.py
│   ├── cli.py
│   ├── data_sources.py
│   ├── manager.py
│   ├── models.py
│   ├── processing.py
│   └── reports.py
├── data/
│   ├── students.csv
│   └── students.json
├── outputs/                 # generated CSV-input example reports
├── outputs_json/            # generated JSON-input example reports
├── outputs_it/              # generated filtered example reports
├── screenshots/             # demo evidence images generated from actual CLI output
├── tests/
│   └── test_data_manager.py
├── main.py
├── requirements.txt
└── README.md
```

## Run the application

Open a terminal in this folder.

### 1. Import CSV and generate reports

```bash
python main.py --input data/students.csv --output outputs
```

### 2. Import JSON and generate reports

```bash
python main.py --input data/students.json --output outputs_json
```

### 3. Filter records

Only IT students with an average score of at least 80:

```bash
python main.py --input data/students.csv --output outputs_it --department IT --min-average 80
```

Sort by name alphabetically:

```bash
python main.py --input data/students.csv --output outputs_by_name --sort-by name --ascending
```

Choose how many top students appear in the report:

```bash
python main.py --input data/students.csv --output outputs_top3 --top 3
```

## Output files

- `report.json` - machine-readable summary with counts, statistics, grade distribution, top students, and validation messages.
- `report.md` - readable Markdown report suitable for review or conversion into a PDF.
- `student_results.csv` - processed student data with average and grade added.

## Grade rules

| Average score | Grade |
|---:|:---|
| 90-100 | A+ |
| 80-89.99 | A |
| 70-79.99 | B |
| 60-69.99 | C |
| 50-59.99 | D |
| Below 50 | F |

The grade is based on the arithmetic mean of Python, SQL, and Statistics scores. This is a project rule for the sample application, not an official college grading rule.

## Run the automated tests

From the project root:

```bash
python -m unittest discover -s tests -v
```

The tests cover model validation, score encapsulation, CSV/JSON readers, invalid JSON structures, sorting, filtering, invalid-row handling, and report generation.

## Python concepts demonstrated

| Week 2 topic | Where it is demonstrated |
|---|---|
| Functions in depth | `processing.py`, `reports.py`, `cli.py` |
| Lambda functions | `processing.sort_students()` |
| List/dictionary comprehensions | `processing.py`, `reports.py` |
| Modules and packages | `student_data_manager/` package |
| Classes and objects | `Person`, `Student`, `DataManager`, `ReportGenerator` |
| Inheritance | `Student(Person)`, `CSVDataSource(DataSource)`, `JSONDataSource(DataSource)` |
| Encapsulation | Private `Student.__student_id`, internal score state, private `DataManager` collections and properties that return copies |
| CSV and JSON | `data_sources.py` |
| Data processing and reports | `processing.py`, `reports.py` |

## Data quality and limitations

- The input JSON must be a list of objects; the input CSV must contain a header row.
- Required fields are student ID, name, Python, SQL, and Statistics scores.
- A record with missing required values or invalid scores/year is skipped and its reason is included in the report.
- Student names and example emails are synthetic sample data.
- This is an educational command-line application, not a hosted database or production student information system.

## Internship submission checklist

- [x] Source code
- [x] Sample dataset (CSV and JSON)
- [x] README
- [x] Demo evidence images generated from actual CLI output
- [x] Automated tests
- [x] Generated reports

## Author note

Review the code, run the app yourself, and make sure you can explain how the data source classes, validation, grade calculation, report generation, and tests work before submitting it as your own internship work. Add your own name and any required internship details before final submission.
