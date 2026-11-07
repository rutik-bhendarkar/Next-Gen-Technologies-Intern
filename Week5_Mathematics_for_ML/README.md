# Week 5 – Mathematics for Machine Learning

## Internship assignment coverage
This project follows Week 5 of the supplied 24-week AI/ML internship plan. It covers mean, median, mode, variance, standard deviation, probability, basic statistics, vectors, matrices, and matrix multiplication. Core calculations are written with explicit Python loops wherever practical; NumPy is used for array representation, input checks, and independent verification.

## Project structure
```text
Week5_Mathematics_for_ML/
├── data/student_performance.csv
├── notebooks/Week5_Mathematics_for_ML.ipynb
├── src/math_operations.py
├── run_analysis.py
├── tests/test_math_operations.py
├── reports/Week5_Mathematics_Analysis_Report.md
├── outputs/results_summary.json
├── outputs/console_results.txt
├── requirements.txt
└── README.md
```

## Setup (Windows PowerShell)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the analysis
From the project root:
```powershell
python run_analysis.py
```
This reads `data/student_performance.csv` and writes the computed summary to `outputs/`.

## Run the notebook
```powershell
python -m notebook
```
Open `notebooks/Week5_Mathematics_for_ML.ipynb` and choose **Run All**.

## Run tests
```powershell
python -m unittest discover -s tests -v
```

## Mathematical conventions
- Population variance divides the sum of squared deviations by `n`.
- Sample variance divides by `n - 1` (and requires at least two values).
- Standard deviation is the square root of the corresponding variance.
- Mode returns every value tied for the highest frequency.
- Empirical probability is `number of observed event cases / total observed cases`.
- Matrix multiplication is defined only when the number of columns in the first matrix equals the number of rows in the second.

## Dataset note
`student_performance.csv` is a small synthetic teaching dataset created for this exercise. It is not real student or institutional data. The observed probabilities describe only these sample rows.

## Important for submission
Run the project in your own environment, inspect all notebook outputs, and make sure you can explain each formula and result. Add your name/registration details only where your institute requests them; do not insert private data into the sample dataset.
