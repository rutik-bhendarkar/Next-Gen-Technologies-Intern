# Week 3 — Numerical Computing with NumPy

## Assignment brief
Perform numerical analysis on a dataset using NumPy. The Week 3 plan asks for arrays, indexing, slicing, broadcasting, mathematical operations, statistical functions, and matrix operations, with at least 10 numerical operations, 5 statistical calculations, and 3 matrix operations.

## Project contents

```text
Week3_NumPy_Numerical_Analysis/
├── data/
│   └── retail_sales.csv
├── notebooks/
│   └── Week3_NumPy_Numerical_Analysis.ipynb
├── outputs/
│   ├── correlation_matrix.csv
│   ├── monthly_revenue.csv
│   └── summary_metrics.json
├── report/
│   └── Week3_Results_Explanation.md
├── src/
│   ├── __init__.py
│   └── numpy_analysis.py
├── tests/
│   └── test_numpy_analysis.py
├── run_analysis.py
└── requirements.txt
```

## Requirements
- Python 3.10 or newer
- NumPy
- Jupyter Notebook (to open/run the notebook)

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the notebook
From the project folder:

```bash
jupyter notebook
```

Open `notebooks/Week3_NumPy_Numerical_Analysis.ipynb` and choose **Run All**.

## Run the analysis script

```bash
python run_analysis.py
```

This regenerates the files inside `outputs/`.

## Run validation tests

```bash
python -m unittest discover -s tests -v
```

## Coverage checklist
- [x] NumPy arrays and array data types/shapes
- [x] Indexing and slicing
- [x] Broadcasting
- [x] At least 10 numerical operations (14 included)
- [x] At least 5 statistical calculations (7 included)
- [x] At least 3 matrix operations (transpose, multiplication, covariance, plus determinant/inverse/solve)
- [x] Jupyter notebook
- [x] Dataset
- [x] Explanation of results

## Dataset note
The included retail dataset is synthetic and reproducibly generated for learning. It must not be described as real company/customer data. Monetary values are in INR.
