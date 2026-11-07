# Week 5 – Mathematics for Machine Learning: Practical Report

## 1. Title
Mathematical Foundations for Machine Learning Using Python and NumPy

## 2. Objective
To study and implement descriptive statistics, probability, vectors, matrices, and matrix multiplication using Python and NumPy. Core calculations are implemented using explicit loops wherever practical to make the underlying mathematical logic clear.

## 3. Tools and libraries
- Python 3.12 (or a compatible Python 3 version)
- NumPy for numerical arrays and verification
- Pandas for reading the sample CSV in the analysis workflow
- Jupyter Notebook for practical demonstration

## 4. Dataset
The project uses `data/student_performance.csv`, a synthetic educational dataset with 12 rows and 5 columns. It contains study hours per day, exam scores, attendance percentage, and assignments completed. It is synthetic and is not sourced from real students.

## 5. Theory and formulas

### Mean
`Mean = sum of observations / number of observations`.

### Median
Sort the observations. For odd `n`, select the middle value. For even `n`, average the two central values.

### Mode
The mode is the value with the highest frequency. This implementation returns all values tied for the highest frequency.

### Variance
Population variance: `sum((x - population_mean)^2) / N`.
Sample variance: `sum((x - sample_mean)^2) / (n - 1)`.

### Standard deviation
`Standard deviation = square root of variance`.

### Probability
Empirical probability: `event count / total observations`, where total observations must be positive and event count must be between zero and total.

### Vectors
A vector is an ordered one-dimensional collection. Operations demonstrated: addition, subtraction, scalar multiplication, dot product, and Euclidean norm.

### Matrices
A matrix is a two-dimensional rectangular array. Operations demonstrated: addition, transpose, and multiplication. For `A(m × n)` and `B(p × q)`, multiplication is valid only if `n = p`; the output has shape `m × q`.

## 6. Implementation
The main implementation is in `src/math_operations.py`. The custom functions use loops for accumulation, insertion sorting, frequency counting, squared-deviation calculations, vector calculations, and matrix calculations. NumPy is used for array storage, numeric validation, and comparison checks.

## 7. Results

### Exam score statistics
- Count: 12
- Minimum: 67.0
- Maximum: 95.0
- Range: 28.0
- Mean: 82.5000
- Median: 85.0000
- Mode(s): [85.0]
- Population variance: 71.5833
- Population standard deviation: 8.4607
- Sample variance: 78.0909
- Sample standard deviation: 8.8369

### Study hours
- Mean: 3.1333
- Median: 3.1000
- Mode(s): [1.5, 1.8, 2.0, 2.5, 2.8, 3.0, 3.2, 3.5, 3.8, 4.0, 4.5, 5.0]
- Population standard deviation: 1.0387

### Attendance percentage
- Mean: 90.0833
- Median: 93.5000
- Mode(s): [95.0]
- Population standard deviation: 7.8151

### Empirical probabilities
- Probability that a sampled exam score is at least 85: `0.5833` (58.33%).
- Probability that a sampled attendance value is at least 90%: `0.6667` (66.67%).

These probabilities describe only the 12 synthetic records in this exercise; they are not population-level claims.

## 8. Verification and testing
The notebook compares manual calculations with NumPy reference calculations for mean, median, variance, standard deviation, dot product, and matrix multiplication. Automated unit tests cover valid calculations, even/odd medians, tied modes, invalid probabilities, empty inputs, and incompatible vector/matrix shapes.

Test output is saved in `outputs/test_results.txt`.

## 9. Conclusion
The practical implements the mathematical building blocks commonly used in machine learning. The exercise clarifies how averages and spread are calculated, how empirical probability is estimated, and how vector/matrix operations are performed. Explicit loops expose the arithmetic while NumPy verification provides an independent correctness check.

## 10. Limitations
- The dataset is small, synthetic, and intended for learning only.
- Empirical probabilities from a small sample can change substantially with additional observations.
- Mode is not always unique; the implementation reports all tied modes.
- This practical focuses on foundational operations and is not a predictive machine-learning model.

## 11. Files submitted
- Python source code
- Jupyter Notebook with explanations and executed outputs
- Sample dataset
- Results JSON and console output
- Automated tests
- README and this report
