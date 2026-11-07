import unittest
import numpy as np
from src.math_operations import (
    manual_sum, manual_mean, manual_median, manual_mode, manual_variance,
    manual_standard_deviation, empirical_probability, probability_from_data,
    vector_add, vector_subtract, vector_scale, vector_dot, vector_norm,
    matrix_add, matrix_transpose, matrix_multiply
)

class TestStatistics(unittest.TestCase):
    def test_sum_mean(self):
        self.assertAlmostEqual(manual_sum([1,2,3,4]), 10)
        self.assertAlmostEqual(manual_mean([1,2,3,4]), 2.5)
    def test_median_odd_even(self):
        self.assertAlmostEqual(manual_median([9,1,5]), 5)
        self.assertAlmostEqual(manual_median([4,1,3,2]), 2.5)
    def test_mode_tie(self):
        self.assertEqual(manual_mode([1,2,2,3,3]), [2.0,3.0])
    def test_variance_population_and_sample(self):
        self.assertAlmostEqual(manual_variance([2,4,6]), 8/3)
        self.assertAlmostEqual(manual_variance([2,4,6], sample=True), 4)
    def test_std(self):
        self.assertAlmostEqual(manual_standard_deviation([2,4,6]), (8/3)**0.5)
    def test_empty_rejected(self):
        with self.assertRaises(ValueError): manual_mean([])
    def test_probability(self):
        self.assertAlmostEqual(empirical_probability(3, 8), 0.375)
        self.assertAlmostEqual(probability_from_data([50,80,90,95], lambda x: x >= 80), 0.75)
    def test_invalid_probability(self):
        with self.assertRaises(ValueError): empirical_probability(5, 4)

class TestLinearAlgebra(unittest.TestCase):
    def test_vector_operations(self):
        np.testing.assert_allclose(vector_add([1,2],[3,4]), [4,6])
        np.testing.assert_allclose(vector_subtract([5,7],[2,3]), [3,4])
        np.testing.assert_allclose(vector_scale([1,2], 3), [3,6])
        self.assertAlmostEqual(vector_dot([1,2,3],[4,5,6]), 32)
        self.assertAlmostEqual(vector_norm([3,4]), 5)
    def test_matrix_add_transpose(self):
        np.testing.assert_allclose(matrix_add([[1,2],[3,4]], [[5,6],[7,8]]), [[6,8],[10,12]])
        np.testing.assert_allclose(matrix_transpose([[1,2,3],[4,5,6]]), [[1,4],[2,5],[3,6]])
    def test_matrix_multiply(self):
        np.testing.assert_allclose(matrix_multiply([[1,2,3],[4,5,6]], [[7,8],[9,10],[11,12]]), [[58,64],[139,154]])
    def test_shape_errors(self):
        with self.assertRaises(ValueError): vector_add([1,2], [1])
        with self.assertRaises(ValueError): matrix_multiply([[1,2]], [[1,2]])

if __name__ == '__main__':
    unittest.main()
