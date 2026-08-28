"""Unit tests for calculator.py using the Python standard library."""

import unittest

from calculator import add


class TestAdd(unittest.TestCase):
    def test_add_positive(self):
        self.assertEqual(add(1, 2), 3)

    def test_add_negative(self):
        self.assertEqual(add(-1, -2), -3)

    def test_add_mixed(self):
        self.assertEqual(add(-1, 1), 0)

    def test_add_float(self):
        self.assertAlmostEqual(add(1.5, 2.25), 3.75)

    def test_add_zero(self):
        self.assertEqual(add(0, 0), 0)


if __name__ == "__main__":
    unittest.main()
