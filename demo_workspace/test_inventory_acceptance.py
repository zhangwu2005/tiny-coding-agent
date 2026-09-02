import unittest

from inventory import reserve_stock


class TestReserveStock(unittest.TestCase):
    def test_aggregates_duplicates_without_mutating_input(self):
        stock = {"A": 5, "B": 3}
        result = reserve_stock(stock, [("A", 2), ("A", 1), ("B", 2)])
        self.assertEqual(result, {"A": 2, "B": 1})
        self.assertEqual(stock, {"A": 5, "B": 3})
        self.assertIsNot(result, stock)

    def test_failure_is_atomic(self):
        stock = {"A": 5, "B": 1}
        with self.assertRaises(ValueError):
            reserve_stock(stock, [("A", 2), ("B", 2)])
        self.assertEqual(stock, {"A": 5, "B": 1})

    def test_missing_sku_raises_key_error(self):
        stock = {"A": 5}
        with self.assertRaises(KeyError):
            reserve_stock(stock, [("B", 1)])
        self.assertEqual(stock, {"A": 5})

    def test_rejects_invalid_quantities(self):
        for quantity in [0, -1, 1.5, True]:
            with self.subTest(quantity=quantity):
                with self.assertRaises(ValueError):
                    reserve_stock({"A": 5}, [("A", quantity)])

    def test_empty_request_returns_a_copy(self):
        stock = {"A": 5}
        result = reserve_stock(stock, [])
        self.assertEqual(result, stock)
        self.assertIsNot(result, stock)
