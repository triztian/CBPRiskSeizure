import unittest

from etl_ipr_data import normalize_string_value


class TestEtl(unittest.TestCase):
    def test_normalize_column_no_changes(self):
        tests = [
            ("abc", "abc"),
            ("some_name", "some_name"),
            ("some_name_1", "some_name_1"),
        ]
        for input, expected in tests:
            result = normalize_string_value(input)
            self.assertEqual(result, expected)

    def test_normalize_column_all_changes(self):
        tests = [
            ("(abc)", "_abc_"),
            ("some name", "some_name"),
        ]
        for input, expected in tests:
            result = normalize_string_value(input)
            self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
