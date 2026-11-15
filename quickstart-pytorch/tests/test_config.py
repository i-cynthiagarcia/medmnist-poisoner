import unittest

from pytorchexample.shared.config import (
    require_config_float,
    require_config_int,
    require_config_string,
)


class ConfigValidationTests(unittest.TestCase):
    def test_integer_validation_rejects_booleans(self):
        with self.assertRaisesRegex(TypeError, "must be an integer"):
            require_config_int(True, "seed")

    def test_string_validation_rejects_non_string_values(self):
        with self.assertRaisesRegex(TypeError, "must be a string"):
            require_config_string(42, "dataset")

    def test_numeric_validation_accepts_integers_as_floats(self):
        self.assertEqual(require_config_float(1, "lr"), 1.0)


if __name__ == "__main__":
    unittest.main()
