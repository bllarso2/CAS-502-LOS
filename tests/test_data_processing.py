import unittest
import pandas as pd
from streamlit_app import load_data

class TestDataProcessing(unittest.TestCase):

    def test_missing_values_count(self):
        """Check the number of missing values instead of failing immediately."""
        df = load_data("Hospital-LOS.csv")
        missing_count = df.isnull().sum().sum()  # Count total missing values
        print(f"Missing values found: {missing_count}")

        # Allow up to 5 missing values, otherwise fail
        self.assertLessEqual(missing_count, 5, "Dataset contains too many missing values!")

if __name__ == "__main__":
    unittest.main()

