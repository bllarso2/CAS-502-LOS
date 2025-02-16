import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Ensure Python finds data_processing

import unittest
import pandas as pd
from data_processing import preprocess_data  # Now Python should recognize it

class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        """Create a sample dataset before each test."""
        self.df = pd.DataFrame({
            "Stay (in days)": [1, 2, None, 4],  # 4 values ✅
            "Department": ["gynecology", "radiotherapy", "surgery", "anesthesia"],  # 4 values ✅
            "Gender": ["M", "F", "M", "F"]  # 4 values ✅
        })

    def test_preprocess_data(self):
        """Test if preprocess_data correctly handles missing values and transformations."""
        processed_df = preprocess_data(self.df)

        # Ensure missing values are handled
        self.assertFalse(processed_df.isnull().values.any(), "Processed data should not have missing values.")

        # Ensure 'Stay (in days)' column is still present
        self.assertIn("Stay (in days)", processed_df.columns, "Processed data should contain 'Stay (in days)' column.")

        # Ensure 'Stay (in days)' column is float type
        self.assertEqual(processed_df["Stay (in days)"].dtype, float, "'Stay (in days)' column should be of type float.")

if __name__ == "__main__":
    unittest.main()

