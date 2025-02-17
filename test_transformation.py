import unittest
import pandas as pd
import numpy as np  # Import NumPy to handle NaN comparison
from data_processing import transform_data

class TestTransformation(unittest.TestCase):

    def test_transform_gender(self):
        """Test that gender encoding works correctly."""
        df = pd.DataFrame({"Gender": ["M", "F", "M", "F", "M"]})  # Sample data
        transformed_df = transform_data(df)

        expected = [0, 1, 0, 1, 0]  # Expected output after transformation
        result = transformed_df["Gender"].tolist()  # Convert column to list
        
        self.assertEqual(result, expected, "Gender should be encoded as 0 (M) and 1 (F).")

    def test_transform_gender_missing_values(self):
        """Test that missing gender values are handled properly (stay NaN)."""
        df = pd.DataFrame({"Gender": ["M", None, "F", "M"]})
        transformed_df = transform_data(df)

        expected = [0, np.nan, 1, 0]  # Use np.nan instead of None
        result = transformed_df["Gender"].tolist()

        for i in range(len(expected)):
            if pd.isna(expected[i]):
                self.assertTrue(pd.isna(result[i]), f"Expected NaN at index {i}, but got {result[i]}")
            else:
                self.assertEqual(result[i], expected[i], f"Mismatch at index {i}: expected {expected[i]}, got {result[i]}")

if __name__ == "__main__":
    unittest.main()

