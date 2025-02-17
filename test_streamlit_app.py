import unittest
import pandas as pd
from streamlit_app import load_data  # Import the function directly

class TestStreamlitFunctions(unittest.TestCase):

    def test_load_data(self):
        """Test if load_data correctly loads a CSV file."""
        test_file = "Hospital-LOS.csv"  # Ensure this file exists
        df = load_data(test_file)  # Call function directly

        self.assertIsInstance(df, pd.DataFrame, "Expected a pandas DataFrame")
        self.assertFalse(df.empty, "DataFrame should not be empty")
        self.assertIn("LOS", df.columns, "Expected 'LOS' column in DataFrame")

if __name__ == "__main__":
    unittest.main()

