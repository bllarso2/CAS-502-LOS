# Running a Pytest on CAS502_FINAL_ST
To ensure the reproducibility of the **CAS502_FINAL_ST** model and facilitate collaborative testing, we recommend following these steps. The latest version of the code is available on **GitHub**, allowing users to clone the repository using:

git clone <repository-url>

Then, navigate to the project directory.

## 1. Setting Up the Environment

A virtual environment should be created and activated to ensure **dependency isolation**. We used:

python -m venv cas502_env

- **macOS/Linux**:
source cas502_env/bin/activate

If using **Anaconda**, as we did, users can set up the environment with:
conda create --name cas502_env python=3.9 conda activate cas502_env

The required dependencies should be installed using:
pip install -r requirements.txt

(This file is located in the **shared repository**.)

Alternatively, users can manually install key dependencies:
pip install pandas scikit-learn jupyter pytest

## 2. Running and Validating the Model

### **A. Running the Jupyter Notebook**
To validate the model, **Jupyter Notebook** can be tested by opening it in the terminal with:

jupyter notebook

Then:
1. Restart the kernel
2. Clear outputs
3. Run all cells sequentially to verify the model's functionality.

### **B. Converting the Notebook to a Python Script**
Users can convert the notebook into a Python script with:
jupyter nbconvert --to script CAS502_FINAL_ST.ipynb

Then execute:
python CAS502_FINAL_ST.py

to confirm that the model runs correctly and produces a **predicted Length of Stay (LOS)**.

### **C. Running Pytest for Automated Validation**
An **automated validation pipeline** is provided in `test_CAS502_FINAL_ST.py`, which can be executed with:

pytest test_CAS502_FINAL_ST.py -v --tb=short

This will check:
- **Preprocessing pipeline**
- **Model training**
- **Prediction outputs**

## 3. Contributing via GitHub

Since the code is hosted on **GitHub**, contributors are encouraged to:
- Report **issues**
- Submit **pull requests** for improvements
- Ensure that any **modifications** are tested before merging.

## 4. Troubleshooting

If the test **fails**, address the following:
- Verify **dataset integrity** (we had to do this multiple times before running a successful test).
- Check for **missing dependencies**.
- Ensure compatibility with **Python 3.9**.

These steps should provide a **comprehensive guideline** for replicating and testing the model while maintaining **transparency** and **version control** through GitHub.

