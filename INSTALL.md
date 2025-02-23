 # Installation Instructions

Below is a step-by-step guide to install and run this project. If you have any questions or encounter issues, please consult the **Troubleshooting** section or open an issue.

---

## Cloning the Repository
### bash:
git clone https://github.com/bllarso2/CAS-502-LOS.git
cd CAS-502-LOS

## Setting up a Virtual Environment
### bash:
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

Note: This project requires Python 3.8+. Make sure you have an appropriate Python version installed.

## Installing Dependencies
### bash:
pip install -r requirements.txt

The requirements.txt file includes all the libraries needed, including Streamlit and any other dependencies for the project.

## Running the Application
### bash:
streamlit run streamlit_app.py

## Testing 
Can do 
### bash
pytest
or
python -m unittest


## Troubleshooting
## Installation Fails
Make sure that you are using a clean virtual environement and have an updated pip:

### bash
pip install -- upgrade pip

If there are still remaining dependency conflicts try:

###bash
rm rf - venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

