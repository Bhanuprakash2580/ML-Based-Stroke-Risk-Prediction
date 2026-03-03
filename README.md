# Heart Disease Predictor Streamlit App

This folder contains a Streamlit application that wraps the code from the `Heart_disease_predictor.ipynb` notebook
and provides a user-friendly web interface.

## Setup

1. **Create an environment** (recommended):
   ```bash
   python -m venv .venv
   # activate it on Windows
   .\.venv\Scripts\activate
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the app

From the workspace root run:

```bash
streamlit run app/app.py
```

The app will start in your default browser. Enter patient data via the form and click **Predict** to
see whether the model estimates the presence of heart disease.

> **Note:** data is read from `Data/heart_disease_data.csv` relative to the workspace root.

## Files

- `app.py` – main Streamlit script.
- `requirements.txt` – Python packages required to run the app.
- `README.md` – this file.
