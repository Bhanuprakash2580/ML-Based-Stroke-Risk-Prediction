# ML‑Based Heart Disease (Stroke Risk) Predictor

This repository contains a machine learning project that trains a classification model on a publicly
available heart disease dataset and exposes the trained predictor via a Streamlit web application.
The original code lives in the Jupyter notebooks found under `Data/` (`Heart_disease_predictor.ipynb`,
`Heart Disease Prediction.ipynb`), and the Streamlit wrapper is implemented in `app/app.py`.

Although the filenames mention "heart disease", the model and interface are intended to illustrate a
**stroke risk prediction** workflow; you are free to adapt or extend it to other medical
classification problems.

---

## 📁 Repository Structure

```
app.py                         # thin entry point for the streamlit app at workspace root
app/                           # streamlit application code
    app.py                     # main application script
    requirements.txt          # dependencies needed by the app
Data/                          # dataset and exploratory notebooks
    heart_disease_data.csv     # raw data used by the models
    ...                       # additional notebooks and data dictionary
```

---

## 🚀 Getting Started

### 1. Environment setup

```bash
python -m venv .venv             # create a virtual environment
.\.venv\Scripts\activate       # activate on Windows
# (use `source .venv/bin/activate` on macOS/Linux)
```

### 2. Install dependencies

```bash
pip install -r requirements.txt  # root requirements install
pip install -r app/requirements.txt  # app-specific packages (streamlit, etc.)
```

> _Tip:_ If you prefer a single command, you can `pip install -r requirements.txt -r app/requirements.txt`.

### 3. Run the Streamlit app

```bash
streamlit run app/app.py
```

A browser window will open automatically. Fill in the patient features on the form and click **Predict**
to view the model's output.

Data used by the app is loaded from `Data/heart_disease_data.csv` relative to the workspace root;
ensure this file is present before launching.

---

## 📊 Dataset

The dataset is a version of the Cleveland heart disease database originally collected by the
UCI Machine Learning Repository. It contains the following columns (see
`Data/Data_Dictionary.md` for full descriptions):

- `age` – age in years
- `sex` – 1 = male, 0 = female
- `cp` – chest pain type (0–3)
- `trestbps` – resting blood pressure
- `chol` – serum cholesterol
- ...
- `target` – diagnosis of heart disease (0 = no disease, 1 = disease)

### 📁 Data Dictionary

Additional information about the variables is available in
`Data/Data_Dictionary.md`.

---

## 🛠 Development

- Notebooks in `Data/` contain the data exploration, preprocessing, and model training steps.
  Feel free to modify them or add new analyses.
- The Streamlit app reads a serialized model artifact (pickled scikit‑learn object) from the
  `app/` directory; regenerate it by re‑running the training notebook and saving the result.

---

## 📘 Contributing

Improvements, bug fixes, or enhancements are very welcome! Here’s a suggested workflow:

1. Fork the repository and create a feature branch.
2. Make your changes and add appropriate tests/notebook outputs.
3. Commit your work with clear messages.
4. Push the branch and open a pull request against the main branch.

> **Git tip:** if you encounter `error: src refspec main does not match any` during `git push`,
> make sure you have committed at least once and either rename your current branch to `main` or
> push the branch you are on (e.g. `git push -u origin master`).

---

## 📄 License

This project is provided under the MIT license. See `LICENSE` for details.
