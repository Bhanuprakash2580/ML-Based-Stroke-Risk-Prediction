import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



@st.cache_data

def load_data():
    """Load and return the heart disease dataset.

    The path is calculated relative to this file so the script can be imported
    from anywhere (e.g. during testing) without failing.
    """
    here = os.path.dirname(__file__)
    data_path = os.path.join(here, "..", "Data", "heart_disease_data.csv")
    return pd.read_csv(data_path)



@st.cache_data

def train_model(df):
    """Train a logistic regression model and return it together with accuracies."""
    X = df.drop(columns="target")
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    train_acc = accuracy_score(model.predict(X_train), y_train)
    test_acc = accuracy_score(model.predict(X_test), y_test)
    return model, train_acc, test_acc


def main():
    st.set_page_config(
        page_title="Heart Disease Predictor",
        page_icon="❤️‍🩹",
        layout="wide",
    )

    st.title("❤️‍🩹 Heart Disease Predictor")
    st.markdown(
        "This demo lets you interact with a logistic regression model trained on the "
        "UCI heart disease dataset. You can enter values manually or upload a file."
    )

    # load and train model once
    df = load_data()
    model, train_acc, test_acc = train_model(df)

    # sidebar controls
    st.sidebar.header("Model performance")
    st.sidebar.write(f"Training accuracy: {train_acc:.2%}")
    st.sidebar.write(f"Test accuracy: {test_acc:.2%}")
    mode = st.sidebar.radio("Input mode", options=["Single patient", "Batch upload"])

    # dataset visualization
    with st.expander("Explore data distribution"):
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="age", nbins=20, title="Age distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.histogram(df, x="chol", nbins=20, title="Cholesterol distribution")
            st.plotly_chart(fig2, use_container_width=True)

    if mode == "Single patient":
        st.header("Patient input")
        with st.form(key="input_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input("Age", min_value=0, max_value=120, value=50)
                sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
                cp = st.number_input("Chest pain type (cp)", min_value=0, max_value=3, value=1)
                trestbps = st.number_input("Resting blood pressure (trestbps)", min_value=0, max_value=300, value=120)
                chol = st.number_input("Serum cholesterol (chol)", min_value=0, max_value=600, value=200)
            with c2:
                fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", options=[0, 1])
                restecg = st.number_input("Resting ECG results", min_value=0, max_value=2, value=1)
                thalach = st.number_input("Max heart rate achieved", min_value=0, max_value=300, value=150)
                exang = st.selectbox("Exercise induced angina", options=[0, 1])
            with c3:
                oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
                slope = st.number_input("Slope of peak exercise ST segment", min_value=0, max_value=2, value=1)
                ca = st.number_input("Major vessels (ca)", min_value=0, max_value=4, value=0)
                thal = st.selectbox("Thalassemia", options=[1, 2, 3])

            submit = st.form_submit_button("Predict")

        if submit:
            input_data = np.array([
                age,
                sex,
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal,
            ]).reshape(1, -1)

            proba = model.predict_proba(input_data)[0, 1]
            pred = int(proba >= 0.5)
            st.metric(
                label="Predicted probability of heart disease",
                value=f"{proba:.1%}",
                delta="✅" if pred == 0 else "⚠️",
            )
            if pred == 0:
                st.success("Model predicts no heart disease.")
            else:
                st.error("Model predicts heart disease.")

    else:  # batch
        st.header("Batch prediction")
        uploaded = st.file_uploader("Upload CSV file with same columns as training data", type="csv")
        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            st.write("First rows of uploaded data:")
            st.dataframe(batch_df.head())
            if st.button("Run batch prediction"):
                results = model.predict(batch_df)
                batch_df["prediction"] = results
                st.write(batch_df)
                st.download_button(
                    "Download results",
                    data=batch_df.to_csv(index=False),
                    file_name="predictions.csv",
                    mime="text/csv",
                )



if __name__ == "__main__":
    main()
