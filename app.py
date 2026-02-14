import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.title("ML Classification Model Comparison")

uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

model_option = st.selectbox(
    "Select Model",
    [
        "Logistic Regression"
    ]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    model = joblib.load(f"{model_option}.pkl")

    X = df.drop("income", axis=1)
    y = df["income"]

    y_pred = model.predict(X)

    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    st.pyplot(fig)

    st.download_button(
        label="Download Test CSV",
        data=df.to_csv(index=False),
        file_name="test_data.csv"
    )
