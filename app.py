import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score
)

st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.metric-box {
    background: #ffffff;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #e6e6e6;
    text-align: center;
}
.metric-value {
    font-size: 24px;
    font-weight: 700;
}
.metric-label {
    font-size: 13px;
    color: #666;
}
</style>
""", unsafe_allow_html=True)

st.title("Machine Learning Model Evaluation Dashboard")
st.caption("Upload labeled test dataset to evaluate trained classification models.")

st.sidebar.header("Configuration")

model_choice = st.sidebar.selectbox(
    "Select Model",
    [
        "logistic_regression",
        "decision_tree",
        "knn",
        "naive_bayes",
        "random_forest",
        "xgboost"
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Labeled Test Dataset (CSV)",
    type=["csv"]
)

@st.cache_resource
def load_model(name):
    return joblib.load(f"model/{name}.pkl")

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if "income" not in df.columns:
        st.error("Target column 'income' not found in dataset.")
        st.stop()

    y_true = df["income"]
    X = df.drop("income", axis=1)

    model = load_model(model_choice)

    y_pred = model.predict(X)

    y_true = y_true.astype(str)
    y_pred = y_pred.astype(str)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    mcc = matthews_corrcoef(y_true, y_pred)

    auc = None
    if hasattr(model, "predict_proba") and len(np.unique(y_true)) == 2:
        try:
            y_prob = model.predict_proba(X)[:, 1]
            y_true_numeric = pd.factorize(y_true)[0]
            auc = roc_auc_score(y_true_numeric, y_prob)
        except:
            auc = None

    st.subheader("Model Performance")

    cols = st.columns(6)

    metrics = [
        ("Accuracy", acc),
        ("Precision", prec),
        ("Recall", rec),
        ("F1 Score", f1),
        ("MCC", mcc),
        ("AUC", auc if auc is not None else 0)
    ]

    for col, (label, value) in zip(cols, metrics):
        col.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

else:
    st.info("Upload a labeled test dataset from the sidebar to begin evaluation.")

st.markdown("---")
st.caption("ML Assignment | Streamlit Deployment | Model Evaluation")
