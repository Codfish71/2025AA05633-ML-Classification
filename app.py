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
    page_title="ML Model Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>

.header-container {
    background: linear-gradient(90deg, #1f3c88, #2a5298);
    padding: 25px;
    border-radius: 12px;
    color: white;
}

.metric-tile {
    background: #ffffff;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #e6e9ef;
    text-align: center;
}

.metric-value {
    font-size: 26px;
    font-weight: 700;
    color: #1f3c88;
}

.metric-label {
    font-size: 13px;
    color: #666;
}

.section-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #eaecef;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# HEADER
# -------------------------------------------------------

st.markdown("""
<div class="header-container">
    <h2>Machine Learning Classification Evaluation</h2>
    <p>Interactive evaluation of classification models on uploaded labeled dataset.</p>
</div>
""", unsafe_allow_html=True)

st.write("")

st.sidebar.title("Configuration Panel")

model_choice = st.sidebar.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Labeled Test Dataset",
    type=["csv"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Ensure dataset contains target column: income")

# -------------------------------------------------------
# MODEL LOADING
# -------------------------------------------------------

@st.cache_resource
def load_model(name):
    filename = name.lower().replace(" ", "_") + ".pkl"
    return joblib.load(f"model/{filename}")

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    if "income" not in df.columns:
        st.error("Target column 'income' not found in dataset.")
    else:
        y = df["income"]
        X = df.drop("income", axis=1)

        model = load_model(model_choice)
        y_pred = model.predict(X)

        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average="macro")
        rec = recall_score(y, y_pred, average="macro")
        f1 = f1_score(y, y_pred, average="macro")
        mcc = matthews_corrcoef(y, y_pred)

        if hasattr(model, "predict_proba"):
            auc = roc_auc_score(y, model.predict_proba(X)[:,1])
        else:
            auc = np.nan

        st.markdown(f"""
        <div class="section-card">
            <b>Model:</b> {model_choice} &nbsp;&nbsp; | &nbsp;&nbsp;
            <b>Samples Evaluated:</b> {len(df)} &nbsp;&nbsp; | &nbsp;&nbsp;
            <b>Unique Classes:</b> {len(np.unique(y))}
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        metric_cols = st.columns(6)

        metric_values = [
            ("Accuracy", acc),
            ("Precision", prec),
            ("Recall", rec),
            ("F1 Score", f1),
            ("MCC", mcc),
            ("AUC", auc)
        ]

        for col, (label, value) in zip(metric_cols, metric_values):
            col.markdown(f"""
            <div class="metric-tile">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value:.3f}</div>
            </div>
            """, unsafe_allow_html=True)


        tab1, tab2 = st.tabs(["Confusion Matrix", "Dataset Preview"])

        with tab1:
            st.write("")
            cm = confusion_matrix(y, y_pred)

            fig, ax = plt.subplots(figsize=(6,5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="viridis",
                ax=ax
            )
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("Actual Label")
            st.pyplot(fig)

        with tab2:
            st.dataframe(df.head())

else:
    st.info("Upload a labeled test dataset from the sidebar to begin evaluation.")


st.markdown("""
<hr>
<div style="text-align:center;color:#999;font-size:12px;">
ML Classification Dashboard | Streamlit Deployment | Academic Evaluation
</div>
""", unsafe_allow_html=True)