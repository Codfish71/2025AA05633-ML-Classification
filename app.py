import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

st.title("ML Classification Model Comparison")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

model_option = st.selectbox(
    "Select Model",
    [
        "Logistic Regression"
    ]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # Clean target
    if df["income"].dtype == "object":
        df["income"] = df["income"].astype(str).str.strip()
        df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})

    X = df.drop("income", axis=1)
    y = df["income"].astype(int)

    categorical_cols = X.select_dtypes(include='object').columns
    numerical_cols = X.select_dtypes(exclude='object').columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        # "Decision Tree": DecisionTreeClassifier(),
        # "KNN": KNeighborsClassifier(),
        # "Naive Bayes": GaussianNB(),
        # "Random Forest": RandomForestClassifier(),
        # "XGBoost": XGBClassifier(eval_metric='logloss')
    }

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', models[model_option])
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    st.pyplot(fig)
