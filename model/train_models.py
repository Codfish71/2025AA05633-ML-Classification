import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

DATA_PATH = "data/adult.csv"
TARGET_COLUMN = "income"
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = df.dropna()

if df[TARGET_COLUMN].dtype == "object":
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"<=50K": 0, ">50K": 1})

y = df[TARGET_COLUMN].astype(int)
X = df.drop(TARGET_COLUMN, axis=1)

categorical_features = X.select_dtypes(include="object").columns
numerical_features = X.select_dtypes(exclude="object").columns

preprocessor_sparse = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

preprocessor_dense = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "logistic_regression": LogisticRegression(max_iter=500),

    "decision_tree": DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=5,
        random_state=42
    ),

    "knn": KNeighborsClassifier(n_neighbors=7),

    "naive_bayes": GaussianNB(),

    "random_forest": RandomForestClassifier(
        n_estimators=40,      
        max_depth=8,
        min_samples_leaf=5,
        random_state=42
    ),

    "xgboost": XGBClassifier(
        n_estimators=40,     
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False
    )
}

for name, model in models.items():

    print(f"Training {name}...")

    if name == "naive_bayes":
        pipeline = Pipeline([
            ("preprocessor", preprocessor_dense),
            ("classifier", model)
        ])
    else:
        pipeline = Pipeline([
            ("preprocessor", preprocessor_sparse),
            ("classifier", model)
        ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {round(acc,4)}")

    save_path = os.path.join(MODEL_DIR, f"{name}.pkl")
    joblib.dump(pipeline, save_path, compress=3)

    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"Saved: {save_path} ({round(size_mb,2)} MB)\n")

print("Training complete.")