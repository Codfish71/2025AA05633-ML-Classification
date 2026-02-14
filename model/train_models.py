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

DATA_PATH = "data/adult.csv"
TARGET = "income"
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = df.dropna()

if df[TARGET].dtype == "object":
    df[TARGET] = df[TARGET].astype(str).str.strip()
    df[TARGET] = df[TARGET].map({"<=50K": 0, ">50K": 1})

y = df[TARGET].astype(int)
X = df.drop(TARGET, axis=1)

cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

pre_sparse = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

pre_dense = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "logistic_regression": LogisticRegression(max_iter=500),
    "decision_tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=7),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(n_estimators=40, max_depth=8, min_samples_leaf=5, random_state=42),
    "xgboost": XGBClassifier(n_estimators=40, max_depth=5, learning_rate=0.1, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="logloss", use_label_encoder=False)
}

for name, model in models.items():
    if name == "naive_bayes":
        pipe = Pipeline([("pre", pre_dense), ("clf", model)])
    else:
        pipe = Pipeline([("pre", pre_sparse), ("clf", model)])

    pipe.fit(X_train, y_train)
    joblib.dump(pipe, f"{MODEL_DIR}/{name}.pkl", compress=3)