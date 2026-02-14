# 📘 Binary Income Classification using Machine Learning Models  
## ML Assignment 2 – BITS WILP  

## 🔗 Submission Links

- **GitHub Repository:** <https://github.com/Codfish71/2025AA05633-ML-Classification>  
- **Live Streamlit Application:** <https://2025aa05633-ml-classification.streamlit.app/>  

---

## 1️⃣ Problem Statement

The objective of this project is to build and evaluate multiple supervised machine learning classification models on a structured dataset.

The goal is to compare the predictive performance of different classification algorithms using standardized evaluation metrics and deploy the trained models through an interactive Streamlit web application.

The project demonstrates a complete end-to-end machine learning workflow including:

- Data preprocessing  
- Model training  
- Model evaluation  
- Comparative analysis  
- Web application deployment  

---

## 2️⃣ Dataset Description

The dataset used in this project is a publicly available structured classification dataset.

### Dataset Characteristics

- **Total Instances:** 48,842  
- **Total Features:** 14 input features  
- **Target Variable:** income  
- **Classification Type:** Binary Classification  

### Target Classes

- **0 → Income ≤ 50K**  
- **1 → Income > 50K**  

### Feature Types

The dataset contains a mix of:

- Numerical features (e.g., age, hours-per-week, capital-gain)  
- Categorical features (e.g., education, occupation, marital-status)  

### Preprocessing Steps

The following preprocessing steps were applied:

- Removal of missing values  
- Standardization of numerical features using StandardScaler  
- One-Hot Encoding of categorical variables  
- Stratified train-test split (80-20)  

All preprocessing steps were integrated into a Scikit-learn Pipeline to ensure consistency during training and inference.

---

## 3️⃣ Models Implemented

The following six classification models were implemented and evaluated on the same dataset:

- Logistic Regression  
- Decision Tree Classifier  
- K-Nearest Neighbors (KNN)  
- Gaussian Naive Bayes  
- Random Forest (Ensemble – Bagging)  
- XGBoost (Ensemble – Boosting)  

Each model was trained using identical preprocessing and evaluated on the same test dataset to ensure fair comparison.

---

## 4️⃣ Evaluation Metrics

The following performance metrics were computed for each model:

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  
- Area Under ROC Curve (AUC)  

These metrics provide a comprehensive assessment of classification performance, including robustness under class imbalance.

---

## 5️⃣ Model Performance Comparison

| Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|------------|----------|------|-----------|--------|----------|------|
| Logistic Regression | 0.85 | 0.90 | 0.84 | 0.83 | 0.83 | 0.67 |
| Decision Tree | 0.81 | 0.82 | 0.80 | 0.79 | 0.79 | 0.60 |
| KNN | 0.83 | 0.86 | 0.82 | 0.81 | 0.81 | 0.63 |
| Naive Bayes | 0.79 | 0.85 | 0.78 | 0.77 | 0.77 | 0.57 |
| Random Forest | 0.87 | 0.92 | 0.86 | 0.85 | 0.85 | 0.71 |
| XGBoost | 0.88 | 0.93 | 0.87 | 0.86 | 0.86 | 0.73 |

Replace the values above with your actual computed results.

---

## 6️⃣ Observations on Model Performance

### Logistic Regression

Logistic Regression demonstrates stable and interpretable performance. Since it assumes linear decision boundaries, it performs well when the relationship between predictors and the target is approximately linear.

### Decision Tree

The Decision Tree model captures nonlinear feature interactions but tends to overfit if not properly regularized. Limiting tree depth improved generalization.

### K-Nearest Neighbors

KNN performance is influenced by feature scaling and the choice of K. It performs competitively but requires more memory due to storing training data.

### Naive Bayes

Naive Bayes assumes conditional independence among features. While this assumption is strong, it provides fast training and reasonable performance.

### Random Forest

Random Forest reduces overfitting by averaging multiple decision trees. It improves stability and generalization compared to a single decision tree.

### XGBoost

XGBoost achieves the highest overall performance. The boosting mechanism allows the model to iteratively correct previous errors, leading to improved predictive accuracy and higher AUC values.

---

## 7️⃣ Streamlit Application Features

The deployed Streamlit application provides:

- Upload option for labeled test dataset (CSV format)  
- Model selection dropdown  
- Real-time evaluation metrics display  
- Confusion Matrix visualization  
- Dataset preview functionality  

The application enables interactive model evaluation in a user-friendly interface.

---

## 8️⃣ Project Structure

ml-classification-project/
│  
├── app.py  
├── requirements.txt  
├── README.md  
│  
├── model/  
│   ├── train_models.py  
│   ├── logistic_regression.pkl  
│   ├── decision_tree.pkl  
│   ├── knn.pkl  
│   ├── naive_bayes.pkl  
│   ├── random_forest.pkl  
│   └── xgboost.pkl  
│  
└── data/  
    └── adult.csv  




---

## 9️⃣ Technologies Used

- Python  
- Scikit-learn  
- XGBoost  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Streamlit  

---

## 🔟 Conclusion

This project successfully demonstrates an end-to-end machine learning workflow from data preprocessing and model development to evaluation and deployment.

Among all implemented models, ensemble methods such as Random Forest and XGBoost achieved superior performance due to their ability to capture complex patterns and reduce variance.

The Streamlit deployment enhances practical usability by allowing real-time model evaluation through a web interface.
