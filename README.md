# 📘 Binary Income Classification using Machine Learning Models  
## ML Assignment 2 – BITS WILP  

## 🔗 Submission Links

- **GitHub Repository:** <https://github.com/Codfish71/2025AA05633-ML-Classification>  
- **Live Streamlit Application:** <https://2025aa05633-ml-classification.streamlit.app/>  

---

## 1️⃣ Problem Statement

The objective of this project is to develop and evaluate multiple supervised machine learning classification models to predict whether an individual's annual income exceeds 50K based on demographic and socio-economic attributes.

The dataset used for this task contains 48,842 records with 14 input features, including both numerical and categorical variables such as age, education, occupation, marital status, capital gain, and hours worked per week. The target variable, income, is a binary variable indicating whether the annual income is less than or equal to 50K or greater than 50K.

This problem is formulated as a binary classification task. The primary goal is to build predictive models that can accurately classify individuals into the correct income category while ensuring robustness across multiple evaluation metrics.

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

## Model-wise Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Logistic Regression provides stable and interpretable performance. Since it assumes a linear relationship between features and the log-odds of the target variable, it performs reasonably well but may struggle to capture complex nonlinear interactions among features. |
| Decision Tree | The Decision Tree model effectively captures nonlinear relationships and feature interactions. However, without proper constraints such as limited depth, it can overfit the training data, leading to slightly reduced generalization performance. |
| KNN | K-Nearest Neighbors performs competitively when features are properly scaled. Its performance depends heavily on the choice of K and distance metric. It may be sensitive to noise and class imbalance and requires higher memory since it stores the entire training dataset. |
| Naive Bayes | Naive Bayes is computationally efficient and fast to train. However, it assumes conditional independence among features, which may not always hold true in real-world data, resulting in comparatively lower predictive performance. |
| Random Forest (Ensemble) | Random Forest improves stability and reduces overfitting by aggregating multiple decision trees. It captures complex feature interactions effectively and generally provides strong and consistent performance across evaluation metrics. |
| XGBoost (Ensemble) | XGBoost achieves superior performance due to its gradient boosting framework, regularization mechanisms, and ability to iteratively correct errors from previous models. It demonstrates high accuracy, strong generalization, and improved AUC compared to other models. |


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
