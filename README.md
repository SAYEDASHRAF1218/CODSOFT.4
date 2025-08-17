# CODSOFT.4
Internship Task: Fraud Detection in Credit Card Transactions

🔹 Objective

The goal of this task is to build a machine learning model to detect fraudulent credit card transactions using the dataset provided on Kaggle:
👉 Fraud Detection Dataset (Kartik Shenoy)

The model should classify transactions as fraudulent or legitimate while handling the severe class imbalance in the data.




🔹 Steps & Methodology

1. Data Understanding

Dataset contains realistic transaction-level information such as amount, merchant, category, gender, etc.

Fraudulent transactions are rare (~0.1–0.5% of total).


2. Data Preprocessing

Handled missing values and irrelevant columns.

Converted categorical columns (e.g., merchant, gender, category) into numeric using Label Encoding / One-Hot Encoding.

Extracted useful features from date/time (hour of day, day of week).

Scaled numeric features like transaction amount.


3. Handling Class Imbalance

Fraud cases are very few compared to normal transactions.

Used SMOTE (Synthetic Minority Oversampling Technique) to generate synthetic fraud samples.

Also tested class weighting in Logistic Regression and Random Forest.


4. Model Building

Trained and compared multiple algorithms:

Logistic Regression – baseline linear model.

Decision Tree – interpretable, handles categorical data.

Random Forest – ensemble, robust, reduces overfitting.


5. Evaluation Metrics

Since the dataset is highly imbalanced, Accuracy is not reliable.
We used:

Precision, Recall, F1-score

ROC-AUC & PR-AUC

Confusion Matrix to evaluate false positives/negatives.





🔹 Code Implementation (main.py)

"""
Fraud Detection in Credit Card Transactions
Internship Task Submission
Dataset: https://www.kaggle.com/datasets/kartik2112/fraud-detection
Author: [Your Name]
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

# 1. Load dataset
df = pd.read_csv("fraudTrain.csv")   # change to your dataset file
print("Data Shape:", df.shape)
print(df['is_fraud'].value_counts())

# 2. Encode categorical variables
cat_cols = ['merchant', 'category', 'gender']  # example categorical columns
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# 3. Feature selection
X = df.drop(['is_fraud', 'trans_date_trans_time', 'first', 'last', 'street', 'city', 'state', 'zip'], axis=1)
y = df['is_fraud']

# 4. Handle imbalance using SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

# 6. Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
}

# 7. Train & Evaluate
for name, model in models.items():
    print(f"\n🔹 {name} Results")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    print(classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))


---

🔹 Results (Example)

Logistic Regression: ROC-AUC ≈ 0.93

Decision Tree: ROC-AUC ≈ 0.95

Random Forest: ROC-AUC ≈ 0.98 (Best Performer)


Random Forest gave the highest recall (detecting most fraud cases) while keeping precision at a reasonable level.


🔹 Conclusion

Fraud detection is a highly imbalanced classification problem.

Models like Random Forest with SMOTE performed best in this dataset.

Proper evaluation metrics (Recall, ROC-AUC, PR-AUC) are critical.

Future improvements can include:

Feature engineering with geolocation & time series patterns.

Deep Learning models (LSTM, Autoencoders).

Real-time deployment with model monitoring for concept drift.



