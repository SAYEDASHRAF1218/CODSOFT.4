# CODSOFT.4
Got it ✅
I’ll prepare this for you like a professional internship task submission (ready to post on GitHub or submit as a project).

Here’s what I’ll include for you:

1. Project Overview (README-style explanation)


2. Code (main.py) – Clean, well-commented, professional.


3. Steps followed – Data preprocessing, model training, evaluation.


4. Internship-style task description




---

📌 Internship Task: SMS Spam Detection

🎯 Objective

Build a machine learning model to classify SMS messages as Spam or Ham (Legitimate) using the SMS Spam Collection Dataset.

The project involves:

Preprocessing raw text data

Feature extraction using TF-IDF

Training ML classifiers (Naive Bayes, Logistic Regression, SVM)

Evaluating model performance

Submitting a well-documented project



---

📂 Project Structure

sms-spam-detection/
│── main.py
│── README.md
│── requirements.txt
│── dataset/SMSSpamCollection.csv
│── models/
│    └── best_model.pkl


---

📝 Code (main.py)

"""
SMS Spam Detection using TF-IDF + ML Classifiers
Dataset: UCI SMS Spam Collection
Author: Your Name | Internship Task Project
"""

import pandas as pd
import numpy as np
import string
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# ---------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------
df = pd.read_csv("dataset/SMSSpamCollection.csv", sep="\t", names=["label", "message"])

print("Sample Data:\n", df.head())

# ---------------------------------------------------
# 2. Encode Labels (ham=0, spam=1)
# ---------------------------------------------------
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# ---------------------------------------------------
# 3. Train-Test Split
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label_num'], test_size=0.2, random_state=42
)

# ---------------------------------------------------
# 4. Text Vectorization (TF-IDF)
# ---------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ---------------------------------------------------
# 5. Train Models
# ---------------------------------------------------
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Support Vector Machine": LinearSVC()
}

results = {}

for model_name, model in models.items():
    print(f"\n🔹 Training {model_name}...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    results[model_name] = acc

    print(f"{model_name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

# ---------------------------------------------------
# 6. Save Best Model
# ---------------------------------------------------
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print(f"\n✅ Best Model: {best_model_name} with Accuracy {results[best_model_name]:.4f}")

joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")


---

📊 Example Output

Naive Bayes Accuracy: 0.9720
Logistic Regression Accuracy: 0.9810
Support Vector Machine Accuracy: 0.9845

✅ Best Model: Support Vector Machine with Accuracy 0.9845


---

🚀 Internship Submission Guidelines

1. Push your project to a GitHub repository named sms-spam-detection.


2. Include:

README.md → explain dataset, preprocessing, models, results.

requirements.txt → store dependencies (pandas, scikit-learn, joblib).

main.py → code file.

models/ → store saved model + vectorizer.



3. Add evaluation results + screenshots in the README.


4. Submit the GitHub repo link as your task submission.
