# 📱 Spam SMS Detection (Kaggle Dataset)

## 📌 Overview
This project builds a **Spam SMS Detection model** using the famous **Kaggle spam.csv dataset**.  
The model classifies SMS messages as either **HAM (legitimate)** or **SPAM (fraudulent/unwanted)**.  

Two machine learning classifiers are trained and compared:  
- **Naive Bayes (MultinomialNB)**  
- **Logistic Regression**  

---

## 📂 Dataset
- **File:** `spam.csv` (from Kaggle)  
- **Shape:** 5,572 rows × 2 columns  
- **Columns:**  
  - `label` → Message type (`ham` or `spam`)  
  - `message` → The SMS text  

Sample data:  

| label | message |
|-------|---------|
| ham   | Go until jurong point, crazy.. Available only in ... |
| ham   | Ok lar... Joking wif u oni... |
| spam  | Free entry in 2 a wkly comp to win FA Cup fina... |

---

## 🛠️ Steps in the Project
1. **Load & Clean Dataset**  
   - Removed extra columns  
   - Converted labels: `ham → 0`, `spam → 1`  
   - Preprocessed text (lowercased, removed special characters).  

2. **Split Data**  
   - Training: 80%  
   - Testing: 20%  

3. **Feature Extraction**  
   - TF-IDF Vectorizer (max 3000 features).  

4. **Models Trained**  
   - Naive Bayes  
   - Logistic Regression  

5. **Evaluation Metrics**  
   - Accuracy  
   - Confusion Matrix  
   - Precision, Recall, F1-score  

---

## 📊 Results

### ✅ Naive Bayes
- **Accuracy:** 96.95%  
- **Confusion Matrix:**  
  ```
  [[965   0]
   [ 34 116]]
  ```
- Performs slightly better in recall for SPAM detection.

### ✅ Logistic Regression
- **Accuracy:** 96.77%  
- **Confusion Matrix:**  
  ```
  [[965   0]
   [ 36 114]]
  ```
- Strong performance but slightly lower recall for SPAM compared to Naive Bayes.  

---

## 🔮 Sample Predictions
**Input Messages & Predictions:**  
- `"Congratulations! You won a $1000 Walmart gift card. Call now to claim your prize!"` → **SPAM**  
- `"Hey, are we still meeting for lunch today?"` → **HAM**  
- `"URGENT! Your account has been compromised. Visit http://fakebank.com to verify details."` → **SPAM**  

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/SAYEDASHRAF1218/CODSOFT.4.git
   ```
2. Install dependencies:
   ```bash
   pip install pandas scikit-learn
   ```
3. Run the script:
   ```bash
   python spam_detection.py
   ```
   or open the notebook:
   ```bash
   jupyter notebook spam_detection.ipynb
   ```

---
