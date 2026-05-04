# 💳 CreditSense – Credit Risk Intelligence Platform

An end-to-end **Machine Learning-powered Credit Risk Assessment system** with an interactive **Streamlit dashboard** to evaluate loan applicants, compare profiles, and perform batch scoring.

---

## 🚀 Overview

CreditSense helps financial institutions and analysts make smarter lending decisions by predicting whether a customer is a **good or bad credit risk** based on personal and financial attributes.

It combines:
- 📊 Machine Learning (Scikit-learn)
- 📈 Data Visualization
- 🧠 Risk Interpretation
- ⚡ Real-time Predictions via Streamlit UI

---

## 🖥️ Application Preview

![CreditSense Dashboard](./assets/UI.png)


## ✨ Features

### 🔮 Predict Credit Risk
- Input applicant details (age, job, savings, loan, etc.)
- Get:
  - Creditworthiness score (%)
  - Risk classification (Good / Bad)
  - Risk grade (A+ → F)
  - Key risk factors

---

### ⚖️ Compare Applicants
- Compare two applicants side-by-side  
- Identify the better lending candidate instantly  

---

### 📋 Batch Scoring
- Upload CSV file  
- Score multiple applicants at once  
- Download results instantly  

---

## 🧠 Machine Learning Pipeline

- Dataset: German Credit Dataset  
- Preprocessing:
  - Handling missing values  
  - Encoding categorical features  
  - Feature scaling  
- Model:
  - Random Forest Classifier  
- Outputs:
  - Prediction (Good / Bad)  
  - Probability scores  

---

## 🗂️ Project Structure
CreditSense/
│
├── app.py
├── german_credit_data.csv
├── requirements.txt
│
├── outputs/
│ ├── rf_model.pkl
│ ├── scaler.pkl
│ ├── le_dict.pkl
│ ├── feature_names.pkl
│
├── assets/
│ └── UI.png
│
└── README.md


---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/anand738/Credit-Risk-Assessment.git
cd Credit-Risk-Assessment

pip install -r requirements.txt

streamlit run app.py
'''

## 🖥️ How It Works
User inputs applicant details via UI
Loan amount is converted from INR → DM
Data is preprocessed and scaled
Model predicts:
Credit Risk (Good / Bad)
Probability scores
Results are displayed with:
Gauge visualization
Risk factors
Final recommendation
