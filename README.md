# Customer Churn Prediction in Telecom Industry

This project focuses on predicting customer churn in the telecom sector using machine learning techniques.  
Customer churn refers to customers leaving a service provider, which directly impacts revenue.  
The goal of this project is to **identify customers who are likely to churn in advance**, so that telecom companies can take proactive retention actions such as offering discounts or personalized plans.

The project is designed to be **practical, interpretable, and deployable**, rather than only accuracy-focused.

---

## Problem Statement

In the telecom industry, acquiring new customers is significantly more expensive than retaining existing ones.  
However, customer churn datasets are often **imbalanced**, and many machine learning models behave like black boxes, making business decisions hard to justify.

This project aims to:
- Predict churn accurately
- Handle class imbalance effectively
- Explain *why* a customer is predicted to churn
- Provide a simple **web-based interface** for real-time churn analysis

---

## Dataset

- **Dataset**: Telco Customer Churn Dataset  
- **Source**: Kaggle  
- **Records**: 7,032 customers  
- **Target Variable**: `Churn` (Yes / No)

### Feature Types
- Demographic details (gender, senior citizen, dependents)
- Account information (tenure, contract type, payment method)
- Service usage (internet service, online security, streaming services)
- Billing details (monthly charges, total charges)

---

## Project Workflow

1. **Exploratory Data Analysis (EDA)**
   - Churn distribution analysis
   - Feature-level churn trends
   - Identification of early churn patterns

2. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical variables
   - Scaling numerical features
   - Class imbalance handling using **SMOTE**

3. **Model Development**
   - Logistic Regression (baseline)
   - Random Forest
   - XGBoost (final model)

4. **Model Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC–AUC
   - Confusion Matrix

5. **Explainability**
   - SHAP values to interpret model predictions
   - Identification of key churn drivers

6. **Deployment**
   - Streamlit-based web application
   - Real-time churn risk scoring
   - Downloadable churn-risk customer list

---

## Models Used

| Model | Purpose |
|------|--------|
| Logistic Regression | Baseline comparison |
| Random Forest | Tree-based ensemble |
| **XGBoost** | Final optimized model |

XGBoost was selected as the final model due to its:
- Strong performance on imbalanced data
- High recall and ROC-AUC
- Ability to capture non-linear feature interactions

---

## Explainability (SHAP)

SHAP (SHapley Additive exPlanations) is used to explain model predictions.

Key insights from SHAP analysis:
- **Contract type**, **tenure**, and **monthly charges** are the strongest churn drivers
- Short-tenure, month-to-month customers show higher churn risk
- SHAP plots help translate model outputs into business-friendly insights

This makes the model suitable for **real-world decision-making**, not just prediction.

---

## Streamlit Web Application

The trained and calibrated model is deployed using **Streamlit**.

### Features:
- Interactive churn probability threshold selection
- Identification of churn-risk customers
- Ranked churn-risk list
- CSV download for retention campaigns

This allows business teams to **use the model without coding knowledge**.

---

## Tech Stack

- **Language**: Python
- **Libraries**:
  - pandas, numpy
  - scikit-learn
  - XGBoost
  - SHAP
  - imbalanced-learn (SMOTE)
  - Streamlit
- **Environment**: VS Code, Jupyter Notebook

---

## How to Run the Project

1. Clone the repository
```bash
git clone https://github.com/gokularaman-c/customer-churn-prediction.git
cd customer-churn-prediction
````

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app

```bash
streamlit run app.py
```

---

## Key Outcomes

* Achieved strong recall and ROC-AUC for churn detection
* Built an interpretable ML pipeline using SHAP
* Developed a deployable, user-friendly churn prediction system
* Designed for both **academic evaluation and industry use**


---

## Author

**Gokularaman C**

---

