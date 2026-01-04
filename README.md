📊 Customer Churn Prediction System (Explainable ML)

An end-to-end Machine Learning project that predicts customer churn and explains why the model made each prediction using Explainable AI (SHAP).

This project goes beyond simple churn prediction by focusing on model transparency, business reasoning, and real-world usability through a deployed web application.

 🚀 Live Demo
🔗 Streamlit App: https://pranav25187-customer-churn-prediction-explainable-appapp-aysluj.streamlit.app/

🧠 Problem Statement
Customer churn is a major challenge for telecom and subscription-based businesses.  
Losing customers directly impacts revenue and growth.

The objective of this project is to:
- Predict whether a customer is likely to churn
- Quantify churn probability
- Explain the prediction in a way business teams can understand
- Enable proactive customer retention strategies

---

 🛠 Tech Stack
- **Programming Language:** Python  
- **Data Processing:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn, XGBoost  
- **Explainable AI:** SHAP  
- **Visualization:** Matplotlib, Seaborn  
- **Web App:** Streamlit  

---

📂 Project Structure


customer_churn_project/
│
├── data/
│ ├── raw/ # Original dataset
│ └── processed/ # Cleaned & transformed data
│
├── notebooks/
│ ├── 01_EDA.ipynb # Business-driven EDA
│ └── 02_Model_Training.ipynb
│
├── app/
│ └── app.py # Streamlit application
│
├── models/
│ ├── churn_model.pkl # Trained ML model
│ ├── scaler.pkl # Feature scaler
│ └── label_encoders.pkl # Encoders for categorical features
│
├── screenshots/
│ ├── app.png # App UI screenshot
│ └── shap.png # SHAP explanation screenshot
│
├── requirements.txt
└── README.md


📊 Dataset
Telco Customer Churn Dataset

- Source: Kaggle / IBM
- ~7,000 customer records
- Features include:
  - Demographics
  - Service subscriptions
  - Contract details
  - Billing information
- Target variable: `Churn` (Yes / No)

---

🔍 Exploratory Data Analysis (EDA)
EDA was performed with a **business-first approach**, focusing on understanding customer behavior rather than just visualizing data.

Key insights:
- Customers on **month-to-month contracts** churn significantly more
- **Low tenure customers** are at the highest churn risk
- **Higher monthly charges**, especially early in the lifecycle, increase churn probability

These insights directly guided feature engineering and model selection.

---

🤖 Model Training & Evaluation
Multiple models were trained and compared:

- Logistic Regression  
- Random Forest  
- XGBoost (final selected model)

Evaluation metrics used:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC (primary metric)

XGBoost was selected due to its superior ROC-AUC score and ability to capture non-linear customer behavior patterns.

---

🔎 Explainable AI (SHAP)
Explainability is the core strength of this project.

SHAP is used to:
- Identify **global churn drivers** (overall feature importance)
- Explain **individual customer predictions**
- Visualize how each feature increases or decreases churn risk

This makes the model transparent and suitable for real business decision-making.

---
🖥 Streamlit Web Application
The project is deployed as an interactive Streamlit app with:

- Customer input form
- Churn prediction (Yes / No)
- Churn probability score
- SHAP waterfall plot for explanation
- End-to-end ML pipeline integration

The app is designed for **non-technical users**, such as business analysts or customer success teams.

---

 📸 Screenshots

<img width="1411" height="945" alt="localhost_8501_ (2)" src="https://github.com/user-attachments/assets/200d2ff4-9c44-444d-9e09-31f616569c7c" />

<img width="1919" height="937" alt="image" src="https://github.com/user-attachments/assets/4dc8072f-ed73-44d5-9ffc-6115aa577d21" />



▶️ Run the Project Locally

bash
pip install -r requirements.txt
streamlit run app/app.py




 🎯 Key Learnings

* End-to-end ML system design
* Business-driven feature engineering
* Model comparison beyond accuracy
* Practical implementation of Explainable AI
* Building and deploying ML-powered web applications



👤 Author

Pranav
Final-year Computer Engineering student
Aspiring Machine Learning Engineer / Data Scientist


