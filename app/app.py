import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Customer Churn Predictor",
    layout="wide"
)

st.title("📊 Customer Churn Prediction System")
st.write("Predict customer churn and understand **why** the model made its decision.")


@st.cache_resource
def load_artifacts():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model_path = os.path.join(BASE_DIR, "models", "churn_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
    encoder_path = os.path.join(BASE_DIR, "models", "label_encoders.pkl")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoders = joblib.load(encoder_path)

    return model, scaler, label_encoders

model, scaler, label_encoders = load_artifacts()




@st.cache_data
def load_feature_template():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "data", "processed", "churn_cleaned.csv")
    df = pd.read_csv(data_path)
    return df.drop("Churn", axis=1)


X_template = load_feature_template()


st.sidebar.header("Customer Details")

user_input = {}

for col in X_template.columns:
    if col in label_encoders:
        user_input[col] = st.sidebar.selectbox(
            col,
            label_encoders[col].classes_.tolist()
        )
    else:
        user_input[col] = st.sidebar.number_input(
            col,
            float(X_template[col].min()),
            float(X_template[col].max())
        )


input_df = pd.DataFrame([user_input])


for col, le in label_encoders.items():
    input_df[col] = le.transform(input_df[col])


num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
input_df[num_cols] = scaler.transform(input_df[num_cols])


if st.button("Predict Churn"):
    prob = model.predict_proba(input_df)[0][1]
    prediction = "Churn" if prob > 0.5 else "No Churn"

    st.subheader("Prediction Result")
    st.write(f"**Prediction:** {prediction}")
    st.write(f"**Churn Probability:** {prob:.2%}")

    
    st.subheader("Explanation")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    fig, ax = plt.subplots()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=input_df.columns
        ),
        show=False
    )
    st.pyplot(fig)
