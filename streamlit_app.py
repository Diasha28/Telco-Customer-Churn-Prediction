import numpy as np
import pickle
import streamlit_app as st
import pandas as pd
from PIL import Image
st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    layout="wide"
    )   
with open("model_pkl", "rb") as f:
    churn_model = pickle.load(f)

final_features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

def churn_predict(input_data):
    df = pd.DataFrame([input_data])

    bin_cols =["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in bin_cols:
        df[col] = df[col].map({"Yes": 1, "No":0})
    df["gender"] =df["gender"].map({"Female": 0, "Male": 1})
    df = pd.get_dummies(df, columns=[
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod"
    ], drop_first=True)
    for col in final_features:
        if col not in df.columns:
            df[col] = 0

    final_df = df[final_features]

    prediction = churn_model.predict(final_df)[0]
    return "Customer is likely to Churn" if prediction ==1 else "Customer is less likely to Churn"

def main():   
    st.markdown( "<h1 style='text-align: center; color:#4A90E2;'>📡 Telco Customer Churn Prediction</h1>", unsafe_allow_html=True)
    image_sidebar = Image.open('sidebarimg.jpg')
    image_banner = Image.open('headerimg.png')
    st.image(image_banner, use_container_width=True )
    st.sidebar.image(image_sidebar, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ("Male", "Female"))
        SeniorCitizen = st.number_input("Senior Citizen", min_value =0, max_value =2)
        Partner = st.selectbox("Do you have a partner", ("Yes", "No"))
        Dependents = st.selectbox("Do you have dependents?", ("Yes", "No"))
        tenure = st.number_input("What is your tenure?", min_value =0, max_value= 70)
        PhoneService = st.selectbox("Phone Service:", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines:", ["No phone service", "No", "Yes"])
        InternetService = st.selectbox("Internet Service:", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security:", ["No internet service", "No", "Yes"])
        OnlineBackup = st.selectbox("Online Backup:", ["No internet service", "No", "Yes"])
        DeviceProtection = st.selectbox("Device Protection:", ["No internet service", "No", "Yes"])
        TechSupport = st.selectbox("Tech Support:", ["No internet service", "No", "Yes"])
    with col2:
        StreamingTV = st.selectbox("Streaming TV:", ["No internet service", "No", "Yes"])
        StreamingMovies = st.selectbox("Streaming Movies:", ["No internet service", "No", "Yes"])
        Contract = st.selectbox("Contract:", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing:", ["Yes", "No"])
        PaymentMethod = st.selectbox("Payment Method:", ["Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ])
        MonthlyCharges = st.number_input("Monthly Charges:", min_value=0.0)
        TotalCharges = st.number_input("Total Charges:", min_value=0.0)
        st.markdown("""
        <style>
            .stButton>button {
                background: linear-gradient(90deg, #4A90E2, #007AFF);
                color: white;
                padding: 10px 24px;
                border-radius: 8px;
                font-size: 18px;
            }
        </style>
    """, unsafe_allow_html=True)
    churn_result =''
    if st.button('Predict Customer Churn'):
        input_data = {
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges
        }
        churn_result =churn_predict(input_data)
    
    st.success(churn_result, icon="✅")

if __name__ == '__main__':
    main()