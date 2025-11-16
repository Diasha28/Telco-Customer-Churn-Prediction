from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import uvicorn

app = FastAPI(
    title ="Telco Customer Churn Prediction",
    description ="The ML model that predicts if the customer is going to churn based on the inputs available",
    version= "1.0.0"
)
# Load model
with open("model_pkl", "rb") as f:
    churn_model = pickle.load(f)

# FINAL FEATURE LIST FROM TRAINING
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

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


def preprocess(data: CustomerData):
    df = pd.DataFrame([data.dict()])

    # Convert Yes/No → 1/0
    bin_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in bin_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # gender: convert to 0/1
    df["gender"] = df["gender"].map({"Female": 0, "Male": 1})

    # One-hot encoding
    df = pd.get_dummies(df, columns=[
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod"
    ], drop_first=True)

    # Add missing columns
    for col in final_features:
        if col not in df.columns:
            df[col] = 0

    df = df[final_features]
    return df


@app.post("/predict")
def predict(data: CustomerData):
    processed = preprocess(data)
    prediction = churn_model.predict(processed)[0]
    return {"prediction": int(prediction),
            "message": "Customer is likely to churn" if prediction == 1 else "Customer is not likely to churn"}


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)