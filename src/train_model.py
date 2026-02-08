import joblib
import logging
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import time
from data_preprocessing import preprocess_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
def train_model(path: str, threshold: float = 0.3):
    logging.info("Loading and preprocessing data...")
    X, y = preprocess_data(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42, stratify=y)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss")
    
    logging.info("Training model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    logging.info(f"Training completed in {training_time:.2f} seconds")

    logging.info("Evaluating model...")
    start_pred = time.time()
    prob = model.predict_proba(X_test)[:, 1]
    y_pred = (prob >= threshold).astype(int)
    pred_time = time.time() - start_pred
    logging.info(f"Prediction completed in {pred_time:.4f} seconds")
    print(classification_report(y_test, y_pred, digits=3))
    logging.info("Saving trained model...")
    joblib.dump(model, "models/churn_model.pkl")
    logging.info("Model saved successfully.")


if __name__ == "__main__":
    path = input("Enter path to dataset CSV file: ")
    train_model(path)


