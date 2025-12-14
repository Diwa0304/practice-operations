import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
import os

def test_model_accuracy():
    model_path = os.path.expanduser("~/oppe_practice/best_transaction_model.pkl")
    model = joblib.load(model_path)
    eval_df = pd.read_csv("data/transactions_v2.csv")
    X_test = eval_df.drop(columns=["Class"])
    y_test = eval_df["Class"]

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    assert acc > 0.7, f"Model accuracy too low: {acc}"
    

DATA_PATH = "data/transactions_v2.csv"

def test_data_shape():
    df = pd.read_csv(DATA_PATH)
    assert df.shape[1] == 32, "Unexpected number of columns"

def test_no_nulls():
    df = pd.read_csv(DATA_PATH)
    assert df.isnull().sum().sum() == 0, "Data contains null values"