import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import count, demographic_parity_difference, demographic_parity_ratio,mean_prediction
from typing import Any, Dict
import joblib

def assess_model_fairness(
    model: Any,
    df_val: pd.DataFrame,
    target_column: str,
    sensitive_feature: str
) -> Dict[str, float]:
    """
    Assesses model fairness using only Accuracy Score across a sensitive feature.
    This avoids the UndefinedMetricWarning tied to Recall/Precision metrics.
    """
    x_test = df_val.drop(columns=[target_column])
    y_pred = model.predict(x_test)
    y_true = df_val[target_column]
    
    mf = MetricFrame(
        metrics=accuracy_score,
        y_true=y_true,
        y_pred= y_pred,
        sensitive_features=df_val[sensitive_feature]
    )
    print("Overall accuracy:", mf.overall)
    print("Demographic parity difference:", mf.difference())
    
    parity_diff_result = demographic_parity_difference(x_test, y_pred, sensitive_features=df_val[sensitive_feature])
    print(parity_diff_result)
    print(demographic_parity_ratio(y_true, y_pred, sensitive_features=df_val[sensitive_feature]))
    print(mean_prediction(y_true, y_pred))
    return {
        "Overall accuracy" : mf.overall,
        "Demographic parity difference": mf.difference(),
        "Demographic parity 2" : demographic_parity_difference(x_test, y_pred, sensitive_features=df_val[sensitive_feature]),
        "Demographic parity ratio" : demographic_parity_ratio(x_test, y_pred, sensitive_features=df_val[sensitive_feature]),
        "Mean prediction" : mean_prediction(y_true, y_pred)
    } 


if __name__=="__main__":
    assess_model_fairness(
        model = joblib.load("best_transaction_model.pkl"),
        df_val = pd.read_csv("data/transactions_v2.csv"),
        target_column = "Class",
        sensitive_feature = "Amount"
    )