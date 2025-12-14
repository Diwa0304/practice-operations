import shap
import numpy as np
from typing import Tuple, Dict, Any, List
import pandas as pd

def calculate_shap_values(
    model: Any, # Your fitted LogisticRegression model
    df_val: pd.DataFrame,
    target_column: str
) -> Tuple[np.ndarray, List[str]]:
    """
    Calculates SHAP values for the validation dataset using the fitted model.

    Args:
        model: The fitted scikit-learn LogisticRegression model.
        df_val: The validation DataFrame containing features and labels.
        target_column: The name of the true label column.

    Returns:
        A tuple containing: (shap_values_array, feature_names_list).
    """
    print("Calculating SHAP values...")
    
    X_val = df_val.drop(columns=[target_column])
    
    # 1. Choose an Explainer (KernelExplainer works for most models, 
    # but LinearExplainer is faster for Logistic Regression)
    try:
        # Use LinearExplainer for Logistic Regression
        explainer = shap.LinearExplainer(
            model, 
            X_val,
            feature_perturbation="interventional"
        )
    except Exception:
        # Fallback for complex models or if the faster explainer fails
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_val, 100))
        
    # 2. Calculate SHAP Values (explainer.shap_values returns a list of arrays for multi-class, 
    # we take the second array [1] for the positive class)
    shap_values = explainer.shap_values(X_val)
    
    
    # Handle the output format (for binary classification, shap_values is often a list of 2 arrays)
    if isinstance(shap_values, list) and len(shap_values) > 1:
         # Assuming binary classification, take the positive class (1) values
        shap_values_pos_class = shap_values[1] 
    else:
        # For single-output models (e.g., L.R. predict)
        shap_values_pos_class = shap_values
    
    print("SHAP values calculated successfully.")
    
    # Note: You can plot this later using shap.summary_plot(shap_values_pos_class, X_val)
    return shap_values_pos_class, list(X_val.columns)