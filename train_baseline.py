import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
from typing import Tuple, Dict, Any
from mlflow.models import infer_signature

def train_simple_baseline(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[LogisticRegression, Dict[str, float]]:
    """
    Trains a Logistic Regression model with default parameters and logs results via MLflow.

    Args:
        df: The input pandas DataFrame.
        target_column: The name of the label column.
        test_size: The proportion of the data to use for the test set.
        random_state: Seed for reproducibility.

    Returns:
        A tuple containing: (fitted_model, metrics_dict).
    """
    
    # 1. Prepare Data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Note: Skipping all preprocessing as requested for OPPE simplicity.
    
    # 2. Start MLflow Run (Begin tracking the experiment)
    with mlflow.start_run(run_name="Simple_LR_Baseline_second") as run:
        
        # 3. Define and Train Model
        model = LogisticRegression(random_state=random_state, solver='liblinear', max_iter=200)
        print("Starting training on Logistic Regression...")
        model.fit(X_train, y_train)
        print("Training complete.")

        # 4. Predict and Calculate Metrics
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='binary', zero_division=0),
            'model_type': type(model).__name__
        }
        # 5. MLflow Logging (Key MLOps step)
        mlflow.log_params(model.get_params())  # Log all default parameters
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != 'model_type'})
    
        signature =  infer_signature(X_train, model.predict(X_train))
        
        model_info = mlflow.sklearn.log_model(
            sk_model = model, 
            artifact_path = "transaction_models",
            signature = signature,
            input_example = X_train,
            registered_model_name = "TRANSACTION-log-reg"
        )
        
        print(f"MLflow Run ID: {run.info.run_id}")
        
        return model, metrics