import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score
import mlflow
from typing import Dict, Any, Tuple
from mlflow.models import infer_signature
import numpy as np

def tune_model_with_gridsearch(
    df: pd.DataFrame,
    target_column: str,
    param_grid: Dict[str, Any],
    cv_folds: int = 3,
    scoring_metric: str = 'f1',
    random_state: int = 42,
    test_size: float = 0.2
) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """
    Performs hyperparameter tuning using GridSearchCV and logs the final best model/run.

    Args:
        df: The input pandas DataFrame.
        target_column: The name of the label column.
        param_grid: Dictionary defining the parameter grid to search.
        cv_folds: The number of cross-validation folds.
        scoring_metric: The metric to optimize during the search (e.g., 'f1', 'accuracy').
        random_state: Seed for reproducibility.
        test_size: Proportion of data to hold out for final evaluation.

    Returns:
        A tuple containing: (best_fitted_model, best_params_and_metrics).
    """
    
    # 1. Prepare Data (Split into Train/Test for final evaluation)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    # NOTE: GridSearchCV will use CV on the full (X, y) set.
    # We will split here to evaluate the final best model on a true held-out set.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 2. Start MLflow Run (Begin tracking the tuning experiment)
    with mlflow.start_run(run_name="Tuned_LR_Best_Candidate") as run:
        
        # 3. Define Model and GridSearch
        estimator = LogisticRegression(random_state=random_state, solver='liblinear', max_iter=500)
        
        # Use make_scorer to ensure we can use it with GridSearchCV
        scorer = make_scorer(f1_score, average='binary', zero_division=0)

        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            # We set refit to the scoring metric to ensure the best model is retrained on all data
            scoring=scorer,
            refit=True, # Critical: Retrains the best model on X_train/y_train
            cv=cv_folds,
            verbose=1,
            n_jobs=-1
        )
        
        print("Starting Grid Search tuning...")
        # Fit on the training data used for the CV splits
        grid_search.fit(X_train, y_train) 
        print("Tuning complete.")

        # 4. Extract and Log Best Parameters
        best_model = grid_search.best_estimator_
        
        # Log all search space parameters and the final best parameters
        mlflow.log_param("param_grid", str(param_grid))
        mlflow.log_params(grid_search.best_params_)
        
        # Log the best CV score metric
        mlflow.log_metric(f'cv_best_score_{scoring_metric}', grid_search.best_score_)
        
        # 5. Evaluate Best Model on Test Set (Final MLOps Validation Step)
        y_pred = best_model.predict(X_test)
        
        final_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='binary', zero_division=0)
        }
        
        # Log final test metrics
        mlflow.log_metrics(final_metrics)

        # 6. Log and Register the Best Model
        signature = infer_signature(X_train, best_model.predict(X_train))
        
        model_info = mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="transaction_models_tuned", # Use a new artifact path
            signature=signature,
            input_example=X_train.head(5),
            # Register the model under a NEW NAME for the tuned version
            registered_model_name="TRANSACTION-log-reg-TUNED" 
        )
        
        print(f"MLflow Run ID: {run.info.run_id}")
        
        # Combine results for the return value
        best_results = {
            'best_params': grid_search.best_params_,
            'test_metrics': final_metrics
        }
        
        return best_model, best_results