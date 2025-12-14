import mlflow
import os
from mlflow import MlflowClient
import mlflow.sklearn
from mlflow.exceptions import MlflowException

def fetch_and_load_latest_model(model_name: str):
    """
    Fetches the latest version of a specific registered model and loads it.
    Args:
        model_name (str): Name of the registered model.
    Returns:
        loaded_model: MLflow model object, or None if not found.
    """
    model_uri = f"models:/{model_name}/latest"
    print(f"Attempting to load latest model from URI: {model_uri}")
    try:
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        print(f"Successfully loaded the latest model for '{model_name}'.")
        return loaded_model
    except MlflowException as e:
        print(f"Error loading model from registry: {e}")
        return None


def fetch_and_load_best_model(model_name: str, metric_key: str = "accuracy"):
    """
    Fetches the model version with the highest value for a specific metric.
    Args:
        model_name (str): Name of the registered model.
        metric_key (str): Metric to sort by (default: 'accuracy').
    Returns:
        loaded_model: MLflow model object, or None if not found.
    """
    client = MlflowClient()
    filter_string = f"name='{model_name}'"
    ordered_versions = client.search_model_versions(
        filter_string=filter_string,
        order_by=[f"metrics.{metric_key} DESC"],
        max_results=1
    )

    if not ordered_versions:
        print(f"No model versions found for registered model: {model_name}")
        return None

    best_version_info = ordered_versions[0]
    best_version = best_version_info.version
    print(f"Best model version found (by {metric_key}): Version {best_version}")

    model_uri = f"models:/{model_name}/{best_version}"
    try:
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        print(f"Successfully loaded best model for '{model_name}'.")
        return loaded_model
    except MlflowException as e:
        print(f"Error loading best model: {e}")
        return None
