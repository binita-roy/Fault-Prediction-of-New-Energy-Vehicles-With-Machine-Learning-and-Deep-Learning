import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable TensorFlow oneDNN optimizations

import mlflow
import mlflow.keras
import mlflow.sklearn
import joblib
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

class ModelDeployer:
    @staticmethod
    def save_model(model, path, model_type="sklearn"):
        """Save a trained model to the given path."""
        if model_type in ["sklearn", "gradient_boosting"]:
            joblib.dump(model, path)
        else:
            model.save(path)
        print(f"✅ Model saved at {path}")

    @staticmethod
    def serve_model(model_name, input_data):
        """Load the model from MLflow and make predictions."""
        model_uri = f"models:/{model_name}/latest"
        print(f"🔄 Loading model from MLflow: {model_uri}")
        
        if model_name in ["best_rf_model", "best_gb_model"]:
            model = mlflow.sklearn.load_model(model_uri)
        else:
            model = mlflow.keras.load_model(model_uri)

        input_data = np.array(input_data, dtype=np.float32)
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)

        predictions = model.predict(input_data)
        return predictions

if __name__ == "__main__":
    print("🔄 Loading trained model for deployment...")
    
    model_paths = {
        "best_rf_model": "F:/Portfolio Projects/NEV_Fault_Prediction/models/best_rf_model.pkl",
        "best_gb_model": "F:/Portfolio Projects/NEV_Fault_Prediction/models/best_gb_model.pkl",
        "best_nn_model": "F:/Portfolio Projects/NEV_Fault_Prediction/models/best_nn_model.keras"
    }
    
    best_model = None
    best_model_name = None
    best_model_type = None

    for model_name, model_path in model_paths.items():
        if os.path.exists(model_path):
            print(f"✅ Best model found: {model_name}")
            if "nn" in model_name:
                best_model = tf.keras.models.load_model(model_path)
                best_model_type = "keras"
            else:
                best_model = joblib.load(model_path)
                best_model_type = "sklearn" if "rf" in model_name else "gradient_boosting"
            best_model_name = model_name
            break

    if best_model is None:
        raise FileNotFoundError("❌ No best model found! Train and save models first.")

    deployed_model_path = f"F:/Portfolio Projects/NEV_Fault_Prediction/models/deployed_model.{best_model_type}"
    print("💾 Saving best model for deployment...")
    ModelDeployer.save_model(best_model, deployed_model_path, best_model_type)

    print("📢 Logging model to MLflow...")
    mlflow.set_tracking_uri("file:F:/Portfolio Projects/NEV_Fault_Prediction/mlruns")
    mlflow.set_experiment("NEV_Fault_Prediction")

    with mlflow.start_run():
        if best_model_type in ["sklearn", "gradient_boosting"]:
            model_info = mlflow.sklearn.log_model(best_model, best_model_name)
        else:
            model_info = mlflow.keras.log_model(best_model, best_model_name)

    client = mlflow.tracking.MlflowClient()
    try:
        client.create_registered_model(best_model_name)
        print(f"✅ Model registered in MLflow as '{best_model_name}'")
    except mlflow.exceptions.MlflowException:
        print(f"⚠️ Model '{best_model_name}' already exists in MLflow, skipping registration.")

model_uri = model_info.model_uri

# ✅ Ensure an MLflow run is active before creating the model version
with mlflow.start_run():
    run_id = mlflow.active_run().info.run_id  # Get the active run ID

try:
    client.create_model_version(name=best_model_name, source=model_uri, run_id=run_id)
    print(f"✅ Model version created for '{best_model_name}'")
except mlflow.exceptions.MlflowException:
    print(f"⚠️ Model version for '{best_model_name}' already exists.")

    
    print("🚀 Loading and serving the model via MLflow...")
    test_input = [[0.5] * 17]
    predictions = ModelDeployer.serve_model(best_model_name, test_input)
    print(f"📌 Model Predictions: {predictions}")
