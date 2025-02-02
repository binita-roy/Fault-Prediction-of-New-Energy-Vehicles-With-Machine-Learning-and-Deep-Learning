from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import mlflow.keras
import os
import tensorflow as tf

# ✅ Create FastAPI app
app = FastAPI()

# ✅ Define input data structure
class PredictionInput(BaseModel):
    data: list

# 🔥 Define model paths
PROJECT_DIR = "F:/Portfolio Projects/fault_prediction_project"
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

BEST_MODELS = {
    "random_forest": os.path.join(MODEL_DIR, "best_rf_model.pkl"),
    "gradient_boosting": os.path.join(MODEL_DIR, "best_gb_model.pkl"),
    "neural_network": os.path.join(MODEL_DIR, "best_nn_model.keras"),
}

# ✅ Load the best model
print("🔄 Loading best model for deployment...")
model, model_type = None, None

for model_name, model_path in BEST_MODELS.items():
    if os.path.exists(model_path):
        print(f"✅ Loading {model_name} model from local file...")
        if "nn" in model_name:
            model = tf.keras.models.load_model(model_path)
            model_type = "neural_network"
        else:
            model = joblib.load(model_path)
            model_type = model_name
        break

if model is None:
    raise FileNotFoundError("❌ No best model found! Train and save models first.")

@app.post("/predict")
def predict(input_data: PredictionInput):
    """Handle prediction requests."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not available")
    
    try:
        # ✅ Ensure correct input format
        data = np.array(input_data.data, dtype=np.float32)

        # ✅ Ensure correct shape
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # ✅ Make predictions
        if model_type == "neural_network":
            predictions = model.predict(data).tolist()
        else:
            predictions = model.predict(data).tolist()
        
        return {"predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
