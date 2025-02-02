# 🚗 Fault Prediction With Machine Learning & Deep Learning

An end-to-end machine learning pipeline for predicting faults in New Energy Vehicles using Random Forest, Gradient Boosting, and Deep Learning models, integrated with MLflow for model tracking and FastAPI for deployment.

## 📌 Project Overview

This project builds a fault prediction system for New Energy Vehicles (NEV) using Random Forest, Gradient Boosting, and Deep Learning (Neural Network) models. The pipeline includes:
- Data preprocessing
- Model training
- Performance tracking with MLflow
- Model deployment using FastAPI
- Visualization of feature importance

## ✅ Key Features

- **ETL Pipeline**: Data preprocessing, feature engineering, and dataset splitting.
- **Machine Learning & Deep Learning Models**: Random Forest, Gradient Boosting, and Neural Networks.
- **Automated Model Tracking**: MLflow for logging experiments and version control.
- **Best Model Selection**: Automatically selects and saves the best-performing model.
- **Model Deployment**: Deploys the best model via FastAPI for real-time predictions.
- **Feature Importance Analysis**: Visualizes key factors influencing predictions.

## 📂 Project Structure

```
 fault_prediction_project/
│── data/                   # Raw and processed data
│── models/                 # Saved best models (.pkl for RF & GB, .keras for NN)
│── reports/                # Model performance reports and visualizations
│── src/                    # Source code for model training and deployment
│   ├── data_processor.py   # Handles data loading and preprocessing
│   ├── model_trainer.py    # Trains RF, GB & NN models, logs experiments in MLflow
│   ├── model_deployer.py   # Saves, logs, and serves the best model
│   ├── visualization.py    # Feature importance plots
│── app/                    # FastAPI application for model serving
│   ├── fastapi_app.py      # REST API for real-time predictions
│── notebooks/              # Jupyter notebooks for exploratory analysis
│── mlruns/                 # MLflow experiment tracking data
│── README.md               # Project documentation (this file)
│── requirements.txt        # Required Python packages
│── config.yaml             # Configuration settings (optional)
```

## 📊 Dataset

**Source**: [Kaggle]

**Data Description**:
- **Features**: Vehicle sensor data, driving patterns, road conditions, environmental factors, maintenance logs.
- **Target Variable**: `fault_type` (categorical, indicating vehicle faults).

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/binita-roy/Fault-Prediction-With-Machine-Learning-Deep-Learning.git
cd NEV_Fault_Prediction
```

### 2️⃣ Create a Virtual Environment (Recommended)
```sh
python -m venv venv
source venv/bin/activate   # For Mac/Linux
venv\Scripts\activate      # For Windows
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

## 🏗️ How to Run the Project

### 1️⃣ Train the Models
Run the training script to preprocess the data and train Random Forest, Gradient Boosting, and Deep Learning models.
```sh
python src/model_trainer.py
```
The best model will be saved in the `models/` directory.

### 2️⃣ Deploy the Best Model using FastAPI
Run the FastAPI server to serve real-time predictions.
```sh
uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```
The API will be available at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 3️⃣ Run MLflow UI for Experiment Tracking
```sh
mlflow ui --backend-store-uri "file:mlruns"
```
Access MLflow UI at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 🔬 Results & Model Comparison

| Model            | Accuracy | F1 Score |
|-----------------|----------|----------|
| Random Forest   | 0.66     | 0.66     |
| Gradient Boosting | 0.63     | 0.64     |
| Neural Network  | 0.61     | 0.60     |

- **Feature Importance (Random Forest & Gradient Boosting)**: Visualizes top factors influencing fault prediction.
- **Neural Network Feature Analysis**: Compares the effect of input features on fault classification.

## 📊 Feature Importance Visualization
Run the following script to generate feature importance plots:
```sh
python src/visualization.py
```

## 🚀 Endpoints & API Usage

| Method | Endpoint  | Description                  |
|--------|----------|------------------------------|
| POST   | `/predict` | Get predictions from the best model |

### Example Request
```json
{
  "data": [[0.5, 1.2, 3.4, 2.1, 4.5, 1.1, 0.8, 3.2, 2.5, 1.0]]
}
```

### Example Response
```json
{
  "predictions": ["engine_overheating"]
}
```

## 🛠️ Technologies Used

- **Programming**: Python (Pandas, NumPy, Scikit-learn, TensorFlow, FastAPI)
- **Machine Learning**: Random Forest, Gradient Boosting, Neural Networks
- **Model Tracking**: MLflow
- **Deployment**: FastAPI, Uvicorn
- **Visualization**: Matplotlib, Seaborn

## 📌 Future Improvements

- Add hyperparameter tuning for optimal model performance.
- Improve feature engineering for better predictions.
- Deploy the API using Docker & cloud services (AWS/Azure/GCP).

## 📝 Acknowledgments

- **Data Source**: Kaggle
- **Libraries**: Scikit-learn, TensorFlow, FastAPI, MLflow

## 💡 Contributions & Issues

- Feel free to fork the repo, open issues, or submit pull requests.
- If you find a bug or have suggestions, please create an issue.

## 📌 Author: Binita Roy

## 📜 License: MIT License

