import sys
import os

# Set the working directory to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
print(f"Project root added to Python path: {project_root}")

# Disable TensorFlow oneDNN optimizations to avoid floating-point rounding issues
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import mlflow
import mlflow.sklearn
import mlflow.keras
import joblib
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


class ModelTrainer:
    def __init__(self, experiment_name="NEV Fault Prediction"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(self.experiment_name)

    def tune_rf(self, X_train, y_train):
        """Hyperparameter tuning for Random Forest."""
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10]
        }
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        print(f"Best parameters for Random Forest: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def tune_gb(self, X_train, y_train):
        """Hyperparameter tuning for Gradient Boosting."""
        param_grid = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 10]
        }
        grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        print(f"Best parameters for Gradient Boosting: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def train_rf(self, X_train, y_train, X_test, y_test):
        print("Initializing Random Forest training...")
        rf_model = self.tune_rf(X_train, y_train)
        rf_model.fit(X_train, y_train)
        print("✅ Random Forest training complete.")

        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        print(f"Random Forest - Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}")
        return rf_model, accuracy

    def train_gb(self, X_train, y_train, X_test, y_test):
        print("Initializing Gradient Boosting training...")
        gb_model = self.tune_gb(X_train, y_train)
        gb_model.fit(X_train, y_train)
        print("✅ Gradient Boosting training complete.")

        y_pred = gb_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        print(f"Gradient Boosting - Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}")
        return gb_model, accuracy

    def train_nn(self, X_train, y_train, X_test, y_test, input_dim):
        print("Initializing Neural Network training...")
        
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        y_train_onehot = tf.keras.utils.to_categorical(y_train_encoded, num_classes=len(label_encoder.classes_))
        y_test_onehot = tf.keras.utils.to_categorical(y_test_encoded, num_classes=len(label_encoder.classes_))

        with mlflow.start_run(run_name="Deep Learning Model"):
            try:
                model = Sequential([
                    Dense(256, activation='relu', input_dim=input_dim),
                    Dropout(0.4),
                    Dense(128, activation='relu'),
                    Dropout(0.3),
                    Dense(64, activation='relu'),
                    Dense(len(label_encoder.classes_), activation='softmax')
                ])
                model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                model.fit(X_train, y_train_onehot, validation_data=(X_test, y_test_onehot), epochs=150, batch_size=64, callbacks=[early_stopping], verbose=1)

                loss, accuracy = model.evaluate(X_test, y_test_onehot, verbose=0)
                return model, accuracy
            except Exception as e:
                print(f"❌ Error during Neural Network training: {e}")


if __name__ == "__main__":
    from src.data_processor import DataProcessor
    data_file_path = "F:/Portfolio Projects/NEV_Fault_Prediction/data/Fault_nev_dataset.csv"
    processor = DataProcessor(data_file_path)
    data = processor.load_data()
    X, y = processor.preprocess(target_column="fault_type", categorical_columns=["road_condition"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    trainer = ModelTrainer()
    
    rf_model, rf_accuracy = trainer.train_rf(X_train, y_train, X_test, y_test)
    gb_model, gb_accuracy = trainer.train_gb(X_train, y_train, X_test, y_test)
    nn_model, nn_accuracy = trainer.train_nn(X_train, y_train, X_test, y_test, input_dim=X_train.shape[1])
    
    best_model, best_model_name = max(
        [(rf_model, "best_rf_model.pkl", rf_accuracy), (gb_model, "best_gb_model.pkl", gb_accuracy), (nn_model, "best_nn_model.keras", nn_accuracy)],
        key=lambda x: x[2]
    )[:2]
    
    best_model_path = f"F:/Portfolio Projects/NEV_Fault_Prediction/models/{best_model_name}"
    
    if isinstance(best_model, (RandomForestClassifier, GradientBoostingClassifier)):
        joblib.dump(best_model, best_model_path)
    else:
        best_model.save(best_model_path)
    
    print(f"✅ Best model saved as {best_model_name} with highest accuracy.")
