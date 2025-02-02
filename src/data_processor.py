import os

# Set the working directory
project_dir = "F:/Portfolio Projects/NEV_Fault_Prediction"
os.chdir(project_dir)
print(f"Updated working directory: {os.getcwd()}")

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        return self.data

    def preprocess(self, target_column, categorical_columns):
        # Encode categorical features
        for column in categorical_columns:
            le = LabelEncoder()
            self.data[column] = le.fit_transform(self.data[column])
            self.label_encoders[column] = le

        # Separate features and target
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]

        # Scale numerical features
        X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)

        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled

    def split_data(self, X, y, test_size=0.2):
        return train_test_split(X, y, test_size=test_size, random_state=42)