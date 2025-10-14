import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

print("--- Starting FINAL Model Training (Gold Standard Method) ---")

# --- 1. Define File Paths ---
DATA_DIR = r"C:\project_2\diabetes_predictor_app1\venv\data"
DATA_FILE_PATH = os.path.join(DATA_DIR, "diabetes_cleaned.csv")

# --- 2. Load the Dataset ---
try:
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"🔴 ERROR: File not found at '{DATA_FILE_PATH}'.")
    exit()

# --- 3. Feature Selection ---
if 'gender' in df.columns and 'Other' in df['gender'].unique():
    df = df[df['gender'] != 'Other']
final_features = ['age', 'bmi', 'hypertension', 'heart_disease', 'HbA1c_level', 'blood_glucose_level', 'diabetes']
df_final = df[final_features].copy()
X = df_final.drop('diabetes', axis=1)
y = df_final['diabetes']
print(f"Selected final features: {X.columns.tolist()}")

# --- 4. Split Data BEFORE Scaling ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data split into {len(X_train)} training and {len(X_test)} testing samples.")

# --- 5. Scale Numerical Features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled correctly using only training data statistics.")

# --- 6. Train the XGBoost Model ---
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Calculated scale_pos_weight for imbalance: {scale_pos_weight:.2f}")
print("\nTraining XGBoost model...")
xgb_model = XGBClassifier(
    objective='binary:logistic', eval_metric='logloss', use_label_encoder=False,
    scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1
)
xgb_model.fit(X_train_scaled, y_train)
print("Model training completed.")

# --- 7. Evaluate Model Performance ---
y_pred = xgb_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("\n--- Model Evaluation ---")
print(f"✅ Final Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- 8. Save All Necessary Artifacts ---
script_dir = os.path.dirname(os.path.abspath(__file__))
MODELS_OUTPUT_DIR = os.path.join(script_dir, "model1")
os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)
print(f"\nSaving final model artifacts to: '{MODELS_OUTPUT_DIR}'")
joblib.dump(xgb_model, os.path.join(MODELS_OUTPUT_DIR, 'diabetes_xgboost_model.pkl'))
joblib.dump(scaler, os.path.join(MODELS_OUTPUT_DIR, 'diabetes_scaler.pkl'))
joblib.dump(X.columns.tolist(), os.path.join(MODELS_OUTPUT_DIR, 'diabetes_model_features.pkl'))
print(f"\n✅ All final model artifacts have been saved successfully.")