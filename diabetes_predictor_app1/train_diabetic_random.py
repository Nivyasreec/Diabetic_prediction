import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

print("Starting model training for 'diabetes_prediction_dataset.csv'...")

# --- 1. Load the Dataset ---
# Using the exact path you provided.
file_path = "C:\\project_2\\diabetes_predictor_app1\\venv\\data\\diabetes_prediction_dataset.csv"

try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f" ERROR: The file was not found at the specified path: {file_path}")
    print(" Please ensure the path is correct and the file exists.")
    exit()

# --- 2. Data Validation and Diagnostics ---
print(f"\n--- Data Diagnostics ---")
print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
if df.shape[0] < 1000: # This dataset should be large
    print("WARNING: This dataset appears smaller than expected. The full version has 100,000 rows.")

if 'diabetes' not in df.columns:
    print("CRITICAL STOP: The target column 'diabetes' was not found in the dataset.")
    exit()

print("\nTarget variable 'diabetes' distribution:")
print(df['diabetes'].value_counts(normalize=True))
print("------------------------\n")

# --- 3. Feature Engineering & Preprocessing ---

# Drop the 'gender' column if it only contains 'Other' as it provides no predictive value
if 'gender' in df.columns and df['gender'].nunique() == 1 and df['gender'].iloc[0] == 'Other':
    df = df.drop('gender', axis=1)
    print("Dropped 'gender' column as it only contained a single value.")

# Convert categorical variables into dummy/indicator variables (One-Hot Encoding)
# This is the correct way to handle text-based columns for machine learning.
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print("Categorical variables have been converted to numerical format.")

# Define features (X) and target (y) AFTER encoding
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# --- 4. Scale Numerical Features ---
# We scale the features to ensure they are all on a similar scale.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert the scaled array back to a DataFrame to keep column names
X = pd.DataFrame(X_scaled, columns=X.columns)
print("All features have been scaled using StandardScaler.")

# --- 5. Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")

# --- 6. Hyperparameter Tuning with GridSearchCV ---
print("\nStarting hyperparameter tuning to find the best model... (This may take a few minutes)")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2']
}
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2) # Using 3 folds for speed
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("\nHyperparameter tuning finished.")
print(f"Best parameters found: {grid_search.best_params_}")

# --- 7. Evaluate the Best Model ---
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\n--- Best Model Evaluation ---")
print(f"✅✅✅ Best Model Accuracy on test set: {accuracy:.4f} ✅✅✅")
if accuracy > 0.9:
    print("Excellent accuracy achieved!")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("------------------------")

# --- 8. Save All Necessary Artifacts ---
# Create a directory for models if it doesn't exist
output_dir = 'models'
os.makedirs(output_dir, exist_ok=True)

# Save the best model, the scaler, and the feature names
with open(os.path.join(output_dir, 'diabetes_rf_model.pkl'), 'wb') as file:
    pickle.dump(best_model, file)
print(f"✅ Best trained model saved to '{output_dir}/diabetes_rf_model.pkl'")

with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as file:
    pickle.dump(scaler, file)
print(f"✅ Scaler object saved to '{output_dir}/scaler.pkl'")

model_features = X.columns.tolist()
with open(os.path.join(output_dir, 'model_features.pkl'), 'wb') as file:
    pickle.dump(model_features, file)
print(f"✅ Model feature names saved to '{output_dir}/model_features.pkl'")

print("\nModel training script finished successfully.")