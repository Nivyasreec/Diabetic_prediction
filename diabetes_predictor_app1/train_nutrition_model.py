import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer

print("Starting nutrition model training...")

# --- Configuration (CORRECTED) ---
# Use a raw string (r"...") to prevent backslash issues for the data path.
DATA_DIR = r"C:\project_2\diabetes_predictor_app1\venv\data"
NUTRITION_CSV_PATH = os.path.join(DATA_DIR, "nutrition_cleaned.csv")
# The output directory will be defined inside the function for robustness.

def train_nutrition_predictor():
    """Trains and saves calorie/sugar models and the TF-IDF vectorizer."""

    # --- THIS IS THE KEY CHANGE FOR SAVING ---
    # Get the directory where this script is located (the 'src' folder).
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Define the output directory to be 'model1' *inside* the script's directory.
    MODELS_OUTPUT_DIR = os.path.join(script_dir, "model1")
    # --- END OF KEY CHANGE ---

    os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)

    try:
        df_food = pd.read_csv(NUTRITION_CSV_PATH)
        print(f"Nutrition dataset loaded successfully from '{NUTRITION_CSV_PATH}'.")
    except FileNotFoundError:
        print(f"🔴 ERROR: Nutrition data not found at '{NUTRITION_CSV_PATH}'. Please ensure the file exists.")
        return

    df_food.columns = df_food.columns.str.strip()
    required_cols = ['Dish Name', 'Calories (kcal)', 'Free Sugar (g)']
    if not all(col in df_food.columns for col in required_cols):
        print(f"🔴 ERROR: Required columns {required_cols} not found in the nutrition data file.")
        return

    df_food = df_food.dropna(subset=required_cols).copy()
    df_food['Dish Name'] = df_food['Dish Name'].astype(str)
    for col in ['Calories (kcal)', 'Free Sugar (g)']:
        df_food[col] = pd.to_numeric(df_food[col], errors='coerce').fillna(df_food[col].median())

    print("Fitting TF-IDF Vectorizer to food names...")
    tfidf_vectorizer = TfidfVectorizer(max_features=1500, stop_words='english', ngram_range=(1, 2))
    X_food_features = tfidf_vectorizer.fit_transform(df_food['Dish Name'])

    print("Training Calorie & Sugar Prediction Models...")
    calorie_model = LinearRegression().fit(X_food_features, df_food['Calories (kcal)'])
    sugar_model = LinearRegression().fit(X_food_features, df_food['Free Sugar (g)'])

    print(f"\nSaving nutrition models to the correct directory: '{MODELS_OUTPUT_DIR}'")

    joblib.dump(tfidf_vectorizer, os.path.join(MODELS_OUTPUT_DIR, 'tfidf_vectorizer.pkl'))
    joblib.dump(calorie_model, os.path.join(MODELS_OUTPUT_DIR, 'calorie_model.pkl'))
    joblib.dump(sugar_model, os.path.join(MODELS_OUTPUT_DIR, 'sugar_model.pkl'))

    print(f"\n✅ Nutrition models ('calorie_model.pkl', 'sugar_model.pkl', 'tfidf_vectorizer.pkl') saved successfully.")
    print("They have been added to the 'model1' folder inside 'src'.")

if __name__ == "__main__":
    train_nutrition_predictor()