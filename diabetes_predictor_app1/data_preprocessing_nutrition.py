import pandas as pd
import os

# --- REPLACE THIS WITH YOUR ACTUAL NUTRITION CSV LINK ---
# Example: "https://raw.githubusercontent.com/datasets/nutrition/main/indian_food_nutrition.csv"
# Or a local path: "C:/path/to/your/Indian_Food_Nutrition_Processed.csv"
NUTRITION_CSV_LINK = "C:\\project_2\\diabetes_predictor_app1\\venv\\data\\Indian_Food_Nutrition_Processed.csv"

def preprocess_nutrition_data(file_path_or_link):
    """
    Loads and preprocesses the food nutrition dataset from a file path or URL.
    Returns the cleaned DataFrame.
    """
    print(f"Loading nutrition data from: {file_path_or_link}")
    try:
        df_food = pd.read_csv(file_path_or_link)
    except FileNotFoundError:
        print(f"Error: The file at {file_path_or_link} was not found. Please check the path/URL.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while loading the dataset from {file_path_or_link}: {e}")
        return pd.DataFrame()

    df_food.columns = df_food.columns.str.strip() # Clean column names

    print("\n--- First 5 Rows of Food Nutrition Data ---")
    print(df_food.head())
    print("\n--- Food Nutrition Dataset Info ---")
    print(df_food.info())
    print("\n--- Food Nutrition Descriptive Statistics ---")
    print(df_food.describe())

    # Ensure 'Calories (kcal)' and 'Free Sugar (g)' are numeric and handle missing/invalid values
    for col in ['Calories (kcal)', 'Free Sugar (g)']:
        if col in df_food.columns:
            df_food[col] = pd.to_numeric(df_food[col], errors='coerce')
            if df_food[col].isnull().any():
                median_val = df_food[col].median()
                df_food[col] = df_food[col].fillna(median_val)
                print(f"Filled missing values in '{col}' with median: {median_val}")
            # Ensure non-negative values for calories and sugar
            df_food[col] = df_food[col].apply(lambda x: max(0, x))
        else:
            print(f"Warning: Column '{col}' not found in food nutrition data for processing.")

    # Ensure 'Dish Name' is present and not empty for text processing
    if 'Dish Name' not in df_food.columns:
        raise ValueError("The 'Dish Name' column is required but not found in the nutrition dataset.")
    df_food = df_food.dropna(subset=['Dish Name']) # Drop rows where dish name is missing
    df_food['Dish Name'] = df_food['Dish Name'].astype(str) # Ensure it's string type

    print("\n✅ Food nutrition data preprocessing complete!")
    return df_food

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    cleaned_food_df = preprocess_nutrition_data(NUTRITION_CSV_LINK)
    if not cleaned_food_df.empty:
        data_save_path = os.path.join(current_dir, '..', 'data', 'nutrition_cleaned.csv')
        cleaned_food_df.to_csv(data_save_path, index=False)
        print(f"Cleaned nutrition data saved to {data_save_path}")