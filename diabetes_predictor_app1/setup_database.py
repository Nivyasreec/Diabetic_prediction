import sqlite3
import pandas as pd
import os

print("--- Database Setup and Population Script ---")

# Define paths relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(script_dir, 'predictions.db')
# Go up one level to find the 'data' folder
NUTRITION_CSV_PATH = os.path.join(script_dir, '..', 'data', 'nutrition_cleaned.csv')

# --- Check if the database file already exists ---
if os.path.exists(DB_PATH):
    print(f"Database file already exists at: {DB_PATH}")
    # Optional: ask the user if they want to delete and recreate it
    # For simplicity, we will just proceed. You can manually delete the .db file if you want a fresh start.

conn = None # Initialize connection to None
try:
    # --- Step 1: Create the Database and Tables ---
    # Connect to the database (this will create the file if it doesn't exist)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    print(f"Successfully connected to database at: {DB_PATH}")

    # SQL command to create the 'predictions' table
    create_predictions_table_sql = """
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        risk_score REAL,
        age INTEGER,
        bmi REAL,
        hypertension TEXT,
        heart_disease TEXT,
        hba1c_level REAL,
        blood_glucose INTEGER,
        total_calories REAL,
        total_sugar REAL,
        food_log TEXT
    );
    """
    
    # SQL command to create the 'food_items' table
    create_food_items_table_sql = """
    CREATE TABLE IF NOT EXISTS food_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE
    );
    """

    cursor.execute(create_predictions_table_sql)
    cursor.execute(create_food_items_table_sql)
    print("Tables 'predictions' and 'food_items' are ensured to exist.")

    # --- Step 2: Populate the 'food_items' Table ---
    print(f"Attempting to read food data from: {NUTRITION_CSV_PATH}")
    df_food = pd.read_csv(NUTRITION_CSV_PATH)
    
    food_names = sorted(df_food['Dish Name'].dropna().unique())
    print(f"Found {len(food_names)} unique food items to insert.")

    # 'INSERT OR IGNORE' prevents errors if a food name is already in the table
    cursor.executemany("INSERT OR IGNORE INTO food_items (name) VALUES (?)", [(name,) for name in food_names])

    # Commit all changes to the database
    conn.commit()
    print("Successfully populated the 'food_items' table.")
    print("\n--- DATABASE SETUP IS COMPLETE ---")

except sqlite3.Error as e:
    print(f"🔴 SQLite error: {e}")
except FileNotFoundError:
    print(f"🔴 CRITICAL ERROR: Nutrition data not found at '{NUTRITION_CSV_PATH}'. Please ensure the 'data' folder and its files are in the correct location.")
except Exception as e:
    print(f"🔴 An unexpected error occurred: {e}")
finally:
    # Close the database connection
    if conn:
        conn.close()
        print("Database connection closed.")