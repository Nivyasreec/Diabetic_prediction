import sqlite3
import pandas as pd
import os

print("--- Database Food Item Population Script ---")

# --- Define Paths ---
# This script is located in the 'src' folder.
script_dir = os.path.dirname(os.path.abspath(__file__))
# The database file is also in the 'src' folder.
DB_PATH = os.path.join(script_dir, 'predictions.db')
# The data file is up one level ('..') from 'src', then inside the 'data' folder.
NUTRITION_CSV_PATH = os.path.join(script_dir, '..', 'data', 'nutrition_cleaned.csv')

conn = None # Initialize connection to None
try:
    # --- Step 1: Connect to the existing database ---
    if not os.path.exists(DB_PATH):
        print(f"🔴 ERROR: Database file not found at '{DB_PATH}'.")
        print("Please create the database and its tables first (e.g., by running setup_database.py).")
        exit()
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    print(f"Successfully connected to the database at: {DB_PATH}")

    # --- Step 2: Read the food data from the CSV file ---
    print(f"Attempting to read food data from: {NUTRITION_CSV_PATH}")
    df_food = pd.read_csv(NUTRITION_CSV_PATH)
    
    # Get a unique, sorted list of all dish names
    food_names = sorted(df_food['Dish Name'].dropna().unique())
    print(f"Found {len(food_names)} unique food items to insert into the database.")

    # --- Step 3: Execute the INSERT query for each food name ---
    # The 'INSERT OR IGNORE' command is crucial. It tells the database:
    # "If this food name already exists in the table, just skip it and don't raise an error."
    # This makes the script safe to run multiple times without creating duplicates.
    
    # This is the SQL INSERT query we will run for every food name.
    insert_query = "INSERT OR IGNORE INTO food_items (name) VALUES (?)"
    
    # For efficiency, we use 'executemany' which runs the query for every item in a list.
    # The data needs to be a list of tuples, e.g., [('Apple',), ('Pizza',), ...]
    data_to_insert = [(name,) for name in food_names]
    
    cursor.executemany(insert_query, data_to_insert)

    # Commit the changes (this saves them permanently to the database file)
    conn.commit()
    
    print("Successfully populated the 'food_items' table.")
    print("The food dropdown menu in your app is now ready.")

except sqlite3.Error as e:
    print(f"🔴 SQLite error: {e}")
except FileNotFoundError:
    print(f"🔴 CRITICAL ERROR: Nutrition data not found at '{NUTRITION_CSV_PATH}'. Please ensure the file path is correct.")
except Exception as e:
    print(f"🔴 An unexpected error occurred: {e}")
finally:
    # Always close the database connection
    if conn:
        conn.close()
        print("Database connection closed.")