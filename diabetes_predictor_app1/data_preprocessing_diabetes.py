import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- REPLACE THIS WITH YOUR ACTUAL DIABETES CSV LINK ---
# Example: "https://raw.githubusercontent.com/datasets/diabetes/main/diabetes.csv"
# Or a local path: "C:/path/to/your/diabetes_prediction_dataset.csv"
DIABETES_CSV_LINK = "C:\\project_2\\diabetes_predictor_app1\\venv\\data\\diabetes_prediction_dataset.csv"

def preprocess_diabetes_data(file_path_or_link, output_dir="figures"):
    """
    Loads, cleans, and visualizes the diabetes dataset.
    
    Args:
        file_path_or_link (str): The path or URL to the diabetes prediction dataset CSV file.
        output_dir (str): Directory to save generated plots.
        
    Returns:
        pd.DataFrame: The cleaned diabetes DataFrame.
    """
    os.makedirs(output_dir, exist_ok=True) # Ensure figures directory exists

    print(f"Loading diabetes data from: {file_path_or_link}")
    try:
        diabetes_df = pd.read_csv(file_path_or_link)
    except FileNotFoundError:
        print(f"Error: The file at {file_path_or_link} was not found. Please check the path/URL.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while loading the dataset from {file_path_or_link}: {e}")
        return pd.DataFrame()

    print("\n--- First 5 Rows ---")
    print(diabetes_df.head())
    print("\n--- Dataset Info ---")
    print(diabetes_df.info())
    print("\n--- Descriptive Statistics ---")
    print(diabetes_df.describe())

    # Columns to check for 0s that represent missing values
    # These column names are based on `diabetes_prediction_dataset.csv`
    cols_to_impute_zeros = ['bmi', 'HbA1c_level', 'blood_glucose_level']
    existing_cols_to_impute = [col for col in cols_to_impute_zeros if col in diabetes_df.columns]

    if existing_cols_to_impute:
        print("\n--- Count of 0s before NaN replacement ---")
        print((diabetes_df[existing_cols_to_impute] == 0).sum())

        for col in existing_cols_to_impute:
            diabetes_df[col] = diabetes_df[col].replace(0, np.nan)

        print("\n--- Count of Missing Values (NaN) after replacement ---")
        print(diabetes_df.isnull().sum())
        
        for col in existing_cols_to_impute:
            if diabetes_df[col].isnull().any():
                median_val = diabetes_df[col].median()
                diabetes_df[col] = diabetes_df[col].fillna(median_val)
                print(f"Filled missing values in '{col}' with median: {median_val}")
    else:
        print("No relevant columns found for 0-replacement and imputation (bmi, HbA1c_level, blood_glucose_level).")

    # Ensure 'diabetes' column exists for subsequent steps
    if 'diabetes' not in diabetes_df.columns:
        print("Error: 'diabetes' target column not found in the dataset. Cannot proceed with target-dependent visualizations or training.")
        return diabetes_df # Return dataframe as is, but with warning

    print("\n--- Distribution of Diabetes Status ---")
    print(diabetes_df['diabetes'].value_counts())
    print(diabetes_df['diabetes'].value_counts(normalize=True) * 100)

    # Plot Distribution of Diabetes
    plt.figure(figsize=(6, 4))
    sns.countplot(x='diabetes', data=diabetes_df)
    plt.title('Distribution of Diabetes (Target Variable)')
    plt.savefig(os.path.join(output_dir, 'diabetes_distribution.png'))
    plt.close()

    if 'gender' in diabetes_df.columns:
        print("\n--- Distribution of Gender ---")
        print(diabetes_df['gender'].value_counts())
        plt.figure(figsize=(7, 5))
        sns.countplot(x='gender', hue='diabetes', data=diabetes_df)
        plt.title('Diabetes Status by Gender')
        plt.savefig(os.path.join(output_dir, 'diabetes_by_gender.png'))
        plt.close()
    else:
        print("Warning: 'gender' column not found for plotting.")

    if 'smoking_history' in diabetes_df.columns:
        print("\n--- Distribution of Smoking History ---")
        print(diabetes_df['smoking_history'].value_counts())
        plt.figure(figsize=(8, 5))
        sns.countplot(x='smoking_history', hue='diabetes', data=diabetes_df)
        plt.title('Diabetes Status by Smoking History')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'diabetes_by_smoking_history.png'))
        plt.close()
    else:
        print("Warning: 'smoking_history' column not found for plotting.")

    numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    existing_numerical_cols = [col for col in numerical_cols if col in diabetes_df.columns]

    if existing_numerical_cols:
        print("\n--- Histograms for Numerical Features ---")
        diabetes_df[existing_numerical_cols].hist(bins=30, figsize=(15, 10))
        plt.suptitle('Histograms of Numerical Features')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, 'numerical_histograms.png'))
        plt.close()

        print("\n--- Box Plots for Numerical Features by Diabetes Status ---")
        plt.figure(figsize=(15, 10))
        num_plots = len(existing_numerical_cols)
        rows = int(np.ceil(num_plots / 2))
        for i, col in enumerate(existing_numerical_cols):
            plt.subplot(rows, 2, i + 1)
            sns.boxplot(x='diabetes', y=col, data=diabetes_df)
            plt.title(f'{col} by Diabetes Status')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'numerical_boxplots.png'))
        plt.close()
    else:
        print("No relevant numerical columns found for histograms and box plots.")

    print("\n--- Correlation Matrix ---")
    numeric_df = diabetes_df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix of Features')
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
        plt.close()
    else:
        print("No numeric columns available to compute correlation matrix.")

    print("✅ Diabetes data preprocessing complete!")
    return diabetes_df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    figures_output_dir = os.path.join(current_dir, '..', 'figures')
    os.makedirs(figures_output_dir, exist_ok=True)
    
    # Pass the global link for preprocessing when run directly
    cleaned_df = preprocess_diabetes_data(DIABETES_CSV_LINK, figures_output_dir)
    
    if not cleaned_df.empty:
        data_save_path = os.path.join(current_dir, '..', 'data', 'diabetes_cleaned.csv')
        cleaned_df.to_csv(data_save_path, index=False)
        print(f"Cleaned diabetes data saved to {data_save_path}")