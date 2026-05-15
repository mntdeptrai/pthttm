import pandas as pd
import os

def preprocess_data(input_file, output_file):
    print(f"Loading data from {input_file}...")
    if not os.path.exists(input_file):
        print(f"Error: {input_file} does not exist.")
        return

    df = pd.read_csv(input_file)
    
    # 1. Remove duplicates
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    duplicates_removed = initial_rows - len(df)
    print(f"Removed {duplicates_removed} duplicate rows.")

    # Define columns by type
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
    continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    # 2. Fill missing values
    missing_total = 0
    
    # Handle Categorical (using Mode)
    for col in categorical_cols:
        if col in df.columns:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                count = df[col].isnull().sum()
                df[col] = df[col].fillna(mode_val)
                print(f"Filled {count} missing values in '{col}' with mode: {mode_val}")
                missing_total += count
            
            # Ensure categorical columns are integers if they don't have fractional parts
            try:
                if (df[col] % 1 == 0).all():
                    df[col] = df[col].astype(int)
            except:
                pass

    # Handle Continuous (using Mean)
    for col in continuous_cols:
        if col in df.columns and df[col].isnull().any():
            mean_val = df[col].mean()
            count = df[col].isnull().sum()
            df[col] = df[col].fillna(mean_val)
            print(f"Filled {count} missing values in '{col}' with mean: {mean_val:.2f}")
            missing_total += count

    print(f"Filled total {missing_total} missing values.")

    # Save cleaned data
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}.")
    print(f"Final dataset shape: {df.shape}")

if __name__ == "__main__":
    input_csv = "data.csv"
    output_csv = "data_cleaned.csv"
    preprocess_data(input_csv, output_csv)
