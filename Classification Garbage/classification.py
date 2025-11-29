import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


print(" 1. Loading Data ")
try:
   
    df = pd.read_csv('heart_dirty.csv')
    print(f"Initial shape: {df.shape}")
except FileNotFoundError:
    print("Error: Please ensure the 'heart_dirty.csv' file is in the correct path.")
    exit()


print("\n 2. Initial Cleanup ")

df_cleaned = df.drop_duplicates()
print(f"After removing duplicates: {df_cleaned.shape}")


numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
target_col = 'HeartDisease'


print("\n 3. Handling Text Inconsistencies ")

df_cleaned.loc[:, 'Sex'] = df_cleaned['Sex'].replace({'m': 'M', 'f': 'F', 'male': 'M', 'female': 'F'})

for col in categorical_cols:
    if col in df_cleaned.columns:
        df_cleaned.loc[:, col] = df_cleaned[col].astype(str).str.strip().str.upper()


print("\n 4. Outlier Handling (Capping Outliers) ")

for col in ['Cholesterol', 'RestingBP']:
    
    Q1 = df_cleaned[col].quantile(0.25)
    Q3 = df_cleaned[col].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR
    
    
    df_cleaned.loc[:, col] = np.where(df_cleaned[col] > upper_bound, upper_bound, df_cleaned[col])
    df_cleaned.loc[:, col] = np.where(df_cleaned[col] < lower_bound, lower_bound, df_cleaned[col])
    print(f"- Handled outliers in {col}")



print("\n 5. Missing Value Imputation ")


imputer_numeric = SimpleImputer(strategy='median')
df_cleaned.loc[:, numerical_cols] = imputer_numeric.fit_transform(df_cleaned[numerical_cols])


imputer_categorical = SimpleImputer(strategy='most_frequent')
df_cleaned.loc[:, categorical_cols] = imputer_categorical.fit_transform(df_cleaned[categorical_cols])

print("Missing values after imputation:")
print(df_cleaned.isnull().sum().sum())


print("\n 6. Encoding and Scaling ")


df_processed = pd.get_dummies(df_cleaned, columns=categorical_cols, drop_first=True)
print(f"Shape after Encoding: {df_processed.shape}")


scaler = StandardScaler()

cols_to_scale = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']


df_processed.loc[:, cols_to_scale] = scaler.fit_transform(df_processed[cols_to_scale])

print(f"\nFinal Processed Data (First 5 rows):")
print(df_processed.head())
print(f"\nFinal Shape: {df_processed.shape}")


df_processed.to_csv('heart_cleaned_processed_final_en.csv', index=False)