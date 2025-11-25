# 1. Import required libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

# Load data
df = pd.read_csv('insurance_dirty.csv')
print(f"Initial shape: {df.shape}")

# 1. Remove completely empty rows (all columns are NaN)
df = df.dropna(how='all')
print(f"After removing empty rows: {df.shape}")

# 2. Remove duplicates
df = df.drop_duplicates()
print(f"After removing duplicates: {df.shape}")

# 3. Clean age column - convert text to numeric, handle invalid values
def clean_age(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        # Handle "twenty" -> 20
        if value.lower() == 'twenty':
            return 20
        # Try to extract number from string
        try:
            return float(value)
        except:
            return np.nan
    try:
        return float(value)
    except:
        return np.nan

df['age'] = df['age'].apply(clean_age)

# 4. Clean BMI column - handle invalid values like "??", 999
def clean_bmi(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        if value == '??' or value.strip() == '':
            return np.nan
        try:
            val = float(value)
            # Remove outliers (BMI > 60 or < 10 is unrealistic)
            if val > 60 or val < 10:
                return np.nan
            return val
        except:
            return np.nan
    try:
        val = float(value)
        # Remove outliers
        if val > 60 or val < 10:
            return np.nan
        return val
    except:
        return np.nan

df['bmi'] = df['bmi'].apply(clean_bmi)

# 5. Clean charges column - handle "free" and outliers
def clean_charges(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        if value.lower() == 'free' or value.strip() == '':
            return np.nan
        try:
            val = float(value)
            # Remove extreme outliers (charges > 100000 or < 1000 seem unrealistic)
            if val > 100000 or val < 1000:
                return np.nan
            return val
        except:
            return np.nan
    try:
        val = float(value)
        # Remove extreme outliers
        if val > 100000 or val < 1000:
            return np.nan
        return val
    except:
        return np.nan

df['charges'] = df['charges'].apply(clean_charges)

# 6. Clean children column - ensure it's numeric
df['children'] = pd.to_numeric(df['children'], errors='coerce')

# 7. Clean categorical columns - remove extra whitespace and standardize
if df['sex'].dtype == 'object':
    df['sex'] = df['sex'].str.strip().str.lower()
if df['smoker'].dtype == 'object':
    df['smoker'] = df['smoker'].str.strip().str.lower()
if df['region'].dtype == 'object':
    df['region'] = df['region'].str.strip().str.lower()

# 8. Remove rows where critical columns are missing (age, charges are essential)
df = df.dropna(subset=['age', 'charges'])
print(f"After removing rows with missing age/charges: {df.shape}")

# 9. Impute missing values for BMI and children using median
imputer = SimpleImputer(strategy='median')
df[['bmi', 'children']] = imputer.fit_transform(df[['bmi', 'children']])

# 10. Ensure data types are correct
df['age'] = df['age'].astype(float)
df['bmi'] = df['bmi'].astype(float)
df['children'] = df['children'].astype(int)
df['charges'] = df['charges'].astype(float)

# 11. Remove any remaining rows with invalid categorical values
valid_sex = ['male', 'female']
valid_smoker = ['yes', 'no']
valid_regions = ['northeast', 'northwest', 'southeast', 'southwest']

df = df[df['sex'].isin(valid_sex) | df['sex'].isna()]
df = df[df['smoker'].isin(valid_smoker) | df['smoker'].isna()]
df = df[df['region'].isin(valid_regions) | df['region'].isna()]

# 12. Fill remaining missing categorical values with mode
if df['sex'].isna().any():
    df['sex'] = df['sex'].fillna(df['sex'].mode()[0] if len(df['sex'].mode()) > 0 else 'male')
if df['smoker'].isna().any():
    df['smoker'] = df['smoker'].fillna(df['smoker'].mode()[0] if len(df['smoker'].mode()) > 0 else 'no')
if df['region'].isna().any():
    df['region'] = df['region'].fillna(df['region'].mode()[0] if len(df['region'].mode()) > 0 else 'southeast')

# print(f"\nFinal shape: {df.shape}")
# print(f"\nMissing values:\n{df.isnull().sum()}")
# print(f"\nData types:\n{df.dtypes}")
# print(f"\nFirst few rows:\n{df.head(10)}")
# print(f"\nSummary statistics:\n{df.describe()}")

# Save cleaned data (before encoding/scaling)
###df.to_csv('insurance.csv', index=False)
# print(f"\nCleaned data saved to 'insurance.csv'")
# print(f"Shape before encoding/scaling: {df.shape}")

# ========== ENCODING AND SCALING ==========

# Create a copy for encoded/scaled data
df_encoded = df.copy()

# 13. Encode categorical variables using LabelEncoder
label_encoders = {}
categorical_cols = ['sex', 'smoker', 'region']

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le
    # print(f"\n{col} encoding:")
    # print(f"  Original values: {df[col].unique()}")
    # print(f"  Encoded values: {df_encoded[col].unique()}")

# 14. Scale numerical features using StandardScaler
numerical_cols = ['age', 'bmi', 'children', 'charges']
scaler = StandardScaler()

# Store original values for reference
df_encoded_scaled = df_encoded.copy()
df_encoded_scaled[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

# print(f"\n\nEncoded and Scaled Data:")
# print(f"Shape: {df_encoded_scaled.shape}")
# print(f"\nFirst 10 rows:")
# print(df_encoded_scaled.head(10))
# print(f"\nData types:")
# print(df_encoded_scaled.dtypes)
# print(f"\nSummary statistics (scaled):")
# print(df_encoded_scaled.describe())

# # Save encoded and scaled data
df_encoded_scaled.to_csv('insurance_encoded_scaled.csv', index=False)
print(f"\nEncoded and scaled data saved to 'insurance_encoded_scaled.csv'")

# Also save encoded but not scaled version
df_encoded.to_csv('insurance_encoded.csv', index=False)
print(f"Encoded (not scaled) data saved to 'insurance_encoded.csv'")
x =df_encoded_scaled.iloc[: , : -1]
y =df.iloc[:, -1]
print(x.head(10))
print(y.head(10))