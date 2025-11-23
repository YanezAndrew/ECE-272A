import pandas as pd
import numpy as np

# Load the diabetes dataset
df = pd.read_csv('diabetes.csv')

# Standardize the BMI column
# Standardization: (x - mean) / std
bmi_mean = df['BMI'].mean()
bmi_std = df['BMI'].std()
df['BMI'] = (df['BMI'] - bmi_mean) / bmi_std

# Save the standardized dataset
df.to_csv('diabetes.csv', index=False)

print("BMI column has been standardized and saved to 'diabetes.csv'")