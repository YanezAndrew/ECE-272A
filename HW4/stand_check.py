import pandas as pd
import numpy as np

# Load the standardized diabetes dataset
df = pd.read_csv('diabetes.csv')

# Calculate mean and standard deviation of BMI column
bmi_mean = df['BMI'].mean()
bmi_std = df['BMI'].std()

# Print the results
print(f"Mean of BMI: {bmi_mean}")
print(f"Standard Deviation of BMI: {bmi_std}")
