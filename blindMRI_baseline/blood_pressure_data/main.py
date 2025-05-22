import pandas as pd
import os

# Load the CSV
df = pd.read_csv('blood_pressure_data/csv/enhanced_health_data.csv') 


bp_data = df[['Systolic BP', 'Diastolic BP']].dropna()
bp_data = bp_data.sample(n=888, random_state=42).reset_index(drop=True)
bp_data.columns = ['SBP', 'DBP']

# Save to CSV
os.makedirs('outputs', exist_ok=True)
bp_data.to_csv('outputs/blood_pressure_data.csv', index=False)


