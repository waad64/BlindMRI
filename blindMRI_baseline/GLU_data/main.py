import pandas as pd
import os

# Load the CSV with glucose data
df = pd.read_csv('GLU_data/csv/diabetes_prediction_dataset.csv') 

glu_data = df[['blood_glucose_level']].sample(n=888, random_state=42).reset_index(drop=True)

glu_data = glu_data.rename(columns={'blood_glucose_level': 'GLU'})

# Save to CSV
os.makedirs('outputs', exist_ok=True)
glu_data.to_csv('outputs/glucose_data.csv', index=False)


