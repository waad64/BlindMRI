import openai
import pandas as pd
import os
from dotenv import load_dotenv
from tqdm import tqdm

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load dataset
csv_path = "merged_dataset.csv"
df = pd.read_csv(csv_path)

# Convert entire dataframe to records
data_json = df.to_dict(orient="records")

# Your big system prompt describing the task and context:
analysis_prompt = f"""
You are a biomedical data analyst and expert in psychophysiology, neuroimaging, and acoustic environments.

You are provided with a dataset of {len(data_json)} patients undergoing MRI scans. Each record includes the following physiological and acoustic features:

- EDA (Electrodermal Activity in microsiemens)
- TEMP (Skin temperature in °C)
- Patient Age
- Patient Gender (0 = male, 1 = female)
- SBP (Systolic Blood Pressure)
- DBP (Diastolic Blood Pressure)
- GLU (Blood glucose mg/dL)
- HR (Heart Rate in bpm)
- IBI_seq (Inter-beat Interval sequence mean in ms)
- MAP (Mean Arterial Pressure)
- HRV (Heart Rate Variability)
- PDB (Peak Decibel Level from MRI)
- NBM (Noise Bursts per Minute)

There are **3 stress states** hypothesized based on combined acoustic and physiological patterns:
- **Likely Non-Stress**: 
    - PDB between 50–65 dB
    - HRV high, HR and SBP lower
    - Stable TEMP and low EDA
- **Borderline Stress**:
    - PDB between 66–80 dB
    - Slight elevation in HR, MAP, and EDA
    - Moderate HRV decline
- **Likely Stress**:
    - PDB >80 dB
    - HR, SBP, DBP, and EDA elevated
    - HRV drops significantly
    - TEMP may decrease (stress-induced vasoconstriction)

🔍 **Your Task**:
- Analyze coherence between **PDB** and physiological markers
- Identify  **stress state** for each sample

Respond ONLY with one of these labels: "Likely Non-Stress", "Borderline Stress", or "Likely Stress".
"""

def format_row_for_prompt(row):
    # Make the user prompt concise and clean
    return (
        f"EDA: {row['EDA']}, TEMP: {row['TEMP']}, Age: {row['Patient Age']}, "
        f"Gender: {row['Patient Gender']}, SBP: {row['SBP']}, DBP: {row['DBP']}, GLU: {row['GLU']}, "
        f"HR: {row['HR']}, IBI_seq: {row['IBI_seq']}, MAP: {row['MAP']}, HRV: {row['HRV']}, "
        f"PDB: {row['PDB']}, NBM: {row['NBM']}"
    )

def detect_stress_state(row):
    user_prompt = format_row_for_prompt(row)

    messages = [
        {"role": "system", "content": analysis_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        max_tokens=10  
    )
    label = response.choices[0].message.content.strip()
    return label

# Detect stress state for all rows, with progress bar
labels = []
print("Starting stress state detection for each row...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    label = detect_stress_state(row)
    labels.append(label)

df['stress_state'] = labels

# Save the enriched dataframe for later use
df.to_csv("training/NBM_training/csv/classified_stress_state.csv", index=False)

