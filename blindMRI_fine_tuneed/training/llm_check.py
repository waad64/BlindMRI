import openai
import pandas as pd
import os
from dotenv import load_dotenv
from tqdm import tqdm

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load dataset
csv_path = "merged_dataset_v2.csv" 
df = pd.read_csv(csv_path)

# Convert entire dataframe to records (dict list)
data_json = df.to_dict(orient="records")

# Big system prompt with context and variables to check
analysis_prompt = f"""
You are a biomedical data analyst and expert in psychophysiology, neuroimaging, and acoustic environments.

You receive data for {len(data_json)} patients undergoing MRI scans, each with physiological features and contextual patient info:

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

Contextual clinical data:
- Blindness Duration (months)
- First MRI Experience (1 = Yes, 0 = No)
- Pre-procedure Briefing (1 = Yes, 0 = No)
- Headphones Provided (1 = Yes, 0 = No)
- Cause of Blindness (e.g., congenital, diabetic retinopathy, trauma)
- Mobility Independence (1 = dependent, 0 = independent)
- Anxiety Level (1 = anxious, 0 = unanxious)

Your task is to analyze the coherence between physiological data and clinical/contextual variables to classify the patient into one of these stress states:

- "Likely Non-Stress"
- "Borderline Stress"
- "Likely Stress"

Only respond with one of these labels, no explanations.
"""

def format_row_for_prompt(row):
    # Format patient data into a concise, clear prompt snippet
    # Note: handle NaNs or missing values gracefully if needed
    return (
        f"EDA: {row.get('EDA', 'NA')}, TEMP: {row.get('TEMP', 'NA')}, Age: {row.get('Patient Age', 'NA')}, "
        f"Gender: {row.get('Patient Gender', 'NA')}, SBP: {row.get('SBP', 'NA')}, DBP: {row.get('DBP', 'NA')}, "
        f"GLU: {row.get('GLU', 'NA')}, HR: {row.get('HR', 'NA')}, IBI_seq: {row.get('IBI_seq', 'NA')}, "
        f"MAP: {row.get('MAP', 'NA')}, HRV: {row.get('HRV', 'NA')}, PDB: {row.get('PDB', 'NA')}, NBM: {row.get('NBM', 'NA')}, "
        f"Blindness Duration (months): {row.get('Blindness Duration', 'NA')}, First MRI Experience: {row.get('First MRI Experience', 'NA')}, "
        f"Pre-procedure Briefing: {row.get('Pre-procedure Briefing', 'NA')}, Headphones Provided: {row.get('Headphones Provided', 'NA')}, "
        f"Cause of Blindness: {row.get('Cause of Blindness', 'NA')}, Mobility Independence: {row.get('Mobility Independence', 'NA')}, "
        f"Anxiety Level: {row.get('Anxiety Level', 'NA')}"
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
        temperature=0,
        max_tokens=10,
        n=1,
        stop=None,
    )
    label = response.choices[0].message.content.strip()
    return label

# Run prediction on entire dataset with progress bar
print("Starting stress state classification...")
labels = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        label = detect_stress_state(row)
    except Exception as e:
        print(f"⚠️ API error at row {idx}: {e}")
        label = "Unknown"
    labels.append(label)

df['stress_state'] = labels

# Save enriched dataset
output_file = "training/csv/checked_data.csv"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
df.to_csv(output_file, index=False)


