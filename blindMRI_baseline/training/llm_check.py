#check coherence of the data 
import pandas as pd
import openai
import os
from datetime import datetime

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  

# Load the dataset
file_path = "training/csv/dataset.csv"  
df = pd.read_csv(file_path)

# Define the full medical rule prompt template
RULES_PROMPT = """
You are a medical data validation assistant. Evaluate whether a given patient record is physiologically and clinically coherent,
based on the patient's stress state and the following medically derived rules for normal and stressed ranges:

--- Feature Ranges ---
For each feature, the normal range is shown for CALM and STRESSED states.

1. Heart Rate (HR, bpm)
   - Calm: 60 - 85
   - Stress: 85 - 130

2. Electrodermal Activity (EDA, µS)
   - Calm: 0.2 - 4.5
   - Stress: 4.5 - 20

3. Skin Temperature (TEMP, °C)
   - Calm: 36.0 - 37.5
   - Stress: 34.0 - 36.5

4. Diastolic BP (DBP, mmHg)
   - Calm: 60 - 80
   - Stress: 80 - 100

5. Systolic BP (SBP, mmHg)
   - Calm: 100 - 125
   - Stress: 125 - 150

6. Inter-beat Interval (mean IBI, sec)
   - Calm: 0.7 - 1.1
   - Stress: 0.5 - 0.75

7. Mean Arterial Pressure (MAP, mmHg)
   - Calm: 70 - 90
   - Stress: 90 - 110

8. Heart Rate Variability (RMSSD, sec)
   - Calm: 0.05 - 0.1
   - Stress: 0.02 - 0.06

9. Glucose (GLU, mmol/L)
   - Calm: 4 - 6
   - Stress: 6 - 9

Check each row for coherence:
- Does the combination of features match the state of borderline , likely stress or non stressed ?
- Are any values contradictory (e.g. high HRV with low IBI in stress)?

Respond with a well structured paragraph that explain the semantic status and the coherence of the data.
If the data is coherent, respond with "OK". If there are inconsistencies, provide a corrected version of the data adding a new column to the row with the name "Stress_State" and the value "Borderline" or "Likely Non-Stress" or "Likely Stress" depending on the state of the data.
"""
def check_record_coherence(row):
    # Extract features for the prompt
    data_str = (
        f"HR: {row['HR']} bpm, "
        f"EDA: {row['EDA']} µS, "
        f"TEMP: {row['TEMP']} °C, "
        f"DBP: {row['DBP']} mmHg, "
        f"SBP: {row['SBP']} mmHg, "
        f"IBI: {row['IBI']} sec, "
        f"MAP: {row['MAP']} mmHg, "
        f"RMSSD: {row['RMSSD']} sec, "
        f"GLU: {row['GLU']} mmol/L"
    )
    
    prompt = (
        RULES_PROMPT + "\n\n"
        f"Patient record data:\n{data_str}\n\n"
        "Please evaluate and respond as instructed."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300,
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        reply = f"Error: {e}"

    return reply

# Apply the function row-wise and collect responses
df['Validation_Response'] = df.apply(check_record_coherence, axis=1)

# Now parse the response to add the Stress_State column if needed
def parse_stress_state(response):
    if "OK" in response:
        return "OK"
    elif "Borderline" in response:
        return "Borderline"
    elif "Likely Non-Stress" in response:
        return "Likely Non-Stress"
    elif "Likely Stress" in response:
        return "Likely Stress"
    else:
        return "Unknown"

df['Stress_State'] = df['Validation_Response'].apply(parse_stress_state)

# Reorganize columns to put Stress_State next to data (optional)
cols = list(df.columns)
data_cols = ['HR', 'EDA', 'TEMP', 'DBP', 'SBP', 'IBI_seq', 'MAP', 'RMSSD', 'GLU']
new_order = data_cols + ['Stress_State'] + [c for c in cols if c not in data_cols + ['Stress_State']]
df = df[new_order]


df.to_csv("training/csv/validated_dataset.csv", index=False)

print("Validation complete, dataset updated with stress state and response explanations.")

