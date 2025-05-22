import openai
import pandas as pd
import json
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load base dataset for context (make sure this CSV includes the new columns too)
df = pd.read_csv("training/PDB_training/csv/merged_dataset.csv")

# Extract a few examples for few-shot context (with all required columns)
few_shot = df.sample(5, random_state=42).to_dict(orient="records")

# Format few-shot prompt examples
few_shot_prompt = "\n".join([
    f"Sample {i+1}:\n{json.dumps(sample, indent=2)}"
    for i, sample in enumerate(few_shot)
])

def build_prompt(num_samples=10):
    return f"""
You are a biomedical data scientist generating synthetic but realistic physiological and psychological data reflecting MRI stress conditions.

Each sample has these fields in this exact order:
EDA, TEMP, Patient Age, Patient Gender, SBP, DBP, GLU, HR, IBI_seq, MAP, HRV, PDB, 
Anxiety_Level (0=unanxious, 1=anxious), 
Blindness_Duration (months, integer), 
Cause_Blindness (1=congenital, 2=diabetic retinopathy, 3=trauma), 
First_MRI_Experience (0=no, 1=yes), 
Headphones_Provided (0=no, 1=yes), 
Mobility_Independence (0=independent, 1=dependent)

Constraints:
- MAP must be calculated as (2 * DBP + SBP)/3
- IBI_seq is a comma-separated string of 100 floats between 0.5 and 1.2
- PDB (Peak Decibel Level) should be between 50 and 100 dB, realistic values based on typical MRI acoustic levels
- Anxiety_Level, First_MRI_Experience, Headphones_Provided, and Mobility_Independence are binary integers (0 or 1)
- Blindness_Duration is an integer representing months (realistic range: 0-480)
- Cause_Blindness is categorical with values 1, 2, or 3 as defined above
- All values must be within physiological and clinical norms consistent with stress states
- Return a JSON array of {num_samples} samples in the exact format and order specified
- No commentary, only raw JSON output

Here are a few real examples:

{few_shot_prompt}
"""

# Generate synthetic data in batches
all_generated = []
batch_size = 50
num_batches = 20  # for 1000 total samples

for i in range(num_batches):
    print(f"Generating batch {i+1}/{num_batches} ...")
    prompt = build_prompt(num_samples=batch_size)
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        temperature=0.7,
        messages=[
            {"role": "system", "content": "You are a biomedical data scientist."},
            {"role": "user", "content": prompt}
        ]
    )
    
    content = response.choices[0].message["content"]
    
    try:
        batch_data = json.loads(content)
        all_generated.extend(batch_data)
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON decode error in batch {i+1}: {e}")
        print("Response content was:", content[:500])
        continue

# Convert to DataFrame
df_aug = pd.DataFrame(all_generated)

# Combine with original data
df_combined = pd.concat([df, df_aug], ignore_index=True)

# Save output
output_path = "training/csv/llm_generate_dataset.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_combined.to_csv(output_path, index=False)

