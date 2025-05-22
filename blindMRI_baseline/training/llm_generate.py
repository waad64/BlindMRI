import openai
import pandas as pd
import json
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load base dataset for context
df = pd.read_csv("csv/dataset.csv")

# Extract a few examples for few-shot context
few_shot = df.sample(5, random_state=42).to_dict(orient="records")

# Format few-shot prompt
few_shot_prompt = "\n".join([
    f"Sample {i+1}:\n{json.dumps(sample, indent=2)}"
    for i, sample in enumerate(few_shot)
])

# GPT prompt
def build_prompt(num_samples=10):
    return f"""
You are an expert biomedical AI system generating synthetic but realistic physiological data.
Each entry represents a patient's physiological readings under stress conditions (low, medium, high).
Here are a few real examples from our dataset:

{few_shot_prompt}

Now, generate {num_samples} new samples in the exact same format and field order:
EDA, TEMP, Patient Age, Patient Gender, SBP, DBP, GLU, HR, IBI_seq, MAP, HRV

Constraints:
- MAP must be calculated as (2 * DBP + SBP)/3
- IBI_seq should be a comma-separated string of 100 floats between 0.5 and 1.2
- All values must be within realistic physiological bounds (as in examples)
- Return a JSON array. No explanations. No commentary.
"""


all_generated = []
batch_size = 50
num_batches = 1000 // batch_size

for i in range(num_batches):
    print(f"Batch {i+1}/{num_batches}")
    prompt = build_prompt(num_samples=batch_size)
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        temperature=0.7,
        messages=[
            {"role": "system", "content": "You are a biomedical data simulator."},
            {"role": "user", "content": prompt}
        ]
    )
    
    content = response.choices[0].message["content"]
    
    try:
        batch_data = json.loads(content)
        all_generated.extend(batch_data)
    except json.JSONDecodeError as e:
        print("⚠️ Error parsing response:", e)
        continue

# Convert generated samples to DataFrame
df_aug = pd.DataFrame(all_generated)

# Concatenate original and synthetic data
df_combined = pd.concat([df, df_aug], ignore_index=True)

# Save the full combined dataset
output_path = "csv/llm_generate_dataset.csv"
df_combined.to_csv(output_path, index=False)

