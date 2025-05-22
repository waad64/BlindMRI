import openai
import pandas as pd
import json
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Build LLM prompt for generating First MRI Experience
def build_prompt(num_samples=10, label="first_mri"):
    label_mapping = {
        "first_mri": "1 (Yes, this is the first MRI session)",
        "not_first_mri": "0 (No, not the first MRI session)"
    }

    return f"""
You are an expert in medical data generation specialized in MRI patient experience.

Generate exactly {num_samples} integer values representing the **First MRI Experience** of patients.

- first_mri: 1 (Yes, this is the first MRI session)
- not_first_mri: 0 (No, not the first MRI session)

Return ONLY a valid JSON array of integers using the codes 1 or 0, with no extra text or explanation.

Example: [1, 0, 1, 1, 0]
"""

# Configuration
n_samples = 7974

# Define proportions for first MRI experience
proportions = {
    'first_mri': 0.3,      # 30% are first-time MRI patients
    'not_first_mri': 0.7,  # 70% have had previous MRI sessions
}

# Calculate counts
counts = {k: int(v * n_samples) for k, v in proportions.items()}
counts['not_first_mri'] += n_samples - sum(counts.values())  # fix rounding

all_values = []

for label, count in counts.items():
    batch_size = 50
    batches = count // batch_size
    remainder = count % batch_size

    print(f"Generating {count} First MRI Experience values for '{label}' in {batches} batches + remainder {remainder}")

    for _ in range(batches):
        prompt = build_prompt(num_samples=batch_size, label=label)
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            temperature=0.3,
            messages=[
                {"role": "system", "content": "You are a medical data generator."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        content = response.choices[0].message["content"]

        try:
            values = json.loads(content)
            if not isinstance(values, list):
                raise ValueError("Output is not a list")
        except Exception as e:
            print(f"⚠️ JSON parsing error: {e}\nOutput was:\n{content}")
            continue

        code = 1 if label == "first_mri" else 0
        cleaned = [code if v != code else v for v in values]
        all_values.extend(cleaned)

    if remainder > 0:
        prompt = build_prompt(num_samples=remainder, label=label)
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            temperature=0.3,
            messages=[
                {"role": "system", "content": "You are a medical data generator."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        content = response.choices[0].message["content"]

        try:
            values = json.loads(content)
            if not isinstance(values, list):
                raise ValueError("Output is not a list")
        except Exception as e:
            print(f"⚠️ JSON parsing error: {e}\nOutput was:\n{content}")
            continue

        code = 1 if label == "first_mri" else 0
        cleaned = [code if v != code else v for v in values]
        all_values.extend(cleaned)

# Shuffle and save dataframe
df_first_mri = pd.DataFrame({'First_MRI_Experience': all_values})
df_first_mri = df_first_mri.sample(frac=1).reset_index(drop=True)

output_path = "First_MRI_Experience/csv/first_mri_experience_generated_by_llm.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_first_mri.to_csv(output_path, index=False)

print(f"✅ First MRI Experience dataset generated and saved to: {output_path}")
