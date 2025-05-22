import openai
import pandas as pd
import json
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def build_prompt(num_samples=10, stress_class="likely_non_stress"):
    pdb_ranges = {
        "likely_non_stress": "50 to 65 decibels (calm environment)",
        "borderline": "66 to 80 decibels (moderate noise)",
        "likely_stress": "81 to 100 decibels (loud and stressful environment)"
    }
    pdb_range = pdb_ranges.get(stress_class, "50 to 65 decibels")

    return f"""
You are an expert in MRI acoustic environments. Generate exactly {num_samples} numeric values representing the Peak Decibel Level (PDB) of MRI machine noise
reflecting the {stress_class.replace('_', ' ')} condition.

- likely_non_stress: PDB values between 50 and 65 decibels (calm environment).
- borderline: PDB values between 66 and 80 decibels (moderate noise).
- likely_stress: PDB values between 81 and 100 decibels (loud and stressful environment).

Return ONLY a JSON array of floats without any extra text, commentary, or labels.

Example: [52.34, 60.12, 64.5]
"""

n_samples = 6776

proportions = {
    'likely_non_stress': 0.4,
    'borderline': 0.3,
    'likely_stress': 0.3,
}

counts = {k: int(v * n_samples) for k, v in proportions.items()}
counts['likely_non_stress'] += n_samples - sum(counts.values())  # rounding fix

all_pdbs = []

for stress_class, count in counts.items():
    batch_size = 50
    batches = count // batch_size
    remainder = count % batch_size

    print(f"Generating {count} PDB values for '{stress_class}' in {batches} batches + remainder {remainder}")

    for _ in range(batches):
        prompt = build_prompt(num_samples=batch_size, stress_class=stress_class)
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            temperature=0.7,
            messages=[
                {"role": "system", "content": "You are an MRI acoustic data generator."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500
        )
        content = response.choices[0].message["content"]

        try:
            pdb_values = json.loads(content)
            if not isinstance(pdb_values, list):
                raise ValueError("Output is not a list")
        except Exception as e:
            print(f"⚠️ JSON parsing error: {e}\nOutput was:\n{content}")
            continue

        low, high = {
            'likely_non_stress': (50, 65),
            'borderline': (66, 80),
            'likely_stress': (81, 100),
        }[stress_class]

        pdb_values = [round(max(low, min(high, float(v))), 2) for v in pdb_values]
        all_pdbs.extend(pdb_values)

    if remainder > 0:
        prompt = build_prompt(num_samples=remainder, stress_class=stress_class)
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            temperature=0.7,
            messages=[
                {"role": "system", "content": "You are an MRI acoustic data generator."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500
        )
        content = response.choices[0].message["content"]

        try:
            pdb_values = json.loads(content)
            if not isinstance(pdb_values, list):
                raise ValueError("Output is not a list")
        except Exception as e:
            print(f"⚠️ JSON parsing error: {e}\nOutput was:\n{content}")
            continue

        low, high = {
            'likely_non_stress': (50, 65),
            'borderline': (66, 80),
            'likely_stress': (81, 100),
        }[stress_class]

        pdb_values = [round(max(low, min(high, float(v))), 2) for v in pdb_values]
        all_pdbs.extend(pdb_values)

# Shuffle and save dataframe
df_generated = pd.DataFrame({'PDB': all_pdbs})
df_generated = df_generated.sample(frac=1).reset_index(drop=True)

output_path = "Peak_dB_Level/csv/pdb_generated_by_llm.csv"
df_generated.to_csv(output_path, index=False)



