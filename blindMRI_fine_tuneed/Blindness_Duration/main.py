import openai
import pandas as pd
import json
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Build LLM prompt for generating Blindness Duration
def build_prompt(num_samples=10, stress_class="likely_non_stress"):
    blindness_ranges = {
        "likely_non_stress": "36 to 240 months (long-term adaptation, lower stress sensitivity)",
        "borderline": "12 to 35 months (partial adaptation)",
        "likely_stress": "0 to 11 months (recent blindness, high sensitivity)"
    }
    blindness_range = blindness_ranges.get(stress_class, "36 to 240 months")

    return f"""
You are an expert in neuro-ophthalmological patient profiling during MRI. Generate exactly {num_samples} numeric values representing the **Blindness Duration in Months** — the number of months a patient has been blind — for patients under the {stress_class.replace('_', ' ')} stress condition during an MRI scan.

- likely_non_stress: 36 to 240 months (long-term adaptation, lower stress sensitivity)
- borderline: 12 to 35 months (partial adaptation)
- likely_stress: 0 to 11 months (recent blindness, high sensitivity)

Return ONLY a valid JSON array of floats. No explanations, comments, labels, or extra text.

Example: [6.0, 2.3, 10.0]
"""

# Configuration
n_samples = 7974 
proportions = {
    'likely_non_stress': 0.4,
    'borderline': 0.3,
    'likely_stress': 0.3,
}

counts = {k: int(v * n_samples) for k, v in proportions.items()}
counts['likely_non_stress'] += n_samples - sum(counts.values())  # rounding fix

all_durations = []

for stress_class, count in counts.items():
    batch_size = 50
    batches = count // batch_size
    remainder = count % batch_size

    print(f"Generating {count} durations for '{stress_class}' in {batches} batches + remainder {remainder}")

    for _ in range(batches):
        prompt = build_prompt(num_samples=batch_size, stress_class=stress_class)
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            temperature=0.7,
            messages=[
                {"role": "system", "content": "You are a medical data generator for MRI patient simulations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500
        )
        content = response.choices[0].message["content"]

        try:
            values = json.loads(content)
            if not isinstance(values, list):
                raise ValueError("Output is not a list")
        except Exception as e:
            print(f"⚠️ JSON parsing error: {e}\nOutput was:\n{content}")
            continue

        low, high = {
            'likely_non_stress': (36, 240),
            'borderline': (12, 35),
            'likely_stress': (0, 11),
        }[stress_class]

        values = [round(max(low, min(high, float(v))), 1) for v in values]
        all_durations.extend(values)

    # handle remainder
    if remainder > 0:
        prompt = build_prompt(num_samples=remainder, stress_class=stress_class)
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            temperature=0.7,
            messages=[
                {"role": "system", "content": "You are a medical data generator for MRI patient simulations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500
        )
        content = response.choices[0].message["content"]

        try:
            values = json.loads(content)
            if not isinstance(values, list):
                raise ValueError("Output is not a list")
        except Exception as e:
            print(f"⚠️ JSON parsing error: {e}\nOutput was:\n{content}")
            continue

        low, high = {
            'likely_non_stress': (36, 240),
            'borderline': (12, 35),
            'likely_stress': (0, 11),
        }[stress_class]

        values = [round(max(low, min(high, float(v))), 1) for v in values]
        all_durations.extend(values)

# Shuffle and save dataframe
df_generated = pd.DataFrame({'Blindness_Duration': all_durations})
df_generated = df_generated.sample(frac=1).reset_index(drop=True)

output_path = "Blindness_Duration/csv/blindness_duration_generated_by_llm.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_generated.to_csv(output_path, index=False)
