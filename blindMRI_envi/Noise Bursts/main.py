import openai
import pandas as pd
import json
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# Build LLM prompt for generating NBM
def build_prompt(num_samples=10, stress_class="likely_non_stress"):
    nbm_ranges = {
        "likely_non_stress": "0 to 2 bursts per minute (calm, minimal acoustic events)",
        "borderline": "3 to 6 bursts per minute (moderate interruptions)",
        "likely_stress": "7 to 12 bursts per minute (frequent, jarring acoustic bursts)"
    }
    nbm_range = nbm_ranges.get(stress_class, "0 to 2 bursts per minute")

    return f"""
You are an expert in MRI acoustic environments. Generate exactly {num_samples} numeric values representing the **Noise Bursts per Minute (NBM)** — the frequency of sudden, loud acoustic events in an MRI suite — under the {stress_class.replace('_', ' ')} condition.

- likely_non_stress: 0 to 2 bursts per minute (calm, minimal acoustic events)
- borderline: 3 to 6 bursts per minute (moderate interruptions)
- likely_stress: 7 to 12 bursts per minute (frequent, jarring acoustic bursts)

Return ONLY a valid JSON array of floats. No explanations, comments, labels, or extra text.

Example: [1.0, 0.3, 2.0]
"""

# Configuration
n_samples = 6776

proportions = {
    'likely_non_stress': 0.4,
    'borderline': 0.3,
    'likely_stress': 0.3,
}

counts = {k: int(v * n_samples) for k, v in proportions.items()}
counts['likely_non_stress'] += n_samples - sum(counts.values())  # rounding fix

all_nbms = []

for stress_class, count in counts.items():
    batch_size = 50
    batches = count // batch_size
    remainder = count % batch_size

    print(f"Generating {count} NBM values for '{stress_class}' in {batches} batches + remainder {remainder}")

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
            nbm_values = json.loads(content)
            if not isinstance(nbm_values, list):
                raise ValueError("Output is not a list")
        except Exception as e:
            print(f"⚠️ JSON parsing error: {e}\nOutput was:\n{content}")
            continue

        low, high = {
            'likely_non_stress': (0, 2),
            'borderline': (3, 6),
            'likely_stress': (7, 12),
        }[stress_class]

        nbm_values = [round(max(low, min(high, float(v))), 2) for v in nbm_values]
        all_nbms.extend(nbm_values)

    # handle remainder
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
            nbm_values = json.loads(content)
            if not isinstance(nbm_values, list):
                raise ValueError("Output is not a list")
        except Exception as e:
            print(f"⚠️ JSON parsing error: {e}\nOutput was:\n{content}")
            continue

        low, high = {
            'likely_non_stress': (0, 2),
            'borderline': (3, 6),
            'likely_stress': (7, 12),
        }[stress_class]

        nbm_values = [round(max(low, min(high, float(v))), 2) for v in nbm_values]
        all_nbms.extend(nbm_values)

# Shuffle and save dataframe
df_generated = pd.DataFrame({'NBM': all_nbms})
df_generated = df_generated.sample(frac=1).reset_index(drop=True)

output_path = "Noise Bursts/csv/nbm_generated_by_llm.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_generated.to_csv(output_path, index=False)


