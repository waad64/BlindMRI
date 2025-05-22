import openai
import pandas as pd
import json
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# Build LLM prompt for generating Anxiety Level
def build_prompt(num_samples=10, anxiety_class="unanxious"):
    examples = {
        "unanxious": "0 (relaxed, no visible signs of distress or worry)",
        "anxious": "1 (elevated pre-scan concern, signs of fear or tension)"
    }
    value = examples.get(anxiety_class, "0")

    return f"""
You are simulating pre-MRI psychological assessment data for blind patients.

Generate exactly {num_samples} binary values representing **baseline self-reported anxiety levels** before the MRI procedure.

- unanxious: 0 (relaxed, no visible signs of distress or worry)
- anxious: 1 (elevated pre-scan concern, signs of fear or tension)

Return ONLY a valid JSON array of integers, 0 or 1. No extra text, no comments.

Example: [0, 0, 1]
"""

# Configuration
n_samples = 7974

proportions = {
    'unanxious': 0.5,
    'anxious': 0.5,
}

counts = {k: int(v * n_samples) for k, v in proportions.items()}
counts['unanxious'] += n_samples - sum(counts.values())  

all_anxiety_labels = []

for anxiety_class, count in counts.items():
    batch_size = 50
    batches = count // batch_size
    remainder = count % batch_size

    print(f"Generating {count} anxiety values for '{anxiety_class}' in {batches} batches + remainder {remainder}")

    for _ in range(batches):
        prompt = build_prompt(num_samples=batch_size, anxiety_class=anxiety_class)
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            temperature=0.5,
            messages=[
                {"role": "system", "content": "You are a psychological data simulator for blind MRI patients."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        content = response.choices[0].message["content"]

        try:
            anxiety_values = json.loads(content)
            if not isinstance(anxiety_values, list):
                raise ValueError("Output is not a list")
            anxiety_values = [int(v) if v in [0, 1] else 0 for v in anxiety_values]
        except Exception as e:
            print(f"⚠️ JSON parsing error: {e}\nOutput was:\n{content}")
            continue

        all_anxiety_labels.extend(anxiety_values)

    if remainder > 0:
        prompt = build_prompt(num_samples=remainder, anxiety_class=anxiety_class)
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            temperature=0.5,
            messages=[
                {"role": "system", "content": "You are a psychological data simulator for blind MRI patients."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        content = response.choices[0].message["content"]

        try:
            anxiety_values = json.loads(content)
            if not isinstance(anxiety_values, list):
                raise ValueError("Output is not a list")
            anxiety_values = [int(v) if v in [0, 1] else 0 for v in anxiety_values]
        except Exception as e:
            print(f"⚠️ JSON parsing error: {e}\nOutput was:\n{content}")
            continue

        all_anxiety_labels.extend(anxiety_values)

# Shuffle and save dataframe
df_anxiety = pd.DataFrame({'Anxiety_Level': all_anxiety_labels})
df_anxiety = df_anxiety.sample(frac=1).reset_index(drop=True)

output_path = "Anxiety_level/csv/anxiety_levels_generated_by_llm.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_anxiety.to_csv(output_path, index=False)


