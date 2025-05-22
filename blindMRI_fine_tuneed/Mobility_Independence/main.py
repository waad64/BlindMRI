import openai
import pandas as pd
import json
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Build LLM prompt for generating Mobility Independence
def build_prompt(num_samples=10, label="independent"):
    description = {
        "dependent": "1 (fully assisted - dependent)",
        "independent": "0 (independent - no assistance needed)"
    }

    return f"""
You are an expert in patient mobility assessment. Generate exactly {num_samples} integer values representing the **Mobility Independence** status of patients.

- Dependent: 1 (fully assisted - dependent)
- Independent: 0 (independent - no assistance needed)

Return ONLY a valid JSON array of integers using codes 1 or 0. No labels, comments, or extra text.

Example: [1, 0, 1, 1, 0]
"""

# Configuration
n_samples = 7974

# Define proportions
proportions = {
    'dependent': 0.4,    # 40% dependent patients
    'independent': 0.6   # 60% independent patients
}

# Calculate counts
counts = {k: int(v * n_samples) for k, v in proportions.items()}
counts['independent'] += n_samples - sum(counts.values())  # fix rounding

all_values = []

for label, count in counts.items():
    batch_size = 50
    batches = count // batch_size
    remainder = count % batch_size

    print(f"Generating {count} mobility independence values for '{label}' in {batches} batches + remainder {remainder}")

    for _ in range(batches):
        prompt = build_prompt(num_samples=batch_size, label=label)
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            temperature=0.3,
            messages=[
                {"role": "system", "content": "You are a medical data generator specialized in patient mobility."},
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

        code = 1 if label == "dependent" else 0
        cleaned = [code if v != code else v for v in values]
        all_values.extend(cleaned)

    if remainder > 0:
        prompt = build_prompt(num_samples=remainder, label=label)
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            temperature=0.3,
            messages=[
                {"role": "system", "content": "You are a medical data generator specialized in patient mobility."},
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

        code = 1 if label == "dependent" else 0
        cleaned = [code if v != code else v for v in values]
        all_values.extend(cleaned)

# Shuffle and save
df_mobility = pd.DataFrame({'Mobility_Independence': all_values})
df_mobility = df_mobility.sample(frac=1).reset_index(drop=True)

output_path = "Mobility_Independence/csv/mobility_independence_generated_by_llm.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_mobility.to_csv(output_path, index=False)


