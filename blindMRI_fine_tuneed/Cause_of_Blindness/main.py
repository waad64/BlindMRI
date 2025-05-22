import openai
import pandas as pd
import json
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Build LLM prompt for generating Causes of Blindness
def build_prompt(num_samples=10, cause_label="congenital"):
    cause_mapping = {
        "congenital": "1 (blind from birth)",
        "diabetic_retinopathy": "2 (due to diabetes-related retinal damage)",
        "trauma": "3 (due to injury or accident)"
    }

    return f"""
You are an expert in ophthalmology and medical data generation. Generate exactly {num_samples} integer values representing the **Cause of Blindness** in patients.

- congenital: 1 (blind from birth)
- diabetic_retinopathy: 2 (due to diabetes-related retinal damage)
- trauma: 3 (due to injury or accident)

Return ONLY a valid JSON array of integers using the code (1, 2, 3) — no explanations, labels, or extra text.

Example: [1, 1, 2, 3, 1]
"""

# Configuration
n_samples = 7974

proportions = {
    'congenital': 0.4,
    'diabetic_retinopathy': 0.35,
    'trauma': 0.25,
}

counts = {k: int(v * n_samples) for k, v in proportions.items()}
counts['congenital'] += n_samples - sum(counts.values())  # rounding fix

all_causes = []

for cause_label, count in counts.items():
    batch_size = 50
    batches = count // batch_size
    remainder = count % batch_size

    print(f"Generating {count} blindness cause values for '{cause_label}' in {batches} batches + remainder {remainder}")

    for _ in range(batches):
        prompt = build_prompt(num_samples=batch_size, cause_label=cause_label)
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
            cause_values = json.loads(content)
            if not isinstance(cause_values, list):
                raise ValueError("Output is not a list")
        except Exception as e:
            print(f"⚠️ JSON parsing error: {e}\nOutput was:\n{content}")
            continue

        # Clip and clean values
        code = {"congenital": 1, "diabetic_retinopathy": 2, "trauma": 3}[cause_label]
        cause_values = [code if v != code else v for v in cause_values]
        all_causes.extend(cause_values)

    if remainder > 0:
        prompt = build_prompt(num_samples=remainder, cause_label=cause_label)
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
            cause_values = json.loads(content)
            if not isinstance(cause_values, list):
                raise ValueError("Output is not a list")
        except Exception as e:
            print(f"⚠️ JSON parsing error: {e}\nOutput was:\n{content}")
            continue

        code = {"congenital": 1, "diabetic_retinopathy": 2, "trauma": 3}[cause_label]
        cause_values = [code if v != code else v for v in cause_values]
        all_causes.extend(cause_values)

# Shuffle and save dataframe
df_causes = pd.DataFrame({'Cause_Blindness': all_causes})
df_causes = df_causes.sample(frac=1).reset_index(drop=True)

output_path = "Blindness Cause/csv/cause_of_blindness_generated_by_llm.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_causes.to_csv(output_path, index=False)


