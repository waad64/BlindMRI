import openai
import pandas as pd
import json
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Build LLM prompt for generating Pre-procedure Briefing status
def build_prompt(num_samples=10, label="yes"):
    description = {
        "yes": "1 (patient received calming verbal explanations before MRI)",
        "no": "0 (patient did NOT receive calming verbal explanations)"
    }

    return f"""
You are an expert in patient psychological preparation for MRI procedures. Generate exactly {num_samples} integer values representing the **Pre-procedure Briefing** status.

- Yes: 1 (patient received calming verbal explanations before MRI)
- No: 0 (patient did NOT receive calming verbal explanations)

Return ONLY a valid JSON array of integers using codes 1 or 0. No labels, comments, or extra text.

Example: [1, 0, 1, 1, 0]
"""

# Configuration
n_samples = 7974

# Define proportions (tu peux ajuster à ta guise)
proportions = {
    'yes': 0.55,   # 55% patients received briefing
    'no': 0.45     # 45% did not
}

# Calculate counts
counts = {k: int(v * n_samples) for k, v in proportions.items()}
counts['yes'] += n_samples - sum(counts.values())  # fix rounding

all_values = []

for label, count in counts.items():
    batch_size = 50
    batches = count // batch_size
    remainder = count % batch_size

    print(f"Generating {count} pre-procedure briefing values for '{label}' in {batches} batches + remainder {remainder}")

    for _ in range(batches):
        prompt = build_prompt(num_samples=batch_size, label=label)
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            temperature=0.3,
            messages=[
                {"role": "system", "content": "You are a medical data generator specialized in patient psychological preparation."},
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

        code = 1 if label == "yes" else 0
        cleaned = [code if v != code else v for v in values]
        all_values.extend(cleaned)

    if remainder > 0:
        prompt = build_prompt(num_samples=remainder, label=label)
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            temperature=0.3,
            messages=[
                {"role": "system", "content": "You are a medical data generator specialized in patient psychological preparation."},
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

        code = 1 if label == "yes" else 0
        cleaned = [code if v != code else v for v in values]
        all_values.extend(cleaned)

# Shuffle and save
df_briefing = pd.DataFrame({'Preprocedure_Briefing': all_values})
df_briefing = df_briefing.sample(frac=1).reset_index(drop=True)

output_path = "Pre_procedure_Briefing/csv/preprocedure_briefing_generated_by_llm.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_briefing.to_csv(output_path, index=False)

print(f"✅ Pre-procedure Briefing dataset (LLM version) generated and saved to: {output_path}")
