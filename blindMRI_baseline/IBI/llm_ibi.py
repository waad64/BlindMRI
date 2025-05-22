import os
from dotenv import load_dotenv
import openai
import pandas as pd
import ast  

# Load environment variables (your OpenAI key)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Read the CSV where IBI sequences are stored as strings representing lists
df = pd.read_csv('IBI/ibi_data.csv')


df['IBI_Seq'] = df['IBI_Seq'].apply(ast.literal_eval)

prompt = f"""
You are provided with a list of Inter-Beat Interval (IBI) sequences, each representing 100 consecutive IBI values (in seconds) from individual patients.

Here are the current IBI sequences for {len(df)} patients:

{df['IBI_Seq'].tolist()}

Your task is to generate additional synthetic IBI sequences to extend this dataset until it contains a total of 888 patients.

Guidelines:
1. Each synthetic IBI sequence should contain exactly 100 positive real values (seconds).
2. Values must be physiologically plausible, reflecting both calm (normal) and stressed (abnormal) states.
3. Include realistic variability and occasional outliers.
4. Output only the new synthetic sequences as a list of lists, do not include the original data.
"""

def generate_synthetic_ibi_sequences(prompt, max_tokens=1500, temperature=0.7, model="gpt-4"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
        stop=None,
    )
    text = response.choices[0].message.content.strip()
    return text

synthetic_data_text = generate_synthetic_ibi_sequences(prompt)

try:
    synthetic_ibi_sequences = eval(synthetic_data_text)
    if not (isinstance(synthetic_ibi_sequences, list) and all(isinstance(seq, list) for seq in synthetic_ibi_sequences)):
        raise ValueError("Output is not a list of lists")
except Exception as e:
    print(f"⚠️ Failed to parse GPT output: {e}")
    synthetic_ibi_sequences = []

print(f"Generated {len(synthetic_ibi_sequences)} synthetic IBI sequences")

# Combine original and synthetic sequences
combined_sequences = df['IBI_Seq'].tolist() + synthetic_ibi_sequences

# Save combined sequences to CSV (convert lists to strings to store properly)
output_path = 'outputs/ibi_sequences.csv'
pd.DataFrame({'IBI_Seq': [str(seq) for seq in combined_sequences]}).to_csv(output_path, index=False)
