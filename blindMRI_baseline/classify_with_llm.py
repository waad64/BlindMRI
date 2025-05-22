import csv
import openai
import os
import time

openai.api_key = os.getenv("OPENAI_API_KEY")  

# Template prompt
def make_prompt(row):
    return f"""
You are a medical assistant AI. Classify the patient's stress state based on these features:

EDA: {row["EDA"]} µS  
TEMP: {row["TEMP"]} °C  
Patient Age: {row["Patient Age"]}  
Gender: {row["Patient Gender"]}  
SBP: {row["SBP"]} mmHg  
DBP: {row["DPB"]} mmHg  
GLU: {row["GLU"]} mmol/L  
HR: {row["HR"]} bpm  
IBI_seq (mean): {sum(map(float, row["IBI_seq"].split(','))) / len(row["IBI_seq"].split(',')):.4f}  
MAP: {row["MAP"]} mmHg  
HRV: {row["HRV"]}  

Follow these ranges:

- HR: Calm (60–85), Stress (85–130)  
- EDA: Calm (0.2–4.5), Stress (4.5–20)  
- TEMP: Calm (36.0–37.5), Stress (34.0–36.5)  
- DBP: Calm (60–80), Stress (80–100)  
- SBP: Calm (100–125), Stress (125–150)  
- IBI (mean): Calm (0.7–1.1), Stress (0.5–0.75)  
- MAP: Calm (70–90), Stress (90–110)  
- HRV: Calm (0.05–0.1), Stress (0.02–0.06)  
- GLU: Calm (4–6), Stress (6–9)

Classify this row as: "Likely Stress", "Likely Non-Stress", or "Borderline". Return just the label.
"""

# Main processing
def classify_with_gpt(input_path, output_path):
    with open(input_path, "r") as infile, open(output_path, "w", newline="") as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ["Stress_State"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(reader):
            prompt = make_prompt(row)
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                classification = response.choices[0].message.content.strip()
                row["Stress_State"] = classification
            except Exception as e:
                print(f"Error on row {i}: {e}")
                row["Stress_State"] = "Error"
            writer.writerow(row)
            time.sleep(0.5)  # to Avoid rate limits


classify_with_gpt("dataset_v1.csv", "classified_outputs.csv")
