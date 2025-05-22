import csv
import openai
import os
import time

openai.api_key = os.getenv("OPENAI_API_KEY")

def safe_mean(ibi_seq_str):
    try:
        values = list(map(float, ibi_seq_str.split(',')))
        return sum(values) / len(values) if values else 0
    except Exception:
        return 0

def make_prompt(row):
    ibi_mean = safe_mean(row.get("IBI_seq", ""))
    return f"""
You are a medical assistant AI. Classify the patient's stress state based on these features:

EDA: {row.get("EDA", "N/A")} µS  
TEMP: {row.get("TEMP", "N/A")} °C  
Patient Age: {row.get("Patient Age", "N/A")}  
Gender: {row.get("Patient Gender", "N/A")}  
SBP: {row.get("SBP", "N/A")} mmHg  
DBP: {row.get("DPB", "N/A")} mmHg  
GLU: {row.get("GLU", "N/A")} mmol/L  
HR: {row.get("HR", "N/A")} bpm  
IBI_seq (mean): {ibi_mean:.4f}  
MAP: {row.get("MAP", "N/A")} mmHg  
HRV: {row.get("HRV", "N/A")}  
PDB (Peak Decibel Level): {row.get("PDB", "N/A")} dB  
NBM (Noise Bursts per Minute): {row.get("NBM", "N/A")}

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
- PDB: Calm (50–65 dB), Borderline (66–80 dB), Stress (>80 dB)  
- NBM: Calm (0–2), Borderline (3–6), Stress (7+)

Classify this row as one of: "Likely Stress", "Likely Non-Stress", or "Borderline". Return just the label.
"""

def classify_with_gpt(input_path, output_path):
    with open(input_path, "r", newline='') as infile, open(output_path, "w", newline='') as outfile:
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
                print(f"Row {i} classified as: {classification}")
                row["Stress_State"] = classification
            except Exception as e:
                print(f"⚠️ Error on row {i}: {e}")
                row["Stress_State"] = "Error"
            writer.writerow(row)
            time.sleep(0.5)  # avoid rate limits

if __name__ == "__main__":
    classify_with_gpt("dataset_v2.csv", "classified_outputs.csv")
