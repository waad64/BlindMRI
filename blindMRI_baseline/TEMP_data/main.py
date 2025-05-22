import pandas as pd
import numpy as np
import os
import glob

# Dynamically fetch the first 8 HR files
hr_files = sorted(glob.glob('TEMP_data/csv/S*TEMP.csv'))[:8]
samples_per_file = 111

all_hr_samples = []

for file in hr_files:
    # Load file (no headers)
    with open(file, 'r') as f:
        lines = f.readlines()
    hr_values = [float(line.strip()) for line in lines]

    np.random.seed(42)  
    np.random.shuffle(hr_values)

    sampled = hr_values[:samples_per_file]

    all_hr_samples.extend(sampled)

df_hr = pd.DataFrame({'temp': all_hr_samples})

os.makedirs("outputs", exist_ok=True)
df_hr.to_csv("outputs/TEMP_data.csv", index=False)
