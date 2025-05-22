import pandas as pd
import numpy as np
import os
import glob


hr_files = sorted(glob.glob('heart_data/csv/S*HR.csv'))[:8]
samples_per_file = 111

all_hr_samples = []

for file in hr_files:
    with open(file, 'r') as f:
        lines = f.readlines()

    hr_values = [float(line.strip()) for line in lines]

    # Shuffle the HR values
    np.random.seed(42)  
    np.random.shuffle(hr_values)
    sampled = hr_values[:samples_per_file]

    all_hr_samples.extend(sampled)


# Convert to DataFrame
df_hr = pd.DataFrame({'HR': all_hr_samples})

os.makedirs("outputs", exist_ok=True)
df_hr.to_csv("outputs/heart_rate_data.csv", index=False)
