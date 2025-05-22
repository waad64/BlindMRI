import pandas as pd
import numpy as np
import os
import glob

ibi_files = sorted(glob.glob('IBI/csv/S*.csv'))[:15]  # 15 patients
ibi_values_list = []

for file in ibi_files:
    with open(file, 'r') as f:
        lines = f.readlines()

    lines = lines[1:]  # skip header

    # Extract IBI (second column) from first 100 rows
    ibi_values = [float(line.strip().split(',')[1]) for line in lines[:100] if ',' in line]

    # Store the full ibi sequence (as a list or string)
    ibi_values_list.append(ibi_values)

# Convert to DataFrame - store as string to keep full sequence per patient
df_summary = pd.DataFrame({'IBI_seq': [','.join(map(str, ibi)) for ibi in ibi_values_list]})

os.makedirs("outputs", exist_ok=True)
df_summary.to_csv("IBI/ibi_data.csv", index=False)
