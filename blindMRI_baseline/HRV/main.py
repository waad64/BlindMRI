import pandas as pd
import numpy as np


# RMSSD calculation function

def calc_rmssd_from_array(ibi_string):
    try:
        ibi = np.array([float(x) for x in ibi_string.strip().split(',')])
        diff = np.diff(ibi)
        rmssd = np.sqrt(np.mean(diff**2))
        return round(rmssd, 6)
    except Exception as e:
        print(f"❌ Error processing IBI row: {e}")
        return np.nan




input_path = 'outputs/raw_dataset_with_map.csv'   
df = pd.read_csv(input_path)


# Apply RMSSD computation

ibi_column = 'IBI_seq'
df['HRV'] = df[ibi_column].apply(calc_rmssd_from_array)



output_path = 'outputs/raw_dataset_with_rmssd.csv'
df.to_csv(output_path, index=False)

