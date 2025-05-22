#test the wasserian distance between the original and synthetic dataset
from scipy.stats import wasserstein_distance
import pandas as pd
import numpy as np

def clean_ibi_sequence(series):
    """Convert string sequences of IBI values to single numeric values (mean)"""
    cleaned = []
    for val in series:
        if isinstance(val, str) and ',' in val:
            # Convert comma-separated string to array of floats and take mean
            try:
                nums = [float(x) for x in val.split(',')]
                cleaned.append(np.mean(nums))
            except ValueError:
                cleaned.append(np.nan)
        else:
            try:
                cleaned.append(float(val))
            except ValueError:
                cleaned.append(np.nan)
    return pd.Series(cleaned, index=series.index)

# Load your original and synthetic datasets
original_df = pd.read_csv("training/csv/dataset.csv")
synthetic_df = pd.read_csv("training/csv/validated_dataset.csv") 

# Define features to compare
features = ['HR', 'EDA', 'TEMP', 'DBP', 'SBP', 'IBI_seq', 'MAP', 'RMSSD', 'GLU']

wd_results = {}

for feature in features:
    if feature not in original_df.columns or feature not in synthetic_df.columns:
        print(f"Warning: Feature '{feature}' missing in one of the datasets, skipping.")
        continue
    
    # Get values for current feature
    orig_vals = original_df[feature].dropna()
    synth_vals = synthetic_df[feature].dropna()
    
    # Special handling for IBI_seq which contains comma-separated values
    if feature == 'IBI_seq':
        orig_vals = clean_ibi_sequence(orig_vals)
        synth_vals = clean_ibi_sequence(synth_vals)
    
    # Ensure we have numeric values
    try:
        orig_vals = pd.to_numeric(orig_vals, errors='coerce').dropna()
        synth_vals = pd.to_numeric(synth_vals, errors='coerce').dropna()
    except Exception as e:
        print(f"Error converting {feature} to numeric: {e}")
        continue
    
    # Calculate Wasserstein distance if we have valid values
    if len(orig_vals) > 0 and len(synth_vals) > 0:
        wd = wasserstein_distance(orig_vals, synth_vals)
        wd_results[feature] = wd
        print(f"Wasserstein distance for {feature}: {wd:.4f}")
    else:
        print(f"Warning: Not enough valid values for {feature}, skipping.")

# Optionally convert to DataFrame for easier visualization
wd_df = pd.DataFrame.from_dict(wd_results, orient='index', columns=['Wasserstein Distance'])
print("\nSummary of Wasserstein Distances per feature:")
print(wd_df)