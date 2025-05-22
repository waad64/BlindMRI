import pandas as pd

# Step 1: Load the merged dataset
df = pd.read_csv('outputs/raw_dataset.csv')


if 'SBP' in df.columns and 'DPB' in df.columns:
    # Step 2: Compute MAP
    df['MAP'] = df['DPB'] + (df['SBP'] - df['DPB']) / 3
else:
    raise ValueError("SBP and/or DPB columns not found in the dataset.")

# Step 3: Save the updated dataset
df.to_csv('outputs/raw_dataset_with_map.csv', index=False)
