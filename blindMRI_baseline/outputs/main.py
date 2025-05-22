import pandas as pd
import glob

# Step 1: Load all CSVs from the 'outputs' folder
csv_files = sorted(glob.glob('outputs/*.csv'))

# Step 2: Read each CSV into a DataFrame
dfs = [pd.read_csv(file) for file in csv_files]

# Step 3: Concatenate all DataFrames horizontally (column-wise)
combined_df = pd.concat(dfs, axis=1)

# Optional: Drop any duplicate columns (e.g., repeated index columns)
combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

# Step 4: Save the merged dataset
combined_df.to_csv('outputs/raw_dataset.csv', index=False)
