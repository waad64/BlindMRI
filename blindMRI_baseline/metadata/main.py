import pandas as pd

# Load dataset
data = pd.read_csv('metadata/csv/Data_Entry_2017.csv')

# Keep only relevant columns and drop missing values
df = data[['Patient Age', 'Patient Gender']].dropna()

# Filter age range: 6 to 70
df = df[df['Patient Age'].between(6, 70)]

# Encode gender: 1 = Female, 0 = Male
df['Patient Gender'] = df['Patient Gender'].map({'F': 1, 'M': 0})

# Define age bins
bins = [6, 15, 25, 35, 45, 55, 60, 65, 70]  
df['age_group'] = pd.cut(df['Patient Age'], bins=bins)

# Determine number of samples per bin to balance dataset
samples_per_bin = 1000 // len(df['age_group'].unique())  

# Sample balanced dataset
balanced_samples = df.groupby('age_group').sample(n=samples_per_bin, random_state=42)

# Drop the age_group column and reset index
balanced_samples = balanced_samples.drop(columns=['age_group']).reset_index(drop=True)

# Save to CSV
balanced_samples.to_csv('outputs/age_gender_data.csv', index=False)


