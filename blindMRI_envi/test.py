import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

# Load data
df = pd.read_csv("Peak_dB_Level/csv/pdb_generated_by_llm.csv")

pdb = df['PDB']

print(f"Total samples: {len(pdb)}")
print(f"PDB basic stats:\n{pdb.describe()}")

# Histogram
plt.figure(figsize=(10,6))
sns.histplot(pdb, bins=30, kde=True, color='skyblue')
plt.title("Histogram & KDE of PDB values")
plt.xlabel("Peak Decibel Level (dB)")
plt.ylabel("Frequency")
plt.show()

# Boxplot
plt.figure(figsize=(6,4))
sns.boxplot(x=pdb, color='lightgreen')
plt.title("Boxplot of PDB values")
plt.xlabel("Peak Decibel Level (dB)")
plt.show()

# Q-Q Plot against Normal Distribution
plt.figure(figsize=(6,6))
stats.probplot(pdb, dist="norm", plot=plt)
plt.title("Q-Q Plot of PDB values vs Normal Distribution")
plt.show()

# K-S Test against Uniform distribution on 50-100 dB (adjust as needed)
low, high = 50, 100
ks_stat, ks_pvalue = stats.kstest(pdb, 'uniform', args=(low, high-low))
print(f"K-S test against Uniform({low},{high}): stat={ks_stat:.4f}, p-value={ks_pvalue:.4f}")

# Anderson-Darling Test for Normality
ad_result = stats.anderson(pdb, dist='norm')
print(f"Anderson-Darling test stat: {ad_result.statistic:.4f}")
for i, crit in enumerate(ad_result.critical_values):
    sig = ad_result.significance_level[i]
    print(f"  {sig}% critical value: {crit:.4f} {'Reject normality' if ad_result.statistic > crit else 'Fail to reject'}")

# Outlier detection using IQR
Q1 = pdb.quantile(0.25)
Q3 = pdb.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR
outliers = pdb[(pdb < lower_bound) | (pdb > upper_bound)]

print(f"Detected {len(outliers)} outliers using IQR method.")
print("Outlier values (first 10):")
print(outliers.head(10))

# Optional: visualize outliers on boxplot
plt.figure(figsize=(6,4))
sns.boxplot(x=pdb, color='lightcoral')
plt.scatter(outliers, [1]*len(outliers), color='blue', alpha=0.6, label='Outliers')
plt.title("Boxplot with Outliers Highlighted")
plt.xlabel("Peak Decibel Level (dB)")
plt.legend()
plt.show()




