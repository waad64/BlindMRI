import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

# Load data
df = pd.read_csv("Noise Bursts/csv/nbm_generated_by_llm.csv")

nbm = df['NBM']

print(f"Total samples: {len(nbm)}")
print(f"NBM basic stats:\n{nbm.describe()}")

# Histogram
plt.figure(figsize=(10,6))
sns.histplot(nbm, bins=30, kde=True, color='orchid')
plt.title("Histogram & KDE of NBM values")
plt.xlabel("Noise Bursts per Minute (NBM)")
plt.ylabel("Frequency")
plt.show()

# Boxplot
plt.figure(figsize=(6,4))
sns.boxplot(x=nbm, color='peachpuff')
plt.title("Boxplot of NBM values")
plt.xlabel("Noise Bursts per Minute (NBM)")
plt.show()

# Q-Q Plot against Normal Distribution
plt.figure(figsize=(6,6))
stats.probplot(nbm, dist="norm", plot=plt)
plt.title("Q-Q Plot of NBM values vs Normal Distribution")
plt.show()

# K-S Test against Uniform distribution on [0, 12] (based on LLM prompt constraints)
low, high = 0, 12
ks_stat, ks_pvalue = stats.kstest(nbm, 'uniform', args=(low, high-low))
print(f"K-S test against Uniform({low},{high}): stat={ks_stat:.4f}, p-value={ks_pvalue:.4f}")

# Anderson-Darling Test for Normality
ad_result = stats.anderson(nbm, dist='norm')
print(f"Anderson-Darling test stat: {ad_result.statistic:.4f}")
for i, crit in enumerate(ad_result.critical_values):
    sig = ad_result.significance_level[i]
    print(f"  {sig}% critical value: {crit:.4f} {'Reject normality' if ad_result.statistic > crit else 'Fail to reject'}")

# Outlier detection using IQR
Q1 = nbm.quantile(0.25)
Q3 = nbm.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR
outliers = nbm[(nbm < lower_bound) | (nbm > upper_bound)]

print(f"Detected {len(outliers)} outliers using IQR method.")
print("Outlier values (first 10):")
print(outliers.head(10))

# Optional: visualize outliers on boxplot
plt.figure(figsize=(6,4))
sns.boxplot(x=nbm, color='lightsteelblue')
plt.scatter(outliers, [1]*len(outliers), color='red', alpha=0.6, label='Outliers')
plt.title("Boxplot of NBM with Outliers Highlighted")
plt.xlabel("Noise Bursts per Minute (NBM)")
plt.legend()
plt.show()
