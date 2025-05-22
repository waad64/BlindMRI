import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.graphics.gofplots import qqplot


file_path = 'outputs/raw_dataset_with_rmssd.csv'
df = pd.read_csv(file_path)

# Check column names to find RMSSD column
print("Columns available:", df.columns.tolist())

if 'HRV' not in df.columns:
    raise ValueError("Column 'HRV' not found in CSV")

rmssd_array = df['HRV'].dropna().values  # drop NaNs just in case


# KDE Plot + Histogram

plt.figure(figsize=(12, 5))
sns.histplot(rmssd_array, kde=True, bins=40, color='coral')
plt.title("HRV Distribution with KDE")
plt.xlabel("HRV (ms)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()


# Boxplot

plt.figure(figsize=(10, 1.5))
sns.boxplot(x=rmssd_array, color='lightblue')
plt.title("Boxplot of HRV")
plt.tight_layout()
plt.show()


# Q-Q Plot

plt.figure(figsize=(6, 6))
qqplot(rmssd_array, line='s', markerfacecolor='green', alpha=0.5)
plt.title("Q-Q Plot of HRV")
plt.grid(True)
plt.tight_layout()
plt.show()


# Outlier Detection (z-score)

z_scores = np.abs(stats.zscore(rmssd_array))
outliers = rmssd_array[z_scores > 3]

print(f"🔍 Detected {len(outliers)} outliers (|z| > 3) out of {len(rmssd_array)} RMSSD values")


# Anderson-Darling Test (normality)

ad_result = stats.anderson(rmssd_array, dist='norm')
print("\n📈 Anderson-Darling Test for HRV:")
print(f"Statistic: {ad_result.statistic:.4f}")
for cv, sig in zip(ad_result.critical_values, ad_result.significance_level):
    print(f"  At {sig}% significance level: critical value = {cv}")


# Kolmogorov-Smirnov Test (normality)

rmssd_norm = (rmssd_array - np.mean(rmssd_array)) / np.std(rmssd_array)
ks_stat, ks_pval = stats.kstest(rmssd_norm, 'norm')

print("\n📊 Kolmogorov-Smirnov Test for HRV:")
print(f"Statistic: {ks_stat:.5f}, p-value: {ks_pval:.5f}")
