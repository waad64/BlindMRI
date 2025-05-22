import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.graphics.gofplots import qqplot


file_path = 'outputs/ibi_sequences.csv'
df = pd.read_csv(file_path, header=None, names=['IBI_seq'], skiprows=1)  


# Flatten all sequences into one big array of floats

ibi_values = []
for row in df['IBI_seq']:
    clean_row = row.strip('"') 
    values = [float(x) for x in clean_row.split(',')]
    ibi_values.extend(values)

ibi_array = np.array(ibi_values)


# KDE Plot + Histogram

plt.figure(figsize=(12, 5))
sns.histplot(ibi_array, kde=True, bins=40, color='skyblue')
plt.title("IBI Distribution with KDE")
plt.xlabel("IBI (seconds)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()


# Boxplot

plt.figure(figsize=(10, 1.5))
sns.boxplot(x=ibi_array, color='lightgreen')
plt.title("Boxplot of IBI")
plt.tight_layout()
plt.show()


# Q-Q Plot

plt.figure(figsize=(6, 6))
qqplot(ibi_array, line='s', markerfacecolor='blue', alpha=0.5)
plt.title("Q-Q Plot of IBI")
plt.grid(True)
plt.tight_layout()
plt.show()


# Outlier Detection using z-scores

z_scores = np.abs(stats.zscore(ibi_array))
outliers = ibi_array[z_scores > 3]

print(f"🔍 Detected {len(outliers)} outliers (|z| > 3) out of {len(ibi_array)} values")


# Anderson-Darling Test (normality)

ad_result = stats.anderson(ibi_array, dist='norm')
print("\n📈 Anderson-Darling Test:")
print(f"Statistic: {ad_result.statistic}")
for cv, sig in zip(ad_result.critical_values, ad_result.significance_level):
    print(f"  At {sig}% significance level: critical value = {cv}")


# Kolmogorov-Smirnov Test (normality)

# Normalize data for K-S test 
ibi_norm = (ibi_array - np.mean(ibi_array)) / np.std(ibi_array)
ks_stat, ks_pval = stats.kstest(ibi_norm, 'norm')

print("\n📊 Kolmogorov-Smirnov Test:")
print(f"Statistic: {ks_stat:.5f}, p-value: {ks_pval:.5f}")
