import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.graphics.gofplots import qqplot

# Charger les données
df = pd.read_csv("First_MRI_Experience/csv/first_mri_experience_generated_by_llm.csv")  
anxiety = df['First_MRI_Experience'].dropna()

# Set the style
sns.set(style="whitegrid")

# Histogram + KDE
plt.figure(figsize=(12, 6))
sns.histplot(anxiety, kde=True, bins=30, color="skyblue")
plt.title("Histogram + KDE of First_MRI_Experience")
plt.xlabel("First_MRI_Experience")
plt.ylabel("Frequency")
plt.show()

# Boxplot
plt.figure(figsize=(6, 4))
sns.boxplot(x=anxiety, color="lightgreen")
plt.title("Boxplot of First_MRI_Experience")
plt.xlabel("First_MRI_Experience")
plt.show()

# Q-Q Plot
plt.figure(figsize=(6, 6))
qqplot(anxiety, line='s')
plt.title("Q-Q Plot of First_MRI_Experience")
plt.show()

# Kolmogorov-Smirnov Test (against normal distribution)
ks_stat, ks_p = stats.kstest(
    (anxiety - anxiety.mean()) / anxiety.std(), 'norm')
print(f"\n📊 Kolmogorov-Smirnov Test:\nStatistic: {ks_stat:.4f}, p-value: {ks_p:.4f}")

# Anderson-Darling Test
ad_result = stats.anderson(anxiety, dist='norm')
print(f"\n📈 Anderson-Darling Test:\nStatistic: {ad_result.statistic:.4f}")
for i in range(len(ad_result.critical_values)):
    sl, cv = ad_result.significance_level[i], ad_result.critical_values[i]
    result = "Reject H0" if ad_result.statistic > cv else "Fail to Reject H0"
    print(f"Significance Level {sl}%: Critical Value = {cv:.3f} --> {result}")
