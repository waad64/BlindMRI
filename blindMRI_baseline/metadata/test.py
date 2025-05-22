import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# Load and clean data
df = pd.read_csv('outputs/age_gender_data.csv')
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
age_data = df['patient_age']


plt.figure(figsize=(16, 12))

# 1. Histogram + KDE
plt.subplot(2, 2, 1)
sns.histplot(age_data, bins=10, kde=True, color='skyblue')
plt.title("Age Distribution (Histogram + KDE)")
plt.xlabel("Patient Age")
plt.ylabel("Frequency")

# 2. KDE Only
plt.subplot(2, 2, 2)
sns.kdeplot(age_data, fill=True, color='orchid')
plt.title("KDE of Patient Age")
plt.xlabel("Patient Age")
plt.ylabel("Density")

# 3. Boxplot
plt.subplot(2, 2, 3)
sns.boxplot(x=age_data, color='lightgreen')
plt.title("Boxplot of Patient Age")

# 4. Q-Q Plot
plt.subplot(2, 2, 4)
sm.qqplot(age_data, line='s', ax=plt.gca())
plt.title("Q-Q Plot of Patient Age")

plt.tight_layout()
plt.show()

# ---  Normality Tests ---

# Anderson-Darling Test
ad_result = stats.anderson(age_data)

# Kolmogorov-Smirnov Test
ks_stat, ks_p = stats.kstest(age_data, 'norm', args=(age_data.mean(), age_data.std()))

# ---  Test Results ---
print("\n📊 Normality Test Results:")
print(f"• Anderson-Darling Statistic: {ad_result.statistic:.4f}")
for sl, cv in zip(ad_result.significance_level, ad_result.critical_values):
    print(f"  At {sl}% significance level: Critical Value = {cv:.4f}")

print(f"\n• Kolmogorov-Smirnov Statistic: {ks_stat:.4f}")
print(f"  p-value: {ks_p:.4f}")

if ks_p < 0.05:
    print("❌ The age data is likely NOT normally distributed (reject H₀).")
else:
    print("✅ The age data may be normally distributed (fail to reject H₀).")
# KDE by gender
sns.kdeplot(data=df, x='patient_age', hue='patient_gender', fill=True)
plt.title("Age Distribution by Gender (KDE)")
plt.show()
